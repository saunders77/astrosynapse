// Compact CPU execution of the existing objective-v2 actor, using Accelerate.
// No rule or model approximation; the NumPy actor remains the reference.
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <vector>

using Vec = std::vector<float>;
struct Net {
    int state, action, hidden, ah, blocks, heads, families;
    float eps;
    std::vector<const float *> weights;
};

static Vec linear(const Vec &x, int rows, int in, int out,
                  const float *w, const float *b) {
    Vec y(rows * out);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out, in,
                1.f, x.data(), in, w, in, 0.f, y.data(), out);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < out; ++j) y[i*out+j] += b[j];
    return y;
}

static void norm(Vec &x, int rows, int width, const float *w, const float *b, float eps) {
    for (int i = 0; i < rows; ++i) {
        float *v = x.data() + i * width;
        float mean = 0.f, variance = 0.f;
        for (int j = 0; j < width; ++j) mean += v[j];
        mean /= width;
        for (int j = 0; j < width; ++j) { float d = v[j]-mean; variance += d*d; }
        float inv = 1.f / std::sqrt(variance / width + eps);
        for (int j = 0; j < width; ++j) v[j] = (v[j]-mean)*inv*w[j]+b[j];
    }
}

static void silu(Vec &x) {
    for (float &v : x) v /= 1.f + std::exp(-std::clamp(v, -40.f, 40.f));
}

static Vec residual(const Vec &x, int rows, int width, Net &net, int &p) {
    auto &w=net.weights;
    Vec y=x;
    norm(y,rows,width,w[p],w[p+1],net.eps);p+=2;
    y=linear(y,rows,width,2*width,w[p],w[p+1]);p+=2;silu(y);
    y=linear(y,rows,2*width,width,w[p],w[p+1]);p+=2;
    for (size_t i=0;i<y.size();++i) y[i]=(y[i]+x[i])*0.7071067811865475244f;
    return y;
}

static Vec trunk(const Vec &x, int rows, int in, int width, int blocks, Net &net, int &p) {
    auto &w=net.weights;
    Vec y=linear(x,rows,in,width,w[p],w[p+1]);p+=2;
    norm(y,rows,width,w[p],w[p+1],net.eps);p+=2;silu(y);
    for(int i=0;i<blocks;++i)y=residual(y,rows,width,net,p);
    return y;
}

extern "C" {
void *astro_create(int state,int action,int hidden,int ah,int blocks,int heads,int families,
                   float eps,const float **weights,int count) {
    return new Net{state,action,hidden,ah,blocks,heads,families,eps,
                   std::vector<const float *>(weights,weights+count)};
}
void astro_destroy(void *ptr) { delete static_cast<Net *>(ptr); }
void astro_options(void *ptr,const float *state,const float *actions,int count,int family,int head,float *out) {
    Net &net=*static_cast<Net *>(ptr);int p=0;
    Vec s=trunk(Vec(state,state+net.state),1,net.state,net.hidden,net.blocks,net,p);
    Vec a=trunk(Vec(actions,actions+count*net.action),count,net.action,net.ah,1,net,p);
    Vec combined(count*(net.hidden+net.ah));
    for(int i=0;i<count;++i) {
        std::copy(s.begin(),s.end(),combined.begin()+i*(net.hidden+net.ah));
        std::copy(a.begin()+i*net.ah,a.begin()+(i+1)*net.ah,combined.begin()+i*(net.hidden+net.ah)+net.hidden);
    }
    Vec fusion=trunk(combined,count,net.hidden+net.ah,net.hidden,net.blocks,net,p);
    for(int h=0;h<net.heads;++h) {
        if(head>=0 && head!=h){p+=8;continue;}
        Vec y=residual(fusion,count,net.hidden,net,p);
        y=linear(y,count,net.hidden,net.families,net.weights[p],net.weights[p+1]);p+=2;
        for(int i=0;i<count;++i)out[head<0?i*net.heads+h:i]=y[i*net.families+family];
    }
}
void astro_values(void *ptr,const float *states,const int *families,int count,float *out) {
    Net &net=*static_cast<Net *>(ptr);int p=0;
    Vec s=trunk(Vec(states,states+count*net.state),count,net.state,net.hidden,net.blocks,net,p);
    p=static_cast<int>(net.weights.size())-2;
    Vec y=linear(s,count,net.hidden,net.heads*net.families,net.weights[p],net.weights[p+1]);
    for(int i=0;i<count;++i)for(int h=0;h<net.heads;++h)
        out[i*net.heads+h]=y[i*net.heads*net.families+families[i]*net.heads+h];
}
}
