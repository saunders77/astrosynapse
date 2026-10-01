"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8765/api";
type Model = { id: string; label: string; playable: boolean; is_champion: boolean; generation?: number };
type Bin = { lower: number; upper: number; positions: number; predicted: number | null; observed: number | null };
type Metrics = { positions: number; brier: number; log_loss: number; predicted: number; observed: number; bins: Bin[] };
type Comparison = { critic: Metrics | null; original: Metrics | null };
type Evaluation = Comparison & { games: number; slices: Record<string, Comparison> };
type History = { epoch: number; train_loss: number; validation_loss: number; validation_brier: number };
type Job = { id: string; model_label: string; status: string; progress: number; games_completed: number; epoch: number; best_epoch?: number; parameters?: number; truncated_games?: number; pause_requested?: boolean; config: { games: number; epochs: number; hidden_size: number; rules_version: number }; history: History[]; error?: string; validation?: Evaluation; result?: { test: Evaluation; validation: Evaluation; best_epoch: number; scope: string } };
const active = (j: Job) => ["queued", "collecting", "fitting", "evaluating"].includes(j.status);
const pct = (x: number | null | undefined) => x == null ? "—" : `${(100 * x).toFixed(1)}%`;
const decimal = (x: number | null | undefined) => x == null ? "—" : x.toFixed(4);
async function api(path: string, init?: RequestInit) {
  const response = await fetch(`${API}${path}`, { cache: "no-store", ...init });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
  return body;
}

export default function CriticsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [games, setGames] = useState(1000);
  const [epochs, setEpochs] = useState(20);
  const [width, setWidth] = useState(128);
  const [seed, setSeed] = useState(20261001);
  const [rules, setRules] = useState(2);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let stopped = false;
    api("/models").then((all: Model[]) => {
      if (stopped) return;
      const available = all.filter(m => m.playable);
      setModels(available);
      setModel([...available].filter(m => m.is_champion).sort((a,b) => (b.generation ?? 0) - (a.generation ?? 0))[0]?.id ?? available[0]?.id ?? "");
    }).catch(e => { if (!stopped) setError(String(e)); });
    return () => { stopped = true; };
  }, []);
  useEffect(() => {
    let stopped = false;
    let pending = false;
    async function refresh() {
      if (pending) return;
      pending = true;
      try {
        const list = await api("/critics") as Job[];
        if (!stopped) { setJobs(list); setDetail(list.find(j => j.id === selected) ?? list[0] ?? null); }
      } catch (e) { if (!stopped) setError(String(e)); }
      finally { pending = false; }
    }
    void refresh();
    const timer = setInterval(refresh, 2500);
    return () => { stopped = true; clearInterval(timer); };
  }, [selected]);
  async function train() {
    setBusy(true); setError("");
    try {
      const job = await api("/critics", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: model, games, epochs, hidden_size: width, seed, rules_version: rules }) }) as Job;
      setJobs(current => [job, ...current]); setSelected(job.id); setDetail(job);
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  async function control(action: string) {
    if (!detail) return;
    setBusy(true); setError("");
    try { const job = await api(`/critics/${detail.id}/${action}`, { method: "POST" }) as Job; setDetail(job); setJobs(current => current.map(j => j.id === job.id ? job : j)); }
    catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  const running = jobs.some(active);
  const evaluation = detail?.result?.test ?? detail?.validation;
  return <main className="progressive-page students-page">
    <header><Link href="/">← Control center & Play</Link><Link href="/progressive">Astro6 progress ↗</Link></header>
    <div className="progressive-heading"><div><p className="progressive-kicker">ASTROSYNAPSE 2 / WIN PROBABILITY</p><h1>Independent critic</h1><p>Train a separate win-probability model while the playing policy stays frozen.</p></div></div>
    {error ? <p role="alert" className="progressive-warning">{error}</p> : null}
    <section className="progressive-panel"><h2>Train a critic</h2>
      <form onSubmit={e => { e.preventDefault(); void train(); }}>
        <div className="student-form">
          <label className="student-model-field"><span>Playing policy</span><select required value={model} disabled={!models.length} onChange={e => setModel(e.target.value)}>{models.map(m => <option key={m.id} value={m.id}>{m.label}{m.is_champion ? " · Champion" : ""}</option>)}</select></label>
          <label><span>Self-play games</span><input required type="number" min="20" max="10000" value={games} onChange={e => setGames(Number(e.target.value))} /></label>
          <label><span>Training epochs</span><input required type="number" min="1" max="200" value={epochs} onChange={e => setEpochs(Number(e.target.value))} /></label>
          <label><span>Hidden width</span><select value={width} onChange={e => setWidth(Number(e.target.value))}>{[32,64,128,256].map(w => <option key={w} value={w}>{w}</option>)}</select></label>
          <label><span>Random seed</span><input required type="number" min="0" max="9007199254740991" value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
          <label><span>Game rules</span><select value={rules} onChange={e => setRules(Number(e.target.value))}><option value={2}>v2 · matches Play</option><option value={1}>v1 · historical campaign</option></select></label>
        </div>
        <p>A new network learns from public positions and final game outcomes. Each game contributes up to 64 uniformly sampled decisions, including forced moves. Whole games are split 70% for training, 15% for checkpoint selection, and 15% for final testing.</p>
        <p>One CPU worker collects games and trains two nonlinear hidden layers. Truncated games are excluded. The final critic is saved separately for review and download; training does not replace the probability display in Play.</p>
        <button type="submit" className="button button-primary" disabled={busy || running || !model}>{running ? "A critic is training" : "Start critic training"}</button>
      </form>
    </section>
    <div className="student-workspace">
      <section className="progressive-panel student-history"><h2>Training runs</h2>
        {!jobs.length ? <p>No critics yet. Start a run to measure calibration against the original model.</p> : jobs.map(j => <button type="button" key={j.id} className={`student-history-item${detail?.id === j.id ? " selected" : ""}`} onClick={() => setSelected(j.id)}><strong>{j.model_label}</strong><span>{j.id.slice(0,8)} · {j.status}</span><small>{j.games_completed.toLocaleString()} / {j.config.games.toLocaleString()} games · epoch {j.epoch} / {j.config.epochs}</small></button>)}
      </section>
      <section className="progressive-panel student-detail"><h2>Progress & calibration</h2>
        {!detail ? <p>Choose a saved run to inspect its progress and results.</p> : <>
          <div className="student-progress" aria-live="polite"><strong>{detail.status} · {pct(detail.progress)}</strong><progress max="1" value={detail.progress} /><span>{detail.games_completed.toLocaleString()} / {detail.config.games.toLocaleString()} games · epoch {detail.epoch} / {detail.config.epochs} · rules v{detail.config.rules_version}</span></div>
          <p>{detail.parameters?.toLocaleString() ?? "—"} critic parameters · {detail.truncated_games ?? 0} truncated games excluded{detail.best_epoch ? ` · selected epoch ${detail.best_epoch}` : ""}</p>
          {active(detail) ? <><p>You can close this page; the worker continues independently. Pausing preserves finished games and epochs. An unfinished epoch restarts on resume.</p><button className="button" disabled={busy || detail.pause_requested} onClick={() => void control("pause")}>{detail.pause_requested ? "Pause requested…" : "Pause training"}</button></> : null}
          {["paused", "interrupted", "failed"].includes(detail.status) ? <button className="button button-primary" disabled={busy || running} onClick={() => void control("resume")}>Resume training</button> : null}
          {detail.error ? <p role="alert" className="progressive-warning">{detail.error}</p> : null}
          {evaluation ? <>
            <h3>{detail.result ? "Final held-out test" : "Validation · used to select the checkpoint"}</h3>
            <p>{evaluation.games} completed games. Lower Brier error and log loss are better. Results describe this policy’s self-play, not arbitrary human opponents. Positions within a game are correlated; small differences need more independent games.</p>
            <div className="student-table-scroll"><table><thead><tr><th>Predictor</th><th>Brier error</th><th>Log loss</th><th>Mean prediction</th><th>Actual wins</th></tr></thead><tbody>{(["critic", "original"] as const).map(key => <tr key={key}><td>{key === "critic" ? "Independent critic" : "Original critic"}</td><td>{decimal(evaluation[key]?.brier)}</td><td>{decimal(evaluation[key]?.log_loss)}</td><td>{pct(evaluation[key]?.predicted)}</td><td>{pct(evaluation[key]?.observed)}</td></tr>)}</tbody></table></div>
            <h3>Calibration by predicted probability</h3><p>A calibrated 70% prediction should win about 70% of the time.</p>
            <div className="student-table-scroll"><table><thead><tr><th>Probability range</th><th>Positions</th><th>Mean prediction</th><th>Actual wins</th></tr></thead><tbody>{evaluation.critic?.bins.map(b => <tr key={b.lower}><td>{pct(b.lower)}–{pct(b.upper)}</td><td>{b.positions}</td><td>{pct(b.predicted)}</td><td>{pct(b.observed)}</td></tr>)}</tbody></table></div>
            <details><summary>Forced moves, game stage & decision families</summary><div className="student-table-scroll"><table><thead><tr><th>Slice</th><th>Positions</th><th>New Brier</th><th>Original Brier</th></tr></thead><tbody>{Object.entries(evaluation.slices).map(([name,m]) => <tr key={name}><td>{name}</td><td>{m.critic?.positions ?? 0}</td><td>{decimal(m.critic?.brier)}</td><td>{decimal(m.original?.brier)}</td></tr>)}</tbody></table></div></details>
          </> : <p>Calibration measurements appear after the first training epoch.</p>}
          {detail.history.length ? <details open><summary>Learning history</summary><div className="student-table-scroll critic-history"><table><thead><tr><th>Epoch</th><th>Training loss</th><th>Validation loss</th><th>Validation Brier</th></tr></thead><tbody>{detail.history.map(h => <tr key={h.epoch}><td>{h.epoch}</td><td>{decimal(h.train_loss)}</td><td>{decimal(h.validation_loss)}</td><td>{decimal(h.validation_brier)}</td></tr>)}</tbody></table></div></details> : null}
          <div className="student-downloads">{detail.result ? <><a href={`${API}/critics/${detail.id}/download/critic`}>Download critic</a><a href={`${API}/critics/${detail.id}/download/report`}>Calibration report</a></> : null}<a href={`${API}/critics/${detail.id}/download/log`}>Worker log</a></div>
        </>}
      </section>
    </div>
  </main>;
}
