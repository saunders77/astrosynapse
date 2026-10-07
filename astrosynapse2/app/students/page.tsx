"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import TreeFlowchart, { type StudentTree } from "./tree-flowchart";
import MiniAstroInspector, { type MiniAstroResult } from "./miniastro-inspector";

const API = "http://127.0.0.1:8765/api";
type Model = { id: string; label: string; run_name?: string; playable: boolean; is_champion: boolean; generation?: number };
type Metrics = { turns: number; covered_turns: number; coverage: number | null; accuracy: number | null; all_turn_accuracy: number | null; purchase_accuracy: number | null; none_accuracy: number | null; always_none_accuracy: number | null };
type Result = Partial<MiniAstroResult> & { student_type?: "tree" | "miniastro"; node_count?: number; tree?: StudentTree; tree_text?: string; metrics: Record<string, Metrics>; limitations: string; sampling: string; trade_definition: string; counts: Record<string, number> };
type Job = { id: string; model_id: string; model_label: string; created_at: number; status: string; progress: number; games_completed: number; config: { games: number; max_nodes: number; rules_version: number; student_type?: "tree" | "miniastro"; parameter_budget?: number; epochs?: number; source_student_id?: string }; epoch?: number; parameter_count?: number; validation_accuracy?: number; counts?: Record<string, number>; error?: string; result?: Result; summary?: Pick<Result, "node_count" | "parameter_count" | "metrics"> };
const active = (j: Job) => ["queued", "collecting", "preparing", "fitting", "evaluating"].includes(j.status);
const pct = (n: number | null | undefined) => n == null ? "—" : `${(n * 100).toFixed(1)}%`;
async function api(path: string, init?: RequestInit) {
  const response = await fetch(`${API}${path}`, { cache: "no-store", ...init });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
  return body;
}

export default function AcquireStudentsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [games, setGames] = useState(10000);
  const [studentType, setStudentType] = useState<"tree" | "miniastro">("miniastro");
  const [budget, setBudget] = useState(10000);
  const [epochs, setEpochs] = useState(40);
  const [source, setSource] = useState("");
  const [nodes, setNodes] = useState(42);
  const [seed, setSeed] = useState(20260924);
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
      const champion = [...available].filter(m => m.is_champion).sort((a, b) => (b.generation ?? 0) - (a.generation ?? 0))[0];
      setModel(champion?.id ?? available[0]?.id ?? "");
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
        const list = await api("/acquire-students") as Job[];
        const id = selected || list[0]?.id;
        const current = id ? await api(`/acquire-students/${id}`) as Job : null;
        if (!stopped) { setJobs(list); setDetail(current); }
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
      const job = await api("/acquire-students", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: model, games, max_nodes: nodes, seed, rules_version: rules, student_type: studentType, parameter_budget: budget, epochs, source_student_id: source || null }) });
      setJobs(current => [job, ...current]); setSelected(job.id); setDetail(job);
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  async function cancel() {
    if (!detail) return;
    setBusy(true);
    try { await api(`/acquire-students/${detail.id}/cancel`, { method: "POST" }); }
    catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  const running = jobs.some(active);
  const result = detail?.result;
  return <main className="progressive-page students-page">
    <header><Link href="/">← Control center & Play</Link><Link href="/progressive">Astro6 progress ↗</Link><Link href="/acquisition-impact">Acquisition impact ↗</Link></header>
    <div className="progressive-heading"><div><p className="progressive-kicker">ASTRO6 / UNDERSTANDING</p><h1>Acquire students</h1><p>Learn the checkpoint’s next acquisition with an inspectable student.</p></div></div>
    {error ? <p role="alert" className="progressive-warning">{error}</p> : null}
    <section className="progressive-panel">
      <h2>Train a student</h2>
      <form onSubmit={e => { e.preventDefault(); void train(); }}>
        <div className="student-form">
          <label className="student-model-field"><span>Teacher checkpoint</span><select value={model} onChange={e => { setModel(e.target.value); setSource(""); }} required disabled={!models.length}>{models.map(m => <option key={m.id} value={m.id}>{m.label}{m.is_champion ? " · Champion" : ""}</option>)}</select></label>
          <label><span>Student architecture</span><select value={studentType} onChange={e => setStudentType(e.target.value as "tree" | "miniastro")}><option value="miniastro">MiniAstro · small additive network</option><option value="tree">Decision tree</option></select></label>
          <label className="student-model-field"><span>Training positions</span><select value={source} onChange={e => { setSource(e.target.value); const saved = jobs.find(j => j.id === e.target.value); if (saved) { setGames(saved.config.games); setRules(saved.config.rules_version); } }}><option value="">Generate new real games</option>{jobs.filter(j => j.status === "complete" && j.model_id === model).map(j => <option key={j.id} value={j.id}>Reuse {j.config.games.toLocaleString()} games · {j.id.slice(0, 6)} · rules v{j.config.rules_version}</option>)}</select></label>
          <label><span>Self-play games</span><input type="number" min="20" max="10000" required disabled={!!source} value={games} onChange={e => setGames(Number(e.target.value))} /></label>
          {studentType === "tree" ? <label><span>Maximum tree nodes</span><input type="number" min="3" max="100" required value={nodes} onChange={e => setNodes(Number(e.target.value))} /></label> : <><label><span>Parameter budget</span><select value={budget} onChange={e => setBudget(Number(e.target.value))}><option value={1000}>1,000 · smallest baseline</option><option value={10000}>10,000 · compact</option><option value={100000}>100,000 · more capacity</option></select></label><label><span>Maximum epochs</span><input type="number" min="1" max="200" value={epochs} onChange={e => setEpochs(Number(e.target.value))} /></label></>}
          <label><span>Random seed</span><input type="number" min="0" max="9007199254740991" required value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
          <label><span>Game rules</span><select disabled={!!source} value={rules} onChange={e => setRules(Number(e.target.value))}><option value={2}>v2 · matches Play</option><option value={1}>v1 · historical Astro6 training</option></select></label>
        </div>
        <p>One uniformly sampled decision moment per player turn, including forced choices. The target is the next card actually acquired that turn, or None. Multi-purchase turns are included.</p>
        {studentType === "miniastro" ? <p>MiniAstro sums five shallow subnetworks and a linear candidate term. Inspect every unit’s weights and each recommendation’s exact score contributions. Whole games are split 70% / 15% / 15%; validation selects the epoch, and the test set measures final agreement. A small model does not guarantee 80–90% fidelity.</p> : <p>One tree scores the current market, Explorer, and None. Validation chooses the tree size within your node cap; test games stay separate.</p>}
        <p>Full-turn trade is an explicit hindsight input, including later draws and abilities. In Play, enter that total or your estimate. Students imitate the teacher; their accuracy does not measure playing strength.</p>
        <button type="submit" className="button button-primary" disabled={busy || running || !model}>{running ? "A student is training" : busy ? "Starting…" : "Train acquire student"}</button>
      </form>
    </section>
    <div className="student-workspace">
      <section className="progressive-panel student-history"><h2>Students</h2>
        {!jobs.length ? <p>No students yet. Choose a checkpoint and start training.</p> : jobs.map(j => <button type="button" className={`student-history-item${detail?.id === j.id ? " selected" : ""}`} key={j.id} onClick={() => setSelected(j.id)}>
          <strong>{j.config.student_type === "miniastro" ? `MiniAstro ${j.config.parameter_budget?.toLocaleString()} · ` : "Tree · "}{j.model_label}</strong><span>{j.id.slice(0, 6)} · {j.status} · {j.games_completed.toLocaleString()} / {j.config.games.toLocaleString()} games</span>
          {j.summary ? <small>{j.summary.parameter_count ? `${j.summary.parameter_count.toLocaleString()} parameters` : `${j.summary.node_count} nodes`} · {pct(j.summary.metrics.test.all_turn_accuracy)} test agreement</small> : null}
        </button>)}
      </section>
      <section className="progressive-panel student-detail" aria-live="polite">
        <h2>{detail ? `${detail.model_label} · ${detail.id.slice(0, 6)}` : "Training progress & model"}</h2>
        {detail ? <>
          <div className="student-progress"><strong>{detail.status} · {pct(detail.progress)}</strong><progress max="1" value={detail.progress} /><span>{detail.epoch ? `Epoch ${detail.epoch} / ${detail.config.epochs} · ${detail.parameter_count?.toLocaleString()} parameters · Validation ${pct(detail.validation_accuracy)} · ` : ""}{detail.games_completed.toLocaleString()} / {detail.config.games.toLocaleString()} games · {(detail.counts?.turns ?? 0).toLocaleString()} sampled turns · rules v{detail.config.rules_version}</span></div>
          {active(detail) ? <><p>You can close this page. Training and saved progress continue independently.</p><button type="button" className="button" disabled={busy} onClick={() => void cancel()}>Cancel training</button></> : null}
          {detail.error ? <p role="alert" className="progressive-warning">{detail.error}</p> : null}
          {result ? <>
            <div className="student-metrics"><div><span>{result.student_type === "miniastro" ? "Trainable parameters" : "Total tree nodes"}</span><strong>{result.parameter_count?.toLocaleString() ?? result.node_count}</strong></div><div><span>Test agreement · all completed turns</span><strong>{pct(result.metrics.test.all_turn_accuracy)}</strong></div><div><span>Test purchase agreement · covered targets</span><strong>{pct(result.metrics.test.purchase_accuracy)}</strong></div><div><span>Always-None baseline · covered turns</span><strong>{pct(result.metrics.test.always_none_accuracy)}</strong></div></div>
            <div className="student-table-scroll"><table><thead><tr><th>Split</th><th>Turns</th><th>Candidate coverage</th><th>Covered agreement</th><th>None agreement</th></tr></thead><tbody>{Object.entries(result.metrics).map(([name, m]) => <tr key={name}><td>{name}</td><td>{m.turns.toLocaleString()}</td><td>{pct(m.coverage)}</td><td>{pct(m.accuracy)}</td><td>{pct(m.none_accuracy)}</td></tr>)}</tbody></table></div>
            <p>{result.limitations}</p><p>{result.counts.future_market_card} sampled targets appeared only after a market replacement; {result.counts.unfinished_turn} unfinished turns were censored.</p>
            {result.student_type === "miniastro" ? <MiniAstroInspector key={detail.id} result={result as MiniAstroResult} studentId={detail.id} apiBase={API} /> : <>
            <h2>The complete tree · {result.node_count} nodes</h2>
            {result.tree ? <TreeFlowchart key={detail.id} tree={result.tree} /> : null}
            <details className="student-tree-text"><summary>English tree and selection rules</summary><pre className="student-tree">{result.tree_text}</pre></details></>}
            <div className="student-downloads">{result.student_type === "miniastro" ? <><a href={`${API}/acquire-students/${detail.id}/download/weights`}>Model weights</a><a href={`${API}/acquire-students/${detail.id}/download/errors`}>Held-out errors</a><a href={`${API}/acquire-students/${detail.id}/download/report`}>Architecture report</a></> : <a href={`${API}/acquire-students/${detail.id}/download/tree`}>Download English tree</a>}<a href={`${API}/acquire-students/${detail.id}/download/student`}>Student JSON</a><a href={`${API}/acquire-students/${detail.id}/download/samples`}>Sampled positions</a></div>
          </> : null}
        </> : <p>Choose a saved student to inspect its results, or train your first one.</p>}
      </section>
    </div>
  </main>;
}
