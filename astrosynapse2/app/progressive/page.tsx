"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Screen = { games: number; score: number; checkpoint: string; stage: number; opponent: string };
type Gate = { model?: string; actor?: string; games: number; score: number; lower: number; confidence: number; pairs: number; passed: boolean; stage: number; opponent: string };
type Progress = {
  available: boolean; name: string; run_name: string; checkpoint_name: string; champion_checkpoint_name: string; path: string; phase: string; running: boolean; stale: boolean; stop_requested: boolean;
  elapsed: number; games: number; stage: number; champion_label: string; error?: string;
  history: Screen[]; gates: Gate[]; promotions: Gate[];
  evaluation?: { pairs: number; score: number; lower?: number; log_evidence?: number; threshold?: number };
  latest?: { games: number; iteration_seconds: number; learning_rate: number; post_update_kl: number; update_rejected: boolean; rollout_score: number; positions: number };
  settings: { hours: number; games: number; round_iterations: number; max_gate_pairs: number; workers: number };
};
const API = "http://127.0.0.1:8765/api/progressive";
const pct = (x?: number) => x == null ? "—" : `${(100 * x).toFixed(2)}%`;
const number = (x?: number) => x == null ? "—" : x.toLocaleString();

export default function ProgressiveTraining() {
  const [data, setData] = useState<Progress | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function refresh() {
    try {
      const response = await fetch(API, { cache: "no-store" });
      if (!response.ok) throw new Error(`Control API returned ${response.status}`);
      setData(await response.json()); setError("");
    } catch (e) { setError(String(e)); }
  }
  useEffect(() => { const initial = setTimeout(refresh, 0); const timer = setInterval(refresh, 5000); return () => { clearTimeout(initial); clearInterval(timer); }; }, []);
  async function control(command: string) {
    setBusy(true);
    try { const response = await fetch(`${API}/${command}`, { method: "POST" }); if (!response.ok) throw new Error(await response.text()); await refresh(); }
    catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  const screens = data?.history ?? [];
  const maximum = Math.max(1, data?.games ?? 1);
  const y = (score: number) => 200 - 160 * (Math.max(.35, Math.min(.75, score)) - .35) / .4;
  const x = (games: number) => 55 + 780 * games / maximum;
  const interval = data ? data.settings.games * data.settings.round_iterations : 0;
  const remaining = data?.latest && interval ? interval - (data.latest.games % interval) : interval;
  const phase = data?.running && data.stop_requested ? "Saving and pausing" : data?.phase?.replaceAll("_", " ");
  return <main className="progressive-page">
    <header><Link href="/">← Astrosynapse 2 control center</Link><span>Refreshes every 5 seconds · Local M4</span></header>
    <div className="progressive-heading"><div><p className="progressive-kicker">ASTRO6 / SUCCESSIVE CHAMPIONS</p><h1>Progressive champion training</h1><p>Learn, verify an improvement, then train against the new champion.</p></div>
      {data?.available && <button disabled={busy || (data.running && data.stop_requested)} onClick={() => control(data.running ? "pause" : "resume")}>{data.running ? "Pause & save" : "Resume training"}</button>}
    </div>
    {error && <p role="alert" className="progressive-warning">Connection interrupted: {error}. Training runs independently of this page.</p>}
    {!data?.available ? <section className="progressive-panel">Waiting for the progressive training process…</section> : <>
      <section className="progressive-status"><span className={data.running ? "progressive-dot" : "progressive-dot idle"} /><strong>{phase}</strong><span>{data.settings.workers} actors · {(data.elapsed / 3600).toFixed(2)} / {data.settings.hours} hours</span></section>
      {data.stale && data.running && <p className="progressive-warning">The process has not updated its heartbeat recently. Saved results remain visible.</p>}
      {data.error && <p role="alert" className="progressive-warning">{data.error}</p>}
      <section className="progressive-panel"><h2>{data.run_name}</h2><dl><dt>Current checkpoint</dt><dd>{data.checkpoint_name}</dd><dt>Champion checkpoint</dt><dd>{data.champion_checkpoint_name}</dd></dl><Link href="/">Open the control center to test and compare →</Link></section>
      <section className="progressive-cards">
        <article><span>Training games</span><strong>{number(data.games)}</strong><small>{data.latest ? `${(data.settings.games / data.latest.iteration_seconds).toFixed(1)} games / second in latest iteration` : "First rollout in progress"}</small></article>
        <article><span>Verified improvements</span><strong>{data.promotions.length}</strong><small>Each clears a lower confidence bound above 50%</small></article>
        <article><span>Current champion</span><strong>{data.stage ? `Generation ${data.stage}` : "Original benchmark"}</strong><small>{data.champion_label}</small></article>
        <article><span>Latest screening score</span><strong>{pct(screens.at(-1)?.score)}</strong><small>Exploratory · versus its incumbent champion</small></article>
      </section>
      <section className="progressive-panel"><div className="progressive-panel-heading"><h2>Checkpoint progress</h2><span>Blue: screening · Green: verified promotion</span></div>
        <svg viewBox="0 0 880 235" role="img" aria-label="Checkpoint win rates against the incumbent champion over training games">
          {[.4, .5, .6, .7].map(score => <g key={score}><line x1="55" x2="835" y1={y(score)} y2={y(score)} stroke={score === .5 ? "#7f90a3" : "#243140"} strokeDasharray={score === .5 ? "6 5" : undefined} /><text x="8" y={y(score) + 4} fill="#9baec2" fontSize="12">{Math.round(score * 100)}%</text></g>)}
          <polyline fill="none" stroke="#70a9ff" strokeWidth="2" points={screens.map(s => `${x(s.games)},${y(s.score)}`).join(" ")} />
          {screens.map((s, i) => <circle key={i} cx={x(s.games)} cy={y(s.score)} r="4" fill="#70a9ff"><title>{s.checkpoint.replace(data.path + "/", "")} · {number(s.games)} games: {pct(s.score)} vs {s.opponent}</title></circle>)}
          {data.promotions.map((s, i) => <circle key={i} cx={x(s.games)} cy={y(s.score)} r="6" fill="#5fe6ca"><title>Verified: {pct(s.score)}; lower bound {pct(s.lower)}</title></circle>)}
          <text x="55" y="225" fill="#9baec2" fontSize="12">0 games</text><text x="835" y="225" textAnchor="end" fill="#9baec2" fontSize="12">{number(data.games)} games</text>
          {!screens.length && <text x="440" y="100" textAnchor="middle" fill="#bdc9d6" fontSize="15">The first checkpoint screen will appear here.</text>}
        </svg><p>Opponents advance after each verified promotion. These scores measure progress against each incumbent; they are not a single fixed-opponent strength curve.</p>
      </section>
      <div className="progressive-columns"><section className="progressive-panel"><h2>{data.phase === "training" ? "Current learning block" : "Current evaluation"}</h2>
        {data.phase === "training" ? <><p>Next screen after approximately <strong>{number(remaining)}</strong> more training games.</p><dl><dt>Learning rate</dt><dd>{data.latest?.learning_rate?.toExponential(2) ?? "—"}</dd><dt>Measured policy KL</dt><dd>{data.latest?.post_update_kl?.toFixed(5) ?? "—"}</dd><dt>Last update</dt><dd>{data.latest ? (data.latest.update_rejected ? "Restored preceding parameters" : "Accepted") : "Collecting games"}</dd></dl></> : <><p>{number(data.evaluation?.pairs)} seat-swapped pairs completed.</p><dl><dt>Observed score</dt><dd>{pct(data.evaluation?.score)}</dd><dt>Lower confidence bound</dt><dd>{pct(data.evaluation?.lower)}</dd></dl></>}
      </section><section className="progressive-panel"><h2>Promotion contract</h2><p>Fresh games against the current champion, with both seats tested. Promotion requires a time-uniform lower bound above 50%, with at least 95% confidence and adjustment for repeated candidates.</p><p>Small improvements can accumulate across checkpoints. After promotion, a separate tournament measures progress against the original champion. Reaching 70% remains a long-term benchmark.</p></section></div>
      <section className="progressive-panel"><h2>Independent verification results</h2><div className="progressive-table"><table><thead><tr><th>Checkpoint</th><th>Training games</th><th>Opponent</th><th>Evaluation games</th><th>Score</th><th>Lower bound</th><th>Confidence</th><th>Result</th></tr></thead><tbody>
        {data.gates.slice(-20).reverse().map((gate, i) => <tr key={i}><td>{(gate.model ?? gate.actor ?? "—").replace(data.path + "/", "")}</td><td>{number(gate.games)}</td><td>{gate.opponent}</td><td>{number(gate.pairs * 2)}</td><td>{pct(gate.score)}</td><td>{pct(gate.lower)}</td><td>{pct(gate.confidence)}</td><td style={{color: gate.passed ? "var(--mint)" : "var(--muted)"}}>{gate.passed ? "Promoted" : "Not proven stronger"}</td></tr>)}
        {!data.gates.length && <tr><td colSpan={8}>No independent promotion test has finished yet.</td></tr>}
      </tbody></table></div></section>
      <p className="progressive-footer">You can close this page while training continues. Pause & save finishes the current work safely. This campaign’s saved training and promotion results use historical rules v1. New tests in Models & Arena and Play use corrected rules v2.</p>
    </>}
  </main>;
}
