"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Screen = { games: number; score: number; checkpoint: string; stage: number; opponent: string; branch?: string };
type Branch = { id: string; recipe: { name: string; temperature?: number; advantage_baseline?: string }; status: string; round: number; games: number; best_score: number; reason: string; screens?: number[]; confirmation?: { score: number } };
type Gate = { model?: string; actor?: string; games: number; score: number; lower: number; confidence: number; pairs: number; passed: boolean; stage: number; opponent: string };
type Progress = {
  available: boolean; name: string; run_name: string; checkpoint_name: string; champion_checkpoint_name: string; path: string; phase: string; running: boolean; stale: boolean; stop_requested: boolean;
  elapsed: number; games: number; stage: number; champion_label: string; error?: string;
  algorithm?: string; generation?: number; accepted_steps?: number; search_games?: number; verification_games?: number; evaluated_games?: number; evaluation_counts?: Record<string, number>;
  search_summary?: { generation: number; operator: string; phase: string; candidate: number; candidates: number; pairs_per_candidate: number; accepted_steps: number; stalled_generations: number };
  mode?: string; inherited_games?: number; inherited_promotions?: number; active_branch?: string; pending_gate?: unknown;
  branches?: Branch[]; events?: { time: number; message: string }[];
  historical_data?: { sampled_positions: number; replay_positions: number; replay_shards: number; replay_sources: string[]; database: { metrics: number; checkpoints: number; arena_jobs: number } };
  history: Screen[]; gates: Gate[]; promotions: Gate[];
  evaluation?: { pairs: number; score: number; lower?: number; log_evidence?: number; threshold?: number };
  latest?: { games: number; iteration_seconds: number; learning_rate: number; post_update_kl: number; update_rejected: boolean; rollout_score: number; positions: number; history_kl?: number; advantage_baseline?: string; critic_calibration?: { all: { prediction: number; outcome: number; brier: number }; forced?: { brier: number } } };
  settings: { hours: number; games: number; round_iterations: number; max_gate_pairs: number; workers: number; max_rounds?: number; max_stalled_generations?: number };
};
const API = "http://127.0.0.1:8765/api/progressive";
const pct = (x?: number) => x == null ? "—" : `${(100 * x).toFixed(2)}%`;
const confidence = (x?: number) => x != null && x > .9999 && x < 1 ? ">99.99%" : pct(x);
const number = (x?: number) => x == null ? "—" : x.toLocaleString();

export default function ProgressiveTraining() {
  const [data, setData] = useState<Progress | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState<{ workers?: number; hours?: number; max_rounds?: number; max_stalled_generations?: number }>({});
  const [notice, setNotice] = useState("");
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
    try { const response = await fetch(`${API}/${command}`, { method: "POST" }); if (!response.ok) throw new Error(await response.text()); setNotice(command === "skip" ? "Branch will stop after the current safe checkpoint." : command === "pause" ? "Pause requested. The current batch will finish and save." : "Supervisor starting."); await refresh(); }
    catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  async function saveSettings() {
    if (!data) return;
    setBusy(true);
    try {
      const response = await fetch(`${API}/settings`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ workers: draft.workers ?? data.settings.workers, hours: draft.hours ?? data.settings.hours, max_rounds: draft.max_rounds ?? data.settings.max_rounds, max_stalled_generations: evolution ? draft.max_stalled_generations ?? data.settings.max_stalled_generations : undefined }) });
      if (!response.ok) throw new Error(await response.text());
      setDraft({}); setNotice("Settings saved. Worker changes apply at the next training or evaluation block."); await refresh();
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  const evolution = data?.algorithm === "greedy_evolution";
  const autonomous = data?.mode === "autonomous";
  const offset = autonomous ? data?.inherited_games ?? 0 : 0;
  const screens = (data?.history ?? []).filter(s => !autonomous || s.branch);
  const promotions = autonomous ? data?.promotions.slice(data.inherited_promotions ?? 0) ?? [] : data?.promotions ?? [];
  const maximum = Math.max(1, (data?.games ?? 1) - offset);
  const y = (score: number) => 200 - 160 * (Math.max(.35, Math.min(.75, score)) - .35) / .4;
  const x = (games: number) => 55 + 780 * (games - offset) / maximum;
  const interval = data ? data.settings.games * data.settings.round_iterations : 0;
  const remaining = data?.latest && interval ? interval - (data.latest.games % interval) : interval;
  const generationGames = (branch: Branch) => Object.entries(data?.evaluation_counts ?? {}).filter(([key]) => key.startsWith(`branches/${branch.id}/`)).reduce((total, [, count]) => total + count, 0) || branch.games;
  const phase = data?.running && data.stop_requested ? "Saving and pausing" : data?.phase?.replaceAll("_", " ");
  return <main className="progressive-page">
    <header><Link href="/">← Astrosynapse 2 control center</Link><Link href="/students">Acquire students ↗</Link><span>Refreshes every 5 seconds · Local M4</span></header>
    <div className="progressive-heading"><div><p className="progressive-kicker">ASTRO6 / SUCCESSIVE CHAMPIONS</p><h1>{evolution ? "Direct policy evolution" : autonomous ? "Autonomous improvement" : "Progressive champion training"}</h1><p>{evolution ? "Improve the policy that actually plays. Retain useful changes. Certify promotions on fresh games." : autonomous ? "Explore training approaches, replace stalled branches, and verify promising candidates." : "Learn, verify an improvement, then train against the new champion."}</p></div>
      {data?.available && <button disabled={busy || (data.running && data.stop_requested)} onClick={() => control(data.running ? "pause" : "resume")}>{data.running ? "Pause & save" : "Resume training"}</button>}
    </div>
    {error && <p role="alert" className="progressive-warning">{error}. Training runs independently of this page.</p>}
    {notice && <p role="status" className="progressive-notice">{notice}</p>}
    {!data?.available ? <section className="progressive-panel">Waiting for the progressive training process…</section> : <>
      <section className="progressive-status"><span className={data.running ? "progressive-dot" : "progressive-dot idle"} /><strong>{phase}</strong><span>{data.settings.workers} actors · {(data.elapsed / 3600).toFixed(2)} / {data.settings.hours} hours</span></section>
      {data.stale && data.running && <p className="progressive-warning">The process has not updated its heartbeat recently. Saved results remain visible.</p>}
      {data.error && <p role="alert" className="progressive-warning">{data.error}</p>}
      <section className="progressive-panel"><h2>{data.run_name}</h2><dl><dt>Current checkpoint</dt><dd>{data.checkpoint_name}</dd><dt>Champion checkpoint</dt><dd>{data.champion_checkpoint_name}</dd></dl><Link href="/">Open the control center to test and compare →</Link></section>
      <section className="progressive-cards">
        <article><span>{evolution ? "Search games this campaign" : "Training games this campaign"}</span><strong>{number(data.games - offset)}</strong><small>{evolution ? `${number(data.verification_games)} additional verification / benchmark games` : data.latest ? `${(data.settings.games / data.latest.iteration_seconds).toFixed(1)} games / second in latest iteration` : "First rollout in progress"}</small></article>
        <article><span>New verified improvements</span><strong>{promotions.length}</strong><small>{autonomous ? `${data.inherited_promotions} inherited champions · strict acceptance gate` : "Each clears a lower confidence bound above 50%"}</small></article>
        <article><span>Current champion</span><strong>{data.stage ? `Generation ${data.stage}` : "Original benchmark"}</strong><small>{data.champion_label}</small></article>
        <article><span>{evolution ? "Latest selection score" : "Latest screening score"}</span><strong>{pct(screens.at(-1)?.score)}</strong><small>Exploratory · versus its incumbent champion</small></article>
      </section>
      {autonomous && !evolution && <>
        <div className="progressive-columns"><section className="progressive-panel"><h2>Autonomous supervisor</h2><p>{data.branches?.find(b => b.id === data.active_branch)?.reason ?? "Preparing the next branch from the verified champion."}</p><p>Four rotating recipes test league exploration, outcome-only learning, broader exploration, and conservative policy heads. Each branch gets at most {data.settings.max_rounds} blocks of {number(interval)} games.</p><button disabled={busy || !data.active_branch || !!data.pending_gate || data.stop_requested} onClick={() => control("skip")}>Skip current branch</button></section>
        <section className="progressive-panel"><h2>Historical experience</h2>{data.historical_data ? <><dl><dt>Retained replay positions</dt><dd>{number(data.historical_data.replay_positions)}</dd><dt>Sampled anchor positions</dt><dd>{number(data.historical_data.sampled_positions)}</dd><dt>Replay sources / shards</dt><dd>{data.historical_data.replay_sources.length} / {number(data.historical_data.replay_shards)}</dd><dt>Historical metric records</dt><dd>{number(data.historical_data.database.metrics)}</dd></dl><p>Compatible observations preserve the incumbent’s behavior. Past verified champions supply 25% of training opponents. New completed games supply policy rewards.</p></> : <p>Checking retained replay and preparing the historical observation sample…</p>}</section></div>
        <section className="progressive-panel"><h2>Training branches</h2><div className="progressive-table"><table><thead><tr><th>Branch</th><th>Recipe</th><th>Status</th><th>Blocks</th><th>Games</th><th>Best screen</th><th>Latest decision</th></tr></thead><tbody>{data.branches?.slice(-16).reverse().map(b => <tr key={b.id}><td>{b.id}</td><td>{b.recipe.name}</td><td>{b.status}</td><td>{b.round} / {data.settings.max_rounds}</td><td>{number(b.id === data.active_branch && data.latest ? data.latest.games : b.games)}</td><td>{b.round ? pct(b.best_score) : "—"}</td><td>{b.reason}</td></tr>)}{!data.branches?.length && <tr><td colSpan={7}>The first branch starts after history preparation.</td></tr>}</tbody></table></div></section>
        <section className="progressive-panel"><h2>Resource controls</h2><form className="progressive-controls" onSubmit={e => { e.preventDefault(); saveSettings(); }}><label>Workers<input type="number" min="1" max="16" required value={draft.workers ?? data.settings.workers} onChange={e => setDraft({ ...draft, workers: e.target.valueAsNumber })} /></label><label>Total campaign hours<input type="number" min="1" max="8760" required value={draft.hours ?? data.settings.hours} onChange={e => setDraft({ ...draft, hours: e.target.valueAsNumber })} /></label><label>Maximum blocks per branch<input type="number" min="1" max="16" required value={draft.max_rounds ?? data.settings.max_rounds ?? 4} onChange={e => setDraft({ ...draft, max_rounds: e.target.valueAsNumber })} /></label><button disabled={busy} type="submit">Save settings</button></form><p>Hours count active runtime across resumes. Increase the total to extend the campaign. Pause saves the active branch or evaluation; resume continues its saved evidence.</p></section>
      </>}
      {evolution && <>
        <div className="progressive-columns"><section className="progressive-panel"><h2>Persistent policy search</h2><p>{data.branches?.find(b => b.id === data.active_branch)?.reason ?? "Preparing the next generation."}</p><dl><dt>Completed generations</dt><dd>{number(data.generation)}</dd><dt>Accepted search steps</dt><dd>{number(data.accepted_steps)}</dd><dt>Current operation</dt><dd>{data.search_summary?.operator.replaceAll("_", " ") ?? "Preparing"}</dd><dt>Candidate progress</dt><dd>{data.pending_gate ? `Promotion test · ${number(data.evaluation?.pairs)} pairs` : data.phase === "confirming_candidate" ? `Confirmation · ${number(data.evaluation?.pairs)} pairs` : data.search_summary ? `${data.search_summary.candidate} / ${data.search_summary.candidates} · ${data.search_summary.phase}` : "—"}</dd></dl><button disabled={busy || !data.active_branch || !!data.pending_gate || data.stop_requested} onClick={() => control("skip")}>Skip current generation</button></section>
        <section className="progressive-panel"><h2>What the search learns</h2><p>Matched full games compare changes to action rankings, ensemble weights, and action features. Individual mutations and combined proposals compete on fresh games. A nominated step must pass another independent comparison with its parent before it becomes the next learner. All candidates play greedily, just as they do when deployed.</p><p>Search and selection scores guide learning. Accepted search steps remain exploratory, including steps retained by earlier search versions, and are counted separately from verified promotions. A separate confirmation sample qualifies a frozen candidate for the promotion test.</p></section></div>
        <section className="progressive-panel"><h2>Search generations</h2><div className="progressive-table"><table><thead><tr><th>Generation</th><th>Operator</th><th>Status</th><th>Games</th><th>Selection score</th><th>Decision</th></tr></thead><tbody>{data.branches?.slice(-12).reverse().map(b => <tr key={b.id}><td>{b.round}</td><td>{b.recipe.name.replaceAll("_", " ")}</td><td>{b.status}</td><td>{number(generationGames(b))}</td><td>{pct(b.screens?.at(-1))}</td><td>{b.reason}</td></tr>)}</tbody></table></div></section>
        <section className="progressive-panel"><h2>Resource controls</h2><form className="progressive-controls" onSubmit={e => { e.preventDefault(); saveSettings(); }}><label>Workers<input type="number" min="1" max="16" required value={draft.workers ?? data.settings.workers} onChange={e => setDraft({ ...draft, workers: e.target.valueAsNumber })} /></label><label>Total campaign hours<input type="number" min="1" max="8760" required value={draft.hours ?? data.settings.hours} onChange={e => setDraft({ ...draft, hours: e.target.valueAsNumber })} /></label><label>Generations per mutation scale<input type="number" min="1" max="16" required value={draft.max_rounds ?? data.settings.max_rounds ?? 16} onChange={e => setDraft({ ...draft, max_rounds: e.target.valueAsNumber })} /></label><label>Generation budget without promotion<input type="number" min="4" max="4096" required value={draft.max_stalled_generations ?? data.settings.max_stalled_generations ?? 64} onChange={e => setDraft({ ...draft, max_stalled_generations: e.target.valueAsNumber })} /></label><button disabled={busy} type="submit">Save settings</button></form><p>Pause preserves completed pairs and the current search. Resume continues the same evidence. The supervisor stops visibly at its time, disk, or sustained-stagnation limit.</p></section>
      </>}
      <section className="progressive-panel"><div className="progressive-panel-heading"><h2>Checkpoint progress</h2><span>{evolution ? "Blue: exploratory selection · Green: verified promotion" : "Blue: screening · Green: verified promotion"}</span></div>
        <svg viewBox="0 0 880 235" role="img" aria-label="Checkpoint win rates against the incumbent champion over training games">
          {[.4, .5, .6, .7].map(score => <g key={score}><line x1="55" x2="835" y1={y(score)} y2={y(score)} stroke={score === .5 ? "#7f90a3" : "#243140"} strokeDasharray={score === .5 ? "6 5" : undefined} /><text x="8" y={y(score) + 4} fill="#9baec2" fontSize="12">{Math.round(score * 100)}%</text></g>)}
          <polyline fill="none" stroke="#70a9ff" strokeWidth="2" points={screens.map(s => `${x(s.games)},${y(s.score)}`).join(" ")} />
          {screens.map((s, i) => <circle key={i} cx={x(s.games)} cy={y(s.score)} r="4" fill="#70a9ff"><title>{s.checkpoint.replace(data.path + "/", "")} · {number(s.games)} games: {pct(s.score)} vs {s.opponent}</title></circle>)}
          {promotions.map((s, i) => <circle key={i} cx={x(s.games)} cy={y(s.score)} r="6" fill="#5fe6ca"><title>Verified: {pct(s.score)}; lower bound {pct(s.lower)}</title></circle>)}
          <text x="55" y="225" fill="#9baec2" fontSize="12">0 new games</text><text x="835" y="225" textAnchor="end" fill="#9baec2" fontSize="12">{number(data.games - offset)} new games</text>
          {!screens.length && <text x="440" y="100" textAnchor="middle" fill="#bdc9d6" fontSize="15">The first checkpoint screen will appear here.</text>}
        </svg><p>Opponents advance after each verified promotion. These scores measure progress against each incumbent; they are not a single fixed-opponent strength curve.</p>
      </section>
      <div className="progressive-columns"><section className="progressive-panel"><h2>{data.phase === "training" ? "Current learning block" : "Current evaluation"}</h2>
        {data.phase === "training" ? <><p>Next screen after approximately <strong>{number(remaining)}</strong> more training games.</p><dl><dt>Learning rate</dt><dd>{data.latest?.learning_rate?.toExponential(2) ?? "—"}</dd><dt>Measured policy KL</dt><dd>{data.latest?.post_update_kl?.toFixed(5) ?? "—"}</dd><dt>Last update</dt><dd>{data.latest ? (data.latest.update_rejected ? "Restored preceding parameters" : "Accepted") : "Collecting games"}</dd></dl></> : <><p>{number(data.evaluation?.pairs)} seat-swapped pairs completed.</p><dl><dt>Observed score</dt><dd>{pct(data.evaluation?.score)}</dd><dt>Lower confidence bound</dt><dd>{pct(data.evaluation?.lower)}</dd></dl></>}
      </section><section className="progressive-panel"><h2>Promotion contract</h2><p>Fresh games against the current champion, with both seats tested. Promotion requires a time-uniform lower bound above 50%, with at least 95% confidence and adjustment for repeated candidates.</p><p>{evolution ? "The error budget is preserved across attempts and resets for each new incumbent. Positive candidates may use the full pair budget; there is no 50.5% score floor." : "Small improvements can accumulate across checkpoints."} After promotion, a separate tournament measures progress against the original champion. Reaching 70% remains a long-term benchmark.</p></section></div>
      {autonomous && data.latest?.critic_calibration && <section className="progressive-panel"><h2>Win-probability diagnostics</h2><dl><dt>Average prediction / observed outcome</dt><dd>{pct(data.latest.critic_calibration.all.prediction)} / {pct(data.latest.critic_calibration.all.outcome)}</dd><dt>Prediction error (Brier; lower is better)</dt><dd>{data.latest.critic_calibration.all.brier.toFixed(4)}</dd><dt>Error on forced decisions</dt><dd>{data.latest.critic_calibration.forced?.brier?.toFixed(4) ?? "—"}</dd><dt>Policy advantage source</dt><dd>{data.latest.advantage_baseline === "constant" ? "Terminal outcome minus 50%; independent of the critic" : "Terminal outcome minus the pre-update critic estimate"}</dd><dt>Historical policy KL</dt><dd>{data.latest.history_kl?.toFixed(5) ?? "—"}</dd></dl><p>Measured on the latest exploratory training games before the critic is fitted. These diagnostics describe prediction quality; promotions depend on independent match results.</p></section>}
      <section className="progressive-panel"><h2>Independent verification results</h2><div className="progressive-table"><table><thead><tr><th>Checkpoint</th><th>Training games</th><th>Opponent</th><th>Evaluation games</th><th>Score</th><th>Lower bound</th><th>Confidence</th><th>Result</th></tr></thead><tbody>
        {data.gates.slice(-20).reverse().map((gate, i) => <tr key={i}><td>{(gate.model ?? gate.actor ?? "—").replace(data.path + "/", "")}</td><td>{number(gate.games)}</td><td>{gate.opponent}</td><td>{number(gate.pairs * 2)}</td><td>{pct(gate.score)}</td><td>{pct(gate.lower)}</td><td>{confidence(gate.confidence)}</td><td style={{color: gate.passed ? "var(--mint)" : "var(--muted)"}}>{gate.passed ? "Promoted" : "Not proven stronger"}</td></tr>)}
        {!data.gates.length && <tr><td colSpan={8}>No independent promotion test has finished yet.</td></tr>}
      </tbody></table></div></section>
      {autonomous && <section className="progressive-panel"><h2>Supervisor decisions</h2><ol className="progressive-events">{data.events?.slice(-12).reverse().map((event, i) => <li key={i}><time>{new Date(event.time * 1000).toLocaleTimeString()}</time> {event.message}</li>)}</ol></section>}
      <p className="progressive-footer">You can close this page while training continues. Pause & save finishes the current work safely. This campaign’s saved training and promotion results use historical rules v1. New tests in Models & Arena and Play use corrected rules v2.</p>
    </>}
  </main>;
}
