"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import "./impact.css";

const API = "http://127.0.0.1:8765/api/acquisition-impact";
type Group = { key: string; label: string; effect: number; ci: number[]; burden: number; burden_ci: number[]; frequency: number; observed_disagreements: number; sampled_positions: number; sampled_games: number; effective_positions: number; supported: boolean };
type Position = { id: string; game: number; root: number; turn: number; player: number; teacher: string; student: string; category: string; pair: string; heat: string; effect: number; ci: number[]; valid_pairs: number; teacher_wins: number; student_wins: number; teacher_only: number; student_only: number; both_win: number; both_lose: number; trade: number; spent: number; own_authority: number; opponent_authority: number; deck_size: number };
type Report = { id: string; reference: string; reference_id: string; student_id: string; completed_at: number; games: number; eligible_positions: number; disagreements: number; disagreement_rate: number; sampled_positions: number; valid_pairs: number; discarded_pairs: number; excluded_free_choices: number; truncated_source_games: number; rollouts_per_position: number; overall: Group; categories: Group[]; pairs: Group[]; stages: Group[]; heat: Group[]; stage_labels: string[]; budget_labels: string[]; positions: Position[]; method: Record<string, string> };
type Brief = Pick<Report, "id" | "reference" | "games" | "completed_at">;
type Detail = { trade_row: { card: string; count: number }[]; hand: { card: string; count: number }[]; own_deck: { card: string; count: number }[]; opponent_deck: { card: string; count: number }[]; candidates: { card: string; score: number; faction_support: number }[]; current_trade: number; total_trade: number; spent: number };
type Filter = { kind: "pair" | "category" | "heat"; key: string; label: string } | null;
const pp = (n: number, digits = 2) => `${n > 0 ? "+" : ""}${(n * 100).toFixed(digits)} pp`;
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
const ci = (g: { ci: number[] }) => `${pp(g.ci[0])} to ${pp(g.ci[1])}`;
const color = (n: number) => n >= 0 ? "#72e3b1" : "#f4bd75";
async function get<T,>(url: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load results (${response.status}). Check that the local API is running.`);
  return response.json();
}

function IntervalChart({ groups, onSelect }: { groups: Group[]; onSelect?: (g: Group) => void }) {
  const bound = Math.max(.025, ...groups.flatMap(g => g.ci.map(Math.abs))) * 1.12;
  const x = (v: number) => 500 + 240 * v / bound;
  return <svg className="impact-svg" viewBox={`0 0 820 ${80 + groups.length * 68}`} role="img" aria-label="Average champion advantage with 95 percent uncertainty intervals">
    <title>Positive values favor the champion choice; negative values favor MiniAstro</title>
    {[-1, -.5, 0, .5, 1].map(t => <g key={t}><line x1={x(t * bound)} x2={x(t * bound)} y1="18" y2={groups.length * 68 + 20} stroke={t === 0 ? "#9aaabb" : "#253648"} strokeDasharray={t === 0 ? "4 4" : undefined} /><text x={x(t * bound)} y={groups.length * 68 + 46} textAnchor="middle">{pp(t * bound, 1)}</text></g>)}
    {groups.map((g, i) => <g key={g.key} role={onSelect ? "button" : undefined} tabIndex={onSelect ? 0 : undefined} aria-label={`${g.label}: ${pp(g.effect)}, interval ${ci(g)}, ${g.sampled_games} sampled games`} onClick={() => onSelect?.(g)} onKeyDown={e => { if (onSelect && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onSelect(g); } }} className={onSelect ? "impact-clickable" : ""}>
      <title>{g.label}: {pp(g.effect)}; 95% interval {ci(g)}; {g.sampled_positions} positions from {g.sampled_games} games</title>
      <rect x="0" y={i * 68 + 5} width="810" height="63" fill="transparent" />
      <text x="5" y={i * 68 + 29} className="impact-label">{g.label}</text><text x="5" y={i * 68 + 49}>{g.sampled_positions} positions · {g.sampled_games} games</text>
      <line x1={x(g.ci[0])} x2={x(g.ci[1])} y1={i * 68 + 30} y2={i * 68 + 30} stroke={color(g.effect)} strokeWidth="3" />
      <circle cx={x(g.effect)} cy={i * 68 + 30} r="6" fill={color(g.effect)} /><text x="810" y={i * 68 + 34} textAnchor="end" fill={color(g.effect)}>{pp(g.effect)}</text>
    </g>)}
  </svg>;
}

function FrequencyChart({ groups, onSelect }: { groups: Group[]; onSelect: (g: Group) => void }) {
  const xmax = Math.max(.01, ...groups.map(g => g.frequency)) * 1.15;
  const ymax = Math.max(.04, ...groups.flatMap(g => g.ci.map(Math.abs))) * 1.1;
  const x = (n: number) => 80 + n / xmax * 650;
  const y = (n: number) => 250 - n / ymax * 185;
  return <svg className="impact-svg" viewBox="0 0 800 510" role="img" aria-label="Frequency of disagreements versus estimated win probability impact">
    <title>Click a comparison to inspect positions. Vertical lines are 95% intervals; larger dots have more sampled games.</title>
    {[-1, -.5, 0, .5, 1].map(t => <g key={t}><line x1="80" x2="730" y1={y(t * ymax)} y2={y(t * ymax)} stroke={t === 0 ? "#a5b6c5" : "#253648"} strokeDasharray={t === 0 ? "5 4" : undefined} /><text x="66" y={y(t * ymax) + 4} textAnchor="end">{pp(t * ymax, 1)}</text></g>)}
    {[0, .25, .5, .75, 1].map(t => <g key={t}><line x1={x(t * xmax)} x2={x(t * xmax)} y1="55" y2="440" stroke="#253648" /><text x={x(t * xmax)} y="463" textAnchor="middle">{pct(t * xmax)}</text></g>)}
    <text x="85" y="29" fill="#72e3b1">Champion choice better ↑</text><text x="85" y="426" fill="#f4bd75">MiniAstro choice better ↓</text>
    <text x="400" y="497" textAnchor="middle">Share of all eligible acquisition decisions</text>
    {groups.map(g => <g key={g.key} className="impact-clickable" role="button" tabIndex={0} aria-label={`${g.label}: frequency ${pct(g.frequency)}, effect ${pp(g.effect)}`} onClick={() => onSelect(g)} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(g); } }}>
      <title>Champion: {g.label}; {g.observed_disagreements} observed disagreements ({pct(g.frequency)} of opportunities); effect {pp(g.effect)}; interval {ci(g)}; {g.sampled_games} sampled games</title>
      <line x1={x(g.frequency)} x2={x(g.frequency)} y1={y(g.ci[0])} y2={y(g.ci[1])} stroke={color(g.effect)} opacity=".55" strokeWidth="2" />
      <circle cx={x(g.frequency)} cy={y(g.effect)} r={4 + Math.min(10, Math.sqrt(g.sampled_games) / 2)} fill={color(g.effect)} stroke="#0d1520" strokeWidth="2" />
    </g>)}
  </svg>;
}

function PositionInspector({ reportId, position }: { reportId: string; position: Position }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    get<Detail>(`${API}/${reportId}/positions/${position.id}`, controller.signal).then(setDetail).catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, [reportId, position.id]);
  return <section className="impact-inspector" aria-label="Selected position">
    <p className="progressive-kicker">POSITION {position.id} · TURN {position.turn} · PLAYER {position.player + 1}</p>
    <h3>{position.teacher} or {position.student}?</h3>
    <p>The champion chose <strong>{position.teacher}</strong>; MiniAstro preferred <strong>{position.student}</strong>. Forcing the champion choice changed rollout win probability by <strong style={{ color: color(position.effect) }}>{pp(position.effect)}</strong> (paired 95% interval {ci(position)}).</p>
    <div className="impact-outcomes"><div><strong>{position.teacher_wins}/{position.valid_pairs}</strong><span>Champion-choice wins</span></div><div><strong>{position.student_wins}/{position.valid_pairs}</strong><span>Student-choice wins</span></div><div><strong>{position.teacher_only} / {position.student_only}</strong><span>Champion-only / student-only wins</span></div><div><strong>{position.both_win} / {position.both_lose}</strong><span>Both win / both lose</span></div></div>
    <p>Authority {position.own_authority} vs {position.opponent_authority} · {position.deck_size} owned cards · {position.trade} trade remaining · {position.spent} already spent.</p>
    <p className="impact-muted">Each paired continuation uses the same sampled hidden state and random streams; the champion plays both seats after the forced choice. One position has only {position.valid_pairs} pairs. Extreme estimates are examples to investigate, not confirmed rules.</p>
    {error ? <p role="alert">{error}</p> : detail ? <><div className="impact-decks">{([['Market', detail.trade_row], ['Hand', detail.hand], ['Your deck · all zones', detail.own_deck], ['Opponent public ownership', detail.opponent_deck]] as const).map(([label, cards]) => <div key={label}><h4>{label}</h4><p>{cards.length ? cards.map(c => `${c.count > 1 ? `${c.count}× ` : ""}${c.card}`).join(" · ") : "Empty"}</p></div>)}</div><details><summary>MiniAstro’s candidate scores</summary><p>These are model preference scores, not win probabilities.</p><div className="student-table-scroll"><table><thead><tr><th>Candidate</th><th>Score</th><th>Matching-faction cards owned</th></tr></thead><tbody>{detail.candidates.map(c => <tr key={c.card}><td>{c.card}</td><td>{c.score.toFixed(3)}</td><td>{c.faction_support}</td></tr>)}</tbody></table></div></details></> : <p role="status">Loading the visible position…</p>}
  </section>;
}

export default function AcquisitionImpactPage() {
  const [reports, setReports] = useState<Brief[]>([]);
  const [run, setRun] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [minimum, setMinimum] = useState(20);
  const [filter, setFilter] = useState<Filter>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("champion");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Position | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    get<Brief[]>(API, controller.signal).then(items => { setReports(items); setRun(items[0]?.id ?? ""); }).catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!run) return;
    const controller = new AbortController();
    get<Report>(`${API}/${run}`, controller.signal).then(value => { setReport(value); setError(""); }).catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, [run]);
  const chooseFilter = (next: Filter) => { setFilter(next); setPage(0); setSelected(null); };
  const supported = report?.pairs.filter(g => g.sampled_games >= minimum) ?? [];
  const positions = report?.positions.filter(p => (!filter || p[filter.kind] === filter.key) && `${p.teacher} ${p.student}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === "student" ? a.effect - b.effect : sort === "turn" ? a.turn - b.turn : b.effect - a.effect) ?? [];
  const pages = Math.ceil(positions.length / 12);
  const activePage = Math.min(page, Math.max(0, pages - 1));
  const purchaseComparison = report?.categories.find(g => g.key === "Different purchases");
  const shuttleComparison = report?.pairs.find(g => g.label === "Buy nothing instead of Federation Shuttle");
  return <main className="progressive-page students-page impact-page">
    <header><Link href="/">← Control center</Link><Link href="/students">Acquire students ↗</Link></header>
    <div className="progressive-heading"><div><p className="progressive-kicker">ASTROSYNAPSE / UNDERSTANDING</p><h1>Acquisition impact</h1><p>Which disagreements matter—and when is the smaller model’s choice just as good?</p></div></div>
    {error ? <p role="alert" className="progressive-warning">{error}</p> : null}
    <label className="impact-run"><span>Completed experiment</span><select value={run} onChange={e => { setRun(e.target.value); setReport(null); chooseFilter(null); }}><option value="" disabled>Select a report</option>{reports.map(r => <option key={r.id} value={r.id}>{r.games.toLocaleString()} games · {r.reference} · {new Date(r.completed_at * 1000).toLocaleDateString()}</option>)}</select></label>
    {!report ? <p role="status">{run ? "Loading paired rollout results…" : "Reports appear here after their rollout analysis has been prepared."}</p> : <>
      <section className="progressive-panel impact-overview"><p className="progressive-kicker">FROZEN CHAMPION VS MINIASTRO1000 · SINGLE-CHOICE COMPARISONS</p><h2>A small average difference, with useful exceptions</h2><p>When the policies disagreed, taking the champion’s choice changed estimated win probability by <strong style={{ color: color(report.overall.effect) }}>{pp(report.overall.effect)}</strong> on average. The 95% interval is <strong>{ci(report.overall)}</strong>. This measures one changed acquisition followed by champion play; it is not a full-game arena win-rate difference.</p>
        <div className="impact-metrics"><div><strong>{report.games.toLocaleString()}</strong><span>Fresh source games</span></div><div><strong>{pct(report.disagreement_rate)}</strong><span>Disagree · {report.disagreements.toLocaleString()} / {report.eligible_positions.toLocaleString()} decisions</span></div><div><strong>{report.sampled_positions.toLocaleString()}</strong><span>Positions tested</span></div><div><strong>{report.valid_pairs.toLocaleString()}</strong><span>Valid paired continuations</span></div></div>
        <p className="impact-muted">Reference: {report.reference}. MiniAstro has 956 parameters and was trained on gen10; the reference is the later promoted champion frozen for this experiment.</p>
      </section>
      <div className="impact-legend"><span><i style={{ background: "#72e3b1" }} />Positive: champion choice better</span><span><i style={{ background: "#f4bd75" }} />Negative: MiniAstro choice better</span><span>pp = percentage points of win probability</span></div>
      {filter ? <div className="impact-filter-banner" role="status"><span>Selected: {filter.label}</span><a href="#positions">View matching positions ↓</a><button className="impact-link" onClick={() => chooseFilter(null)}>Clear selection</button></div> : null}
      <section className="progressive-panel"><h2>What stands out in this run</h2><div className="impact-findings">
        {purchaseComparison ? <div><h3>Choosing which card</h3><p>Different-purchase disagreements favor the champion by <strong>{pp(purchaseComparison.effect)}</strong> on average, with a 95% interval of {ci(purchaseComparison)}.</p><button className="impact-link" onClick={() => chooseFilter({ kind: "category", key: purchaseComparison.key, label: purchaseComparison.label })}>Inspect these decisions</button></div> : null}
        <div><h3>Buy versus stop is mixed</h3><p>The category charts below separate stopping too early from buying when the champion stops. A small average or an interval crossing zero does not establish that every such purchase is harmless.</p></div>
        {shuttleComparison ? <div><h3>A candidate for follow-up</h3><p>In {shuttleComparison.sampled_games} sampled games where the champion stopped and MiniAstro wanted Federation Shuttle, stopping averaged <strong>{pp(shuttleComparison.effect)}</strong> (95% interval {ci(shuttleComparison)}). This is a selected pattern to retest, not a universal card rule.</p><button className="impact-link" onClick={() => chooseFilter({ kind: "pair", key: shuttleComparison.key, label: shuttleComparison.label })}>Inspect Federation Shuttle examples</button></div> : null}
      </div></section>
      <div className="impact-grid"><section className="progressive-panel"><h2>Which kind of disagreement matters?</h2><p>Average impact within each disagreement type. Lines show 95% intervals. Click a row to inspect examples.</p><IntervalChart groups={report.categories} onSelect={g => chooseFilter({ kind: "category", key: g.key, label: g.label })} /><p className="impact-muted">An interval crossing zero is compatible with either choice being better on average. These averages can conceal important exceptions.</p></section>
      <section className="progressive-panel"><h2>Does the stage of the game matter?</h2><p>Average impact per disagreement, grouped by the game’s turn counter (each player turn increments it).</p><IntervalChart groups={report.stages} /><p className="impact-muted">The stage groups contain different positions and alternatives. This is an observed pattern, not the effect of changing the turn number alone.</p></section></div>
      <section className="progressive-panel"><div className="impact-section-title"><div><h2>Frequent mistakes or expensive mistakes?</h2><p>Each dot is a directed comparison: the champion’s purchase instead of MiniAstro’s. Hover or focus for names; click to inspect positions.</p></div><label>Minimum sampled games<select value={minimum} onChange={e => setMinimum(Number(e.target.value))}>{[5, 10, 20, 50].map(n => <option key={n} value={n}>{n} games</option>)}</select></label></div>
        <div className="impact-grid impact-scatter"><div><FrequencyChart groups={supported} onSelect={g => chooseFilter({ kind: "pair", key: g.key, label: g.label })} /><p className="impact-muted">Dot size = sampled games. Vertical lines = 95% intervals. Frequency counts every observed disagreement, including positions not selected for rollouts.</p></div><div className="student-table-scroll impact-pair-table"><table><thead><tr><th>Champion instead of student</th><th>Frequency</th><th>Impact / disagreement</th><th>Sampled games</th></tr></thead><tbody>{supported.map(g => <tr key={g.key}><td><button className="impact-link" onClick={() => chooseFilter({ kind: "pair", key: g.key, label: g.label })}>{g.label}</button></td><td>{pct(g.frequency)}</td><td style={{ color: color(g.effect) }}>{pp(g.effect)}<small>{ci(g)}</small></td><td>{g.sampled_games}</td></tr>)}</tbody></table>{!supported.length ? <p>No comparisons meet this sample threshold.</p> : null}</div></div>
        <details><summary>How comparisons are prioritized</summary><p>Rows are sorted by their estimated signed contribution across all eligible decisions: sampled effects are weighted by their sampling probability and divided by total opportunities. Frequent costly choices rise to the top. Negative values remain negative. This measure cannot be added up across a game to predict an arena result. Rankings and intervals are exploratory across many comparisons.</p></details>
      </section>
      <section className="progressive-panel"><h2>When is the choice consequential?</h2><p>Average champion advantage per disagreement, by game stage and remaining trade. Cells with fewer than 20 sampled games are marked as sparse.</p><div className="student-table-scroll"><table className="impact-heatmap"><thead><tr><th>Game stage / budget</th>{report.budget_labels.map(b => <th key={b}>{b}</th>)}</tr></thead><tbody>{report.stage_labels.map(stage => <tr key={stage}><th>{stage}</th>{report.budget_labels.map(budget => { const g = report.heat.find(g => g.key === `${stage}|${budget}`); return <td key={budget}>{g ? <button style={{ background: g.effect >= 0 ? `rgba(114,227,177,${.06 + Math.min(.35, Math.abs(g.effect) * 5)})` : `rgba(244,189,117,${.06 + Math.min(.35, Math.abs(g.effect) * 5)})` }} onClick={() => chooseFilter({ kind: "heat", key: g.key, label: `${stage} · ${budget}` })}><strong style={{ color: color(g.effect) }}>{pp(g.effect)}</strong><span>{ci(g)}</span><small>{g.sampled_games} games · {g.sampled_positions} positions{!g.supported ? " · sparse" : ""}</small></button> : <span>No sampled evidence</span>}</td>; })}</tr>)}</tbody></table></div></section>
      <section className="progressive-panel" id="positions"><h2>Inspect the actual decisions</h2><p>Explore both directions. A disagreement is not automatically a mistake, and the largest single-position estimates are noisy.</p><div className="impact-controls"><label>Find a card<input value={query} placeholder="e.g. Explorer" onChange={e => { setQuery(e.target.value); setPage(0); }} /></label><label>Order examples<select value={sort} onChange={e => { setSort(e.target.value); setPage(0); }}><option value="champion">Largest champion advantage</option><option value="student">Largest MiniAstro advantage</option><option value="turn">Earliest turn</option></select></label>{filter ? <button className="button" onClick={() => chooseFilter(null)}>Clear: {filter.label} ×</button> : <span>All comparisons</span>}</div>
        <p>{positions.length.toLocaleString()} sampled positions{filter ? ` · ${filter.label}` : ""}</p>
        {selected ? <PositionInspector key={`${run}/${selected.id}`} reportId={run} position={selected} /> : <p className="impact-muted">Select a position below to see the market, decks, candidate scores and paired outcomes.</p>}
        <div className="student-table-scroll"><table><thead><tr><th>Position</th><th>Champion choice</th><th>MiniAstro choice</th><th>Remaining trade</th><th>Champion advantage</th><th>95% interval</th></tr></thead><tbody>{positions.slice(activePage * 12, activePage * 12 + 12).map(p => <tr key={p.id} className={selected?.id === p.id ? "impact-selected" : ""}><td><button className="impact-link" aria-pressed={selected?.id === p.id} onClick={() => setSelected(p)}>Game {p.game + 1} · turn {p.turn}</button></td><td>{p.teacher}</td><td>{p.student}</td><td>{p.trade}</td><td style={{ color: color(p.effect) }}>{pp(p.effect)}</td><td>{ci(p)}</td></tr>)}</tbody></table></div>
        <div className="impact-pagination"><button className="button" disabled={activePage === 0} onClick={() => setPage(activePage - 1)}>Previous</button><span>{pages ? `Page ${activePage + 1} of ${pages}` : "No matching positions"}</span><button className="button" disabled={activePage + 1 >= pages} onClick={() => setPage(activePage + 1)}>Next</button></div>
      </section>
      <section className="progressive-panel"><h2>How to read this evidence</h2><p>{report.method.scope}</p><p>{report.method.weighting}</p><p>{report.method.interval}</p><p>{report.method.position_interval}</p><p>{report.discarded_pairs} paired samples discarded · {report.truncated_source_games} truncated source games · {report.excluded_free_choices} free-acquisition decisions excluded.</p><p>Sampling weights correct the two-position-per-game cap. They do not correct distribution changes when MiniAstro plays full games. Many disagreements come from the same source game; intervals therefore resample whole games. The same saved positions are used to discover patterns, so surprising findings should be retested on fresh positions.</p><details><summary>Frozen models and experiment identifiers</summary><p>Reference: {report.reference_id}</p><p>Student: {report.student_id}</p><p>Experiment: {report.id}</p><p>Completed {new Date(report.completed_at * 1000).toLocaleString()}</p></details><a href={`${API}/${run}/download`}>Download analysis and sampled-position summaries</a></section>
    </>}
  </main>;
}
