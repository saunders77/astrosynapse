"use client";

import { useEffect, useState } from "react";

export type MiniAstroResult = {
  parameter_count: number;
  best_epoch: number;
  report_text: string;
  architecture: { base_indices: number[]; normalization: string; groups: { name: string; width: number; indices: number[]; parameters: number }[] };
  history: { epoch: number; training_loss: number; validation_accuracy: number; validation_purchase_accuracy: number }[];
  error_examples: { sample_index: number; game: number; turn: number; target: number; predicted: number | null; excluded_reason: string | null }[];
  fresh_audit?: { games: number; turns: number; all_turn_accuracy: number; purchase_accuracy: number };
};
type Unit = { group: string; unit: number; bias: number; output_weight: number; inputs: { feature: string; weight: number; mean: number; scale: number }[] };
const signed = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(4)}`;

export default function MiniAstroInspector({ result, studentId, apiBase }: { result: MiniAstroResult; studentId: string; apiBase: string }) {
  const [group, setGroup] = useState(0);
  const [unit, setUnit] = useState(0);
  const [detail, setDetail] = useState<{ key: string; value: Unit } | null>(null);
  const [error, setError] = useState("");
  const [cards, setCards] = useState<Record<number, string>>({ [-1]: "None" });
  const key = `${studentId}/${group}/${unit}`;
  const current = detail?.key === key ? detail.value : null;
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/acquire-students/${studentId}/unit?group=${group}&unit=${unit}`, { signal: controller.signal }).then(async response => {
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Unit unavailable");
      setDetail({ key, value: body }); setError("");
    }).catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, [apiBase, studentId, group, unit, key]);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/cards`, { signal: controller.signal }).then(r => r.json()).then((items: { card_id: number; name: string }[]) => setCards({ [-1]: "None", ...Object.fromEntries(items.map(c => [c.card_id, c.name])) })).catch(() => undefined);
    return () => controller.abort();
  }, [apiBase]);
  const points = result.history.map((h, i) => `${50 + 660 * i / Math.max(1, result.history.length - 1)},${230 - 200 * h.validation_accuracy}`).join(" ");
  return <section className="miniastro-inspector" aria-label="MiniAstro model inspector">
    <h2>Inside MiniAstro · {result.parameter_count.toLocaleString()} parameters</h2>
    {result.fresh_audit ? <p>Fresh-game check: <strong>{(100 * result.fresh_audit.all_turn_accuracy).toFixed(1)}% agreement</strong> across {result.fresh_audit.turns.toLocaleString()} sampled turns in {result.fresh_audit.games} new games; {(100 * result.fresh_audit.purchase_accuracy).toFixed(1)}% on covered purchase targets. Weights were frozen before these games.</p> : null}
    <p>Each candidate’s score is a sum of a linear term and five small subnetworks. Group names describe which inputs they receive; they are not automatically discovered strategies.</p>
    <div className="miniastro-architecture"><div><strong>Linear candidate terms</strong><span>{result.architecture.base_indices.length} weights</span></div>{result.architecture.groups.map(g => <div key={g.name}><strong>{g.name}</strong><span>{g.indices.length} inputs → {g.width} ReLU units → score</span><small>{g.parameters.toLocaleString()} parameters</small></div>)}</div>
    <p className="miniastro-equation">Candidate score = linear terms + economy + deck composition + timing + matchup + market</p>
    <details><summary>Validation learning curve · selected epoch {result.best_epoch}</summary><svg viewBox="0 0 750 260" role="img" aria-label="Held-out validation agreement by training epoch" className="miniastro-learning-curve"><title>Validation agreement by epoch; dotted guides show 80% and 90%</title>{[0, .5, .8, .9, 1].map(level => <g key={level}><line x1="50" x2="710" y1={230 - level * 200} y2={230 - level * 200} stroke="#344458" strokeDasharray={level === .8 || level === .9 ? "4 4" : undefined} /><text x="4" y={235 - level * 200} fill="#bdc9d6" fontSize="13">{level * 100}%</text></g>)}<polyline points={points} fill="none" stroke="#5fe6ca" strokeWidth="3" /><text x="50" y="252" fill="#bdc9d6" fontSize="13">Epoch 1</text><text x="650" y="252" fill="#bdc9d6" fontSize="13">{result.history.length}</text></svg></details>
    <h3>Inspect a hidden unit</h3>
    <div className="student-form"><label><span>Subnetwork</span><select value={group} onChange={e => { setGroup(Number(e.target.value)); setUnit(0); }}>{result.architecture.groups.map((g, i) => <option key={g.name} value={i}>{g.name}</option>)}</select></label><label><span>Unit</span><select value={unit} onChange={e => setUnit(Number(e.target.value))}>{Array.from({ length: result.architecture.groups[group].width }, (_, i) => <option key={i} value={i}>{i + 1}</option>)}</select></label></div>
    {error ? <p role="alert">{error}</p> : null}
    {current ? <><p>Contribution = <strong>{signed(current.output_weight)}</strong> × max(0, <strong>{signed(current.bias)}</strong> + weighted input sum).</p><p>All input weights are shown below, sorted by magnitude. Inputs use {result.architecture.normalization}. These weights describe the model’s calculation; correlated inputs prevent treating them as causal effects.</p><div className="miniastro-weights student-table-scroll"><table><thead><tr><th>Input</th><th>Weight</th><th>Training mean · log scale</th><th>Training std · log scale</th></tr></thead><tbody>{current.inputs.map(input => <tr key={input.feature}><td>{input.feature}</td><td className={input.weight >= 0 ? "miniastro-positive" : "miniastro-negative"}>{signed(input.weight)}</td><td>{input.mean.toFixed(4)}</td><td>{input.scale.toFixed(4)}</td></tr>)}</tbody></table></div></> : <p>Loading unit weights…</p>}
    <details><summary>Held-out mistakes · first {result.error_examples.length} examples</summary><p>The full downloadable error set retains each sampled public position for later concept discovery.</p><div className="student-table-scroll"><table><thead><tr><th>Game / turn</th><th>Teacher’s actual acquisition</th><th>MiniAstro prediction</th></tr></thead><tbody>{result.error_examples.map(e => <tr key={e.sample_index}><td>{e.game} / {e.turn}</td><td>{cards[e.target] ?? `Card ${e.target}`}</td><td>{e.predicted === null ? "Target outside available candidates" : cards[e.predicted] ?? `Card ${e.predicted}`}</td></tr>)}</tbody></table></div></details>
    <details><summary>Architecture and selection rules</summary><pre className="student-tree">{result.report_text}</pre></details>
  </section>;
}
