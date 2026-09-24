"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Student = { id: string; model_label: string; status: string; config: { max_nodes: number; rules_version: number } };
type Candidate = { card_id: number; name: string; score: number; leaf: number; path: { node: number; feature: string; value: number; threshold: number; answer: string }[] };
type Advice = { recommendation: Candidate; candidates: Candidate[]; rules_version: number; selection_rule: string };

export default function AcquireStudentAdvice({ apiBase, observation, spent: knownSpent, positionKey }: {
  apiBase: string; observation: Record<string, unknown> | null; spent?: number; positionKey: string;
}) {
  const [students, setStudents] = useState<Student[]>([]);
  const [student, setStudent] = useState("");
  const [total, setTotal] = useState("");
  const [manualSpent, setManualSpent] = useState("0");
  const [reply, setReply] = useState<{ key: string; advice: Advice } | null>(null);
  const [error, setError] = useState<{ key: string; message: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const spent = knownSpent ?? Number(manualSpent);
  const body = observation ? JSON.stringify({ observation, total_trade: Number(total), spent }) : "";
  const requestKey = JSON.stringify([positionKey, student, body, total]);
  const advice = reply?.key === requestKey ? reply.advice : null;

  useEffect(() => {
    let stopped = false;
    async function refresh() {
      try {
        const response = await fetch(`${apiBase}/acquire-students`, { signal: AbortSignal.timeout(5000) });
        if (!response.ok) throw new Error("Student service unavailable");
        const jobs = await response.json() as Student[];
        if (!stopped) {
          setStudents(jobs.filter(j => j.status === "complete"));
          setStudent(current => current || window.localStorage.getItem("astro-acquire-student") || "");
        }
      } catch { /* Keep the checkpoint advisor usable when this service is offline. */ }
    }
    void refresh();
    const timer = setInterval(refresh, 10000);
    return () => { stopped = true; clearInterval(timer); };
  }, [apiBase]);

  useEffect(() => {
    if (!student || !body || total === "" || !Number.isFinite(Number(total))) return;
    const controller = new AbortController();
    let stopped = false;
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const response = await fetch(`${apiBase}/acquire-students/${student}/recommend`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body, signal: controller.signal,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Check the position and trade inputs.");
        if (!stopped) { setReply({ key: requestKey, advice: result }); setError(null); }
      } catch (e) {
        if (!stopped) setError({ key: requestKey, message: e instanceof Error ? e.message : "Student recommendation unavailable" });
      } finally { if (!stopped) setLoading(false); }
    }, 200);
    return () => { stopped = true; controller.abort(); clearTimeout(timer); };
  }, [apiBase, body, student, total, requestKey]);

  return <section className="student-advice" aria-label="Acquire student recommendation">
    <div className="student-advice-heading"><strong>Acquire student</strong><Link href="/students">Train & inspect ↗</Link></div>
    <label><span>Student</span><select value={students.some(s => s.id === student) ? student : ""} onChange={e => { setStudent(e.target.value); window.localStorage.setItem("astro-acquire-student", e.target.value); }}>
      <option value="">No student selected</option>
      {students.map(s => <option key={s.id} value={s.id}>{s.model_label} · {s.id.slice(0, 6)} · rules v{s.config.rules_version}</option>)}
    </select></label>
    {!students.length ? <p>No trained students yet. Train one on the Students page.</p> : null}
    {student ? <>
      <label><span>Total trade for this whole turn</span><input type="number" min={spent} max="10000" step="1" value={total} placeholder="Enter total or estimate" onChange={e => setTotal(e.target.value)} /></label>
      {knownSpent === undefined ? <label><span>Trade already spent this turn</span><input type="number" min="0" max="10000" step="1" value={manualSpent} onChange={e => setManualSpent(e.target.value)} /></label> : <p>Trade already spent: {knownSpent}</p>}
      <p>Include trade from cards still to be played or drawn. Training uses the teacher’s actual full-turn total. This forecast follows that same convention.</p>
      {!observation ? <p>Available during your turn once the visible position is complete.</p> : total === "" ? <p>Enter the full-turn trade total to see a recommendation.</p> : null}
      {loading && !advice && observation && total !== "" ? <p role="status">Following the tree…</p> : null}
      {error?.key === requestKey ? <p role="alert">{error.message}</p> : null}
      {advice ? <div aria-live="polite">
        <p className="student-choice">Next acquisition: <strong>{advice.recommendation.name}</strong></p>
        <p>{advice.recommendation.card_id === -1 ? "Predicts no further acquisition this turn." : "Predicts the next card later this turn; it may require more plays or a free acquisition ability."}</p>
        {advice.rules_version !== 2 ? <p>Trained under historical rules v1; this Play experience uses rules v2.</p> : null}
        <details><summary>Why this choice?</summary>
          <p>{advice.selection_rule}</p>
          <ol>{advice.recommendation.path.map(step => <li key={step.node}>{step.feature} ≤ {step.threshold.toLocaleString()}? <strong>{step.answer}</strong> (here: {step.value.toLocaleString()})</li>)}<li>Leaf {advice.recommendation.leaf}: score {advice.recommendation.score.toFixed(6)}.</li></ol>
          <table><thead><tr><th>Candidate</th><th>Tree score</th><th>Leaf</th></tr></thead><tbody>{advice.candidates.map(c => <tr key={c.card_id}><td>{c.name}</td><td>{c.score.toFixed(6)}</td><td>{c.leaf}</td></tr>)}</tbody></table>
        </details>
      </div> : null}
    </> : null}
  </section>;
}
