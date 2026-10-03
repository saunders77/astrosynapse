"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import "./style.css";

const API = "http://127.0.0.1:8765/api";
type Model = { id: string; label: string; playable?: boolean; path?: string; architecture?: string; encoder_version?: number; is_champion?: boolean; was_champion?: boolean; critic_id?: string };
type Critic = { id: string; label: string; encoder_version?: number; kind: string; is_champion?: boolean };
type Config = { name: string; policy_id: string; critic_id: string; opponent_ids: string[]; panel_ids: string[]; policy_fraction: number; block_games: number; batch_games: number; probe_games: number; workers: number; max_hours: number; storage_gb: number };
type Event = { at: number; kind: string; candidate?: string; reason?: string; gate?: { score: number; lower: number }; playing_strength?: { difference: number; lower: number } };
type Campaign = { id: string; name: string; status: string; phase: string; running: boolean; champion_id: string; critic_id: string; models: Model[]; critics: Critic[]; config: Config; budget?: { policy: number; critic: number; evaluation: number }; blocks: number; training_games?: number; elapsed_seconds?: number; active_job?: string; progress?: { games?: number; pairs?: number; target?: number; score?: number; epoch?: number }; promotions: Event[]; events: Event[]; error?: string; pause_requested: boolean; stop_after_block: boolean; storage_bytes?: number; log_url: string };
type Match = { id: string; mode: string; label_a: string; label_b: string; status: string; pairs: number; pairs_completed: number; score?: number; truncated?: number; error?: string; critic_metrics?: ({ brier: number; log_loss: number; predicted: number; observed: number } | null)[] };
const pct = (v?: number) => v === undefined ? "—" : `${(v * 100).toFixed(2)}%`;
const duration = (v = 0) => `${(v / 3600).toFixed(2)} h`;
async function api<T>(path: string, body?: unknown, method = "POST"): Promise<T> {
  const r = await fetch(API + path, body === undefined ? { cache: "no-store" } : { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const value = await r.json();
  if (!r.ok) throw new Error(typeof value.detail === "string" ? value.detail : JSON.stringify(value.detail));
  return value;
}
const initial: Config = { name: "Arch3 champion autopilot", policy_id: "", critic_id: "", opponent_ids: [], panel_ids: [], policy_fraction: .8, block_games: 20000, batch_games: 1000, probe_games: 2000, workers: 6, max_hours: 24, storage_gb: 30 };

export default function AutopilotPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [critics, setCritics] = useState<Critic[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [selected, setSelected] = useState("");
  const [config, setConfig] = useState(initial);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [match, setMatch] = useState({ mode: "arena", model_a: "", model_b: "", critic_a: "", critic_b: "", pairs: 2000, workers: 2, seed: 20261004 });
  const current = campaigns.find(c => c.id === selected) ?? campaigns[0];
  useEffect(() => {
    let stopped = false, pending = false;
    async function refresh() {
      if (pending) return;
      pending = true;
      try {
        const [ms, cs, runs, games] = await Promise.all([api<Model[]>("/models"), api<Critic[]>("/autopilot/critics"), api<Campaign[]>("/autopilot"), api<Match[]>("/autopilot/matches")]);
        if (!stopped) {
          const playable = ms.filter(m => m.playable);
          setModels(playable); setCritics(cs); setCampaigns(runs); setMatches(games);
          setConfig(c => ({ ...c, policy_id: c.policy_id || playable.find(m => m.architecture === "arch3")?.id || playable[0]?.id || "", critic_id: c.critic_id || cs.find(c => c.kind === "independent" && c.encoder_version === 3)?.id || "" }));
          setMatch(m => ({ ...m, model_a: m.model_a || playable[0]?.id || "", model_b: m.model_b || playable[1]?.id || playable[0]?.id || "" }));
        }
      } catch (e) { if (!stopped) setError(String(e)); }
      finally { pending = false; }
    }
    void refresh(); const timer = setInterval(refresh, 3000);
    return () => { stopped = true; clearInterval(timer); };
  }, []);
  async function act(action: () => Promise<unknown>) {
    setBusy(true); setError("");
    try { await action(); setCampaigns(await api<Campaign[]>("/autopilot")); setMatches(await api<Match[]>("/autopilot/matches")); }
    catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  function control(action: string) { if (current) void act(() => api(`/autopilot/${current.id}/${action}`, {})); }
  function toggle(key: "opponent_ids" | "panel_ids", id: string) { setConfig(c => ({ ...c, [key]: c[key].includes(id) ? c[key].filter(x => x !== id) : [...c[key], id] })); }
  const budget = current?.budget ?? { policy: 0, critic: 0, evaluation: 0 };
  const training = budget.policy + budget.critic;
  const policyShare = training ? budget.policy / training : .8;
  const modelOptions = models.map(m => <option key={m.id} value={m.id}>{m.label} · {m.architecture ?? `arch${m.encoder_version ?? "?"}`}{m.is_champion ? " · champion" : ""}</option>);
  const criticOptions = critics.map(c => <option key={c.id} value={c.id}>{c.label} · arch{c.encoder_version ?? "?"}{c.is_champion ? " · champion" : ""}</option>);

  return <main className="autopilot-page">
    <nav><Link href="/">← Control center / Arena</Link><Link href="/critics">Independent critics</Link><Link href="/progressive">Previous campaigns</Link></nav>
    <header className="auto-heading"><div><span className="auto-kicker">AUTONOMOUS TRAINING</span><h1>Champion autopilot</h1><p>Train policies, test critics, and promote stronger players with durable evaluation evidence.</p></div><button className="button button-primary" onClick={() => setShowCreate(!showCreate)}>New campaign</button></header>
    {error && <div role="alert" className="auto-error">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
    {showCreate && <section className="auto-panel"><h2>Start a campaign</h2><p>Two training seeds. Critics remain frozen within each policy block. Promotion requires fresh paired games; inconclusive candidates keep the current champion in place.</p>
      <div className="auto-grid"><label>Name<input value={config.name} onChange={e => setConfig({ ...config, name: e.target.value })}/></label><label>Starting policy<select value={config.policy_id} onChange={e => setConfig({ ...config, policy_id: e.target.value })}>{modelOptions}</select></label><label>Independent arch3 critic<select value={config.critic_id} onChange={e => setConfig({ ...config, critic_id: e.target.value })}>{critics.filter(c => c.kind === "independent" && c.encoder_version === 3).map(c => <option key={c.id} value={c.id}>{c.label}</option>)}</select></label>
      {([['block_games','Games per policy block'],['workers','Workers'],['max_hours','Campaign hours'],['storage_gb','Storage limit (GB)']] as const).map(([key,label]) => <label key={key}>{label}<input type="number" min="1" value={config[key]} onChange={e => setConfig({ ...config, [key]: Number(e.target.value) })}/></label>)}
      <label>Policy share of training compute<input type="number" min="50" max="95" value={Math.round(config.policy_fraction*100)} onChange={e => setConfig({ ...config, policy_fraction: Number(e.target.value)/100 })}/></label></div>
      <div className="auto-grid">{([['opponent_ids','Historical training opponents'],['panel_ids','Fixed evaluation panel']] as const).map(([key,label]) => <fieldset key={key}><legend>{label}</legend><div className="auto-checklist">{models.filter(m => m.is_champion || m.was_champion).map(m => <label key={m.id}><input type="checkbox" checked={config[key].includes(m.id)} onChange={() => toggle(key,m.id)}/>{m.label}</label>)}</div></fieldset>)}</div>
      <p>75% of training games face the current champion; 25% use history. Evaluation is measured separately from the training split. Existing arch2 policies are ported to arch3 when selected as the learner.</p>
      <button className="button button-primary" disabled={busy || !config.critic_id || !config.opponent_ids.length || !config.panel_ids.length} onClick={() => void act(async () => { const run = await api<Campaign>("/autopilot", config); setSelected(run.id); await api(`/autopilot/${run.id}/start`, {}); setShowCreate(false); })}>Create and start with automatic promotions</button>
    </section>}
    <label className="auto-run-select">Campaign<select value={current?.id ?? ""} onChange={e => setSelected(e.target.value)}>{campaigns.map(c => <option key={c.id} value={c.id}>{c.name} · {c.status}</option>)}</select></label>
    {current ? <>
      <section className="auto-panel"><div className="auto-toolbar"><div><span className={`auto-status ${current.running ? "running" : ""}`}>{current.status}</span><h2>{current.phase.replaceAll("_", " ")}</h2><p>{current.active_job ?? "Between tasks"}{current.progress?.games !== undefined ? ` · ${current.progress.games.toLocaleString()} training games` : ""}{current.progress?.pairs !== undefined ? ` · ${current.progress.pairs.toLocaleString()} / ${current.progress.target?.toLocaleString()} pairs · ${pct(current.progress.score)}` : ""}{current.progress?.epoch !== undefined ? ` · critic epoch ${current.progress.epoch}` : ""}</p></div><div className="auto-buttons"><button className="button" disabled={busy || current.running || current.status === "complete"} onClick={() => control("resume")}>Start / resume</button><button className="button" disabled={busy || !current.running || current.pause_requested} onClick={() => control("pause")}>{current.pause_requested ? "Pausing at checkpoint…" : "Pause"}</button><button className="button" disabled={busy || !current.running || current.stop_after_block} onClick={() => control("drain")}>{current.stop_after_block ? "Will stop after block" : "Stop after block"}</button><a className="button" href={`http://127.0.0.1:8765${current.log_url}`} target="_blank" rel="noreferrer">Worker log</a></div></div>{current.error && <p role="alert" className="auto-error">{current.error}</p>}
        <div className="auto-stats"><div><span>Policy training</span><strong>{duration(budget.policy)}</strong><small>{pct(training ? budget.policy/training : undefined)} actual · {pct(current.config.policy_fraction)} target</small></div><div><span>Critic development</span><strong>{duration(budget.critic)}</strong><small>Fitting + matched training probes</small></div><div><span>Evaluation</span><strong>{duration(budget.evaluation)}</strong><small>Separate from 80/20 allocation</small></div><div><span>Training games</span><strong>{(current.training_games ?? 0).toLocaleString()}</strong><small>{current.blocks} policy blocks completed</small></div></div>
        <div className="auto-budget" aria-label={`Policy compute share ${pct(policyShare)}`}><div style={{ width: `${policyShare*100}%` }}/></div><p>Critic time is an allowance, not a requirement. Early stopping returns unused time to policy work; complete probe comparisons can temporarily exceed the target share.</p>
        <div className="auto-grid"><div><h3>Playing champion</h3><p>{current.models.find(m => m.id === current.champion_id)?.label}</p><code>{current.champion_id}</code></div><div><h3>Accepted training critic</h3><p>{current.critics.find(c => c.id === current.critic_id)?.label}</p><code>{current.critic_id}</code></div></div>
        <p>{duration(current.elapsed_seconds)} elapsed / {current.config.max_hours} h limit · {((current.storage_bytes ?? 0)/1024**3).toFixed(2)} / {current.config.storage_gb} GB retained · {current.promotions.length} promotions</p>
        <details><summary>Resource limits (pause before editing)</summary><form key={current.id} onSubmit={event => { event.preventDefault(); const form = new FormData(event.currentTarget); void act(() => api(`/autopilot/${current.id}/settings`, { max_hours: Number(form.get("hours")), storage_gb: Number(form.get("storage")), workers: Number(form.get("workers")), policy_fraction: Number(form.get("share"))/100 }, "PATCH")); }}><div className="auto-grid"><label>Total campaign hours<input name="hours" type="number" min="0.1" max="720" step="0.1" defaultValue={current.config.max_hours}/></label><label>Storage limit (GB)<input name="storage" type="number" min="1" max="500" defaultValue={current.config.storage_gb}/></label><label>Workers<input name="workers" type="number" min="1" max="8" defaultValue={current.config.workers}/></label><label>Policy compute share (%)<input name="share" type="number" min="50" max="95" defaultValue={current.config.policy_fraction*100}/></label></div><button className="button" disabled={busy || current.running}>Save limits</button></form></details>
        <details><summary>Restore previous champions</summary><p>Stop after the current block before restoring. Only previously accepted champions are selectable.</p><form key={`restore-${current.id}`} onSubmit={event => { event.preventDefault(); const form = new FormData(event.currentTarget); void act(() => api(`/autopilot/${current.id}/champions/restore`, { policy_id: form.get("policy"), critic_id: form.get("critic") })); }}><div className="auto-grid"><label>Retained policy champion<select name="policy" defaultValue={current.champion_id}>{current.models.filter(m => m.was_champion).map(m => <option key={m.id} value={m.id}>{m.label}</option>)}</select></label><label>Retained accepted critic<select name="critic" defaultValue={current.critic_id}>{current.critics.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}</select></label></div><button className="button" disabled={busy || current.running}>Restore selected champions</button></form></details>
      </section>
      <section className="auto-panel"><h2>Decisions and promotion evidence</h2><p>Fresh gates account for repeated attempts. Policy promotions also require evidence of no material regression on the historical panel. Critic promotions require stronger play in matched training probes.</p>{current.events.length ? <div className="auto-events">{[...current.events].reverse().map((event,i) => <details key={`${event.at}-${i}`}><summary><time>{new Date(event.at*1000).toLocaleString()}</time><strong>{event.kind.replaceAll("_"," ")}</strong><span>{event.reason ?? (event.gate ? `${pct(event.gate.score)} vs incumbent` : "")}</span></summary><pre>{JSON.stringify(event,null,2)}</pre></details>)}</div> : <p>No promotion decisions yet. Training and evaluation are in progress.</p>}</section>
    </> : <section className="auto-panel"><h2>No campaigns yet</h2><p>Create a campaign to begin policy training and independent critic development.</p></section>}
    <section className="auto-panel" id="matches"><h2>Arena & self-play: policies and critics</h2><p>Choose each policy and value predictor independently, including older architectures. Policies choose moves; critics measure win probability and prediction error. Self-play runs the same policy in both seats and can compare two different critics.</p>
      <div className="auto-grid"><label>Mode<select value={match.mode} onChange={e => setMatch({ ...match, mode: e.target.value })}><option value="arena">Arena — different policies</option><option value="selfplay">Self-play — same policy, selectable critics</option></select></label><label>Policy A<select value={match.model_a} onChange={e => setMatch({ ...match, model_a: e.target.value })}>{modelOptions}</select></label><label>Policy B<select disabled={match.mode === "selfplay"} value={match.mode === "selfplay" ? match.model_a : match.model_b} onChange={e => setMatch({ ...match, model_b: e.target.value })}>{modelOptions}</select></label><label>Critic A<select value={match.critic_a} onChange={e => setMatch({ ...match, critic_a: e.target.value })}><option value="">No critic diagnostics</option>{criticOptions}</select></label><label>Critic B<select value={match.critic_b} onChange={e => setMatch({ ...match, critic_b: e.target.value })}><option value="">No critic diagnostics</option>{criticOptions}</select></label><label>Seat-swapped pairs<input type="number" min="1" max="20000" value={match.pairs} onChange={e => setMatch({ ...match, pairs: Number(e.target.value) })}/></label></div>
      <button className="button button-primary" disabled={busy || !match.model_a || matches.some(m => ["running","queued"].includes(m.status))} onClick={() => void act(() => api("/autopilot/matches", { ...match, model_b: match.mode === "selfplay" ? match.model_a : match.model_b, critic_a: match.critic_a || null, critic_b: match.critic_b || null }))}>Run {match.mode === "selfplay" ? "self-play" : "arena"}</button>
      <div className="auto-events">{matches.slice(0,10).map(m => <article key={m.id}><h3>{m.mode} · {m.status} · {m.pairs_completed}/{m.pairs} pairs</h3><p>{m.label_a} vs {m.label_b}</p><p>Policy A score: <strong>{pct(m.score)}</strong> · truncated games: {m.truncated ?? 0}</p>{m.critic_metrics?.map((c,i) => c && <p key={i}>Critic {i === 0 ? "A" : "B"}: Brier {c.brier.toFixed(4)} · log loss {c.log_loss.toFixed(4)} · predicted {pct(c.predicted)} / observed {pct(c.observed)}</p>)}{m.error && <p role="alert">{m.error}</p>}{["running","queued"].includes(m.status) ? <button className="button" onClick={() => void act(() => api(`/autopilot/matches/${m.id}/pause`, {}))}>Pause match</button> : ["paused","interrupted"].includes(m.status) ? <button className="button" onClick={() => void act(() => api(`/autopilot/matches/${m.id}/resume`, {}))}>Resume match</button> : null}</article>)}</div>
    </section>
  </main>;
}
