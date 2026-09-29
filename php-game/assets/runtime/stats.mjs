// One key per game makes recording idempotent and avoids cross-tab counter races.
export function recordResult(storage, prefix, game) {
  if (game?.status !== 'complete' || !game.result || !game.model_id) return;
  const outcome = game.result.truncated || game.result.winner === null ? 'draws' : game.result.winner === 0 ? 'wins' : game.result.winner === 1 ? 'losses' : null;
  if (!outcome) return;
  const key = prefix + game.id;
  if (storage.getItem(key) === null) storage.setItem(key, JSON.stringify({model: game.model_id, label: game.model_label, outcome}));
}
export function readStats(storage, prefix) {
  const stats = new Map();
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (!key?.startsWith(prefix)) continue;
    let record;
    try { record = JSON.parse(storage.getItem(key)); } catch { continue; }
    if (!record || typeof record.model !== 'string' || !['wins', 'losses', 'draws'].includes(record.outcome)) continue;
    if (!stats.has(record.model)) stats.set(record.model, {label: record.label || record.model, wins: 0, losses: 0, draws: 0});
    stats.get(record.model)[record.outcome]++;
  }
  return stats;
}
