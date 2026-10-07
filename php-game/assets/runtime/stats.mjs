// One key per game makes recording idempotent and avoids cross-tab counter races.
export function recordResult(storage, prefix, game) {
  if (game?.status !== 'complete' || !game.result || !game.model_id) return;
  const outcome = game.result.truncated || game.result.winner === null ? 'draws' : game.result.winner === 0 ? 'wins' : game.result.winner === 1 ? 'losses' : null;
  if (!outcome) return;
  const key = prefix + game.id;
  if (storage.getItem(key) === null) storage.setItem(key, JSON.stringify({model: game.model_id, label: game.model_label, outcome, completedAt: Date.now()}));
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

// Wilson score intervals stay meaningful even after all wins or all losses.
export function winInterval(wins, count) {
  if (!count) return null;
  const z = 1.959963984540054, p = wins / count, denominator = 1 + z * z / count;
  const center = (p + z * z / (2 * count)) / denominator;
  const radius = z * Math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator;
  return [Math.max(0, center - radius), Math.min(1, center + radius)];
}
export function readLevelStats(storage, prefix) {
  const rows = Array.from({length: 5}, (_, i) => ({level: 5 - i, wins: 0, losses: 0, draws: 0, points: []}));
  const records = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (!key?.startsWith(prefix)) continue;
    try {
      const record = JSON.parse(storage.getItem(key));
      const match = /^level-0*([1-5])$/.exec(record?.model);
      if (match && ['wins', 'losses', 'draws'].includes(record.outcome)) records.push({...record, level: Number(match[1]), key});
    } catch { /* Ignore malformed results. */ }
  }
  records.sort((a, b) => (Number.isFinite(a.completedAt) ? a.completedAt : 0) - (Number.isFinite(b.completedAt) ? b.completedAt : 0) || a.key.localeCompare(b.key));
  for (const record of records) {
    const row = rows[5 - record.level]; row[record.outcome]++;
    if (record.outcome === 'draws') continue;
    const count = row.wins + row.losses;
    row.points.push({count, rate: row.wins / count, interval: winInterval(row.wins, count), completedAt: record.completedAt});
  }
  return rows;
}
