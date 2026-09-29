export function opponentTurnSummary(game, cards) {
  const entries = game.action_log.filter(entry => entry.player_id === 1);
  const completed = entries.filter(entry => entry.kind === 'end_turn' || entry.turn < game.observation.turn || game.status === 'complete');
  const turn = completed.at(-1)?.turn;
  if (turn === undefined) return 'No AI turn yet · Game log';
  const played = [], acquired = [];
  for (const entry of entries.filter(entry => entry.turn === turn)) {
    const id = entry.kind === 'free_acquire' ? entry.target_card_id : entry.card_id;
    if (id < 2 || !cards[id]) continue;
    if (entry.kind === 'play_card') played.push(cards[id].name);
    if (entry.kind === 'acquire' || entry.kind === 'free_acquire') acquired.push(cards[id].name);
  }
  const parts = [];
  if (played.length) parts.push(`played ${played.join(', ')}`);
  if (acquired.length) parts.push(`acquired ${acquired.join(', ')}`);
  return `Last turn: ${parts.join('; ') || 'no non-default cards played or acquired'}`;
}
