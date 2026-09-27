export type CardAction = {
  id: number;
  kind: string;
  card_id?: number;
  target_card_id?: number;
  source_zone?: string;
  label: string;
};

export type CardControl = { id: number; label: string; title: string; onClick: () => void };

export function matchesCardAction(action: CardAction, cardId: number, zone: string): boolean {
  switch (action.kind) {
    case "activate_base":
    case "activate_ally":
    case "choose_mode":
      return zone === "in_play" && action.card_id === cardId;
    case "destroy_base":
      return zone === "opponent_in_play" && action.target_card_id === cardId;
    case "free_acquire":
    case "scrap_trade_row":
      return zone === "trade_row" && action.target_card_id === cardId;
    case "copy_ship":
      return zone === "in_play" && action.target_card_id === cardId;
    case "discard_card":
      return zone === "hand" && action.card_id === cardId;
    default:
      return false;
  }
}

export function cardActionLabel(action: CardAction): string {
  if (["destroy_base", "free_acquire", "scrap_trade_row", "copy_ship", "discard_card"].includes(action.kind)) return "SELECT TARGET";
  if (action.kind === "activate_base") return "USE ABILITY";
  if (action.kind === "activate_ally") return "USE ALLY ABILITY";
  return action.label;
}
