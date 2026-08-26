"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";

export type ManualModelGroup = {
  runId: string;
  runName: string;
  models: Array<{ id: string; label: string }>;
};

type Side = "astro5" | "hard";
type ZoneId =
  | "tradeRow"
  | "astroHand"
  | "astroDeck"
  | "astroKnownTop"
  | "astroDiscard"
  | "astroInPlay"
  | "hardHidden"
  | "hardKnownHand"
  | "hardKnownTop"
  | "hardDiscard"
  | "hardInPlay"
  | "scrapHeap";

type CardDefinition = {
  card_id: number;
  name: string;
  cost: number;
  combat: number;
  authority: number;
  trade: number;
  faction: string;
  card_type: "ship" | "base" | "outpost";
  primary: string;
  ally: string;
  ally_amount: number;
  scrap: string;
  scrap_amount: number;
  defense: number;
  copies: number;
};

type TrackedCard = {
  uid: string;
  cardId: number | null;
  reservedCardId?: number | null;
  copiedCardId?: number | null;
  activated?: boolean;
  allyTriggered?: boolean;
  playedTurn?: number;
};

type PlayerBoard = {
  authority: number;
  trade: number;
  combat: number;
  pendingDiscard: number;
  nextShipToTop: boolean;
  hand: TrackedCard[];
  deck: TrackedCard[];
  knownTop: TrackedCard[];
  discard: TrackedCard[];
  inPlay: TrackedCard[];
};

type HardBoard = {
  authority: number;
  trade: number;
  combat: number;
  pendingDiscard: number;
  nextShipToTop: boolean;
  handCount: number;
  deckCount: number;
  hidden: TrackedCard[];
  knownHand: TrackedCard[];
  knownTop: TrackedCard[];
  discard: TrackedCard[];
  inPlay: TrackedCard[];
};

type MatchEvent = {
  id: string;
  turn: number;
  side: Side | "system";
  text: string;
};

type PendingDecision = {
  family: string;
  prompt: string;
  actions: SemanticAction[];
  effect?: string;
  sourceCardId?: number;
  remaining?: number;
  completed?: number;
  followupEffect?: string;
};

type ManualMatch = {
  id: string;
  checkpointId: string;
  startingSide: Side;
  activeSide: Side;
  turn: number;
  actionNumber: number;
  astro: PlayerBoard;
  hard: HardBoard;
  tradeRow: TrackedCard[];
  explorersRemaining: number;
  scrapHeap: TrackedCard[];
  pendingDecision: PendingDecision | null;
  events: MatchEvent[];
  startedAt: number;
};

type SemanticAction = {
  id: number;
  label: string;
  kind: string;
  card_id?: number;
  target_card_id?: number;
  ability?: string;
  source_zone?: string;
  amount?: number;
  amount2?: number;
};

type ScoredAction = SemanticAction & {
  model_value: number | null;
  model_recommended: boolean;
};

type Recommendation = {
  family: string;
  prompt: string;
  score_semantics: "policy_probability" | "win_outcome" | null;
  expected_win_rate: number | null;
  actions: ScoredAction[];
};

type HardActionKind =
  | "play"
  | "acquire"
  | "attack_player"
  | "attack_base"
  | "discard"
  | "scrap"
  | "scrap_row"
  | "ability"
  | "end_turn";

type HardQueuedDecision = {
  effect: string;
  cardId: number;
  cardName: string;
};

type Props = {
  apiBase: string;
  connected: boolean;
  modelGroups: ManualModelGroup[];
  onToast: (message: string) => void;
};

const STORAGE_KEY = "astrosynapse.manual-hard-ai-match.v1";
let uidSequence = 0;

function uid(prefix = "card") {
  uidSequence += 1;
  return `${prefix}-${uidSequence.toString(36)}`;
}

function card(cardId: number | null, prefix = "card"): TrackedCard {
  return { uid: uid(prefix), cardId };
}

function undefinedCards(count: number, prefix: string): TrackedCard[] {
  return Array.from({ length: Math.max(0, count) }, () => card(null, prefix));
}

function starterPool(prefix: string): TrackedCard[] {
  return [
    ...Array.from({ length: 8 }, () => card(0, `${prefix}-scout`)),
    ...Array.from({ length: 2 }, () => card(1, `${prefix}-viper`)),
  ];
}

const FALLBACK_CARDS: CardDefinition[] = [
  { card_id: 0, name: "Scout", cost: 0, combat: 0, authority: 0, trade: 1, faction: "unaligned", card_type: "ship", primary: "", ally: "", ally_amount: 0, scrap: "", scrap_amount: 0, defense: 0, copies: 0 },
  { card_id: 1, name: "Viper", cost: 0, combat: 1, authority: 0, trade: 0, faction: "unaligned", card_type: "ship", primary: "", ally: "", ally_amount: 0, scrap: "", scrap_amount: 0, defense: 0, copies: 0 },
  { card_id: 2, name: "Explorer", cost: 2, combat: 0, authority: 0, trade: 2, faction: "unaligned", card_type: "ship", primary: "", ally: "", ally_amount: 0, scrap: "gain_combat", scrap_amount: 2, defense: 0, copies: 0 },
  { card_id: 4, name: "Battle Pod", cost: 2, combat: 4, authority: 0, trade: 0, faction: "blob", card_type: "ship", primary: "scrap_trade_row", ally: "gain_combat", ally_amount: 2, scrap: "", scrap_amount: 0, defense: 0, copies: 2 },
  { card_id: 13, name: "Trade Pod", cost: 2, combat: 0, authority: 0, trade: 3, faction: "blob", card_type: "ship", primary: "", ally: "gain_combat", ally_amount: 2, scrap: "", scrap_amount: 0, defense: 0, copies: 3 },
  { card_id: 20, name: "Missile Bot", cost: 2, combat: 2, authority: 0, trade: 0, faction: "machine_cult", card_type: "ship", primary: "scrap_any", ally: "gain_combat", ally_amount: 2, scrap: "", scrap_amount: 0, defense: 0, copies: 3 },
  { card_id: 30, name: "Imperial Fighter", cost: 1, combat: 2, authority: 0, trade: 0, faction: "star_empire", card_type: "ship", primary: "opponent_discard", ally: "gain_combat", ally_amount: 2, scrap: "", scrap_amount: 0, defense: 0, copies: 3 },
  { card_id: 31, name: "Imperial Frigate", cost: 3, combat: 4, authority: 0, trade: 0, faction: "star_empire", card_type: "ship", primary: "opponent_discard", ally: "gain_combat", ally_amount: 2, scrap: "draw", scrap_amount: 0, defense: 0, copies: 3 },
  { card_id: 36, name: "War World", cost: 5, combat: 3, authority: 0, trade: 0, faction: "star_empire", card_type: "outpost", primary: "", ally: "gain_combat", ally_amount: 4, scrap: "", scrap_amount: 0, defense: 4, copies: 1 },
  { card_id: 40, name: "Cutter", cost: 2, combat: 0, authority: 4, trade: 2, faction: "trade_federation", card_type: "ship", primary: "", ally: "gain_combat", ally_amount: 4, scrap: "", scrap_amount: 0, defense: 0, copies: 3 },
  { card_id: 43, name: "Federation Shuttle", cost: 1, combat: 0, authority: 0, trade: 2, faction: "trade_federation", card_type: "ship", primary: "", ally: "gain_authority", ally_amount: 4, scrap: "", scrap_amount: 0, defense: 0, copies: 3 },
  { card_id: 48, name: "Trading Post", cost: 3, combat: 0, authority: 0, trade: 0, faction: "trade_federation", card_type: "outpost", primary: "trading_post", ally: "", ally_amount: 0, scrap: "gain_combat", scrap_amount: 3, defense: 4, copies: 2 },
];

function createMatch(checkpointId: string, startingSide: Side): ManualMatch {
  const astroHandSize = startingSide === "astro5" ? 3 : 5;
  const hardHandSize = startingSide === "hard" ? 3 : 5;
  const astroStarters = starterPool("astro-starter");
  const astroReservedHand = astroStarters.slice(-astroHandSize).map((item) => ({
    uid: uid("astro-hand"),
    cardId: null,
    reservedCardId: item.cardId,
  }));
  return {
    id: uid("match"),
    checkpointId,
    startingSide,
    activeSide: startingSide,
    turn: 1,
    actionNumber: 0,
    astro: {
      authority: 50,
      trade: 0,
      combat: 0,
      pendingDiscard: 0,
      nextShipToTop: false,
      hand: astroReservedHand,
      deck: astroStarters.slice(0, 10 - astroHandSize).map((item) => ({ ...item, uid: uid("astro-deck") })),
      knownTop: [],
      discard: [],
      inPlay: [],
    },
    hard: {
      authority: 50,
      trade: 0,
      combat: 0,
      pendingDiscard: 0,
      nextShipToTop: false,
      handCount: hardHandSize,
      deckCount: 10 - hardHandSize,
      hidden: starterPool("hard-hidden"),
      knownHand: [],
      knownTop: [],
      discard: [],
      inPlay: [],
    },
    tradeRow: undefinedCards(5, "trade-row"),
    explorersRemaining: 10,
    scrapHeap: [],
    pendingDecision: null,
    events: [{ id: uid("event"), turn: 1, side: "system", text: `New match · ${startingSide === "astro5" ? "Astro5" : "Hard AI"} goes first` }],
    startedAt: 0,
  };
}

function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatPercent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function apiErrorMessage(payload: unknown, fallback: string): string {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) return fallback;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail.flatMap((item) => typeof item === "object" && item !== null && "msg" in item && typeof (item as { msg?: unknown }).msg === "string" ? [(item as { msg: string }).msg] : []);
    if (messages.length) return messages.join(" · ");
  }
  return fallback;
}

function factionClass(definition: CardDefinition | undefined): string {
  const faction = definition?.faction ?? "unaligned";
  if (faction.includes("blob")) return "blob";
  if (faction.includes("machine")) return "machine";
  if (faction.includes("star")) return "star";
  if (faction.includes("trade")) return "trade";
  return "neutral";
}

function effectiveCardId(tracked: TrackedCard): number | null {
  return tracked.copiedCardId ?? tracked.cardId;
}

function effectiveDefinition(tracked: TrackedCard, definitions: Map<number, CardDefinition>): CardDefinition | undefined {
  const cardId = effectiveCardId(tracked);
  return cardId === null ? undefined : definitions.get(cardId);
}

function baseRequiresActivation(definition: CardDefinition): boolean {
  return definition.card_type !== "ship"
    && definition.card_id !== 19
    && definition.card_id !== 29
    && Boolean(definition.combat || definition.authority || definition.trade || definition.primary);
}

function originalCard(tracked: TrackedCard): TrackedCard {
  return {
    uid: tracked.uid,
    cardId: tracked.cardId,
    ...(tracked.reservedCardId !== undefined ? { reservedCardId: tracked.reservedCardId } : {}),
  };
}

function abilityLabel(effect: string, amount = 0): string {
  if (!effect) return "";
  if (effect === "gain_combat") return `Gain ${amount} combat`;
  if (effect === "gain_trade") return `Gain ${amount} trade`;
  if (effect === "gain_authority") return `Gain ${amount} authority`;
  const labels: Record<string, string> = {
    draw: "Draw a card",
    draw_two: "Draw two cards",
    opponent_discard: "Opponent discards",
    scrap_any: "Scrap from hand or discard",
    scrap_trade_row: "Scrap a trade-row card",
    trading_post: "Choose authority or trade",
    patrol_mech: "Choose combat or trade",
    barter_world: "Choose authority or trade",
    defense_center: "Choose combat or authority",
    recycle: "Choose trade or cycle cards",
    blob_world: "Choose combat or draw",
    copy_ship: "Copy a ship",
    destroy_base: "Optionally destroy a base",
    destroy_and_scrap: "Destroy a base and scrap the row",
    draw_then_scrap: "Draw, then scrap from hand",
    draw_destroy: "Draw, then optionally destroy a base",
    scrap_two_draw: "Scrap up to two, then draw",
    free_ship: "Acquire a ship free to deck top",
    ship_top: "Next acquired ship goes on top",
    all_ally: "All factions count as allied",
    fleet_hq: "Ships gain one combat",
    embassy_yacht: "Draw two with two bases",
  };
  return labels[effect] ?? titleCase(effect);
}

function cardRules(definition: CardDefinition | undefined): string {
  if (!definition) return "Choose this card from the catalog";
  return [
    definition.primary ? abilityLabel(definition.primary) : "",
    definition.ally ? `Ally: ${abilityLabel(definition.ally, definition.ally_amount)}` : "",
    definition.scrap ? `Scrap: ${abilityLabel(definition.scrap, definition.scrap_amount)}` : "",
  ].filter(Boolean).join(" · ") || "No additional ability";
}

function CardNameEditor({
  value,
  catalog,
  onSelect,
  onCancel,
  allowUndefined = true,
  listboxId = "relay-card-options",
  autoFocus = false,
  showAllOptions = false,
}: {
  value: CardDefinition | undefined;
  catalog: CardDefinition[];
  onSelect: (cardId: number | null) => void;
  onCancel: () => void;
  allowUndefined?: boolean;
  listboxId?: string;
  autoFocus?: boolean;
  showAllOptions?: boolean;
}) {
  const [query, setQuery] = useState(value?.name ?? "");
  const inputRef = useRef<HTMLInputElement>(null);
  const matches = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const ranked = catalog.filter((item) => !normalized || item.name.toLowerCase().includes(normalized));
    return ranked.sort((left, right) => {
      const leftStarts = left.name.toLowerCase().startsWith(normalized) ? 0 : 1;
      const rightStarts = right.name.toLowerCase().startsWith(normalized) ? 0 : 1;
      return leftStarts - rightStarts || left.name.localeCompare(right.name);
    });
    return showAllOptions ? ranked : ranked.slice(0, 7);
  }, [catalog, query, showAllOptions]);
  const quickCards = useMemo(
    () => [0, 1, 2].flatMap((cardId) => catalog.find((item) => item.card_id === cardId) ?? []),
    [catalog],
  );

  useEffect(() => {
    if (!autoFocus) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [autoFocus]);

  const chooseFirst = () => {
    const exact = catalog.find((item) => item.name.toLowerCase() === query.trim().toLowerCase());
    const selected = exact ?? matches[0];
    if (selected) onSelect(selected.card_id);
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      chooseFirst();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      onCancel();
    }
  };

  return (
    <div className="relay-card-editor">
      {quickCards.length ? (
        <div className="relay-card-quick-picks" aria-label="Common cards">
          {quickCards.map((item) => <button key={item.card_id} type="button" onClick={() => onSelect(item.card_id)}>{item.name}</button>)}
        </div>
      ) : null}
      <label>
        <span>{showAllOptions ? "Possible cards" : "Card name"}</span>
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Start typing…"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded="true"
          aria-controls={listboxId}
        />
      </label>
      <div className="relay-card-options" id={listboxId} role="listbox">
        {matches.map((item) => (
          <button key={item.card_id} type="button" role="option" aria-selected={value?.card_id === item.card_id} onClick={() => onSelect(item.card_id)}>
            <span className={`relay-faction-dot faction-${factionClass(item)}`} />
            <strong>{item.name}</strong>
            <small>{item.card_type} · cost {item.cost}</small>
          </button>
        ))}
        {!matches.length ? <p>No catalog match. Leave it Undefined until the card is supported.</p> : null}
      </div>
      <footer>
        {allowUndefined ? <button type="button" onClick={() => onSelect(null)}>Leave Undefined</button> : <span />}
        <button type="button" onClick={onCancel}>Cancel</button>
      </footer>
    </div>
  );
}

function EditableCard({
  tracked,
  definition,
  catalog,
  compact = false,
  topLabel,
  onChange,
  onDelete,
  onToggleActivated,
  showAllOptions = false,
}: {
  tracked: TrackedCard;
  definition: CardDefinition | undefined;
  catalog: CardDefinition[];
  compact?: boolean;
  topLabel?: string;
  onChange: (cardId: number | null) => void;
  onDelete: () => void;
  onToggleActivated?: () => void;
  showAllOptions?: boolean;
}) {
  const [editing, setEditing] = useState(tracked.cardId === null);

  if (editing || tracked.cardId === null) {
    return (
      <article className={`relay-card relay-card-undefined${compact ? " is-compact" : ""}`}>
        {topLabel ? <span className="relay-top-label">{topLabel}</span> : null}
        <CardNameEditor
          value={definition}
          catalog={catalog}
          onSelect={(cardId) => {
            onChange(cardId);
            if (cardId !== null) setEditing(false);
          }}
          onCancel={() => {
            if (tracked.cardId !== null) setEditing(false);
          }}
          listboxId={`relay-card-options-${tracked.uid}`}
          autoFocus={tracked.cardId !== null}
          showAllOptions={showAllOptions}
        />
        <button type="button" className="relay-card-delete" onClick={onDelete} aria-label="Delete undefined card">×</button>
      </article>
    );
  }

  return (
    <article className={`relay-card faction-${factionClass(definition)}${compact ? " is-compact" : ""}`}>
      {topLabel ? <span className="relay-top-label">{topLabel}</span> : null}
      <span className="relay-card-cost">{definition?.cost ?? "?"}</span>
      <span className="relay-card-kind">{definition?.card_type ?? "undefined"}</span>
      <strong title={definition?.name}>{definition?.name ?? "Undefined"}</strong>
      {!compact ? <p>{cardRules(definition)}</p> : null}
      <div className="relay-card-stats">
        {definition?.trade ? <span><b>{definition.trade}</b> trade</span> : null}
        {definition?.combat ? <span><b>{definition.combat}</b> combat</span> : null}
        {definition?.authority ? <span><b>{definition.authority}</b> auth</span> : null}
        {definition?.defense ? <span><b>{definition.defense}</b> defense</span> : null}
      </div>
      {tracked.activated !== undefined ? <button type="button" className={`relay-card-state ${tracked.activated ? "is-used" : ""}`} onClick={onToggleActivated} disabled={!onToggleActivated} title="Toggle activation state">{tracked.activated ? "activated" : "ready"}</button> : null}
      <div className="relay-card-controls">
        <button type="button" onClick={() => setEditing(true)} aria-label={`Edit ${definition?.name ?? "card"}`} title="Edit card">✎</button>
        <button type="button" onClick={onDelete} aria-label={`Delete ${definition?.name ?? "card"}`} title="Delete card">×</button>
      </div>
    </article>
  );
}

function CardZone({
  label,
  detail,
  cards,
  zone,
  catalog,
  definitions,
  compact = true,
  emptyText = "Empty",
  onChange,
  onDelete,
  onAdd,
  onToggleActivated,
  catalogForCard,
}: {
  label: string;
  detail?: string;
  cards: TrackedCard[];
  zone: ZoneId;
  catalog: CardDefinition[];
  definitions: Map<number, CardDefinition>;
  compact?: boolean;
  emptyText?: string;
  onChange: (zone: ZoneId, uid: string, cardId: number | null) => void;
  onDelete: (zone: ZoneId, uid: string) => void;
  onAdd: (zone: ZoneId) => void;
  onToggleActivated?: (zone: ZoneId, uid: string) => void;
  catalogForCard?: (card: TrackedCard) => CardDefinition[];
}) {
  return (
    <section className="relay-zone">
      <header>
        <div><strong>{label}</strong>{detail ? <span>{detail}</span> : null}</div>
        <button type="button" onClick={() => onAdd(zone)}>+ Card</button>
      </header>
      <div className="relay-card-row">
        {cards.map((item, index) => (
          <EditableCard
            key={item.uid}
            tracked={item}
            definition={zone === "astroInPlay" || zone === "hardInPlay" ? effectiveDefinition(item, definitions) : item.cardId === null ? undefined : definitions.get(item.cardId)}
            catalog={catalogForCard?.(item) ?? catalog}
            compact={compact}
            topLabel={item.copiedCardId !== undefined && item.copiedCardId !== null ? "Stealth Needle copy" : zone === "astroKnownTop" || zone === "hardKnownTop" ? `Top ${index + 1}` : undefined}
            onChange={(cardId) => onChange(zone, item.uid, cardId)}
            onDelete={() => onDelete(zone, item.uid)}
            onToggleActivated={onToggleActivated ? () => onToggleActivated(zone, item.uid) : undefined}
            showAllOptions={Boolean(catalogForCard)}
          />
        ))}
        {!cards.length ? <span className="relay-zone-empty">{emptyText}</span> : null}
      </div>
    </section>
  );
}

function StatInput({ label, value, onChange, tone }: { label: string; value: number; onChange: (value: number) => void; tone?: string }) {
  return (
    <label className={`relay-stat${tone ? ` is-${tone}` : ""}`}>
      <span>{label}</span>
      <input type="number" min="0" value={value} onChange={(event) => onChange(Math.max(0, Number(event.target.value) || 0))} />
    </label>
  );
}

function RelayModal({ title, kicker, onClose, children, wide = false }: { title: string; kicker: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="relay-modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className={`relay-modal${wide ? " is-wide" : ""}`} role="dialog" aria-modal="true" aria-label={title}>
        <header><div><span>{kicker}</span><h2>{title}</h2></div><button type="button" onClick={onClose} aria-label={`Close ${title}`}>×</button></header>
        {children}
      </section>
    </div>
  );
}

function definitionList(cards: TrackedCard[], definitions: Map<number, CardDefinition>): CardDefinition[] {
  return cards.flatMap((item) => item.cardId === null ? [] : [definitions.get(item.cardId)]).filter((item): item is CardDefinition => Boolean(item));
}

function remainingTradeDeck(match: ManualMatch, catalog: CardDefinition[]): CardDefinition[] {
  const pool = catalog.flatMap((item) => Array.from({ length: item.copies || 0 }, () => item));
  const occupied = [
    ...match.tradeRow,
    ...match.astro.hand,
    ...match.astro.deck,
    ...match.astro.knownTop,
    ...match.astro.discard,
    ...match.astro.inPlay,
    ...match.hard.hidden,
    ...match.hard.knownHand,
    ...match.hard.knownTop,
    ...match.hard.discard,
    ...match.hard.inPlay,
    ...match.scrapHeap,
  ];
  for (const tracked of occupied) {
    if (tracked.cardId === null || tracked.cardId <= 2) continue;
    const index = pool.findIndex((item) => item.card_id === tracked.cardId);
    if (index >= 0) pool.splice(index, 1);
  }
  return pool;
}

function unresolvedCards(match: ManualMatch, definitions: Map<number, CardDefinition>): Array<{ zone: ZoneId; uid: string; label: string }> {
  const zones: Array<[ZoneId, string, TrackedCard[]]> = [
    ["tradeRow", "trade row", match.tradeRow],
    ["astroHand", "Astro5 hand", match.astro.hand],
    ["astroDeck", "Astro5 deck", match.astro.deck],
    ["astroKnownTop", "Astro5 known top", match.astro.knownTop],
    ["astroDiscard", "Astro5 discard", match.astro.discard],
    ["astroInPlay", "Astro5 in play", match.astro.inPlay],
    ["hardHidden", "Hard AI hidden pool", match.hard.hidden],
    ["hardKnownHand", "Hard AI known hand", match.hard.knownHand],
    ["hardKnownTop", "Hard AI known top", match.hard.knownTop],
    ["hardDiscard", "Hard AI discard", match.hard.discard],
    ["hardInPlay", "Hard AI in play", match.hard.inPlay],
    ["scrapHeap", "scrap heap", match.scrapHeap],
  ];
  return zones.flatMap(([zone, label, cards]) => cards.filter((item) => {
    const cardId = zone === "astroInPlay" || zone === "hardInPlay" ? effectiveCardId(item) : item.cardId;
    return cardId === null || !definitions.has(cardId);
  }).map((item) => ({ zone, uid: item.uid, label })));
}

function astroHandCandidateCatalog(match: ManualMatch, target: TrackedCard, catalog: CardDefinition[]): CardDefinition[] {
  const possibleIds = new Set<number>();
  const targetCardId = target.cardId ?? target.reservedCardId;
  if (targetCardId !== null && targetCardId !== undefined) possibleIds.add(targetCardId);
  for (const item of match.astro.deck) {
    if (item.cardId !== null) possibleIds.add(item.cardId);
  }
  for (const item of match.astro.hand) {
    if (item.uid === target.uid || item.cardId !== null || item.reservedCardId === null || item.reservedCardId === undefined) continue;
    possibleIds.add(item.reservedCardId);
  }
  // Legacy opening positions stored before reservations were introduced still
  // have a known Scout/Viper starter composition.
  if (!possibleIds.size && match.turn === 1 && !match.astro.discard.length && !match.astro.inPlay.length) {
    possibleIds.add(0);
    possibleIds.add(1);
  }
  return catalog.filter((definition) => possibleIds.has(definition.card_id));
}

function buildMainActions(match: ManualMatch, definitions: Map<number, CardDefinition>): SemanticAction[] {
  const actions: SemanticAction[] = [];
  const semanticKeys = new Set<string>();
  const add = (action: Omit<SemanticAction, "id">) => {
    const key = [action.kind, action.card_id ?? -1, action.target_card_id ?? -1, action.ability ?? "", action.source_zone ?? "", action.amount ?? 0, action.amount2 ?? 0].join(":");
    if (semanticKeys.has(key)) return;
    semanticKeys.add(key);
    actions.push({ id: actions.length, ...action });
  };

  for (const item of match.astro.hand) {
    if (item.cardId === null) continue;
    const definition = definitions.get(item.cardId);
    add({ kind: "play_card", card_id: item.cardId, source_zone: "hand", label: `Play ${definition?.name ?? "card"}` });
  }
  for (const item of match.astro.inPlay) {
    const definition = effectiveDefinition(item, definitions);
    if (!definition) continue;
    if (definition && definition.card_type !== "ship" && !item.activated && (definition.combat || definition.trade || definition.authority || definition.primary)) {
      add({ kind: "activate_base", card_id: definition.card_id, source_zone: "in_play", label: `Activate ${definition.name}` });
    }
    if (definition?.scrap) {
      add({ kind: "scrap_for_ability", card_id: definition.card_id, source_zone: "in_play", ability: definition.scrap, amount: definition.scrap_amount, label: `Scrap ${definition.name} · ${abilityLabel(definition.scrap, definition.scrap_amount)}` });
    }
  }

  if (match.astro.combat > 0) {
    const outposts = match.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "outpost");
    const bases = outposts.length ? outposts : match.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "base");
    for (const target of bases) {
      const definition = effectiveDefinition(target, definitions);
      if (definition && match.astro.combat >= definition.defense) {
        add({ kind: "attack_base", target_card_id: definition.card_id, source_zone: "opponent_in_play", amount: definition.defense, amount2: match.astro.combat, label: `Attack ${definition.name}` });
      }
    }
    if (!outposts.length) add({ kind: "attack_player", amount: match.astro.combat, label: `Attack Hard AI for ${match.astro.combat}` });
  }

  for (const item of match.tradeRow) {
    if (item.cardId === null) continue;
    const definition = definitions.get(item.cardId);
    if (definition && definition.cost <= match.astro.trade) {
      add({ kind: "acquire", card_id: definition.card_id, source_zone: "trade_row", amount: definition.cost, label: `Acquire ${definition.name}` });
    }
  }
  if (match.explorersRemaining > 0 && match.astro.trade >= 2) {
    add({ kind: "acquire", card_id: 2, source_zone: "explorer_supply", amount: 2, label: "Acquire Explorer" });
  }
  add({ kind: "end_turn", label: "End turn" });
  return actions;
}

function buildObservation(match: ManualMatch, catalog: CardDefinition[], definitions: Map<number, CardDefinition>) {
  const inPlay = (cards: TrackedCard[]) => cards.flatMap((item) => {
    const definition = effectiveDefinition(item, definitions);
    return definition ? [{ card: definition, activated: Boolean(item.activated), ally_triggered: Boolean(item.allyTriggered), copied_from_stealth_needle: item.copiedCardId !== undefined && item.copiedCardId !== null }] : [];
  });
  const tradeDeck = remainingTradeDeck(match, catalog);
  return {
    version: 2,
    player_id: 0,
    active_player: match.activeSide === "astro5" ? 0 : 1,
    starting_player: match.startingSide === "astro5" ? 0 : 1,
    is_starting_player: match.startingSide === "astro5",
    turn: match.turn,
    action_number: match.actionNumber,
    own_authority: match.astro.authority,
    opponent_authority: match.hard.authority,
    opponent_pending_discard: match.hard.pendingDiscard,
    combat: match.astro.combat,
    trade: match.astro.trade,
    pending_discard: match.astro.pendingDiscard,
    hand: definitionList(match.astro.hand, definitions),
    own_deck_count: match.astro.deck.length + match.astro.knownTop.length,
    own_deck: definitionList(match.astro.deck, definitions),
    own_known_top: definitionList(match.astro.knownTop, definitions),
    own_discard: definitionList(match.astro.discard, definitions),
    own_in_play: inPlay(match.astro.inPlay),
    opponent_hand_count: match.hard.handCount,
    opponent_known_hand: definitionList(match.hard.knownHand, definitions),
    opponent_hidden: definitionList(match.hard.hidden, definitions),
    opponent_deck_count: match.hard.deckCount + match.hard.knownTop.length,
    opponent_known_top: definitionList(match.hard.knownTop, definitions),
    opponent_discard: definitionList(match.hard.discard, definitions),
    opponent_in_play: inPlay(match.hard.inPlay),
    trade_row: match.tradeRow.map((item) => item.cardId === null ? null : definitions.get(item.cardId) ?? null),
    trade_deck_count: tradeDeck.length,
    trade_deck: tradeDeck,
    explorers_remaining: match.explorersRemaining,
    explorer_supply: Array.from({ length: match.explorersRemaining }, () => definitions.get(2)).filter(Boolean),
    scrap_heap: definitionList(match.scrapHeap, definitions),
    next_ship_to_top: Boolean(match.astro.nextShipToTop),
    blob_cards_played: match.astro.inPlay.filter((item) => item.playedTurn === match.turn && item.cardId !== 23 && effectiveDefinition(item, definitions)?.faction === "blob").length,
    all_allied: match.astro.inPlay.some((item) => effectiveCardId(item) === 19),
    fleet_active: match.astro.inPlay.some((item) => effectiveCardId(item) === 29),
  };
}

function zoneArray(match: ManualMatch, zone: ZoneId): TrackedCard[] {
  if (zone === "tradeRow") return match.tradeRow;
  if (zone === "scrapHeap") return match.scrapHeap;
  if (zone === "astroHand") return match.astro.hand;
  if (zone === "astroDeck") return match.astro.deck;
  if (zone === "astroKnownTop") return match.astro.knownTop;
  if (zone === "astroDiscard") return match.astro.discard;
  if (zone === "astroInPlay") return match.astro.inPlay;
  if (zone === "hardHidden") return match.hard.hidden;
  if (zone === "hardKnownHand") return match.hard.knownHand;
  if (zone === "hardKnownTop") return match.hard.knownTop;
  if (zone === "hardDiscard") return match.hard.discard;
  return match.hard.inPlay;
}

function replaceZone(match: ManualMatch, zone: ZoneId, cards: TrackedCard[]): ManualMatch {
  if (zone === "tradeRow") return { ...match, tradeRow: cards };
  if (zone === "scrapHeap") return { ...match, scrapHeap: cards };
  if (zone === "astroHand") return { ...match, astro: { ...match.astro, hand: cards } };
  if (zone === "astroDeck") return { ...match, astro: { ...match.astro, deck: cards } };
  if (zone === "astroKnownTop") return { ...match, astro: { ...match.astro, knownTop: cards } };
  if (zone === "astroDiscard") return { ...match, astro: { ...match.astro, discard: cards } };
  if (zone === "astroInPlay") return { ...match, astro: { ...match.astro, inPlay: cards } };
  if (zone === "hardHidden") return { ...match, hard: { ...match.hard, hidden: cards } };
  if (zone === "hardKnownHand") return { ...match, hard: { ...match.hard, knownHand: cards } };
  if (zone === "hardKnownTop") return { ...match, hard: { ...match.hard, knownTop: cards } };
  if (zone === "hardDiscard") return { ...match, hard: { ...match.hard, discard: cards } };
  return { ...match, hard: { ...match.hard, inPlay: cards } };
}

function withEvent(match: ManualMatch, side: MatchEvent["side"], text: string, eventTurn = match.turn): ManualMatch {
  return {
    ...match,
    events: [{ id: uid("event"), turn: eventTurn, side, text }, ...match.events].slice(0, 100),
  };
}

function reconcileInitialAstroDeck(match: ManualMatch): ManualMatch {
  const isOpeningState = match.turn === 1 && !match.astro.discard.length && !match.astro.inPlay.length;
  const definedOpeningHand = match.astro.hand.length > 0 && match.astro.hand.every((item) => item.cardId !== null);
  const untouchedDeck = match.astro.deck.every((item) => item.cardId === null);
  if (!isOpeningState || !definedOpeningHand || !untouchedDeck) return match;
  const handIds = match.astro.hand.map((item) => item.cardId);
  if (handIds.some((cardId) => cardId !== 0 && cardId !== 1)) return match;
  const scouts = Math.max(0, 8 - handIds.filter((cardId) => cardId === 0).length);
  const vipers = Math.max(0, 2 - handIds.filter((cardId) => cardId === 1).length);
  return {
    ...match,
    astro: {
      ...match.astro,
      deck: [
        ...Array.from({ length: scouts }, () => card(0, "astro-deck-scout")),
        ...Array.from({ length: vipers }, () => card(1, "astro-deck-viper")),
      ],
    },
  };
}

function drawAstroCards(match: ManualMatch, count: number): ManualMatch {
  let deck = [...match.astro.deck];
  const knownTop = [...match.astro.knownTop];
  let discard = [...match.astro.discard];
  const drawn: TrackedCard[] = [];
  for (let index = 0; index < count; index += 1) {
    const known = knownTop.shift();
    if (known) {
      drawn.push({ ...originalCard(known), uid: uid("astro-known-draw"), reservedCardId: known.cardId });
      continue;
    }
    if (!deck.length && discard.length) {
      deck = discard;
      discard = [];
    }
    if (!deck.length) break;
    const reserved = deck.pop();
    drawn.push({ uid: uid("astro-draw"), cardId: null, reservedCardId: reserved?.cardId ?? null });
  }
  return {
    ...match,
    astro: { ...match.astro, deck, knownTop, discard, hand: [...match.astro.hand, ...drawn] },
  };
}

function drawHardCards(match: ManualMatch, count: number): ManualMatch {
  let hidden = [...match.hard.hidden];
  const knownHand = [...match.hard.knownHand];
  const knownTop = [...match.hard.knownTop];
  let discard = [...match.hard.discard];
  let handCount = match.hard.handCount;
  let deckCount = match.hard.deckCount;
  for (let index = 0; index < count; index += 1) {
    if (deckCount <= 0 && discard.length) {
      hidden = [...hidden, ...discard.map(originalCard)];
      deckCount += discard.length;
      discard = [];
    }
    if (deckCount <= 0) break;
    const known = knownTop.shift();
    if (known) knownHand.push(originalCard(known));
    deckCount -= 1;
    handCount += 1;
  }
  return {
    ...match,
    hard: { ...match.hard, hidden, knownHand, knownTop, discard, handCount, deckCount },
  };
}

function applyAutomaticHardEffect(match: ManualMatch, effect: string, amount: number, definitions: Map<number, CardDefinition>): ManualMatch {
  if (!effect || HARD_DECISION_EFFECTS.has(effect)) {
    if (effect === "draw_then_scrap" || effect === "draw_destroy") return drawHardCards(match, 1);
    return match;
  }
  if (effect === "gain_combat") return { ...match, hard: { ...match.hard, combat: match.hard.combat + amount } };
  if (effect === "gain_trade") return { ...match, hard: { ...match.hard, trade: match.hard.trade + amount } };
  if (effect === "gain_authority") return { ...match, hard: { ...match.hard, authority: match.hard.authority + amount } };
  if (effect === "draw") return drawHardCards(match, 1);
  if (effect === "draw_two") return drawHardCards(match, 2);
  if (effect === "opponent_discard") return { ...match, astro: { ...match.astro, pendingDiscard: match.astro.pendingDiscard + 1 } };
  if (effect === "ship_top") return { ...match, hard: { ...match.hard, nextShipToTop: true } };
  if (effect === "embassy_yacht") {
    const baseCount = match.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type !== "ship").length;
    return baseCount >= 2 ? drawHardCards(match, 2) : match;
  }
  return match;
}

function triggerAutomaticHardAllies(match: ManualMatch, definitions: Map<number, CardDefinition>): ManualMatch {
  let next = match;
  while (true) {
    const allAllied = next.hard.inPlay.some((item) => effectiveCardId(item) === 19);
    const candidate = next.hard.inPlay.find((item) => {
      const definition = effectiveDefinition(item, definitions);
      if (item.allyTriggered || !definition?.ally || definition.faction === "unaligned") return false;
      return allAllied || next.hard.inPlay.some((other) => other.uid !== item.uid && effectiveDefinition(other, definitions)?.faction === definition.faction);
    });
    if (!candidate) return next;
    const definition = effectiveDefinition(candidate, definitions);
    if (!definition) return next;
    next = {
      ...next,
      hard: { ...next.hard, inPlay: next.hard.inPlay.map((item) => item.uid === candidate.uid ? { ...item, allyTriggered: true } : item) },
    };
    next = applyAutomaticHardEffect(next, definition.ally, definition.ally_amount, definitions);
  }
}

function newlyTriggeredHardAllyDecision(match: ManualMatch, played: CardDefinition, definitions: Map<number, CardDefinition>): CardDefinition | undefined {
  const existing = match.hard.inPlay.map((item) => ({ item, definition: effectiveDefinition(item, definitions) }));
  const eventual = [...existing.map(({ definition }) => definition), played].filter((item): item is CardDefinition => Boolean(item));
  const allAllied = eventual.some((definition) => definition.card_id === 19);
  return [...existing, { item: { uid: "new", cardId: played.card_id, allyTriggered: false }, definition: played }].find(({ item, definition }, index, entries) => {
    if (item.allyTriggered || !definition?.ally || !HARD_DECISION_EFFECTS.has(definition.ally) || definition.faction === "unaligned") return false;
    return allAllied || entries.some(({ definition: other }, otherIndex) => otherIndex !== index && other?.faction === definition.faction);
  })?.definition;
}

function effectDecision(
  effect: string,
  source: CardDefinition,
  match: ManualMatch,
  definitions: Map<number, CardDefinition>,
  progress: Pick<PendingDecision, "remaining" | "completed" | "followupEffect"> = {},
): PendingDecision | null {
  const action = (details: Omit<SemanticAction, "id">, index: number): SemanticAction => ({ id: index, ...details });
  const pending = (family: string, prompt: string, actions: SemanticAction[], extras: Partial<PendingDecision> = {}): PendingDecision => {
    const semanticKeys = new Set<string>();
    const uniqueActions = actions.filter((item) => {
      const key = [item.kind, item.card_id ?? -1, item.target_card_id ?? -1, item.ability ?? "", item.source_zone ?? "", item.amount ?? 0, item.amount2 ?? 0].join(":");
      if (semanticKeys.has(key)) return false;
      semanticKeys.add(key);
      return true;
    }).map((item, index) => ({ ...item, id: index }));
    return {
      family,
      prompt,
      actions: uniqueActions,
      effect,
      sourceCardId: source.card_id,
      ...progress,
      ...extras,
    };
  };

  if (effect === "scrap_trade_row") {
    const actions = match.tradeRow.flatMap((item) => item.cardId === null ? [] : [item]).map((item, index) => action({ kind: "scrap_trade_row", card_id: source.card_id, target_card_id: item.cardId ?? -1, ability: "scrap_trade_row", source_zone: "trade_row", label: `Scrap ${definitions.get(item.cardId ?? -1)?.name ?? "trade-row card"}` }, index));
    actions.push(action({ kind: "decline", card_id: source.card_id, ability: "scrap_trade_row", label: "Decline" }, actions.length));
    return pending("scrap_trade_row", `${source.name}: optionally scrap a trade-row card`, actions);
  }

  if (effect === "scrap_any" || effect === "scrap_two_draw") {
    const candidates = [
      ...match.astro.discard.map((item) => ({ item, zone: "discard" })),
      ...match.astro.hand.map((item) => ({ item, zone: "hand" })),
    ].filter(({ item }) => item.cardId !== null);
    const actions = candidates.map(({ item, zone }, index) => action({ kind: "scrap_card", card_id: item.cardId ?? -1, source_zone: zone, label: `Scrap ${definitions.get(item.cardId ?? -1)?.name ?? "card"} from ${zone}` }, index));
    actions.push(action({ kind: "decline", card_id: source.card_id, ability: "scrap_any", label: effect === "scrap_two_draw" ? "Finish scrapping" : "Decline" }, actions.length));
    return pending("scrap", `${source.name}: ${effect === "scrap_two_draw" ? "scrap up to two cards" : "scrap from hand or discard"}`, actions, effect === "scrap_two_draw" ? { remaining: progress.remaining ?? 2, completed: progress.completed ?? 0 } : {});
  }

  if (effect === "draw_then_scrap") {
    const actions = match.astro.hand.filter((item) => item.cardId !== null).map((item, index) => action({ kind: "scrap_card", card_id: item.cardId ?? -1, source_zone: "hand", label: `Scrap ${definitions.get(item.cardId ?? -1)?.name ?? "card"} from hand` }, index));
    return match.astro.hand.length ? pending("scrap", `${source.name}: scrap a card from hand`, actions) : null;
  }

  if (effect === "recycle_cycle") {
    if (!match.astro.hand.length) return null;
    const actions = match.astro.hand.filter((item) => item.cardId !== null).map((item, index) => action({ kind: "discard_card", card_id: item.cardId ?? -1, source_zone: "hand", label: `Discard ${definitions.get(item.cardId ?? -1)?.name ?? "card"}` }, index));
    actions.push(action({ kind: "decline", card_id: source.card_id, ability: "cycle", label: "Finish cycling" }, actions.length));
    return pending("discard", "Discard a card to replace", actions, { remaining: progress.remaining ?? 2, completed: progress.completed ?? 0 });
  }

  const modes: Record<string, Array<[string, number, string]>> = {
    trading_post: [["gain_authority", 1, "Gain 1 authority"], ["gain_trade", 1, "Gain 1 trade"]],
    patrol_mech: [["gain_combat", 5, "Gain 5 combat"], ["gain_trade", 3, "Gain 3 trade"]],
    barter_world: [["gain_authority", 2, "Gain 2 authority"], ["gain_trade", 2, "Gain 2 trade"]],
    defense_center: [["gain_combat", 2, "Gain 2 combat"], ["gain_authority", 3, "Gain 3 authority"]],
  };
  if (modes[effect]) {
    return pending("ability_mode", `${source.name}: choose a resource`, modes[effect].map(([ability, amount, label], index) => action({ kind: "choose_mode", card_id: source.card_id, ability, amount, label }, index)));
  }

  if (effect === "blob_world") {
    const blobCount = match.astro.inPlay.filter((item) => item.playedTurn === match.turn && item.cardId !== 23 && effectiveDefinition(item, definitions)?.faction === "blob").length;
    return pending("ability_mode", `${source.name}: choose combat or cards`, [
      action({ kind: "choose_mode", card_id: source.card_id, ability: "gain_combat", amount: 5, label: "Gain 5 combat" }, 0),
      action({ kind: "choose_mode", card_id: source.card_id, ability: "draw", amount: blobCount, label: `Draw ${blobCount}` }, 1),
    ]);
  }

  if (effect === "recycle") {
    return pending("ability_mode", `${source.name}: gain trade or cycle cards`, [
      action({ kind: "choose_mode", card_id: source.card_id, ability: "gain_trade", amount: 1, label: "Gain 1 trade" }, 0),
      action({ kind: "choose_mode", card_id: source.card_id, ability: "cycle", amount: 2, label: "Discard up to 2, then draw" }, 1),
    ]);
  }

  if (effect === "free_ship") {
    const ships = match.tradeRow.filter((item) => item.cardId !== null && definitions.get(item.cardId)?.card_type === "ship");
    if (!ships.length) return null;
    const actions = ships.map((item, index) => action({ kind: "free_acquire", card_id: source.card_id, target_card_id: item.cardId ?? -1, ability: "free_ship_to_top", source_zone: "trade_row", label: `Acquire ${definitions.get(item.cardId ?? -1)?.name ?? "ship"} free to deck top` }, index));
    actions.push(action({ kind: "decline", card_id: source.card_id, ability: "free_ship_to_top", label: "Decline" }, actions.length));
    return pending("free_acquire", `${source.name}: optionally acquire a ship free`, actions);
  }

  if (effect === "copy_ship") {
    const sourceItem = [...match.astro.inPlay].reverse().find((item) => item.cardId === source.card_id && !item.activated)
      ?? [...match.astro.inPlay].reverse().find((item) => item.cardId === source.card_id);
    const ships = match.astro.inPlay.filter((item) => item.uid !== sourceItem?.uid && effectiveDefinition(item, definitions)?.card_type === "ship");
    if (!ships.length) return null;
    return pending("copy_ship", `${source.name}: copy a ship`, ships.map((item, index) => {
      const copied = effectiveDefinition(item, definitions);
      return action({ kind: "copy_ship", card_id: source.card_id, target_card_id: copied?.card_id ?? -1, ability: "copy_ship", source_zone: "in_play", label: `Copy ${copied?.name ?? "ship"}` }, index);
    }));
  }

  if (effect === "destroy_base" || effect === "draw_destroy") {
    const outposts = match.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "outpost");
    const bases = outposts.length ? outposts : match.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "base");
    if (!bases.length) return null;
    const actions = bases.map((item, index) => {
      const target = effectiveDefinition(item, definitions);
      return action({ kind: "destroy_base", card_id: source.card_id, target_card_id: target?.card_id ?? -1, ability: "destroy_base", source_zone: "opponent_in_play", label: `Destroy ${target?.name ?? "base"}` }, index);
    });
    actions.push(action({ kind: "decline", card_id: source.card_id, ability: "destroy_base", label: "Decline" }, actions.length));
    return pending("destroy_base", `${source.name}: optionally destroy a base`, actions);
  }
  return null;
}

function refreshPendingDecision(pending: PendingDecision, match: ManualMatch, definitions: Map<number, CardDefinition>): PendingDecision {
  if (!pending.effect || pending.sourceCardId === undefined) return pending;
  const source = definitions.get(pending.sourceCardId);
  if (!source) return pending;
  return effectDecision(pending.effect, source, match, definitions, pending) ?? pending;
}

function hardFollowup(effect: string, cardName: string): { kind: HardActionKind; notice: string } | null {
  if (effect === "scrap_trade_row") return { kind: "scrap_row", notice: `${cardName}: which trade-row card did the Hard AI scrap?` };
  if (effect === "scrap_any" || effect === "draw_then_scrap" || effect === "scrap_two_draw") return { kind: "scrap", notice: `${cardName}: which card did the Hard AI scrap?` };
  if (effect === "recycle") return { kind: "ability", notice: `${cardName}: which mode did the Hard AI choose? Record the mode, then any discarded cards.` };
  if (effect === "patrol_mech" || effect === "trading_post" || effect === "barter_world" || effect === "defense_center" || effect === "blob_world" || effect === "copy_ship") return { kind: "ability", notice: `${cardName}: record the option the Hard AI chose.` };
  if (effect === "destroy_base" || effect === "destroy_and_scrap" || effect === "draw_destroy") return { kind: "attack_base", notice: `${cardName}: which base did the Hard AI choose to destroy?` };
  if (effect === "free_ship") return { kind: "acquire", notice: `${cardName}: which ship did the Hard AI acquire for free?` };
  return null;
}

function hardAbilityOptions(effect: string): Array<{ value: string; label: string }> {
  const options: Record<string, Array<{ value: string; label: string }>> = {
    patrol_mech: [{ value: "gain_combat:5", label: "Gain 5 combat" }, { value: "gain_trade:3", label: "Gain 3 trade" }],
    trading_post: [{ value: "gain_authority:1", label: "Gain 1 authority" }, { value: "gain_trade:1", label: "Gain 1 trade" }],
    barter_world: [{ value: "gain_authority:2", label: "Gain 2 authority" }, { value: "gain_trade:2", label: "Gain 2 trade" }],
    defense_center: [{ value: "gain_combat:2", label: "Gain 2 combat" }, { value: "gain_authority:3", label: "Gain 3 authority" }],
    blob_world: [{ value: "gain_combat:5", label: "Gain 5 combat" }, { value: "draw:0", label: "Draw cards (enter count below)" }],
    recycle: [{ value: "gain_trade:1", label: "Gain 1 trade" }, { value: "cycle:0", label: "Discard up to 2, then draw" }],
  };
  return options[effect] ?? [];
}

const HARD_DECISION_EFFECTS = new Set([
  "scrap_trade_row",
  "scrap_any",
  "scrap_two_draw",
  "draw_then_scrap",
  "destroy_base",
  "patrol_mech",
  "copy_ship",
  "recycle",
  "barter_world",
  "defense_center",
  "trading_post",
  "blob_world",
  "free_ship",
  "destroy_and_scrap",
  "draw_destroy",
]);

function hardAttackTargets(match: ManualMatch, definitions: Map<number, CardDefinition>, requireCombat: boolean): TrackedCard[] {
  const outposts = match.astro.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "outpost");
  const candidates = outposts.length ? outposts : match.astro.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "base");
  return requireCombat
    ? candidates.filter((item) => (effectiveDefinition(item, definitions)?.defense ?? Number.POSITIVE_INFINITY) <= match.hard.combat)
    : candidates;
}

function affordableHardAcquisitions(match: ManualMatch, definitions: Map<number, CardDefinition>): TrackedCard[] {
  return match.tradeRow.filter((item) => {
    const definition = item.cardId === null ? undefined : definitions.get(item.cardId);
    return Boolean(definition && definition.cost <= match.hard.trade);
  });
}

function hardLegalActionKinds(match: ManualMatch, definitions: Map<number, CardDefinition>, decisionEffect: string): HardActionKind[] {
  if (decisionEffect) {
    if (decisionEffect === "scrap_trade_row") return match.tradeRow.some((item) => item.cardId !== null) ? ["scrap_row"] : [];
    if (["scrap_any", "scrap_two_draw", "draw_then_scrap"].includes(decisionEffect)) {
      const hasCandidate = match.hard.handCount > 0 || (decisionEffect !== "draw_then_scrap" && match.hard.discard.length > 0);
      return hasCandidate ? ["scrap"] : [];
    }
    if (["destroy_base", "destroy_and_scrap", "draw_destroy"].includes(decisionEffect)) return hardAttackTargets(match, definitions, false).length ? ["attack_base"] : [];
    if (decisionEffect === "free_ship") return match.tradeRow.some((item) => item.cardId !== null && definitions.get(item.cardId)?.card_type === "ship") ? ["acquire"] : [];
    if (decisionEffect === "recycle_cycle") return match.hard.handCount > 0 ? ["discard"] : [];
    return ["ability"];
  }
  if (match.hard.pendingDiscard > 0 && match.hard.handCount > 0) return ["discard"];
  const kinds: HardActionKind[] = [];
  if (match.hard.handCount > 0) kinds.push("play");
  if (affordableHardAcquisitions(match, definitions).length || (match.explorersRemaining > 0 && match.hard.trade >= 2)) kinds.push("acquire");
  if (match.hard.combat > 0 && !match.astro.inPlay.some((item) => effectiveDefinition(item, definitions)?.card_type === "outpost")) kinds.push("attack_player");
  if (hardAttackTargets(match, definitions, true).length) kinds.push("attack_base");
  if (match.hard.inPlay.some((item) => Boolean(effectiveDefinition(item, definitions)?.scrap))) kinds.push("scrap");
  if (match.hard.inPlay.some((item) => {
    const definition = effectiveDefinition(item, definitions);
    return definition && !item.activated && baseRequiresActivation(definition);
  })) kinds.push("ability");
  kinds.push("end_turn");
  return kinds;
}

function playAllIsDecisionFree(match: ManualMatch, definitions: Map<number, CardDefinition>): boolean {
  if (match.astro.hand.length < 2 || match.astro.hand.some((item) => item.cardId === null)) return false;
  const playedDefinitions = match.astro.hand.flatMap((item) => item.cardId === null ? [] : definitions.get(item.cardId) ?? []);
  if (playedDefinitions.length !== match.astro.hand.length) return false;
  const eventualInPlay = [...match.astro.inPlay.map((item) => effectiveDefinition(item, definitions)), ...playedDefinitions].filter((item): item is CardDefinition => Boolean(item));
  const hasNeedleTarget = eventualInPlay.filter((item) => item.card_type === "ship").length > 1;
  if (playedDefinitions.some((definition) => definition.card_type === "ship" && (HARD_DECISION_EFFECTS.has(definition.primary) || ["draw", "draw_two", "embassy_yacht"].includes(definition.primary)))) return false;
  if (playedDefinitions.some((definition) => definition.card_id === 23 && hasNeedleTarget)) return false;
  const allAllied = eventualInPlay.some((definition) => definition.card_id === 19);
  return !eventualInPlay.some((definition, index) => {
    if (!definition.ally || definition.faction === "unaligned") return false;
    const allied = allAllied || eventualInPlay.some((other, otherIndex) => otherIndex !== index && other.faction === definition.faction);
    return allied && (HARD_DECISION_EFFECTS.has(definition.ally) || ["draw", "draw_two"].includes(definition.ally));
  });
}

function loadSavedMatch(fallback: ManualMatch): ManualMatch {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    for (const match of raw.matchAll(/"(?:uid|id)":"[^"]*-([0-9a-z]+)"/g)) {
      const sequence = Number.parseInt(match[1], 36);
      if (Number.isFinite(sequence)) uidSequence = Math.max(uidSequence, sequence);
    }
    const saved = JSON.parse(raw) as ManualMatch;
    return {
      ...saved,
      actionNumber: saved.actionNumber ?? saved.events.filter((event) => event.turn === saved.turn && event.side === "astro5").length,
      astro: { ...saved.astro, nextShipToTop: Boolean(saved.astro.nextShipToTop) },
      hard: { ...saved.hard, nextShipToTop: Boolean(saved.hard.nextShipToTop), knownHand: saved.hard.knownHand ?? [] },
      pendingDecision: saved.pendingDecision ?? null,
    };
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return fallback;
  }
}


export default function ManualHardAiMatch({ apiBase, connected, modelGroups, onToast }: Props) {
  const firstModelId = modelGroups.flatMap((group) => group.models)[0]?.id ?? "";
  const [catalog, setCatalog] = useState<CardDefinition[]>(FALLBACK_CARDS);
  const [match, setMatch] = useState<ManualMatch>(() => loadSavedMatch(createMatch(firstModelId, "astro5")));
  const [setupFirst, setSetupFirst] = useState<Side>("astro5");
  const [setupCheckpoint, setSetupCheckpoint] = useState(firstModelId);
  const [inventoryOpen, setInventoryOpen] = useState(false);
  const [hardActionOpen, setHardActionOpen] = useState(match.activeSide === "hard");
  const [hardActionKind, setHardActionKind] = useState<HardActionKind>(match.hard.pendingDiscard > 0 ? "discard" : "play");
  const [hardCardId, setHardCardId] = useState<number | null>(null);
  const [hardAmount, setHardAmount] = useState(0);
  const [hardTargetUid, setHardTargetUid] = useState("");
  const [hardSourceZone, setHardSourceZone] = useState<"hand" | "discard" | "in_play">("hand");
  const [hardDecisionEffect, setHardDecisionEffect] = useState("");
  const [hardAbilityChoice, setHardAbilityChoice] = useState("");
  const [hardDeclined, setHardDeclined] = useState(false);
  const [hardScrapCount, setHardScrapCount] = useState(0);
  const [hardDecisionNotice, setHardDecisionNotice] = useState(match.hard.pendingDiscard > 0 ? `The Hard AI must discard ${match.hard.pendingDiscard}.` : "");
  const [hardQueuedDecision, setHardQueuedDecision] = useState<HardQueuedDecision | null>(null);
  const [pendingTradeRefillSlot, setPendingTradeRefillSlot] = useState<number | null>(null);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [recommendationState, setRecommendationState] = useState<"idle" | "loading" | "offline" | "error">("idle");
  const [advisorError, setAdvisorError] = useState("");
  const [advisorAttempt, setAdvisorAttempt] = useState(0);
  const definitions = useMemo(() => new Map(catalog.map((item) => [item.card_id, item])), [catalog]);
  const unresolved = useMemo(() => unresolvedCards(match, definitions), [definitions, match]);
  const effectiveSetupCheckpoint = setupCheckpoint || firstModelId;
  const invalidateAdvice = useCallback(() => {
    setRecommendation(null);
    setAdvisorError("");
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiBase}/cards`, { signal: AbortSignal.timeout(3_000) })
      .then(async (response) => {
        if (!response.ok) throw new Error("card catalog unavailable");
        const raw: unknown = await response.json();
        const cards = Array.isArray(raw) ? raw : typeof raw === "object" && raw !== null && Array.isArray((raw as { cards?: unknown }).cards) ? (raw as { cards: unknown[] }).cards : [];
        if (!cancelled && cards.length) setCatalog(cards as CardDefinition[]);
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [apiBase, connected]);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(match));
  }, [match]);

  const updateCard = useCallback((zone: ZoneId, cardUid: string, cardId: number | null) => {
    setMatch((current) => {
      const currentCards = zoneArray(current, zone);
      const target = currentCards.find((item) => item.uid === cardUid);
      let next = replaceZone(current, zone, currentCards.map((item) => {
        if (item.uid !== cardUid) return item;
        if ((zone === "astroInPlay" || zone === "hardInPlay") && item.copiedCardId !== undefined && item.copiedCardId !== null) {
          return { ...item, copiedCardId: cardId };
        }
        return { ...item, cardId, copiedCardId: undefined };
      }));
      if (zone === "astroHand" && target && cardId !== null) {
        const outgoingCardId = target.cardId ?? target.reservedCardId;
        const swapIndex = next.astro.deck.findIndex((item) => item.cardId === cardId);
        if (outgoingCardId !== null && outgoingCardId !== undefined && outgoingCardId !== cardId && swapIndex >= 0) {
          const deck = [...next.astro.deck];
          deck[swapIndex] = { ...deck[swapIndex], cardId: outgoingCardId };
          next = { ...next, astro: { ...next.astro, deck } };
        } else if (outgoingCardId !== null && outgoingCardId !== undefined && outgoingCardId !== cardId) {
          const reservation = next.astro.hand.find((item) => item.uid !== cardUid && item.cardId === null && item.reservedCardId === cardId);
          if (reservation) {
            next = { ...next, astro: { ...next.astro, hand: next.astro.hand.map((item) => item.uid === reservation.uid ? { ...item, reservedCardId: outgoingCardId } : item) } };
          }
        }
      }
      next = reconcileInitialAstroDeck(next);
      return withEvent(next, "system", `${cardId === null ? "Marked" : "Defined"} a card in ${zone}`);
    });
    invalidateAdvice();
  }, [invalidateAdvice]);

  const deleteCard = useCallback((zone: ZoneId, cardUid: string) => {
    setMatch((current) => withEvent(replaceZone(current, zone, zoneArray(current, zone).filter((item) => item.uid !== cardUid)), "system", `Deleted a card from ${zone}`));
    invalidateAdvice();
  }, [invalidateAdvice]);

  const addCard = useCallback((zone: ZoneId) => {
    setMatch((current) => {
      const added = card(null, zone);
      const tracked = zone === "astroInPlay" || zone === "hardInPlay" ? { ...added, activated: false, allyTriggered: false } : added;
      return replaceZone(current, zone, [...zoneArray(current, zone), tracked]);
    });
    invalidateAdvice();
  }, [invalidateAdvice]);

  const toggleActivated = useCallback((zone: ZoneId, cardUid: string) => {
    setMatch((current) => replaceZone(current, zone, zoneArray(current, zone).map((item) => item.uid === cardUid ? { ...item, activated: !item.activated } : item)));
    invalidateAdvice();
  }, [invalidateAdvice]);

  const updateAstroStat = (key: keyof Pick<PlayerBoard, "authority" | "trade" | "combat" | "pendingDiscard">, value: number) => {
    setMatch((current) => ({ ...current, astro: { ...current.astro, [key]: value } }));
    invalidateAdvice();
  };

  const updateHardStat = (key: keyof Pick<HardBoard, "authority" | "trade" | "combat" | "pendingDiscard" | "handCount" | "deckCount">, value: number) => {
    setMatch((current) => ({ ...current, hard: { ...current.hard, [key]: value } }));
    invalidateAdvice();
  };

  const startNewMatch = () => {
    if (!effectiveSetupCheckpoint) {
      onToast("Choose a playable checkpoint first");
      return;
    }
    const next = createMatch(effectiveSetupCheckpoint, setupFirst);
    setMatch(next);
    setRecommendation(null);
    setPendingTradeRefillSlot(null);
    setHardDecisionNotice("");
    setHardQueuedDecision(null);
    setHardDecisionEffect("");
    setHardAbilityChoice("");
    setHardDeclined(false);
    setHardSourceZone("hand");
    setHardScrapCount(0);
    setInventoryOpen(false);
    setHardActionOpen(setupFirst === "hard");
    onToast("Hard AI companion match ready");
  };

  const currentDecision = useMemo<PendingDecision>(() => {
    if (match.pendingDecision) return refreshPendingDecision(match.pendingDecision, match, definitions);
    if (match.astro.pendingDiscard > 0) {
      return {
        family: "discard",
        prompt: `Choose ${match.astro.pendingDiscard === 1 ? "a card" : `${match.astro.pendingDiscard} cards`} to discard`,
        actions: match.astro.hand.flatMap((item) => {
          if (item.cardId === null) return [];
          return [{ id: 0, kind: "discard_card", card_id: item.cardId, source_zone: "hand", label: `Discard ${definitions.get(item.cardId)?.name ?? "card"}` }];
        }).filter((action, index, actions) => actions.findIndex((candidate) => candidate.card_id === action.card_id) === index).map((action, index) => ({ ...action, id: index })),
      };
    }
    return {
      family: "main",
      prompt: "Choose Astro5's next action",
      actions: buildMainActions(match, definitions),
    };
  }, [definitions, match]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      setRecommendation(null);
      setAdvisorError("");
      if (match.activeSide !== "astro5" || unresolved.length || !match.checkpointId || !currentDecision.actions.length) {
        setRecommendationState("idle");
        return;
      }
      setRecommendationState("loading");
      try {
        const response = await fetch(`${apiBase}/advisor/evaluate`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model_id: match.checkpointId,
            observation: buildObservation(match, catalog, definitions),
            decision: currentDecision,
          }),
          signal: AbortSignal.timeout(10_000),
        });
        if (!response.ok) throw new Error(apiErrorMessage(await response.json().catch(() => null), `Advisor returned ${response.status}`));
        const result = await response.json() as Recommendation;
        if (!cancelled) {
          setRecommendation(result);
          setRecommendationState("idle");
        }
      } catch (error) {
        if (!cancelled) {
          setAdvisorError(error instanceof Error ? error.message : "The advisor could not score this position.");
          setRecommendationState(connected ? "error" : "offline");
        }
      }
    }, 220);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [advisorAttempt, apiBase, catalog, connected, currentDecision, definitions, match, unresolved.length]);

  const applySimpleEffect = useCallback((current: ManualMatch, effect: string, amount: number, source: CardDefinition): ManualMatch => {
    let next = current;
    if (effect === "gain_combat") next = { ...next, astro: { ...next.astro, combat: next.astro.combat + amount } };
    else if (effect === "gain_trade") next = { ...next, astro: { ...next.astro, trade: next.astro.trade + amount } };
    else if (effect === "gain_authority") next = { ...next, astro: { ...next.astro, authority: next.astro.authority + amount } };
    else if (effect === "draw") next = drawAstroCards(next, Math.max(1, amount));
    else if (effect === "draw_two") next = drawAstroCards(next, 2);
    else if (effect === "opponent_discard") next = { ...next, hard: { ...next.hard, pendingDiscard: next.hard.pendingDiscard + Math.max(1, amount) } };
    else if (effect === "ship_top") next = { ...next, astro: { ...next.astro, nextShipToTop: true } };
    else if (effect === "embassy_yacht") {
      const bases = next.astro.inPlay.filter((item) => item.cardId !== null && definitions.get(item.cardId)?.card_type !== "ship").length;
      if (bases >= 2) next = drawAstroCards(next, 2);
    }
    else if (effect === "draw_destroy" || effect === "draw_then_scrap") {
      next = drawAstroCards(next, 1);
      const decision = effectDecision(effect, source, next, definitions);
      if (decision) next = { ...next, pendingDecision: decision };
    }
    else if (effect === "destroy_and_scrap") {
      const destroy = effectDecision("destroy_base", source, next, definitions);
      if (destroy) next = { ...next, pendingDecision: { ...destroy, followupEffect: "scrap_trade_row" } };
      else {
        const scrap = effectDecision("scrap_trade_row", source, next, definitions);
        if (scrap) next = { ...next, pendingDecision: scrap };
      }
    }
    else {
      const decision = effectDecision(effect, source, next, definitions);
      if (decision) next = { ...next, pendingDecision: decision };
    }
    return next;
  }, [definitions]);

  const applyAstroAction = useCallback((action: SemanticAction) => {
    setMatch((current) => {
      const priorDecision = current.pendingDecision;
      let next: ManualMatch = { ...current, pendingDecision: null, actionNumber: current.actionNumber + 1 };
      const definition = action.card_id === undefined ? undefined : definitions.get(action.card_id);
      const triggerAvailableAllies = (state: ManualMatch): ManualMatch => {
        let result = state;
        while (!result.pendingDecision) {
          const allAllied = result.astro.inPlay.some((item) => effectiveCardId(item) === 19);
          const candidate = result.astro.inPlay.find((item) => {
            const itemDefinition = effectiveDefinition(item, definitions);
            if (item.allyTriggered || !itemDefinition?.ally || itemDefinition.faction === "unaligned") return false;
            return allAllied || result.astro.inPlay.some((other) => other.uid !== item.uid && effectiveDefinition(other, definitions)?.faction === itemDefinition.faction);
          });
          if (!candidate) break;
          const candidateDefinition = effectiveDefinition(candidate, definitions);
          if (!candidateDefinition) break;
          result = {
            ...result,
            astro: {
              ...result.astro,
              inPlay: result.astro.inPlay.map((item) => item.uid === candidate.uid ? { ...item, allyTriggered: true } : item),
            },
          };
          result = applySimpleEffect(result, candidateDefinition.ally, candidateDefinition.ally_amount, candidateDefinition);
        }
        return result;
      };
      if (action.kind === "play_card" && definition) {
        const played = next.astro.hand.find((item) => item.cardId === definition.card_id);
        if (!played) return current;
        const isNeedle = definition.card_id === 23;
        const isShip = definition.card_type === "ship";
        const inPlayCard = { ...played, activated: isShip ? !isNeedle : !baseRequiresActivation(definition), allyTriggered: false, playedTurn: next.turn };
        next = {
          ...next,
          astro: {
            ...next.astro,
            hand: next.astro.hand.filter((item) => item.uid !== played.uid),
            inPlay: [...next.astro.inPlay, inPlayCard],
          },
        };
        if (isNeedle) {
          const decision = effectDecision("copy_ship", definition, next, definitions);
          if (decision) next = { ...next, pendingDecision: decision };
          else next = { ...next, astro: { ...next.astro, inPlay: next.astro.inPlay.map((item) => item.uid === inPlayCard.uid ? { ...item, activated: true } : item) } };
        } else if (isShip) {
          const fleetBonus = next.astro.inPlay.some((item) => effectiveCardId(item) === 29) ? 1 : 0;
          next = { ...next, astro: {
            ...next.astro,
            authority: next.astro.authority + definition.authority,
            trade: next.astro.trade + definition.trade,
            combat: next.astro.combat + definition.combat + fleetBonus,
          } };
          if (definition.primary) next = applySimpleEffect(next, definition.primary, 0, definition);
        }
      } else if (action.kind === "activate_base" && definition) {
        const target = next.astro.inPlay.find((item) => effectiveCardId(item) === definition.card_id && !item.activated);
        if (!target) return current;
        next = { ...next, astro: {
          ...next.astro,
          authority: next.astro.authority + definition.authority,
          trade: next.astro.trade + definition.trade,
          combat: next.astro.combat + definition.combat,
          inPlay: next.astro.inPlay.map((item) => item.uid === target.uid ? { ...item, activated: true } : item),
        } };
        if (definition.primary) next = applySimpleEffect(next, definition.primary, 0, definition);
      } else if (action.kind === "scrap_for_ability" && definition) {
        const target = next.astro.inPlay.find((item) => effectiveCardId(item) === definition.card_id);
        if (target) next = { ...next, astro: { ...next.astro, inPlay: next.astro.inPlay.filter((item) => item.uid !== target.uid) }, scrapHeap: [...next.scrapHeap, originalCard(target)] };
        next = applySimpleEffect(next, action.ability ?? definition.scrap, action.amount ?? definition.scrap_amount, definition);
      } else if (action.kind === "attack_player") {
        const amount = action.amount ?? next.astro.combat;
        next = { ...next, astro: { ...next.astro, combat: Math.max(0, next.astro.combat - amount) }, hard: { ...next.hard, authority: Math.max(0, next.hard.authority - amount) } };
      } else if (action.kind === "attack_base") {
        const target = next.hard.inPlay.find((item) => effectiveCardId(item) === action.target_card_id);
        if (target) next = { ...next, astro: { ...next.astro, combat: Math.max(0, next.astro.combat - (action.amount ?? 0)) }, hard: { ...next.hard, inPlay: next.hard.inPlay.filter((item) => item.uid !== target.uid), discard: [...next.hard.discard, originalCard(target)] } };
      } else if (action.kind === "acquire" && definition) {
        if (action.source_zone === "explorer_supply") {
          next = { ...next, explorersRemaining: Math.max(0, next.explorersRemaining - 1), astro: { ...next.astro, trade: Math.max(0, next.astro.trade - (action.amount ?? 2)), discard: [...next.astro.discard, card(2, "astro-explorer")] } };
        } else {
          const target = next.tradeRow.find((item) => item.cardId === definition.card_id);
          if (target) {
            const goesOnTop = next.astro.nextShipToTop && definition.card_type === "ship";
            next = { ...next, tradeRow: next.tradeRow.map((item) => item.uid === target.uid ? card(null, "trade-row-refill") : item), astro: { ...next.astro, trade: Math.max(0, next.astro.trade - (action.amount ?? definition.cost)), nextShipToTop: goesOnTop ? false : next.astro.nextShipToTop, discard: goesOnTop ? next.astro.discard : [...next.astro.discard, target], knownTop: goesOnTop ? [target, ...next.astro.knownTop] : next.astro.knownTop } };
          }
        }
      } else if (action.kind === "choose_mode") {
        if (action.ability === "draw") next = drawAstroCards(next, action.amount ?? 1);
        else if (action.ability === "gain_combat") next = { ...next, astro: { ...next.astro, combat: next.astro.combat + (action.amount ?? 0) } };
        else if (action.ability === "gain_trade") next = { ...next, astro: { ...next.astro, trade: next.astro.trade + (action.amount ?? 0) } };
        else if (action.ability === "gain_authority") next = { ...next, astro: { ...next.astro, authority: next.astro.authority + (action.amount ?? 0) } };
        else if (action.ability === "cycle") {
          const source = action.card_id === undefined ? undefined : definitions.get(action.card_id);
          const decision = source ? effectDecision("recycle_cycle", source, next, definitions) : null;
          if (decision) next = { ...next, pendingDecision: decision };
        }
      } else if (action.kind === "scrap_card") {
        const source = action.source_zone === "hand" ? next.astro.hand : next.astro.discard;
        const target = source.find((item) => item.cardId === action.card_id);
        if (target) next = { ...next, astro: { ...next.astro, [action.source_zone === "hand" ? "hand" : "discard"]: source.filter((item) => item.uid !== target.uid) }, scrapHeap: [...next.scrapHeap, target] };
        if (target && priorDecision?.effect === "scrap_two_draw" && priorDecision.sourceCardId !== undefined) {
          const completed = (priorDecision.completed ?? 0) + 1;
          const remaining = Math.max(0, (priorDecision.remaining ?? 2) - 1);
          const sourceDefinition = definitions.get(priorDecision.sourceCardId);
          const decision = sourceDefinition && remaining > 0 ? effectDecision("scrap_two_draw", sourceDefinition, next, definitions, { remaining, completed }) : null;
          next = decision ? { ...next, pendingDecision: decision } : drawAstroCards(next, completed);
        }
      } else if (action.kind === "discard_card") {
        const target = next.astro.hand.find((item) => item.cardId === action.card_id);
        if (target) next = { ...next, astro: { ...next.astro, hand: next.astro.hand.filter((item) => item.uid !== target.uid), discard: [...next.astro.discard, target], pendingDiscard: Math.max(0, next.astro.pendingDiscard - 1) } };
        if (target && priorDecision?.effect === "recycle_cycle" && priorDecision.sourceCardId !== undefined) {
          const completed = (priorDecision.completed ?? 0) + 1;
          const remaining = Math.max(0, (priorDecision.remaining ?? 2) - 1);
          const sourceDefinition = definitions.get(priorDecision.sourceCardId);
          const decision = sourceDefinition && remaining > 0 && next.astro.hand.length ? effectDecision("recycle_cycle", sourceDefinition, next, definitions, { remaining, completed }) : null;
          next = decision ? { ...next, pendingDecision: decision } : drawAstroCards(next, completed);
        }
      } else if (action.kind === "scrap_trade_row") {
        const target = next.tradeRow.find((item) => item.cardId === action.target_card_id);
        if (target) next = { ...next, tradeRow: next.tradeRow.map((item) => item.uid === target.uid ? card(null, "trade-row-refill") : item), scrapHeap: [...next.scrapHeap, target] };
      } else if (action.kind === "destroy_base") {
        const target = next.hard.inPlay.find((item) => effectiveCardId(item) === action.target_card_id);
        if (target) next = { ...next, hard: { ...next.hard, inPlay: next.hard.inPlay.filter((item) => item.uid !== target.uid), discard: [...next.hard.discard, originalCard(target)] } };
        if (priorDecision?.followupEffect && priorDecision.sourceCardId !== undefined) {
          const sourceDefinition = definitions.get(priorDecision.sourceCardId);
          const scrap = sourceDefinition ? effectDecision(priorDecision.followupEffect, sourceDefinition, next, definitions) : null;
          if (scrap) next = { ...next, pendingDecision: scrap };
        }
      } else if (action.kind === "decline") {
        if (priorDecision?.effect === "scrap_two_draw" && priorDecision.sourceCardId !== undefined) {
          const completed = priorDecision.completed ?? 0;
          const remaining = Math.max(0, (priorDecision.remaining ?? 2) - 1);
          const sourceDefinition = definitions.get(priorDecision.sourceCardId);
          const decision = sourceDefinition && remaining > 0 ? effectDecision("scrap_two_draw", sourceDefinition, next, definitions, { remaining, completed }) : null;
          next = decision ? { ...next, pendingDecision: decision } : drawAstroCards(next, completed);
        } else if (priorDecision?.effect === "recycle_cycle" && (priorDecision.completed ?? 0) > 0) {
          next = drawAstroCards(next, priorDecision.completed ?? 0);
        }
        if (priorDecision?.followupEffect && priorDecision.sourceCardId !== undefined) {
          const sourceDefinition = definitions.get(priorDecision.sourceCardId);
          const followup = sourceDefinition ? effectDecision(priorDecision.followupEffect, sourceDefinition, next, definitions) : null;
          if (followup) next = { ...next, pendingDecision: followup };
        }
      } else if (action.kind === "free_acquire") {
        const target = next.tradeRow.find((item) => item.cardId === action.target_card_id);
        if (target) next = { ...next, tradeRow: next.tradeRow.map((item) => item.uid === target.uid ? card(null, "trade-row-refill") : item), astro: { ...next.astro, nextShipToTop: false, knownTop: [target, ...next.astro.knownTop] } };
      } else if (action.kind === "copy_ship") {
        const copied = action.target_card_id === undefined ? undefined : definitions.get(action.target_card_id);
        if (copied) {
          const needle = [...next.astro.inPlay].reverse().find((item) => item.cardId === (action.card_id ?? 23) && !item.activated)
            ?? [...next.astro.inPlay].reverse().find((item) => item.cardId === (action.card_id ?? 23));
          const fleetBonus = next.astro.inPlay.some((item) => effectiveCardId(item) === 29) ? 1 : 0;
          next = { ...next, astro: {
            ...next.astro,
            authority: next.astro.authority + copied.authority,
            trade: next.astro.trade + copied.trade,
            combat: next.astro.combat + copied.combat + fleetBonus,
            inPlay: next.astro.inPlay.map((item) => item.uid === needle?.uid ? { ...item, copiedCardId: copied.card_id, activated: true } : item),
          } };
          if (copied.primary) next = applySimpleEffect(next, copied.primary, 0, copied);
        }
      } else if (action.kind === "end_turn") {
        const ships = next.astro.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "ship").map(originalCard);
        const bases = next.astro.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type !== "ship").map((item) => {
          const base = effectiveDefinition(item, definitions);
          return { ...item, activated: base ? !baseRequiresActivation(base) : false, allyTriggered: false };
        });
        const hardBases = next.hard.inPlay.map((item) => {
          const base = effectiveDefinition(item, definitions);
          return { ...item, activated: base ? !baseRequiresActivation(base) : false, allyTriggered: false };
        });
        next = { ...next, activeSide: "hard", turn: next.turn + 1, actionNumber: 0, astro: { ...next.astro, trade: 0, combat: 0, nextShipToTop: false, discard: [...next.astro.discard, ...next.astro.hand.map(originalCard), ...ships], hand: [], inPlay: bases }, hard: { ...next.hard, inPlay: hardBases } };
        next = drawAstroCards(next, 5);
      }
      if (!next.pendingDecision && action.kind !== "end_turn") next = triggerAvailableAllies(next);
      return withEvent(next, "astro5", action.label, current.turn);
    });
    if (action.kind === "end_turn") {
      const pendingDiscard = match.hard.pendingDiscard;
      setHardActionKind(pendingDiscard > 0 ? "discard" : "play");
      setHardDecisionEffect("");
      setHardAbilityChoice("");
      setHardDeclined(false);
      setHardDecisionNotice(pendingDiscard > 0 ? `The Hard AI must discard ${pendingDiscard}.` : "");
      setHardActionOpen(true);
    }
    setRecommendation(null);
  }, [applySimpleEffect, definitions, match.hard.pendingDiscard]);

  const playAll = useCallback(() => {
    for (const item of match.astro.hand) {
      if (item.cardId === null) return;
      const definition = definitions.get(item.cardId);
      applyAstroAction({ id: 0, kind: "play_card", card_id: item.cardId, source_zone: "hand", label: `Play ${definition?.name ?? "card"}` });
    }
  }, [applyAstroAction, definitions, match.astro.hand]);

  const recordHardAction = () => {
    const acquiredTradeRowSlot = hardActionKind === "acquire" && hardTargetUid !== "explorer-supply"
      ? match.tradeRow.findIndex((item) => item.uid === hardTargetUid)
      : -1;
    setMatch((current) => {
      let next = current;
      let eventText = "Hard AI action recorded";
      const definition = hardCardId === null ? undefined : definitions.get(hardCardId);
      if (hardDeclined) {
        eventText = `Hard AI declined ${hardDecisionEffect ? abilityLabel(hardDecisionEffect) : "the optional action"}`;
        if (hardDecisionEffect === "scrap_two_draw" && hardScrapCount > 0) next = drawHardCards(next, hardScrapCount);
        if (hardDecisionEffect === "recycle_cycle" && hardScrapCount > 0) next = drawHardCards(next, hardScrapCount);
      } else if (hardActionKind === "play" && definition) {
        const known = next.hard.knownHand.find((item) => item.cardId === definition.card_id);
        const existing = known ?? next.hard.hidden.find((item) => item.cardId === definition.card_id);
        const played = existing ?? card(definition.card_id, "hard-played");
        const hidden = known ? next.hard.hidden : existing ? next.hard.hidden.filter((item) => item.uid !== existing.uid) : next.hard.hidden.slice(0, -1);
        const knownHand = known ? next.hard.knownHand.filter((item) => item.uid !== known.uid) : next.hard.knownHand;
        const isShip = definition.card_type === "ship";
        const activated = isShip ? definition.card_id !== 23 : !baseRequiresActivation(definition);
        next = { ...next, hard: {
          ...next.hard,
          authority: next.hard.authority + (isShip ? definition.authority : 0),
          trade: next.hard.trade + (isShip ? definition.trade : 0),
          combat: next.hard.combat + (isShip ? definition.combat + (next.hard.inPlay.some((item) => effectiveCardId(item) === 29) ? 1 : 0) : 0),
          handCount: Math.max(0, next.hard.handCount - 1),
          hidden,
          knownHand,
          inPlay: [...next.hard.inPlay, { ...played, activated, allyTriggered: false, playedTurn: next.turn }],
        } };
        if (isShip && definition.card_id !== 23) next = applyAutomaticHardEffect(next, definition.primary, 0, definitions);
        next = triggerAutomaticHardAllies(next, definitions);
        eventText = `Hard AI played ${definition.name}`;
      } else if (hardActionKind === "acquire") {
        if (hardTargetUid === "explorer-supply" && next.explorersRemaining > 0) {
          next = { ...next, explorersRemaining: next.explorersRemaining - 1, hard: { ...next.hard, trade: Math.max(0, next.hard.trade - 2), discard: [...next.hard.discard, card(2, "hard-explorer")] } };
          eventText = "Hard AI acquired Explorer";
        }
        const target = next.tradeRow.find((item) => item.uid === hardTargetUid);
        const targetDefinition = target?.cardId === null || target?.cardId === undefined ? undefined : definitions.get(target.cardId);
        if (target && targetDefinition) {
          const freeToTop = hardDecisionEffect === "free_ship";
          const effectToTop = next.hard.nextShipToTop && targetDefinition.card_type === "ship";
          const goesOnTop = freeToTop || effectToTop;
          next = { ...next, tradeRow: next.tradeRow.map((item) => item.uid === target.uid ? { uid: item.uid, cardId: null } : item), hard: {
            ...next.hard,
            trade: freeToTop ? next.hard.trade : Math.max(0, next.hard.trade - targetDefinition.cost),
            nextShipToTop: effectToTop ? false : next.hard.nextShipToTop,
            deckCount: goesOnTop ? next.hard.deckCount + 1 : next.hard.deckCount,
            discard: goesOnTop ? next.hard.discard : [...next.hard.discard, target],
            knownTop: goesOnTop ? [target, ...next.hard.knownTop] : next.hard.knownTop,
          } };
          eventText = `Hard AI acquired ${targetDefinition.name}${freeToTop ? " free to deck top" : effectToTop ? " to deck top" : ""}`;
        }
      } else if (hardActionKind === "attack_player") {
        const amount = Math.max(0, hardAmount || next.hard.combat);
        next = { ...next, astro: { ...next.astro, authority: Math.max(0, next.astro.authority - amount) }, hard: { ...next.hard, combat: Math.max(0, next.hard.combat - amount) } };
        eventText = `Hard AI attacked Astro5 for ${amount}`;
      } else if (hardActionKind === "attack_base") {
        const target = next.astro.inPlay.find((item) => item.uid === hardTargetUid);
        const targetDefinition = target?.cardId === null || target?.cardId === undefined ? undefined : definitions.get(target.cardId);
        if (target && targetDefinition) {
          const freeDestroy = hardDecisionEffect === "destroy_base" || hardDecisionEffect === "destroy_and_scrap" || hardDecisionEffect === "draw_destroy";
          next = { ...next, astro: { ...next.astro, inPlay: next.astro.inPlay.filter((item) => item.uid !== target.uid), discard: [...next.astro.discard, originalCard(target)] }, hard: { ...next.hard, combat: freeDestroy ? next.hard.combat : Math.max(0, next.hard.combat - targetDefinition.defense) } };
          eventText = `Hard AI ${freeDestroy ? "destroyed" : "attacked"} ${targetDefinition.name}`;
        }
      } else if (hardActionKind === "discard" && definition) {
        const known = next.hard.knownHand.find((item) => item.cardId === definition.card_id);
        const existing = known ?? next.hard.hidden.find((item) => item.cardId === definition.card_id);
        const discarded = existing ?? card(definition.card_id, "hard-discarded");
        next = { ...next, hard: { ...next.hard, hidden: known ? next.hard.hidden : existing ? next.hard.hidden.filter((item) => item.uid !== existing.uid) : next.hard.hidden.slice(0, -1), knownHand: known ? next.hard.knownHand.filter((item) => item.uid !== known.uid) : next.hard.knownHand, handCount: Math.max(0, next.hard.handCount - 1), pendingDiscard: Math.max(0, next.hard.pendingDiscard - 1), discard: [...next.hard.discard, originalCard(discarded)] } };
        if (hardDecisionEffect === "recycle_cycle" && hardScrapCount >= 1) next = drawHardCards(next, hardScrapCount + 1);
        eventText = `Hard AI discarded ${definition.name}`;
      } else if (hardActionKind === "scrap" && definition) {
        const zoneKey = hardSourceZone === "hand" ? "hidden" : hardSourceZone === "discard" ? "discard" : "inPlay";
        const sourceCards = next.hard[zoneKey];
        const known = hardSourceZone === "hand" ? next.hard.knownHand.find((item) => effectiveCardId(item) === definition.card_id) : undefined;
        const target = known ?? sourceCards.find((item) => effectiveCardId(item) === definition.card_id);
        if (target) {
          next = { ...next, hard: {
            ...next.hard,
            [zoneKey]: known ? sourceCards : sourceCards.filter((item) => item.uid !== target.uid),
            knownHand: known ? next.hard.knownHand.filter((item) => item.uid !== known.uid) : next.hard.knownHand,
            handCount: hardSourceZone === "hand" ? Math.max(0, next.hard.handCount - 1) : next.hard.handCount,
          }, scrapHeap: [...next.scrapHeap, originalCard(target)] };
          if (hardSourceZone === "in_play" && definition.scrap === "gain_combat") next = { ...next, hard: { ...next.hard, combat: next.hard.combat + definition.scrap_amount } };
          if (hardSourceZone === "in_play" && definition.scrap === "gain_trade") next = { ...next, hard: { ...next.hard, trade: next.hard.trade + definition.scrap_amount } };
          if (hardSourceZone === "in_play" && definition.scrap === "draw") next = drawHardCards(next, 1);
          if (hardSourceZone === "in_play" && definition.scrap === "draw_destroy") next = drawHardCards(next, 1);
          if (hardSourceZone === "in_play" && definition.scrap === "opponent_discard") next = { ...next, astro: { ...next.astro, pendingDiscard: next.astro.pendingDiscard + 1 } };
          if (hardDecisionEffect === "scrap_two_draw" && hardScrapCount >= 1) next = drawHardCards(next, hardScrapCount + 1);
          eventText = `Hard AI scrapped ${definition.name}`;
        }
      } else if (hardActionKind === "ability" && definition) {
        if (hardDecisionEffect === "copy_ship") {
          const target = next.hard.inPlay.find((item) => item.uid === hardTargetUid);
          const copied = target ? effectiveDefinition(target, definitions) : undefined;
          const needle = [...next.hard.inPlay].reverse().find((item) => item.cardId === definition.card_id && !item.copiedCardId);
          if (copied && needle) {
            const fleetBonus = next.hard.inPlay.some((item) => effectiveCardId(item) === 29) ? 1 : 0;
            next = { ...next, hard: {
              ...next.hard,
              authority: next.hard.authority + copied.authority,
              trade: next.hard.trade + copied.trade,
              combat: next.hard.combat + copied.combat + fleetBonus,
              inPlay: next.hard.inPlay.map((item) => item.uid === needle.uid ? { ...item, copiedCardId: copied.card_id, activated: true } : item),
            } };
            next = applyAutomaticHardEffect(next, copied.primary, 0, definitions);
            next = triggerAutomaticHardAllies(next, definitions);
            eventText = `Hard AI copied ${copied.name} with Stealth Needle`;
          }
        } else {
          const baseTarget = next.hard.inPlay.find((item) => effectiveCardId(item) === definition.card_id && !item.activated && definition.card_type !== "ship");
          if (baseTarget) {
            next = { ...next, hard: {
              ...next.hard,
              authority: next.hard.authority + definition.authority,
              trade: next.hard.trade + definition.trade,
              combat: next.hard.combat + definition.combat,
              inPlay: next.hard.inPlay.map((item) => item.uid === baseTarget.uid ? { ...item, activated: true } : item),
            } };
          }
          const recordedEffect = hardDecisionEffect || (baseTarget ? definition.primary : definition.ally);
          const [ability, encodedAmount] = hardAbilityChoice.split(":");
          const resolvedAbility = ability || recordedEffect;
          const amount = Number(encodedAmount) || hardAmount || (recordedEffect === definition.ally ? definition.ally_amount : 0);
          if (resolvedAbility === "gain_combat") next = { ...next, hard: { ...next.hard, combat: next.hard.combat + amount } };
          else if (resolvedAbility === "gain_trade") next = { ...next, hard: { ...next.hard, trade: next.hard.trade + amount } };
          else if (resolvedAbility === "gain_authority") next = { ...next, hard: { ...next.hard, authority: next.hard.authority + amount } };
          else if (resolvedAbility === "draw") next = drawHardCards(next, Math.max(1, amount));
          else if (resolvedAbility === "opponent_discard") next = { ...next, astro: { ...next.astro, pendingDiscard: next.astro.pendingDiscard + 1 } };
          else if (resolvedAbility === "ship_top") next = { ...next, hard: { ...next.hard, nextShipToTop: true } };
          next = triggerAutomaticHardAllies(next, definitions);
          eventText = `Hard AI triggered ${definition.name}${resolvedAbility ? ` · ${abilityLabel(resolvedAbility, amount)}` : hardAmount ? ` · ${hardAmount}` : ""}`;
        }
      } else if (hardActionKind === "scrap_row") {
        const target = next.tradeRow.find((item) => item.uid === hardTargetUid);
        const targetDefinition = target?.cardId === null || target?.cardId === undefined ? undefined : definitions.get(target.cardId);
        if (target && targetDefinition) {
          next = { ...next, tradeRow: next.tradeRow.map((item) => item.uid === target.uid ? card(null, "trade-row-refill") : item), scrapHeap: [...next.scrapHeap, target] };
          eventText = `Hard AI scrapped ${targetDefinition.name} from the trade row`;
        }
      } else if (hardActionKind === "end_turn") {
        const ships = next.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "ship").map(originalCard);
        const bases = next.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type !== "ship").map((item) => ({ ...item, activated: true, allyTriggered: false }));
        next = { ...next, activeSide: "astro5", turn: next.turn + 1, hard: { ...next.hard, trade: 0, combat: 0, pendingDiscard: 0, nextShipToTop: false, discard: [...next.hard.discard, ...ships], inPlay: bases } };
        next = drawHardCards(next, Math.max(0, 5 - next.hard.handCount));
        eventText = "Hard AI ended its turn";
      }
      return withEvent(next, "hard", eventText, current.turn);
    });
    setHardCardId(null);
    setHardAmount(0);
    setHardTargetUid("");
    setHardAbilityChoice("");
    setHardDeclined(false);
    setHardSourceZone("hand");
    const playedDefinition = hardActionKind === "play" && hardCardId !== null ? definitions.get(hardCardId) : undefined;
    const scrappedDefinition = hardActionKind === "scrap" && hardSourceZone === "in_play" && hardCardId !== null ? definitions.get(hardCardId) : undefined;
    const abilityDefinition = hardActionKind === "ability" && hardCardId !== null ? definitions.get(hardCardId) : undefined;
    const manuallyTriggeredEffect = hardActionKind === "ability" && !hardDecisionEffect ? abilityDefinition?.ally ?? "" : "";
    const activatedFollowupEffect = hardActionKind === "ability" && ["scrap_trade_row", "scrap_any", "scrap_two_draw", "draw_then_scrap", "destroy_base", "free_ship", "destroy_and_scrap", "draw_destroy", "copy_ship"].includes(hardDecisionEffect) ? hardDecisionEffect : "";
    const allyDecisionDefinition = playedDefinition ? newlyTriggeredHardAllyDecision(match, playedDefinition, definitions) : undefined;
    const copiedTarget = hardActionKind === "ability" && hardDecisionEffect === "copy_ship" ? match.hard.inPlay.find((item) => item.uid === hardTargetUid) : undefined;
    const copiedDefinition = copiedTarget ? effectiveDefinition(copiedTarget, definitions) : undefined;
    const playedFollowup = playedDefinition?.primary ? hardFollowup(playedDefinition.primary, playedDefinition.name) : null;
    const scrappedFollowup = scrappedDefinition?.scrap ? hardFollowup(scrappedDefinition.scrap, scrappedDefinition.name) : null;
    const activatedFollowup = activatedFollowupEffect && abilityDefinition ? hardFollowup(activatedFollowupEffect, abilityDefinition.name) : null;
    const manualFollowup = manuallyTriggeredEffect && abilityDefinition ? hardFollowup(manuallyTriggeredEffect, abilityDefinition.name) : null;
    const allyFollowup = allyDecisionDefinition?.ally ? hardFollowup(allyDecisionDefinition.ally, allyDecisionDefinition.name) : null;
    const copiedFollowup = copiedDefinition?.primary ? hardFollowup(copiedDefinition.primary, copiedDefinition.name) : null;
    const queuedFollowup = hardQueuedDecision ? hardFollowup(hardQueuedDecision.effect, hardQueuedDecision.cardName) : null;
    if (playedFollowup && allyFollowup && allyDecisionDefinition) {
      setHardQueuedDecision({ effect: allyDecisionDefinition.ally, cardId: allyDecisionDefinition.card_id, cardName: allyDecisionDefinition.name });
    }
    const nextHardDecisionEffect = playedFollowup ? playedDefinition?.primary ?? ""
      : scrappedFollowup ? scrappedDefinition?.scrap ?? ""
        : activatedFollowup ? activatedFollowupEffect
          : manualFollowup ? manuallyTriggeredEffect
            : allyFollowup ? allyDecisionDefinition?.ally ?? ""
              : copiedFollowup ? copiedDefinition?.primary ?? ""
                : queuedFollowup ? hardQueuedDecision?.effect ?? ""
                : "";
    const chainedFollowup = hardActionKind === "attack_base" && hardDecisionEffect === "destroy_and_scrap"
      ? { kind: "scrap_row" as HardActionKind, notice: "Blob Destroyer: which trade-row card did the Hard AI scrap?" }
      : null;
    const repeatedScrap = hardActionKind === "scrap" && hardDecisionEffect === "scrap_two_draw" && !hardDeclined && hardScrapCount === 0
      ? { kind: "scrap" as HardActionKind, notice: "Brain World: choose a second card to scrap, or finish." }
      : null;
    const startedCycle = hardActionKind === "ability" && hardAbilityChoice.startsWith("cycle")
      ? { kind: "discard" as HardActionKind, notice: "Recycling Station: choose a card to discard, or finish cycling." }
      : null;
    const repeatedCycle = hardActionKind === "discard" && hardDecisionEffect === "recycle_cycle" && !hardDeclined && hardScrapCount === 0
      ? { kind: "discard" as HardActionKind, notice: "Recycling Station: choose a second card to discard, or finish." }
      : null;
    const repeatedForcedDiscard = hardActionKind === "discard" && match.hard.pendingDiscard > 1
      ? { kind: "discard" as HardActionKind, notice: `The Hard AI must discard ${match.hard.pendingDiscard - 1} more.` }
      : null;
    const followup = playedFollowup
      ?? scrappedFollowup
      ?? activatedFollowup
      ?? manualFollowup
      ?? allyFollowup
      ?? copiedFollowup
      ?? chainedFollowup
      ?? repeatedScrap
      ?? startedCycle
      ?? repeatedCycle
      ?? repeatedForcedDiscard
      ?? queuedFollowup;
    if (followup) {
      const usesQueuedFollowup = followup === queuedFollowup;
      setHardActionKind(followup.kind);
      setHardDecisionEffect(chainedFollowup ? "scrap_trade_row" : repeatedScrap ? "scrap_two_draw" : startedCycle || repeatedCycle ? "recycle_cycle" : nextHardDecisionEffect);
      setHardAbilityChoice("");
      setHardCardId(startedCycle || repeatedCycle ? null : usesQueuedFollowup ? hardQueuedDecision?.cardId ?? null : playedDefinition?.card_id ?? scrappedDefinition?.card_id ?? copiedDefinition?.card_id ?? abilityDefinition?.card_id ?? null);
      setHardScrapCount(repeatedScrap || repeatedCycle ? 1 : 0);
      setHardDecisionNotice(followup.notice);
      setHardActionOpen(true);
      if (usesQueuedFollowup) setHardQueuedDecision(null);
    } else {
      setHardDecisionEffect("");
      setHardAbilityChoice("");
      setHardScrapCount(0);
      setHardDecisionNotice("");
      setHardActionOpen(false);
    }
    if (acquiredTradeRowSlot >= 0) {
      setPendingTradeRefillSlot(acquiredTradeRowSlot);
      setHardActionOpen(false);
    }
    setRecommendation(null);
  };

  const checkpointLabel = modelGroups.flatMap((group) => group.models).find((model) => model.id === match.checkpointId)?.label ?? "Choose checkpoint";
  const recommended = recommendation?.actions.find((action) => action.model_recommended) ?? null;
  const hardModeOptions = hardAbilityOptions(hardDecisionEffect);
  const legalHardKinds = hardLegalActionKinds(match, definitions, hardDecisionEffect);
  const legalHardAttackTargets = hardAttackTargets(match, definitions, !hardDecisionEffect);
  const legalHardAcquisitions = hardDecisionEffect === "free_ship"
    ? match.tradeRow.filter((item) => item.cardId !== null && definitions.get(item.cardId)?.card_type === "ship")
    : affordableHardAcquisitions(match, definitions);
  const hardCardCatalog = hardActionKind === "scrap" && hardSourceZone === "in_play"
    ? catalog.filter((definition) => match.hard.inPlay.some((item) => effectiveCardId(item) === definition.card_id && Boolean(definition.scrap)))
    : hardActionKind === "ability"
      ? hardDecisionEffect && hardCardId !== null
        ? catalog.filter((definition) => definition.card_id === hardCardId)
        : catalog.filter((definition) => match.hard.inPlay.some((item) => effectiveCardId(item) === definition.card_id && !item.activated && baseRequiresActivation(definition)))
      : catalog;
  const possibleAstroHandCards = (item: TrackedCard) => astroHandCandidateCatalog(match, item, catalog);

  return (
    <div className="relay-app">
      <header className="relay-intro">
        <div>
          <span className="relay-eyebrow"><i /> Hard AI companion</span>
          <h1>Play the checkpoint. Record the iPad.</h1>
          <p>Astro5 advises one move at a time; you transcribe the Hard AI and every revealed card.</p>
        </div>
        <div className="relay-setup">
          <label>
            <span>Astro5 checkpoint</span>
            <select value={effectiveSetupCheckpoint} onChange={(event) => setSetupCheckpoint(event.target.value)}>
              {!firstModelId ? <option value="">No playable checkpoints</option> : null}
              {modelGroups.map((group) => (
                <optgroup key={group.runId} label={group.runName}>
                  {group.models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
                </optgroup>
              ))}
            </select>
          </label>
          <fieldset>
            <legend>First player</legend>
            <button type="button" className={setupFirst === "astro5" ? "is-active" : ""} onClick={() => setSetupFirst("astro5")}>Astro5</button>
            <button type="button" className={setupFirst === "hard" ? "is-active" : ""} onClick={() => setSetupFirst("hard")}>Hard AI</button>
          </fieldset>
          <button type="button" className="relay-new-match" onClick={startNewMatch}>Start new match</button>
        </div>
      </header>

      <div className="relay-statusbar">
        <div><span className={`relay-turn-dot is-${match.activeSide}`} /><strong>{match.activeSide === "astro5" ? "Astro5 turn" : "Hard AI turn"}</strong><small>Turn {match.turn}</small></div>
        <p><span>Checkpoint</span>{checkpointLabel}</p>
        <p><span>Position</span>{unresolved.length ? `${unresolved.length} Undefined` : recommendationState === "loading" ? "Scoring…" : "Reconciled"}</p>
        <button type="button" onClick={() => setInventoryOpen(true)}>Decks & hidden cards <b>{match.astro.deck.length + match.astro.knownTop.length} / {match.hard.hidden.length + match.hard.knownHand.length + match.hard.knownTop.length}</b></button>
      </div>

      <div className="relay-shell">
        <div className="relay-table">
          <section className="relay-player relay-hard" aria-label="Hard AI board">
            <header>
              <div className="relay-player-name"><span>H</span><p><strong>Hard AI</strong><small>Enter exactly what it does on the iPad</small></p></div>
              <div className="relay-stats">
                <StatInput label="Authority" value={match.hard.authority} onChange={(value) => updateHardStat("authority", value)} />
                <StatInput label="Trade" value={match.hard.trade} onChange={(value) => updateHardStat("trade", value)} tone="trade" />
                <StatInput label="Combat" value={match.hard.combat} onChange={(value) => updateHardStat("combat", value)} tone="combat" />
              </div>
              <button type="button" className="relay-hidden-button" onClick={() => setInventoryOpen(true)}><strong>{match.hard.handCount}</strong><span>hand</span><strong>{match.hard.deckCount}</strong><span>deck</span></button>
            </header>
            {match.hard.pendingDiscard ? <div className="relay-alert"><span>Hard AI must discard {match.hard.pendingDiscard}</span><button type="button" onClick={() => updateHardStat("pendingDiscard", 0)}>Clear</button></div> : null}
            <CardZone label="Hard AI cards in play" detail={`${match.hard.inPlay.length} visible`} cards={match.hard.inPlay} zone="hardInPlay" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} onToggleActivated={toggleActivated} />
            <CardZone label="Hard AI discard" detail={`${match.hard.discard.length} cards`} cards={match.hard.discard} zone="hardDiscard" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
          </section>

          <section className="relay-market" aria-label="Trade row">
            <header><div><span>Shared market</span><h2>Trade row</h2></div><label>Explorers <input type="number" min="0" max="10" value={match.explorersRemaining} onChange={(event) => { setMatch((current) => ({ ...current, explorersRemaining: Math.min(10, Math.max(0, Number(event.target.value) || 0)) })); invalidateAdvice(); }} /></label></header>
            <div className="relay-market-row">
              {match.tradeRow.map((item, index) => <EditableCard key={item.uid} tracked={item} definition={item.cardId === null ? undefined : definitions.get(item.cardId)} catalog={catalog} onChange={(cardId) => updateCard("tradeRow", item.uid, cardId)} onDelete={() => deleteCard("tradeRow", item.uid)} topLabel={`Slot ${index + 1}`} />)}
              <button type="button" className="relay-add-slot" onClick={() => addCard("tradeRow")}>+ Trade-row card</button>
            </div>
          </section>

          <section className="relay-player relay-astro" aria-label="Astro5 board">
            <CardZone label="Astro5 cards in play" detail={`${match.astro.inPlay.length} active`} cards={match.astro.inPlay} zone="astroInPlay" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} onToggleActivated={toggleActivated} />
            <header>
              <div className="relay-player-name"><span>A5</span><p><strong>Astro5</strong><small>Your side · {checkpointLabel}</small></p></div>
              <div className="relay-stats">
                <StatInput label="Authority" value={match.astro.authority} onChange={(value) => updateAstroStat("authority", value)} />
                <StatInput label="Trade" value={match.astro.trade} onChange={(value) => updateAstroStat("trade", value)} tone="trade" />
                <StatInput label="Combat" value={match.astro.combat} onChange={(value) => updateAstroStat("combat", value)} tone="combat" />
              </div>
              <button type="button" className="relay-hidden-button" onClick={() => setInventoryOpen(true)}><strong>{match.astro.hand.length}</strong><span>hand</span><strong>{match.astro.deck.length + match.astro.knownTop.length}</strong><span>deck</span></button>
            </header>
            {match.astro.pendingDiscard ? <div className="relay-alert"><span>Astro5 must discard {match.astro.pendingDiscard}</span><button type="button" onClick={() => updateAstroStat("pendingDiscard", 0)}>Clear</button></div> : null}
            <CardZone label="Your hand" detail="One-click choices are limited to cards possible from the tracked deck" cards={match.astro.hand} zone="astroHand" catalog={catalog} catalogForCard={possibleAstroHandCards} definitions={definitions} compact={false} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
            <CardZone label="Astro5 discard" detail={`${match.astro.discard.length} cards`} cards={match.astro.discard} zone="astroDiscard" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
          </section>
        </div>

        <aside className="relay-command" aria-label="Turn command center">
          <div className="relay-command-head">
            <span>{match.activeSide === "astro5" ? "Checkpoint advisor" : "External turn recorder"}</span>
            <strong>{match.activeSide === "astro5" ? "Next Astro5 action" : "What did the Hard AI do?"}</strong>
            <small>{match.activeSide === "astro5" ? "Nothing changes until you confirm it happened on the iPad." : "Record one action, then continue until it ends the turn."}</small>
            {match.activeSide === "astro5" && match.pendingDecision ? <button type="button" className="relay-clear-decision" onClick={() => { setMatch((current) => ({ ...current, pendingDecision: null })); setRecommendation(null); }}>Clear this prompted decision</button> : null}
          </div>

          {match.activeSide === "hard" ? (
            <div className="relay-hard-console">
              <div className="relay-hard-pulse"><i /><span>{match.hard.combat > 0 ? `${match.hard.combat} damage currently available` : "Hard AI turn is waiting for input"}</span></div>
              {legalHardKinds.length ? <button type="button" className="relay-primary-command" onClick={() => { const kind = legalHardKinds[0]; setHardActionKind(kind); if (kind === "attack_player") setHardAmount(match.hard.combat); if (kind === "scrap" && !hardDecisionEffect) setHardSourceZone("in_play"); setHardActionOpen(true); }}>Record Hard AI action</button> : null}
              <div className="relay-quick-actions">
                {legalHardKinds.map((kind) => <button key={kind} type="button" onClick={() => { setHardActionKind(kind); if (kind === "attack_player") setHardAmount(match.hard.combat); if (kind === "scrap" && !hardDecisionEffect) setHardSourceZone("in_play"); setHardAbilityChoice(""); setHardDeclined(false); setHardActionOpen(true); }}>{kind === "play" ? "Play from hand" : kind === "attack_player" ? `Attack Astro5 · ${match.hard.combat}` : kind === "attack_base" ? "Attack base" : kind === "end_turn" ? "End turn" : kind === "scrap" ? "Scrap for ability" : kind === "ability" ? "Activate base" : titleCase(kind)}</button>)}
              </div>
            </div>
          ) : unresolved.length ? (
            <div className="relay-blocked">
              <span>{unresolved.length}</span>
              <strong>Define cards to continue</strong>
              <p>Astro5 will not score an incomplete position. Undefined cards are highlighted directly on the table.</p>
              <ul>{unresolved.slice(0, 5).map((item) => <li key={item.uid}>{item.label}</li>)}</ul>
              {unresolved.length > 5 ? <small>+ {unresolved.length - 5} more</small> : null}
            </div>
          ) : recommendationState === "loading" ? (
            <div className="relay-thinking"><i /><strong>Astro5 is scoring the position</strong><span>Every legal action is being compared.</span></div>
          ) : recommendation ? (
            <>
              <div className="relay-win-rate">
                <span>Expected win rate</span>
                <strong>{formatPercent(recommendation.expected_win_rate)}</strong>
                <small>Current position · before the next move</small>
              </div>
              {currentDecision.family === "main" && playAllIsDecisionFree(match, definitions) ? <button type="button" className="relay-play-all" onClick={playAll}>Play all cards in hand</button> : null}
              {recommended ? (
                <button type="button" className="relay-recommendation" onClick={() => applyAstroAction(recommended)}>
                  <span>Recommended</span>
                  <strong>{recommended.label}</strong>
                  <p><b>{formatPercent(recommended.model_value)}</b>{recommendation.score_semantics === "policy_probability" ? " checkpoint action score" : " action value"}</p>
                </button>
              ) : null}
              <div className="relay-alternatives">
                <header><span>All legal actions</span><small>{recommendation.score_semantics === "policy_probability" ? "policy share" : "value"}</small></header>
                {recommendation.actions.map((action) => (
                  <button key={action.id} type="button" className={action.model_recommended ? "is-best" : ""} onClick={() => applyAstroAction(action)}>
                    <span>{action.label}</span><b>{formatPercent(action.model_value)}</b>
                  </button>
                ))}
              </div>
              <p className="relay-command-note">Click the action once after entering it in the Hard AI game.</p>
            </>
          ) : (
            <div className="relay-advisor-offline">
              <strong>{recommendationState === "error" ? "Position could not be scored" : "Advisor is offline"}</strong>
              <p>{recommendationState === "error" ? advisorError || "Check the edited card counts and try again. The board remains saved." : "Start the local Astro5 service to see checkpoint values and expected win rate."}</p>
              <button type="button" onClick={() => setAdvisorAttempt((attempt) => attempt + 1)}>Retry checkpoint advisor</button>
              <div className="relay-unscored-actions">{currentDecision.actions.slice(0, 8).map((action) => <span key={action.id}>{action.label}</span>)}</div>
            </div>
          )}

          <section className="relay-log">
            <header><span>Match log</span><button type="button" onClick={() => setMatch((current) => ({ ...current, events: [] }))}>Clear</button></header>
            <div>{match.events.slice(0, 10).map((event) => <p key={event.id}><b>{event.side === "astro5" ? "A5" : event.side === "hard" ? "H" : "·"}</b><span>{event.text}</span><small>T{event.turn}</small></p>)}</div>
          </section>
        </aside>
      </div>

      {inventoryOpen ? (
        <RelayModal title="Decks & hidden cards" kicker="Scrambled information sets" onClose={() => setInventoryOpen(false)} wide>
          <div className="relay-inventory-note"><strong>Order stays hidden.</strong><span>Known top cards are separated and shown top-first. Every copy remains independently editable.</span></div>
          <div className="relay-inventory-grid">
            <article>
              <header><span className="relay-avatar is-astro">A5</span><div><strong>Astro5 cards</strong><small>Your exact hand, unordered deck, and known top</small></div></header>
              <CardZone label="Known top cards" detail="Top to bottom" cards={match.astro.knownTop} zone="astroKnownTop" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
              <CardZone label="Scrambled deck" detail={`${match.astro.deck.length} cards · unordered`} cards={match.astro.deck} zone="astroDeck" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
              <CardZone label="Hand" detail={`${match.astro.hand.length} cards · tracked possibilities only`} cards={match.astro.hand} zone="astroHand" catalog={catalog} catalogForCard={possibleAstroHandCards} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
            </article>
            <article>
              <header><span className="relay-avatar is-hard">H</span><div><strong>Hard AI cards</strong><small>Unknown hand + deck stay combined; revealed top draws remain known</small></div></header>
              <div className="relay-count-editor"><StatInput label="Hand count" value={match.hard.handCount} onChange={(value) => updateHardStat("handCount", value)} /><StatInput label="Deck count" value={match.hard.deckCount} onChange={(value) => updateHardStat("deckCount", value)} /></div>
              <CardZone label="Known cards in hand" detail={`${match.hard.knownHand.length} revealed from deck top`} cards={match.hard.knownHand} zone="hardKnownHand" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
              <CardZone label="Known top cards" detail="Top to bottom" cards={match.hard.knownTop} zone="hardKnownTop" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
              <CardZone label="Scrambled hand + deck" detail={`${match.hard.hidden.length} hidden cards`} cards={match.hard.hidden} zone="hardHidden" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
            </article>
          </div>
          <CardZone label="Shared scrap heap" detail={`${match.scrapHeap.length} cards removed from the game`} cards={match.scrapHeap} zone="scrapHeap" catalog={catalog} definitions={definitions} onChange={updateCard} onDelete={deleteCard} onAdd={addCard} />
        </RelayModal>
      ) : null}

      {pendingTradeRefillSlot !== null ? (
        <RelayModal title="What replaced the acquired card?" kicker={`Trade-row slot ${pendingTradeRefillSlot + 1}`} onClose={() => setPendingTradeRefillSlot(null)}>
          <div className="relay-refill-prompt">
            <strong>Choose the new trade-row card</strong>
            <p>The slot stays unresolved until you enter the replacement shown on the iPad.</p>
            <CardNameEditor
              value={undefined}
              catalog={[...new Map(remainingTradeDeck(match, catalog).map((definition) => [definition.card_id, definition])).values()]}
              onSelect={(cardId) => {
                const target = match.tradeRow[pendingTradeRefillSlot];
                if (target) updateCard("tradeRow", target.uid, cardId);
                setPendingTradeRefillSlot(null);
                setHardActionOpen(false);
              }}
              onCancel={() => setPendingTradeRefillSlot(null)}
              allowUndefined={false}
              listboxId="relay-trade-refill-options"
              autoFocus
            />
          </div>
        </RelayModal>
      ) : null}

      {hardActionOpen && pendingTradeRefillSlot === null && match.activeSide === "hard" ? (
        <RelayModal title="Record the Hard AI's next action" kicker={`Turn ${match.turn} · live iPad input`} onClose={() => setHardActionOpen(false)}>
          <div className="relay-action-kinds">
            {([
              ["play", "Play from hand"],
              ["acquire", "Acquire"],
              ["attack_player", "Attack Astro5"],
              ["attack_base", "Attack base"],
              ["discard", "Discard"],
              ["scrap", "Scrap"],
              ["scrap_row", "Scrap trade row"],
              ["ability", "Trigger ability"],
              ["end_turn", "End turn"],
            ] as Array<[HardActionKind, string]>).filter(([kind]) => legalHardKinds.includes(kind)).map(([kind, label]) => <button key={kind} type="button" className={hardActionKind === kind ? "is-active" : ""} onClick={() => { setHardActionKind(kind); setHardCardId(null); setHardTargetUid(""); if (kind === "attack_player") setHardAmount(match.hard.combat); if (kind === "scrap" && !hardDecisionEffect) setHardSourceZone("in_play"); setHardAbilityChoice(""); setHardDeclined(false); }}>{label}</button>)}
          </div>

          <div className="relay-action-form">
            {hardDecisionNotice ? <div className="relay-decision-notice"><strong>Decision required</strong><span>{hardDecisionNotice}</span></div> : null}
            {["scrap_trade_row", "scrap_any", "scrap_two_draw", "destroy_base", "destroy_and_scrap", "draw_destroy", "free_ship", "recycle_cycle"].includes(hardDecisionEffect) ? (
              <button type="button" className={`relay-decline-decision${hardDeclined ? " is-active" : ""}`} onClick={() => setHardDeclined((value) => !value)}>
                {hardDeclined ? "Decline selected" : "The Hard AI declined this optional choice"}
              </button>
            ) : null}
            {hardActionKind === "play" || hardActionKind === "discard" || hardActionKind === "scrap" || hardActionKind === "ability" ? (
              <div><span>Which card?</span><CardNameEditor key={`${hardActionKind}-${hardDecisionEffect}-${hardCardId ?? "empty"}`} value={hardCardId === null ? undefined : definitions.get(hardCardId)} catalog={hardCardCatalog} onSelect={(cardId) => { setHardCardId(cardId); if (hardActionKind === "ability" && !hardDecisionEffect && cardId !== null) setHardDecisionEffect(definitions.get(cardId)?.primary ?? ""); }} onCancel={() => setHardCardId(null)} allowUndefined={false} listboxId="relay-hard-card-options" autoFocus /></div>
            ) : null}
            {hardActionKind === "scrap" ? <label><span>Card came from</span><select value={hardSourceZone} onChange={(event) => setHardSourceZone(event.target.value as "hand" | "discard" | "in_play")}><option value="hand">Hand</option>{hardDecisionEffect !== "draw_then_scrap" ? <option value="discard">Discard pile</option> : null}{!hardDecisionEffect ? <option value="in_play">In play (scrap ability)</option> : null}</select></label> : null}
            {hardActionKind === "ability" && hardModeOptions.length ? <label><span>Option chosen</span><select value={hardAbilityChoice} onChange={(event) => setHardAbilityChoice(event.target.value)}><option value="">Choose an option…</option>{hardModeOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label> : null}
            {hardActionKind === "ability" && hardDecisionEffect === "copy_ship" ? <label><span>Ship copied</span><select value={hardTargetUid} onChange={(event) => setHardTargetUid(event.target.value)}><option value="">Choose a ship…</option>{match.hard.inPlay.filter((item) => effectiveDefinition(item, definitions)?.card_type === "ship" && !(item.cardId === hardCardId && !item.copiedCardId)).map((item) => <option key={item.uid} value={item.uid}>{effectiveDefinition(item, definitions)?.name}</option>)}</select></label> : null}
            {hardActionKind === "acquire" ? <label><span>Trade-row card</span><select value={hardTargetUid} onChange={(event) => setHardTargetUid(event.target.value)}><option value="">Choose a card…</option>{hardDecisionEffect !== "free_ship" && match.explorersRemaining > 0 && match.hard.trade >= 2 ? <option value="explorer-supply">Explorer</option> : null}{legalHardAcquisitions.map((item) => <option key={item.uid} value={item.uid}>{definitions.get(item.cardId ?? -1)?.name ?? "Unsupported card"}</option>)}</select></label> : null}
            {hardActionKind === "scrap_row" ? <label><span>Trade-row card to scrap</span><select value={hardTargetUid} onChange={(event) => setHardTargetUid(event.target.value)}><option value="">Choose a card…</option>{match.tradeRow.map((item) => <option key={item.uid} value={item.uid} disabled={item.cardId === null}>{item.cardId === null ? "Undefined" : definitions.get(item.cardId)?.name ?? "Unsupported card"}</option>)}</select></label> : null}
            {hardActionKind === "attack_base" ? <label><span>Astro5 base</span><select value={hardTargetUid} onChange={(event) => setHardTargetUid(event.target.value)}><option value="">Choose a base…</option>{legalHardAttackTargets.map((item) => <option key={item.uid} value={item.uid}>{effectiveDefinition(item, definitions)?.name}</option>)}</select></label> : null}
            {hardActionKind === "attack_player" || hardActionKind === "ability" ? <label><span>{hardActionKind === "attack_player" ? "Combat dealt" : "Optional amount / result"}</span><input type="number" min="0" value={hardAmount} onChange={(event) => setHardAmount(Math.max(0, Number(event.target.value) || 0))} /></label> : null}
            {hardActionKind === "end_turn" ? <div className="relay-end-turn-copy"><strong>Finish the Hard AI turn?</strong><p>Ships move to discard, bases remain, resources clear, and Astro5 becomes active.</p></div> : null}
          </div>
          <footer className="relay-modal-actions">
            <button type="button" onClick={() => setHardActionOpen(false)}>Cancel</button>
            <button type="button" className="is-primary" onClick={recordHardAction} disabled={!hardDeclined && (((hardActionKind === "play" || hardActionKind === "discard" || hardActionKind === "scrap" || hardActionKind === "ability") && hardCardId === null) || (hardActionKind === "ability" && hardModeOptions.length > 0 && !hardAbilityChoice) || (hardActionKind === "ability" && hardDecisionEffect === "copy_ship" && !hardTargetUid) || ((hardActionKind === "acquire" || hardActionKind === "attack_base" || hardActionKind === "scrap_row") && !hardTargetUid))}>Record action</button>
          </footer>
        </RelayModal>
      ) : null}
    </div>
  );
}
