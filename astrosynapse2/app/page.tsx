"use client";

import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type ReactNode,
} from "react";

/**
 * Local trainer API contract (all routes are rooted at API_BASE):
 *
 * GET   /health -> { ok, active_run_id }
 * GET   /system -> local M4 / memory / accelerator telemetry
 * GET   /presets -> validated m4_24h and quick RunConfig recipes
 * GET   /runs -> Run[]
 * POST  /runs <- { preset, name, overrides, start }
 * GET   /runs/:id -> { run, live, latest_metric, active }
 * GET   /runs/:id/metrics?after=<sequence> -> MetricPoint[]
 * GET   /runs/:id/events?limit=100 -> persisted audit events
 * GET   /models?run_id=:id -> ModelCheckpoint[]
 * PATCH /models/:id <- { pinned }
 * GET   /models/:id/actor -> downloadable .actor.npz
 * POST  /runs/:id/start|pause|resume|stop|checkpoint
 *   -> updated persisted run
 * POST  /arena
 *   <- { model_a, model_b, pairs } (service enforces paired seeds and seat reversal)
 * GET   /arena/:id -> persistent paired-evaluation progress and result
 * POST  /games
 *   <- { model_id, human_starts }
 * GET   /games/:id -> GameSession
 * POST  /games/:id/choice
 *   <- { action_id }
 * PATCH /runs/:id/config
 *   <- { changes: safe live RunConfig fields }
 *
 * The normalizers below intentionally accept snake_case and common nested
 * variants so the dashboard remains useful while the local Python service is
 * upgraded. If the service cannot be reached, the complete UI switches to a
 * deterministic demo session and labels that state prominently.
 */

const API_BASE = "http://127.0.0.1:8765/api";
const POLL_INTERVAL_MS = 1_000;

const jargon = {
  seed: "A number used to control randomness. Reusing it recreates the same shuffled cards and game conditions, which makes tests fair and repeatable.",
  pairedSeeds: "Each seed is played twice with the models swapping seats. This helps separate model skill from luck and first-player advantage.",
  champion: "The current best approved model. A newer checkpoint becomes champion only after it proves stronger in a sufficiently large evaluation.",
  heldOutStrength: "Win rate in evaluation games that are kept separate from training. It is a better test of real improvement than performance on games the model learned from.",
  confidenceInterval: "A range around the measured score that shows statistical uncertainty. More evaluation games usually make this range narrower.",
  checkpoint: "A saved copy of the neural network at a particular moment in training. You can evaluate, compare, download, or resume from it.",
  selfPlay: "Games where the training model plays against itself or other saved versions, creating examples it can learn from.",
  replayBuffer: "Stored game decisions used as training examples. The percentage shows how much of this memory is currently filled.",
  outcomeBce: "Outcome Binary Cross-Entropy: how wrong the model's predicted chance of winning was compared with the final win or loss. Lower is better when comparing similar data; about 0.693 is coin-flip performance.",
  brierScore: "The average squared error of predicted win probabilities. Lower is better; 0 is perfect, and 0.25 corresponds to always predicting a 50% chance for win/loss games.",
  explainedVariance: "How much of the variation in game results the model's predictions account for. 1 is ideal, 0 means no useful explanation, and negative values mean it is doing worse than a simple average prediction.",
  bootstrapUncertainty: "How much the model's independently trained prediction heads disagree. More disagreement suggests the model is less certain about unfamiliar or ambiguous positions.",
  curriculum: "The current stage of the training plan. Early stages use helpful opponents or examples before shifting toward ordinary self-play.",
  safetyTruncations: "Games stopped by a turn or action safety limit instead of ending normally. They receive a neutral learning result; lower is generally better.",
  draws: "Games recorded without a winner. They contribute a neutral 50% outcome to learning and evaluation.",
  meanTurns: "The average number of turns in completed games. A sudden change can reveal a shift in play style or a game-engine problem.",
  forcedChoices: "One-option decisions omitted from training because there was nothing meaningful for the model to choose.",
  uncertainty: "Disagreement among the model's multiple win-prediction heads. Higher values usually mean it is less sure about the position.",
  actions: "The moves the game rules currently allow. During training, the model scores these choices and selects one; forced one-option choices are skipped.",
  modelActions: "Controls for this saved model: the diamond pins it so cleanup will keep it, and the down arrow downloads its lightweight game-playing file.",
  actionValue: "The model's estimated chance that the player making this choice will eventually win. It is a prediction, not a guaranteed result.",
  outcomeLearning: "Training the model to predict the final win, loss, or draw from each decision it made during a game.",
  lineage: "The family tree of saved models. A parent is the checkpoint a newer model continued training from.",
  elo: "A familiar rating-style conversion of the measured head-to-head score. Here it is only a comparison between these models, not a universal rating.",
  actors: "Separate CPU worker processes that play simulated games and send the resulting decisions to the learner.",
  learner: "The part of the system that updates the neural network from stored game examples.",
  hiddenSize: "The width of the neural network's internal representation. Larger values can learn more complex patterns but use more memory and computation.",
  residualBlocks: "Repeated neural-network layers with shortcut connections that help deeper models train reliably. More blocks increase capacity and cost.",
  bootstrapHeads: "Several separate win predictors sharing one network. Their disagreement provides an estimate of uncertainty and encourages varied play.",
  batchSize: "How many stored decisions the learner processes together for one network update. Larger batches use more memory but give smoother updates.",
  learningRate: "How large each adjustment to the network is. Too high can make learning unstable; too low can make it very slow.",
  replayCapacity: "The maximum number of past decisions kept as training examples. Old examples are replaced when it fills.",
  replayWarmup: "How many examples must be collected before neural-network updates begin, preventing learning from a tiny, unrepresentative sample.",
  epsilon: "The probability of deliberately trying a random legal move instead of the model's favorite, so training explores alternatives.",
  bootstrapUpdates: "Early learner updates based on games from hand-written starter opponents. They give a new random network useful examples before self-play takes over.",
  league: "A collection of older saved models used as opponents, helping the current model avoid forgetting how to beat earlier strategies.",
  baseline: "A fixed hand-written opponent used as a stable reference and to provide useful early training games.",
  promotionConfidence: "How certain the evaluation must be before a challenger may become champion. Higher confidence requires stronger or more plentiful evidence.",
  promotionMargin: "The extra win-rate advantage above 50% a challenger must prove before it can become champion.",
  evaluationPairs: "The number of paired seeds used for comparison. Every pair plays two games with seats swapped, so 5,000 pairs means 10,000 games.",
  stratifiedReplay: "Stored examples are separated by type of decision so common moves do not crowd out rare but important choices.",
  decisionFamilies: "Groups of choices with the same meaning, such as buying, discarding, or scrapping. The model learns each group from its own balanced supply of examples.",
  outcomeEstimate: "The model's predicted chance that the player making the decision will ultimately win the game.",
} as const;

type JargonKey = keyof typeof jargon;

function Jargon({
  term,
  children,
  side = "top",
  align = "center",
}: {
  term: JargonKey;
  children: ReactNode;
  side?: "top" | "bottom";
  align?: "center" | "right";
}) {
  const explanation = jargon[term];
  return (
    <span
      className={`jargon-help tooltip-${side} tooltip-${align}`}
      tabIndex={0}
      data-tooltip={explanation}
      aria-label={`${typeof children === "string" ? children : term}: ${explanation}`}
    >
      {children}
    </span>
  );
}

type TabId = "overview" | "train" | "models" | "play" | "diagnostics";
type RunStatus =
  | "ready"
  | "running"
  | "pausing"
  | "paused"
  | "stopping"
  | "interrupted"
  | "complete"
  | "error";

type MetricPoint = {
  seq: number;
  games: number;
  gamesPerSecond: number;
  decisionsPerSecond: number;
  winRate: number;
  ciLow: number;
  ciHigh: number;
  hasEvaluation: boolean;
  hasLearnerDiagnostics: boolean;
  outcomeLoss: number;
  brier: number;
  explainedVariance: number;
  uncertainty: number;
  truncationRate: number;
  drawRate: number;
  meanTurns: number;
  forcedChoices: number;
  curriculumPhase: string;
  bootstrapUpdatesRemaining: number;
  replayFill: number;
  replayFamilies: Record<string, number>;
  cpuPercent: number;
  memoryGb: number;
};

type RunView = {
  id: string;
  name: string;
  status: RunStatus;
  phase: string;
  games: number;
  decisions: number;
  updates: number;
  elapsedSeconds: number;
  durationSeconds: number;
  championId: string;
  seed: number;
  lastError?: string;
};

type HardwareView = {
  machine: string;
  chip: string;
  backend: string;
  memoryTotalGb: number;
  memoryUsedGb: number;
  cpuPercent: number;
  actorProcesses: number;
  learnerDevice: string;
  metalAvailable: boolean | null;
  metalActiveGb: number;
  metalPeakGb: number;
};

type ModelCheckpoint = {
  id: string;
  label: string;
  parentId?: string;
  games: number;
  created: string;
  score: number;
  ciLow: number;
  ciHigh: number;
  eloDelta: number;
  hasElo: boolean;
  isChampion: boolean;
  isPinned: boolean;
  sizeMb: number | null;
  evaluated: boolean;
};

type ArenaResultView = {
  id: string;
  status: string;
  progress: number;
  pairsCompleted: number;
  pairsRequested: number;
  modelALabel: string;
  modelBLabel: string;
  score: number;
  ciLow: number;
  ciHigh: number;
  elo: number;
  firstSeatScore: number;
  secondSeatScore: number;
  truncatedGames: number;
  gamesCompleted: number;
  intervalMethod: string;
  recommendation: string;
};

type AuditEvent = {
  id: string;
  at: string;
  kind: "success" | "info" | "warning" | "error";
  title: string;
  detail: string;
};

type DashboardSnapshot = {
  run: RunView;
  hardware: HardwareView;
  metrics: MetricPoint[];
  models: ModelCheckpoint[];
  events: AuditEvent[];
};

type TrainerConfig = {
  name: string;
  preset: "m4_24h" | "quick" | "custom";
  durationMinutes: number;
  actorProcesses: number;
  gamesPerActorBatch: number;
  hiddenSize: number;
  residualBlocks: number;
  bootstrapHeads: number;
  batchSize: number;
  learningRate: number;
  replayCapacity: number;
  replayWarmup: number;
  heuristicBootstrapUpdates: number;
  currentSelfplayFraction: number;
  leagueFraction: number;
  baselineFraction: number;
  evaluationPairs: number;
  evaluateEveryGames: number;
  checkpointEveryGames: number;
  promotionConfidence: number;
  promotionMargin: number;
  epsilonStart: number;
  epsilonEnd: number;
};

type GameCard = {
  id: string;
  name: string;
  faction: "trade" | "blob" | "machine" | "star" | "neutral";
  cost: number;
  attack?: number;
  trade?: number;
  authority?: number;
  defense?: number;
  text: string;
  kind: "ship" | "base";
};

type GameState = {
  turn: number;
  humanAuthority: number;
  opponentAuthority: number;
  trade: number;
  attack: number;
  deckCount: number;
  discardCount: number;
  opponentDeckCount: number;
  explorersRemaining: number;
  hand: GameCard[];
  ownDeck: GameCard[];
  opponentHand: GameCard[];
  opponentDeck: GameCard[];
  market: GameCard[];
  humanBases: GameCard[];
  opponentBases: GameCard[];
  log: string[];
};

type RemoteGameAction = {
  id: number;
  label: string;
  modelValue: number | null;
  recommended: boolean;
};

type RemoteGameSession = {
  id: string;
  status: "your_turn" | "model_thinking" | "complete" | "error";
  prompt: string;
  family: string;
  actions: RemoteGameAction[];
  modelLabel: string;
  result?: string;
  error?: string;
};

const numberFormatter = new Intl.NumberFormat("en-US");
const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const arenaBaselines = [
  { id: "baseline:balanced", label: "Balanced baseline" },
  { id: "baseline:economy", label: "Economy baseline" },
  { id: "baseline:aggressive", label: "Aggressive baseline" },
  { id: "baseline:random", label: "Random baseline" },
];

const tabs: Array<{ id: TabId; label: string; short: string }> = [
  { id: "overview", label: "Overview", short: "01" },
  { id: "train", label: "Train", short: "02" },
  { id: "models", label: "Models & Arena", short: "03" },
  { id: "play", label: "Play", short: "04" },
  { id: "diagnostics", label: "Diagnostics / Settings", short: "05" },
];

const initialConfig: TrainerConfig = {
  name: "M4 24-hour run",
  preset: "m4_24h",
  durationMinutes: 1_440,
  actorProcesses: 8,
  gamesPerActorBatch: 16,
  hiddenSize: 192,
  residualBlocks: 3,
  bootstrapHeads: 3,
  batchSize: 2_048,
  learningRate: 0.0003,
  replayCapacity: 900_000,
  replayWarmup: 50_000,
  heuristicBootstrapUpdates: 2_000,
  currentSelfplayFraction: 0.55,
  leagueFraction: 0.3,
  baselineFraction: 0.15,
  evaluationPairs: 5_000,
  evaluateEveryGames: 500_000,
  checkpointEveryGames: 100_000,
  promotionConfidence: 0.95,
  promotionMargin: 0,
  epsilonStart: 0.2,
  epsilonEnd: 0.025,
};

const demoModels: ModelCheckpoint[] = [
  {
    id: "champion-042",
    label: "Champion 042",
    parentId: "champion-038",
    games: 190_000,
    created: "18 min ago",
    score: 0.618,
    ciLow: 0.586,
    ciHigh: 0.649,
    eloDelta: 83,
    hasElo: true,
    isChampion: true,
    isPinned: true,
    sizeMb: 6.4,
    evaluated: true,
  },
  {
    id: "candidate-846k",
    label: "Candidate 846k",
    parentId: "champion-042",
    games: 205_400,
    created: "just now",
    score: 0.604,
    ciLow: 0.561,
    ciHigh: 0.645,
    eloDelta: 74,
    hasElo: true,
    isChampion: false,
    isPinned: false,
    sizeMb: 6.4,
    evaluated: true,
  },
  {
    id: "champion-038",
    label: "Champion 038",
    parentId: "champion-031",
    games: 145_000,
    created: "1h 26m ago",
    score: 0.571,
    ciLow: 0.538,
    ciHigh: 0.603,
    eloDelta: 50,
    hasElo: true,
    isChampion: false,
    isPinned: true,
    sizeMb: 6.4,
    evaluated: true,
  },
  {
    id: "champion-031",
    label: "Champion 031",
    parentId: "bootstrap-v1",
    games: 100_000,
    created: "3h 11m ago",
    score: 0.548,
    ciLow: 0.515,
    ciHigh: 0.581,
    eloDelta: 34,
    hasElo: true,
    isChampion: false,
    isPinned: false,
    sizeMb: 6.4,
    evaluated: true,
  },
  {
    id: "bootstrap-v1",
    label: "Heuristic bootstrap",
    games: 0,
    created: "5h 42m ago",
    score: 0.5,
    ciLow: 0.5,
    ciHigh: 0.5,
    eloDelta: 0,
    hasElo: true,
    isChampion: false,
    isPinned: true,
    sizeMb: 6.4,
    evaluated: true,
  },
];

const demoArenaResult: ArenaResultView = {
  id: "demo-arena-042",
  status: "complete",
  progress: 100,
  pairsCompleted: 5_000,
  pairsRequested: 5_000,
  modelALabel: "Champion 042",
  modelBLabel: "Champion 038",
  score: 0.618,
  ciLow: 0.586,
  ciHigh: 0.649,
  elo: 83,
  firstSeatScore: 0.611,
  secondSeatScore: 0.625,
  truncatedGames: 4,
  gamesCompleted: 10_000,
  intervalMethod: "nonparametric_bootstrap",
  recommendation: "Model A advantage is supported by the paired interval",
};

function buildDemoMetrics(count = 48): MetricPoint[] {
  return Array.from({ length: count }, (_, index) => {
    const progress = index / Math.max(count - 1, 1);
    const ripple = Math.sin(index * 0.72) * 0.012;
    const winRate = 0.492 + progress * 0.124 + ripple;
    return {
      seq: index,
      games: Math.round(176_600 + progress * 28_800),
      gamesPerSecond: 9.8 + Math.sin(index * 0.47) * 0.55 + progress * 0.9,
      decisionsPerSecond: 2_320 + Math.cos(index * 0.38) * 130 + progress * 150,
      winRate,
      ciLow: winRate - (0.055 - progress * 0.024),
      ciHigh: winRate + (0.055 - progress * 0.024),
      hasEvaluation: true,
      hasLearnerDiagnostics: true,
      outcomeLoss: 0.69 - progress * 0.14 + Math.sin(index) * 0.008,
      brier: 0.25 - progress * 0.065 + Math.cos(index * 0.8) * 0.004,
      explainedVariance: -0.06 + progress * 0.39 + Math.sin(index * 0.31) * 0.015,
      uncertainty: 0.13 - progress * 0.045 + Math.sin(index * 0.27) * 0.004,
      truncationRate: 0.018 - progress * 0.011,
      drawRate: 0.024 - progress * 0.009,
      meanTurns: 34.2 - progress * 5.7,
      forcedChoices: Math.round(84_000 + progress * 18_000),
      curriculumPhase: "self_play",
      bootstrapUpdatesRemaining: 0,
      replayFill: Math.min(0.94, 0.28 + progress * 0.66),
      replayFamilies: {
        main_phase: 31_000 + index * 520,
        choose_discard: 9_000 + index * 145,
        choose_scrap: 13_000 + index * 210,
        choose_trade_row_scrap: 6_000 + index * 98,
        choose_free_ship: 4_000 + index * 64,
        choose_topdeck: 7_000 + index * 112,
        choose_ability: 10_000 + index * 166,
      },
      cpuPercent: 76 + Math.sin(index * 0.4) * 6,
      memoryGb: 9.4 + Math.sin(index * 0.24) * 0.6 + progress * 0.7,
    };
  });
}

const emptyMetric: MetricPoint = {
  seq: -1,
  games: 0,
  gamesPerSecond: 0,
  decisionsPerSecond: 0,
  winRate: 0.5,
  ciLow: 0,
  ciHigh: 1,
  hasEvaluation: false,
  hasLearnerDiagnostics: false,
  outcomeLoss: 0,
  brier: 0,
  explainedVariance: 0,
  uncertainty: 0,
  truncationRate: 0,
  drawRate: 0,
  meanTurns: 0,
  forcedChoices: 0,
  curriculumPhase: "waiting",
  bootstrapUpdatesRemaining: 0,
  replayFill: 0,
  replayFamilies: {},
  cpuPercent: 0,
  memoryGb: 0,
};

const demoSnapshot: DashboardSnapshot = {
  run: {
    id: "m4-24h-20260807",
    name: "M4 24-hour run",
    status: "running",
    phase: "Self-play collection",
    games: 205_400,
    decisions: 51_300_000,
    updates: 3_184,
    elapsedSeconds: 5 * 3_600 + 42 * 60 + 18,
    durationSeconds: 24 * 3_600,
    championId: "champion-042",
    seed: 20260807,
  },
  hardware: {
    machine: "Mac mini",
    chip: "Apple M4 · 10-core",
    backend: "MLX / Metal",
    memoryTotalGb: 16,
    memoryUsedGb: 10.1,
    cpuPercent: 81,
    actorProcesses: 8,
    learnerDevice: "Metal GPU",
    metalAvailable: true,
    metalActiveGb: 1.8,
    metalPeakGb: 2.4,
  },
  metrics: buildDemoMetrics(),
  models: demoModels,
  events: [
    {
      id: "evt-1",
      at: "18 min",
      kind: "success",
      title: "Champion 042 promoted",
      detail: "61.8% · 95% CI 58.6–64.9 · 5,000 paired games",
    },
    {
      id: "evt-2",
      at: "31 min",
      kind: "info",
      title: "Replay buffer crossed 250k",
      detail: "Recent sample mix holding at 35%",
    },
    {
      id: "evt-3",
      at: "52 min",
      kind: "warning",
      title: "Throughput recovered",
      detail: "Actor 6 restarted after a stalled game seed",
    },
    {
      id: "evt-4",
      at: "1h 26m",
      kind: "success",
      title: "Champion 038 promoted",
      detail: "57.1% · 95% CI 53.8–60.3",
    },
  ],
};

const demoCards: Record<string, GameCard> = {
  federation: {
    id: "federation-shuttle",
    name: "Federation Shuttle",
    faction: "trade",
    cost: 1,
    trade: 2,
    authority: 4,
    text: "Ally: gain 4 authority",
    kind: "ship",
  },
  battlePod: {
    id: "battle-pod",
    name: "Battle Pod",
    faction: "blob",
    cost: 2,
    attack: 4,
    text: "Scrap a card in the trade row · Ally: +2 combat",
    kind: "ship",
  },
  missileBot: {
    id: "missile-bot",
    name: "Missile Bot",
    faction: "machine",
    cost: 2,
    attack: 2,
    text: "Scrap from hand or discard · Ally: +2 combat",
    kind: "ship",
  },
  imperial: {
    id: "imperial-frigate",
    name: "Imperial Frigate",
    faction: "star",
    cost: 3,
    attack: 4,
    text: "Opponent discards · Ally: +2 combat · Scrap: draw",
    kind: "ship",
  },
  cutter: {
    id: "cutter",
    name: "Cutter",
    faction: "trade",
    cost: 2,
    trade: 2,
    authority: 4,
    text: "Ally: +4 combat",
    kind: "ship",
  },
  tradePod: {
    id: "trade-pod",
    name: "Trade Pod",
    faction: "blob",
    cost: 2,
    trade: 3,
    text: "Ally: +2 combat",
    kind: "ship",
  },
  scoutA: {
    id: "scout-a",
    name: "Scout",
    faction: "neutral",
    cost: 0,
    trade: 1,
    text: "Gain 1 trade",
    kind: "ship",
  },
  scoutB: {
    id: "scout-b",
    name: "Scout",
    faction: "neutral",
    cost: 0,
    trade: 1,
    text: "Gain 1 trade",
    kind: "ship",
  },
  viper: {
    id: "viper",
    name: "Viper",
    faction: "neutral",
    cost: 0,
    attack: 1,
    text: "Gain 1 combat",
    kind: "ship",
  },
  tradingPost: {
    id: "trading-post",
    name: "Trading Post",
    faction: "trade",
    cost: 3,
    defense: 4,
    text: "Choose 1 authority or 1 trade · Scrap: +3 combat",
    kind: "base",
  },
  warWorld: {
    id: "war-world",
    name: "War World",
    faction: "star",
    cost: 5,
    attack: 3,
    defense: 4,
    text: "Outpost · Ally: +4 combat",
    kind: "base",
  },
};

const initialGame: GameState = {
  turn: 7,
  humanAuthority: 43,
  opponentAuthority: 37,
  trade: 0,
  attack: 0,
  deckCount: 4,
  discardCount: 7,
  opponentDeckCount: 8,
  explorersRemaining: 10,
  hand: [demoCards.scoutA, demoCards.federation, demoCards.viper, demoCards.tradePod, demoCards.scoutB],
  ownDeck: [demoCards.scoutA, demoCards.scoutB, demoCards.viper, demoCards.federation],
  opponentHand: [demoCards.scoutA, demoCards.viper, demoCards.imperial, demoCards.cutter, demoCards.scoutB],
  opponentDeck: [demoCards.scoutA, demoCards.scoutB, demoCards.viper, demoCards.tradePod, demoCards.missileBot, demoCards.battlePod, demoCards.cutter, demoCards.federation],
  market: [demoCards.battlePod, demoCards.missileBot, demoCards.imperial, demoCards.cutter, demoCards.tradingPost],
  humanBases: [demoCards.tradingPost],
  opponentBases: [demoCards.warWorld],
  log: [
    "Turn 7 · Your main phase",
    "Orion acquired War World for 5 trade",
    "You drew 5 cards",
  ],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asNumber(value: unknown, fallback: number): number {
  const converted = Number(value);
  return Number.isFinite(converted) ? converted : fallback;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function asGigabytes(value: unknown, fallback: number): number {
  const bytes = Number(value);
  return Number.isFinite(bytes) && bytes > 0 ? bytes / 1_073_741_824 : fallback;
}

function displayTime(value: unknown, fallback: string): string {
  if (typeof value === "string" && value.trim()) return value;
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) return fallback;
  return new Date(seconds * 1_000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function normalizeStatus(value: unknown, fallback: RunStatus): RunStatus {
  const status = String(value ?? "").toLowerCase();
  if (["training", "active", "started"].includes(status)) return "running";
  if (["pause", "paused"].includes(status)) return "paused";
  if (["pause_requested", "pausing"].includes(status)) return "pausing";
  if (["stop_requested", "stopping"].includes(status)) return "stopping";
  if (["done", "finished", "complete", "completed"].includes(status)) return "complete";
  if (["failed", "error"].includes(status)) return "error";
  if (status === "interrupted") return "interrupted";
  if (["idle", "ready", "created", "stopped"].includes(status)) return "ready";
  return fallback;
}

function normalizeMetric(raw: unknown, fallback: MetricPoint): MetricPoint {
  const item = isRecord(raw) ? raw : {};
  const evaluation = isRecord(item.evaluation) ? item.evaluation : {};
  const system = isRecord(item.system) ? item.system : {};
  const replay = isRecord(item.replay) ? item.replay : {};
  const familyRoot = isRecord(replay.families) ? replay.families : {};
  const replayFamilies = Object.fromEntries(
    Object.entries(familyRoot).map(([name, value]) => {
      const family = isRecord(value) ? value : {};
      return [name, asNumber(family.size ?? family.writes ?? value, 0)];
    }),
  );
  const hasEvaluation = Object.keys(evaluation).length > 0
    || item.win_rate !== undefined
    || item.ci_low !== undefined
    || item.score !== undefined;
  const hasLearnerDiagnostics = item.loss !== undefined || item.brier !== undefined;
  const totalBytes = asNumber(system.memory_total_bytes, 0);
  const availableBytes = asNumber(system.memory_available_bytes, 0);
  const games = asNumber(item.games ?? item.total_games, fallback.games);
  const draws = asNumber(item.draws, fallback.drawRate * games);
  const truncations = asNumber(item.truncations, fallback.truncationRate * games);
  return {
    seq: asNumber(item.seq ?? item.sequence, fallback.seq),
    games,
    gamesPerSecond: asNumber(item.games_per_second ?? item.games_per_sec ?? item.games_s, fallback.gamesPerSecond),
    decisionsPerSecond: asNumber(
      item.decisions_per_second ?? item.decisions_per_sec ?? item.decisions_s,
      fallback.decisionsPerSecond,
    ),
    winRate: hasEvaluation ? asNumber(item.win_rate ?? item.score ?? evaluation.estimate, 0.5) : 0.5,
    ciLow: hasEvaluation ? asNumber(item.ci_low ?? item.score_low ?? evaluation.low, 0) : 0,
    ciHigh: hasEvaluation ? asNumber(item.ci_high ?? item.score_high ?? evaluation.high, 1) : 1,
    hasEvaluation,
    hasLearnerDiagnostics,
    outcomeLoss: asNumber(item.loss ?? item.outcome_bce, fallback.outcomeLoss),
    brier: asNumber(item.brier, fallback.brier),
    explainedVariance: asNumber(item.explained_variance, fallback.explainedVariance),
    uncertainty: asNumber(item.uncertainty, fallback.uncertainty),
    truncationRate: asNumber(
      item.truncation_rate,
      games > 0 ? truncations / games : fallback.truncationRate,
    ),
    drawRate: games > 0 ? draws / games : fallback.drawRate,
    meanTurns: asNumber(item.mean_turns, fallback.meanTurns),
    forcedChoices: asNumber(item.forced_choices, fallback.forcedChoices),
    curriculumPhase: asString(item.curriculum_phase, fallback.curriculumPhase),
    bootstrapUpdatesRemaining: asNumber(
      item.heuristic_bootstrap_updates_remaining,
      fallback.bootstrapUpdatesRemaining,
    ),
    replayFill: asNumber(replay.utilization ?? item.replay_fill ?? item.replay_fraction, fallback.replayFill),
    replayFamilies: Object.keys(replayFamilies).length ? replayFamilies : fallback.replayFamilies,
    cpuPercent: asNumber(item.cpu_percent ?? system.cpu_percent, fallback.cpuPercent),
    memoryGb: asNumber(
      item.memory_gb ?? system.memory_gb,
      totalBytes > 0 && availableBytes >= 0 ? (totalBytes - availableBytes) / 1_073_741_824 : fallback.memoryGb,
    ),
  };
}

const emptyModel: ModelCheckpoint = {
  id: "",
  label: "Unnamed checkpoint",
  games: 0,
  created: "—",
  score: 0.5,
  ciLow: 0,
  ciHigh: 1,
  eloDelta: 0,
  hasElo: false,
  isChampion: false,
  isPinned: false,
  sizeMb: null,
  evaluated: false,
};

function normalizeModel(raw: unknown, fallback: ModelCheckpoint = emptyModel): ModelCheckpoint {
  const item = isRecord(raw) ? raw : {};
  const evaluation = isRecord(item.evaluation) ? item.evaluation : {};
  const latestArena = isRecord(evaluation.latest_arena) ? evaluation.latest_arena : {};
  const latestPaired = isRecord(latestArena.paired_interval) ? latestArena.paired_interval : {};
  const directPaired = isRecord(evaluation.paired_interval) ? evaluation.paired_interval : {};
  const hasLatestArena = latestArena.model_a_score !== undefined;
  const hasLegacyEvaluation = evaluation.evaluated === true || (
    evaluation.evaluated === undefined
    && (evaluation.estimate !== undefined
      || evaluation.score !== undefined
      || evaluation.win_rate !== undefined
      || evaluation.model_a_score !== undefined)
  );
  const evaluated = hasLatestArena
    || item.evaluated === true
    || hasLegacyEvaluation
    || (item.evaluated === undefined && item.score !== undefined);
  const eloValue = item.elo_delta
    ?? latestArena.elo_difference_a_minus_b
    ?? evaluation.elo_delta
    ?? (fallback.hasElo ? fallback.eloDelta : undefined);
  const hasElo = eloValue !== null && eloValue !== undefined && Number.isFinite(Number(eloValue));
  const sizeValue = item.size_mb !== null && item.size_mb !== undefined
    ? asNumber(item.size_mb, 0)
    : item.size_bytes !== null && item.size_bytes !== undefined
      ? asNumber(item.size_bytes, 0) / 1_048_576
      : fallback.sizeMb;
  return {
    id: asString(item.id, fallback.id),
    label: asString(item.label ?? item.name, fallback.label),
    parentId: typeof item.parent_id === "string" && item.parent_id ? item.parent_id : undefined,
    games: asNumber(item.games, fallback.games),
    created: displayTime(item.created ?? item.created_at ?? item.created_at_display, fallback.created),
    score: evaluated
      ? asNumber(
          item.score
            ?? latestArena.model_a_score
            ?? evaluation.model_a_score
            ?? evaluation.estimate
            ?? evaluation.score
            ?? evaluation.win_rate,
          0.5,
        )
      : 0.5,
    ciLow: evaluated
      ? asNumber(item.ci_low ?? latestPaired.low ?? directPaired.low ?? evaluation.low ?? evaluation.ci_low, 0)
      : 0,
    ciHigh: evaluated
      ? asNumber(item.ci_high ?? latestPaired.high ?? directPaired.high ?? evaluation.high ?? evaluation.ci_high, 1)
      : 1,
    eloDelta: evaluated && hasElo ? asNumber(eloValue, 0) : 0,
    hasElo: evaluated && hasElo,
    isChampion: Boolean(item.is_champion ?? fallback.isChampion),
    isPinned: Boolean(item.is_pinned ?? fallback.isPinned),
    sizeMb: sizeValue === null ? null : Math.max(0, sizeValue),
    evaluated,
  };
}

function normalizeSnapshot(raw: unknown, previous: DashboardSnapshot): DashboardSnapshot {
  const root = isRecord(raw) ? raw : {};
  const live = isRecord(root.live) ? root.live : {};
  const sourceRun = isRecord(root.run)
    ? { ...root.run, ...live }
    : isRecord(root.active_run)
      ? root.active_run
      : root;
  const latestRecord = isRecord(root.latest_metric) ? root.latest_metric : {};
  const metricSystem = isRecord(latestRecord.system) ? latestRecord.system : {};
  const metricMetal = isRecord(latestRecord.metal) ? latestRecord.metal : {};
  const baseHardware = isRecord(root.hardware)
    ? root.hardware
    : isRecord(root.system)
      ? root.system
      : {};
  const sourceHardware: Record<string, unknown> = {
    ...baseHardware,
    ...metricSystem,
    learner_device: metricMetal.device ?? baseHardware.learner_device,
  };
  const sourceConfig = isRecord(sourceRun.config) ? sourceRun.config : {};
  const sourceMetrics = Array.isArray(root.metrics) ? root.metrics : [];
  const latestRaw = root.latest_metric;
  const incomingRunId = asString(sourceRun.id ?? sourceRun.run_id, previous.run.id);
  const sameRun = incomingRunId === previous.run.id;
  const priorMetrics = sameRun ? previous.metrics : [];
  const lastMetric = priorMetrics.at(-1) ?? emptyMetric;
  let metrics = priorMetrics;
  if (sourceMetrics.length) {
    const bySequence = new Map(metrics.map((point) => [point.seq, point]));
    sourceMetrics.forEach((point, index) => {
      const normalized = normalizeMetric(point, { ...lastMetric, seq: lastMetric.seq + index + 1 });
      bySequence.set(normalized.seq, normalized);
    });
    metrics = [...bySequence.values()].sort((a, b) => a.seq - b.seq).slice(-180);
  }
  if (latestRaw) {
    const latest = normalizeMetric(latestRaw, lastMetric);
    if (latest.seq !== metrics.at(-1)?.seq || latest.games !== metrics.at(-1)?.games) {
      metrics = [...metrics, latest].slice(-180);
    }
  }

  const rawModels = Array.isArray(root.models) ? root.models : null;
  const models = rawModels
    ? rawModels.map((model) => {
        const modelId = isRecord(model) ? asString(model.id, "") : "";
        const priorModel = sameRun ? previous.models.find((item) => item.id === modelId) : undefined;
        return normalizeModel(model, priorModel);
      })
    : previous.models;

  const rawEvents = Array.isArray(root.events) ? root.events : null;
  const events = rawEvents
    ? rawEvents.slice(0, 20).map((event, index): AuditEvent => {
        const item = isRecord(event) ? event : {};
        const rawKind = String(item.kind ?? "info").toLowerCase();
        const payload = isRecord(item.payload) ? item.payload : {};
        const payloadDetail = Object.keys(payload).length ? JSON.stringify(payload) : "";
        const kind: AuditEvent["kind"] = rawKind.includes("error")
          ? "error"
          : rawKind.includes("warn")
            ? "warning"
            : rawKind.includes("promot") || rawKind.includes("success")
              ? "success"
              : "info";
        return {
          id: item.id === null || item.id === undefined ? `api-event-${index}` : String(item.id),
          at: displayTime(item.at ?? item.created_at ?? item.created_at_display, "recently"),
          kind,
          title: asString(item.title ?? item.message, "Trainer event"),
          detail: asString(item.detail, payloadDetail || rawKind.replaceAll("_", " ")),
        };
      })
    : previous.events;

  return {
    run: {
      id: incomingRunId,
      name: asString(sourceRun.name ?? sourceRun.run_name, previous.run.name),
      status: normalizeStatus(sourceRun.status, previous.run.status),
      phase: asString(sourceRun.phase ?? sourceRun.stage, previous.run.phase),
      games: asNumber(sourceRun.games ?? sourceRun.total_games, metrics.at(-1)?.games ?? previous.run.games),
      decisions: asNumber(sourceRun.decisions ?? sourceRun.total_decisions, previous.run.decisions),
      updates: asNumber(sourceRun.updates ?? sourceRun.iteration, previous.run.updates),
      elapsedSeconds: asNumber(
        sourceRun.active_elapsed_seconds ?? sourceRun.elapsed_seconds ?? sourceRun.elapsed,
        metrics.at(-1)?.seq === asNumber(latestRaw && isRecord(latestRaw) ? latestRaw.seq : -2, -2)
          ? asNumber(
              isRecord(latestRaw) ? latestRaw.active_elapsed_seconds ?? latestRaw.elapsed_seconds : undefined,
              sameRun ? previous.run.elapsedSeconds : 0,
            )
          : sameRun ? previous.run.elapsedSeconds : 0,
      ),
      durationSeconds: asNumber(
        sourceRun.duration_seconds ?? sourceConfig.duration_seconds,
        asNumber(sourceRun.duration_minutes ?? sourceConfig.duration_minutes, previous.run.durationSeconds / 60) * 60,
      ),
      championId: asString(
        sourceRun.champion_id ?? sourceRun.best_checkpoint,
        sameRun ? previous.run.championId : "",
      ),
      seed: asNumber(sourceRun.seed ?? sourceConfig.seed, sameRun ? previous.run.seed : 0),
      lastError: asString(sourceRun.last_error, previous.run.lastError ?? "") || undefined,
    },
    hardware: {
      machine: asString(sourceHardware.machine ?? sourceHardware.host, previous.hardware.machine),
      chip: asString(sourceHardware.chip ?? sourceHardware.cpu, previous.hardware.chip),
      backend: asString(sourceHardware.backend ?? sourceHardware.accelerator, previous.hardware.backend),
      memoryTotalGb: asNumber(
        sourceHardware.memory_total_gb,
        asGigabytes(sourceHardware.memory_total_bytes, previous.hardware.memoryTotalGb),
      ),
      memoryUsedGb: asNumber(
        sourceHardware.memory_used_gb,
        sourceHardware.memory_total_bytes && sourceHardware.memory_available_bytes
          ? asGigabytes(
              Number(sourceHardware.memory_total_bytes) - Number(sourceHardware.memory_available_bytes),
              previous.hardware.memoryUsedGb,
            )
          : previous.hardware.memoryUsedGb,
      ),
      cpuPercent: asNumber(sourceHardware.cpu_percent, metrics.at(-1)?.cpuPercent ?? previous.hardware.cpuPercent),
      actorProcesses: asNumber(
        sourceHardware.actor_processes ?? sourceConfig.actor_processes ?? sourceHardware.recommended_actor_processes,
        previous.hardware.actorProcesses,
      ),
      learnerDevice: asString(
        sourceHardware.learner_device ?? sourceHardware.device ?? sourceHardware.accelerator,
        previous.hardware.learnerDevice,
      ),
      metalAvailable: Object.keys(metricMetal).length ? Boolean(metricMetal.metal_available) : null,
      metalActiveGb: Object.keys(metricMetal).length
        ? asGigabytes(metricMetal.active_memory_bytes, 0)
        : 0,
      metalPeakGb: Object.keys(metricMetal).length
        ? asGigabytes(metricMetal.peak_memory_bytes, 0)
        : 0,
    },
    metrics,
    models,
    events,
  };
}

function formatDuration(totalSeconds: number, compact = false): string {
  const seconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(seconds / 3_600);
  const minutes = Math.floor((seconds % 3_600) / 60);
  const remainder = seconds % 60;
  if (compact) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function normalizeArenaJob(raw: unknown): ArenaResultView | null {
  if (!isRecord(raw)) return null;
  const result = isRecord(raw.result) ? raw.result : {};
  const paired = isRecord(result.paired_interval) ? result.paired_interval : {};
  const modelA = isRecord(result.model_a) ? result.model_a : {};
  const modelB = isRecord(result.model_b) ? result.model_b : {};
  const pairsRequested = asNumber(result.pairs_requested, 0);
  const pairsCompleted = asNumber(result.pairs_completed, 0);
  const progressFraction = asNumber(
    result.progress,
    pairsRequested ? pairsCompleted / pairsRequested : 0,
  );
  return {
    id: asString(raw.id ?? raw.job_id, ""),
    status: asString(raw.status, "queued"),
    progress: Math.max(0, Math.min(100, progressFraction <= 1 ? progressFraction * 100 : progressFraction)),
    pairsCompleted,
    pairsRequested,
    modelALabel: asString(modelA.label ?? modelA.ref, "Model A"),
    modelBLabel: asString(modelB.label ?? modelB.ref, "Model B"),
    score: asNumber(result.model_a_score, 0.5),
    ciLow: asNumber(paired.low, 0),
    ciHigh: asNumber(paired.high, 1),
    elo: asNumber(result.elo_difference_a_minus_b, 0),
    firstSeatScore: asNumber(result.model_a_first_seat_score, 0.5),
    secondSeatScore: asNumber(result.model_a_second_seat_score, 0.5),
    truncatedGames: asNumber(result.truncated_games, 0),
    gamesCompleted: asNumber(result.games_completed, pairsCompleted * 2),
    intervalMethod: asString(result.paired_interval_method, "paired interval"),
    recommendation: asString(
      raw.error ?? (isRecord(result.promotion) ? result.promotion.recommendation : undefined),
      pairsCompleted ? "Evaluation in progress" : "Waiting for the first paired result",
    ),
  };
}

function cardFromApi(raw: unknown, fallbackId: string): GameCard {
  const item = isRecord(raw) ? raw : {};
  const faction = String(item.faction ?? "unaligned");
  const cardFaction: GameCard["faction"] = faction.includes("blob")
    ? "blob"
    : faction.includes("machine")
      ? "machine"
      : faction.includes("star")
        ? "star"
        : faction.includes("trade")
          ? "trade"
          : "neutral";
  const text = [item.primary, item.ally ? `Ally: ${item.ally}` : "", item.scrap ? `Scrap: ${item.scrap}` : ""]
    .filter(Boolean)
    .map((part) => titleCase(String(part)))
    .join(" · ");
  return {
    id: `${asNumber(item.card_id, -1)}-${fallbackId}`,
    name: asString(item.name, "Unknown card"),
    faction: cardFaction,
    cost: asNumber(item.cost, 0),
    attack: asNumber(item.combat, 0) || undefined,
    trade: asNumber(item.trade, 0) || undefined,
    authority: asNumber(item.authority, 0) || undefined,
    defense: asNumber(item.defense, 0) || undefined,
    text: text || "No additional ability",
    kind: String(item.card_type ?? "ship") === "ship" ? "ship" : "base",
  };
}

function normalizeRemoteGame(raw: unknown, previous: GameState): {
  session: RemoteGameSession;
  game: GameState;
} | null {
  if (!isRecord(raw)) return null;
  const observation = isRecord(raw.observation) ? raw.observation : {};
  const board = isRecord(raw.board) ? raw.board : {};
  const players = Array.isArray(board.players) ? board.players : [];
  const human = isRecord(players[0]) ? players[0] : {};
  const opponent = isRecord(players[1]) ? players[1] : {};
  const decision = isRecord(raw.decision) ? raw.decision : {};
  const rawActions = Array.isArray(decision.actions) ? decision.actions : [];
  const rawLog = Array.isArray(raw.action_log) ? raw.action_log : [];
  const result = isRecord(raw.result) ? raw.result : {};
  const cardZones = isRecord(raw.card_zones) ? raw.card_zones : {};
  const ownZones = isRecord(cardZones.own) ? cardZones.own : {};
  const opponentZones = isRecord(cardZones.opponent) ? cardZones.opponent : {};
  const hand = Array.isArray(observation.hand)
    ? observation.hand.map((card, index) => cardFromApi(card, `hand-${index}`))
    : previous.hand;
  const market = Array.isArray(observation.trade_row)
    ? observation.trade_row.filter(Boolean).map((card, index) => cardFromApi(card, `market-${index}`))
    : previous.market;
  const ownInPlay = Array.isArray(observation.own_in_play) ? observation.own_in_play : [];
  const opponentInPlay = Array.isArray(observation.opponent_in_play) ? observation.opponent_in_play : [];
  const status = asString(raw.status, "model_thinking") as RemoteGameSession["status"];
  const resultText = raw.result
    ? result.winner === null
      ? "Draw"
      : Number(result.winner) === 0
        ? "You won"
        : "Opponent won"
    : undefined;
  return {
    session: {
      id: asString(raw.id, ""),
      status,
      prompt: asString(decision.prompt, status === "model_thinking" ? "Opponent is choosing…" : "Choose a legal action"),
      family: asString(decision.family, "main_phase"),
      actions: rawActions.map((action, index) => {
        const item = isRecord(action) ? action : {};
        return {
          id: asNumber(item.id, index),
          label: asString(item.label, `Action ${index + 1}`),
          modelValue: item.model_value === null || item.model_value === undefined ? null : asNumber(item.model_value, 0.5),
          recommended: Boolean(item.model_recommended),
        };
      }),
      modelLabel: asString(raw.model_label, "Opponent"),
      result: resultText,
      error: asString(raw.error, "") || undefined,
    },
    game: {
      turn: asNumber(observation.turn ?? board.turns, previous.turn),
      humanAuthority: asNumber(observation.own_authority ?? human.authority, previous.humanAuthority),
      opponentAuthority: asNumber(observation.opponent_authority ?? opponent.authority, previous.opponentAuthority),
      trade: asNumber(observation.trade ?? human.trade, previous.trade),
      attack: asNumber(observation.combat ?? human.combat, previous.attack),
      deckCount: asNumber(observation.own_deck_count ?? human.deck_count, previous.deckCount),
      discardCount: Array.isArray(observation.own_discard)
        ? observation.own_discard.length
        : Array.isArray(human.discard)
          ? human.discard.length
          : previous.discardCount,
      opponentDeckCount: asNumber(observation.opponent_deck_count ?? opponent.deck_count, previous.opponentDeckCount),
      explorersRemaining: asNumber(observation.explorers_remaining ?? board.explorers_remaining, previous.explorersRemaining),
      hand,
      ownDeck: Array.isArray(ownZones.deck)
        ? ownZones.deck.map((card, index) => cardFromApi(card, `own-deck-${index}`))
        : previous.ownDeck,
      opponentHand: Array.isArray(opponentZones.hand)
        ? opponentZones.hand.map((card, index) => cardFromApi(card, `opponent-hand-${index}`))
        : previous.opponentHand,
      opponentDeck: Array.isArray(opponentZones.deck)
        ? opponentZones.deck.map((card, index) => cardFromApi(card, `opponent-deck-${index}`))
        : previous.opponentDeck,
      market,
      humanBases: ownInPlay
        .map((entry, index) => cardFromApi(isRecord(entry) ? entry.card : entry, `own-base-${index}`))
        .filter((card) => card.kind === "base"),
      opponentBases: opponentInPlay
        .map((entry, index) => cardFromApi(isRecord(entry) ? entry.card : entry, `opponent-base-${index}`))
        .filter((card) => card.kind === "base"),
      log: rawLog.length
        ? [...rawLog].reverse().map((entry) => {
            const item = isRecord(entry) ? entry : {};
            return `${asString(item.player, "Game")} · ${asString(item.label, "action")}`;
          })
        : previous.log,
    },
  };
}

function configToApi(config: TrainerConfig): Record<string, string | number> {
  return {
    preset: config.preset,
    duration_minutes: config.durationMinutes,
    actor_processes: config.actorProcesses,
    games_per_actor_batch: config.gamesPerActorBatch,
    hidden_size: config.hiddenSize,
    residual_blocks: config.residualBlocks,
    bootstrap_heads: config.bootstrapHeads,
    batch_size: config.batchSize,
    learning_rate: config.learningRate,
    replay_capacity: config.replayCapacity,
    replay_warmup: config.replayWarmup,
    heuristic_bootstrap_updates: config.heuristicBootstrapUpdates,
    current_selfplay_fraction: config.currentSelfplayFraction,
    league_fraction: config.leagueFraction,
    baseline_fraction: config.baselineFraction,
    evaluation_pairs: config.evaluationPairs,
    evaluate_every_games: config.evaluateEveryGames,
    checkpoint_every_games: config.checkpointEveryGames,
    promotion_confidence: config.promotionConfidence,
    promotion_margin: config.promotionMargin,
    epsilon_start: config.epsilonStart,
    epsilon_end: config.epsilonEnd,
  };
}

function configFromApi(raw: unknown, previous: TrainerConfig): TrainerConfig {
  const item = isRecord(raw) ? raw : {};
  const presetValue = String(item.preset ?? previous.preset);
  const preset: TrainerConfig["preset"] = ["m4_24h", "quick", "custom"].includes(presetValue)
    ? presetValue as TrainerConfig["preset"]
    : previous.preset;
  return {
    name: asString(item.name, previous.name),
    preset,
    durationMinutes: asNumber(item.duration_minutes, previous.durationMinutes),
    actorProcesses: asNumber(item.actor_processes, previous.actorProcesses),
    gamesPerActorBatch: asNumber(item.games_per_actor_batch, previous.gamesPerActorBatch),
    hiddenSize: asNumber(item.hidden_size, previous.hiddenSize),
    residualBlocks: asNumber(item.residual_blocks, previous.residualBlocks),
    bootstrapHeads: asNumber(item.bootstrap_heads, previous.bootstrapHeads),
    batchSize: asNumber(item.batch_size, previous.batchSize),
    learningRate: asNumber(item.learning_rate, previous.learningRate),
    replayCapacity: asNumber(item.replay_capacity, previous.replayCapacity),
    replayWarmup: asNumber(item.replay_warmup, previous.replayWarmup),
    heuristicBootstrapUpdates: asNumber(
      item.heuristic_bootstrap_updates,
      previous.heuristicBootstrapUpdates,
    ),
    currentSelfplayFraction: asNumber(item.current_selfplay_fraction, previous.currentSelfplayFraction),
    leagueFraction: asNumber(item.league_fraction, previous.leagueFraction),
    baselineFraction: asNumber(item.baseline_fraction, previous.baselineFraction),
    evaluationPairs: asNumber(item.evaluation_pairs, previous.evaluationPairs),
    evaluateEveryGames: asNumber(item.evaluate_every_games, previous.evaluateEveryGames),
    checkpointEveryGames: asNumber(item.checkpoint_every_games, previous.checkpointEveryGames),
    promotionConfidence: asNumber(item.promotion_confidence, previous.promotionConfidence),
    promotionMargin: asNumber(item.promotion_margin, previous.promotionMargin),
    epsilonStart: asNumber(item.epsilon_start, previous.epsilonStart),
    epsilonEnd: asNumber(item.epsilon_end, previous.epsilonEnd),
  };
}

async function fetchJson(path: string, options?: RequestInit): Promise<unknown> {
  const controller = new AbortController();
  const method = (options?.method ?? "GET").toUpperCase();
  const timeoutMs = method === "GET" ? 3_000 : 10_000;
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      cache: "no-store",
      headers: {
        Accept: "application/json",
        ...(options?.body ? { "Content-Type": "application/json" } : {}),
        ...options?.headers,
      },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } finally {
    window.clearTimeout(timeout);
  }
}

function MetricCanvas({
  points,
  mode,
  height = 220,
}: {
  points: MetricPoint[];
  mode: "throughput" | "quality" | "loss";
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const draw = () => {
      const rect = parent.getBoundingClientRect();
      const width = Math.max(280, rect.width);
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);

      const padding = { top: 16, right: 12, bottom: 24, left: 12 };
      const chartWidth = width - padding.left - padding.right;
      const chartHeight = height - padding.top - padding.bottom;
      const visible = points.slice(-72);
      if (visible.length < 2) return;

      context.strokeStyle = "rgba(140, 167, 193, 0.12)";
      context.lineWidth = 1;
      for (let row = 0; row <= 4; row += 1) {
        const y = padding.top + (chartHeight / 4) * row;
        context.beginPath();
        context.moveTo(padding.left, y + 0.5);
        context.lineTo(width - padding.right, y + 0.5);
        context.stroke();
      }

      const xAt = (index: number) => padding.left + (index / (visible.length - 1)) * chartWidth;
      const series =
        mode === "throughput"
          ? [
              { values: visible.map((point) => point.gamesPerSecond), color: "#5fe6ca" },
              { values: visible.map((point) => point.decisionsPerSecond / 240), color: "#6aa9ff" },
            ]
          : mode === "quality"
            ? [{ values: visible.map((point) => point.winRate), color: "#e8bd64" }]
            : [
                { values: visible.map((point) => point.outcomeLoss), color: "#8fb2ff" },
                { values: visible.map((point) => point.brier), color: "#e58ad4" },
              ];

      const allValues = series.flatMap((item) => item.values);
      if (mode === "quality") {
        allValues.push(...visible.map((point) => point.ciLow), ...visible.map((point) => point.ciHigh), 0.5);
      }
      const rawMin = Math.min(...allValues);
      const rawMax = Math.max(...allValues);
      const range = Math.max(rawMax - rawMin, 0.05);
      const min = mode === "quality" ? Math.max(0.42, rawMin - range * 0.12) : Math.max(0, rawMin - range * 0.12);
      const max = rawMax + range * 0.12;
      const yAt = (value: number) => padding.top + chartHeight - ((value - min) / Math.max(max - min, 0.001)) * chartHeight;

      if (mode === "quality") {
        context.fillStyle = "rgba(232, 189, 100, 0.12)";
        context.beginPath();
        visible.forEach((point, index) => {
          const x = xAt(index);
          const y = yAt(point.ciHigh);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        for (let index = visible.length - 1; index >= 0; index -= 1) {
          context.lineTo(xAt(index), yAt(visible[index].ciLow));
        }
        context.closePath();
        context.fill();

        context.setLineDash([4, 5]);
        context.strokeStyle = "rgba(229, 236, 244, 0.25)";
        context.beginPath();
        context.moveTo(padding.left, yAt(0.5));
        context.lineTo(width - padding.right, yAt(0.5));
        context.stroke();
        context.setLineDash([]);
      }

      series.forEach((item, seriesIndex) => {
        const gradient = context.createLinearGradient(0, padding.top, 0, height - padding.bottom);
        gradient.addColorStop(0, `${item.color}2d`);
        gradient.addColorStop(1, `${item.color}00`);
        context.beginPath();
        item.values.forEach((value, index) => {
          const x = xAt(index);
          const y = yAt(value);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        if (seriesIndex === 0 && mode !== "loss") {
          context.lineTo(xAt(item.values.length - 1), height - padding.bottom);
          context.lineTo(xAt(0), height - padding.bottom);
          context.closePath();
          context.fillStyle = gradient;
          context.fill();
          context.beginPath();
          item.values.forEach((value, index) => {
            const x = xAt(index);
            const y = yAt(value);
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
          });
        }
        context.strokeStyle = item.color;
        context.lineWidth = seriesIndex === 0 ? 2 : 1.5;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.stroke();
      });

      const latest = series[0].values.at(-1) ?? 0;
      context.fillStyle = series[0].color;
      context.beginPath();
      context.arc(xAt(visible.length - 1), yAt(latest), 3.5, 0, Math.PI * 2);
      context.fill();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [height, mode, points]);

  const label =
    mode === "throughput"
      ? "Training throughput over recent samples"
      : mode === "quality"
        ? "Held-out win rate and confidence interval over recent evaluations"
        : "Outcome binary cross-entropy and Brier score over recent updates";

  return <canvas ref={canvasRef} className="metric-canvas" role="img" aria-label={label} />;
}

function StatusDot({ status }: { status: RunStatus }) {
  return <span className={`status-dot status-${status}`} aria-hidden="true" />;
}

function Gauge({ value, label }: { value: number; label: string }) {
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div className="gauge" style={{ "--gauge": `${clamped * 360}deg` } as CSSProperties}>
      <div className="gauge-inner">
        <strong>{Math.round(clamped * 100)}%</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

function CardTile({
  card,
  compact = false,
  selected = false,
  disabled = false,
  onClick,
}: {
  card: GameCard;
  compact?: boolean;
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}) {
  const content = (
    <>
      <span className="card-cost" aria-label={`Cost ${card.cost}`}>{card.cost}</span>
      <span className="card-kind">{card.kind}</span>
      <strong>{card.name}</strong>
      <span className="card-rule">{card.text}</span>
      <span className="card-stats">
        {card.trade ? <span><b>{card.trade}</b> trade</span> : null}
        {card.attack ? <span><b>{card.attack}</b> combat</span> : null}
        {card.authority ? <span><b>{card.authority}</b> auth</span> : null}
        {card.defense ? <span><b>{card.defense}</b> defense</span> : null}
      </span>
    </>
  );
  const className = `game-card faction-${card.faction}${compact ? " card-compact" : ""}${selected ? " is-selected" : ""}`;
  if (onClick) {
    return (
      <button type="button" className={className} onClick={onClick} disabled={disabled} aria-pressed={selected}>
        {content}
      </button>
    );
  }
  return <div className={className}>{content}</div>;
}

function CardCollection({ label, cards }: { label: string; cards: GameCard[] }) {
  const groups = Array.from(
    cards.reduce((items, card) => {
      const current = items.get(card.name);
      items.set(card.name, current ? { card, count: current.count + 1 } : { card, count: 1 });
      return items;
    }, new Map<string, { card: GameCard; count: number }>()),
  ).map(([, value]) => value);

  return (
    <section className="card-collection" aria-label={`${label}, unordered`}>
      <header><strong>{label}</strong><span>{cards.length} cards · unordered</span></header>
      <div>
        {groups.map(({ card, count }) => (
          <span className={`card-chip faction-${card.faction}`} key={card.name}>
            <b>{count}×</b><span>{card.name}</span><small>{card.kind} · cost {card.cost}</small>
          </span>
        ))}
        {!cards.length ? <span className="empty-card-zone">No cards</span> : null}
      </div>
    </section>
  );
}

function CardInventory({ game, onClose }: { game: GameState; onClose: () => void }) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="inventory-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section className="card-inventory" role="dialog" aria-modal="true" aria-labelledby="inventory-title">
        <header>
          <div><span className="panel-kicker">Card visibility</span><h2 id="inventory-title">Hands & decks</h2><p>Cards are grouped by name. Deck order is intentionally not shown.</p></div>
          <button type="button" className="inventory-close" onClick={onClose} aria-label="Close hands and decks">×</button>
        </header>
        <div className="inventory-players">
          <article><h3><span className="player-avatar human-avatar">YOU</span>Your cards</h3><CardCollection label="Hand" cards={game.hand} /><CardCollection label="Deck" cards={game.ownDeck} /></article>
          <article><h3><span className="player-avatar opponent-avatar">AI</span>Opponent cards</h3><CardCollection label="Hand" cards={game.opponentHand} /><CardCollection label="Deck" cards={game.opponentDeck} /></article>
        </div>
      </section>
    </div>
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <span className="empty-orbit" aria-hidden="true">◎</span>
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [snapshot, setSnapshot] = useState<DashboardSnapshot>(demoSnapshot);
  const [connected, setConnected] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [commandBusy, setCommandBusy] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [config, setConfig] = useState<TrainerConfig>(initialConfig);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [arenaA, setArenaA] = useState("champion-042");
  const [arenaB, setArenaB] = useState("champion-038");
  const [arenaPairs, setArenaPairs] = useState(5_000);
  const [arenaRunning, setArenaRunning] = useState(false);
  const [arenaProgress, setArenaProgress] = useState(0);
  const [arenaJobId, setArenaJobId] = useState<string | null>(null);
  const [arenaResult, setArenaResult] = useState<ArenaResultView | null>(demoArenaResult);
  const [game, setGame] = useState<GameState>(initialGame);
  const [selectedCard, setSelectedCard] = useState<string | null>(null);
  const [playModel, setPlayModel] = useState("champion-042");
  const [humanStarts, setHumanStarts] = useState(false);
  const [remoteRunId, setRemoteRunId] = useState<string | null>(null);
  const [remoteGame, setRemoteGame] = useState<RemoteGameSession | null>(null);
  const [inventoryOpen, setInventoryOpen] = useState(false);
  const pollInFlight = useRef(false);
  const hasConnectedRef = useRef(false);
  const activeRunIdRef = useRef<string | null>(null);
  const metricsSeqRef = useRef(-1);
  const gameRef = useRef(game);
  const basePresetRef = useRef<"m4_24h" | "quick">("m4_24h");
  const presetCatalogRef = useRef<Record<string, unknown>>({});

  const latestMetric = snapshot.metrics.at(-1) ?? (connected ? emptyMetric : buildDemoMetrics(1)[0]);
  const evaluatedMetricPoints = snapshot.metrics.filter((point) => point.hasEvaluation);
  const arenaHasSamples = Boolean(arenaResult && arenaResult.pairsCompleted > 0);
  const heldOutScore = arenaHasSamples ? arenaResult!.score : latestMetric.hasEvaluation ? latestMetric.winRate : 0.5;
  const heldOutLow = arenaHasSamples ? arenaResult!.ciLow : latestMetric.hasEvaluation ? latestMetric.ciLow : 0;
  const heldOutHigh = arenaHasSamples ? arenaResult!.ciHigh : latestMetric.hasEvaluation ? latestMetric.ciHigh : 1;
  const replayFamilyEntries = Object.entries(latestMetric.replayFamilies)
    .map(([name, size]) => ({ name: titleCase(name), size }))
    .sort((a, b) => b.size - a.size);
  const replayFamilyTotal = replayFamilyEntries.reduce((sum, item) => sum + item.size, 0);
  const progress = snapshot.run.durationSeconds
    ? Math.min(1, snapshot.run.elapsedSeconds / snapshot.run.durationSeconds)
    : 0;
  const remaining = Math.max(0, snapshot.run.durationSeconds - snapshot.run.elapsedSeconds);

  const showToast = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast((current) => (current === message ? null : current)), 3_000);
  };

  useEffect(() => {
    gameRef.current = game;
  }, [game]);

  useEffect(() => {
    let cancelled = false;

    const tickDemo = () => {
      setSnapshot((current) => {
        if (current.run.status !== "running") return current;
        const previous = current.metrics.at(-1) ?? buildDemoMetrics(1)[0];
        const nextSeq = previous.seq + 1;
        const nextGamesPerSecond = Math.max(4, previous.gamesPerSecond + Math.sin(nextSeq * 0.37) * 0.08);
        const gamesAdded = Math.round(nextGamesPerSecond);
        const drift = Math.min(0.0002, nextSeq / 1_000_000);
        const next: MetricPoint = {
          ...previous,
          seq: nextSeq,
          games: previous.games + gamesAdded,
          gamesPerSecond: nextGamesPerSecond,
          decisionsPerSecond: 2_440 + Math.cos(nextSeq * 0.32) * 130,
          winRate: Math.min(0.67, previous.winRate + drift + Math.sin(nextSeq * 0.59) * 0.0006),
          ciLow: Math.min(0.65, previous.ciLow + drift),
          ciHigh: Math.min(0.69, previous.ciHigh + drift),
          outcomeLoss: Math.max(0.42, previous.outcomeLoss + Math.sin(nextSeq * 0.9) * 0.002),
          brier: Math.max(0.11, previous.brier + Math.cos(nextSeq * 0.7) * 0.001),
          explainedVariance: Math.min(0.8, previous.explainedVariance + 0.0002),
          uncertainty: Math.max(0.025, previous.uncertainty - 0.00003),
          replayFill: Math.min(0.98, previous.replayFill + 0.0004),
          cpuPercent: 78 + Math.sin(nextSeq * 0.44) * 5,
          memoryGb: Math.min(12.8, previous.memoryGb + 0.0008),
        };
        return {
          ...current,
          run: {
            ...current.run,
            games: current.run.games + gamesAdded,
            decisions: current.run.decisions + Math.round(next.decisionsPerSecond),
            elapsedSeconds: current.run.elapsedSeconds + 1,
          },
          hardware: {
            ...current.hardware,
            cpuPercent: next.cpuPercent,
            memoryUsedGb: next.memoryGb,
          },
          metrics: [...current.metrics, next].slice(-180),
        };
      });
    };

    const poll = async () => {
      if (pollInFlight.current) return;
      pollInFlight.current = true;
      try {
        const [healthRaw, systemRaw, presetsRaw, runsRaw] = await Promise.all([
          fetchJson("/health"),
          fetchJson("/system"),
          Object.keys(presetCatalogRef.current).length
            ? Promise.resolve(presetCatalogRef.current)
            : fetchJson("/presets").catch(() => ({})),
          fetchJson("/runs"),
        ]);
        if (cancelled) return;
        if (isRecord(presetsRaw) && Object.keys(presetsRaw).length) presetCatalogRef.current = presetsRaw;
        const health = isRecord(healthRaw) ? healthRaw : {};
        const runs = Array.isArray(runsRaw) ? runsRaw : [];
        const activeId = typeof health.active_run_id === "string" ? health.active_run_id : null;
        const rememberedId = activeRunIdRef.current;
        const rememberedExists = rememberedId
          ? runs.some((run) => isRecord(run) && run.id === rememberedId)
          : false;
        const runId = activeId
          ?? (rememberedExists ? rememberedId : null)
          ?? (isRecord(runs[0]) ? asString(runs[0].id, "") || null : null);

        if (!runId) {
          setRemoteRunId(null);
          activeRunIdRef.current = null;
          metricsSeqRef.current = -1;
          setArenaResult(null);
          setArenaProgress(0);
          setArenaRunning(false);
          setArenaJobId(null);
          setSnapshot((current) => {
            const next = normalizeSnapshot({ system: systemRaw }, current);
            return {
              ...next,
              run: {
                ...next.run,
                id: "",
                name: "No run selected",
                status: "ready",
                phase: "Awaiting launch",
                games: 0,
                decisions: 0,
                updates: 0,
                elapsedSeconds: 0,
                championId: "",
                seed: 0,
              },
              metrics: [],
              models: [],
              events: [],
            };
          });
        } else {
          const changedRun = activeRunIdRef.current !== runId;
          const after = changedRun ? -1 : metricsSeqRef.current;
          const [detailRaw, metricsRaw, modelsRaw, eventsRaw, arenaJobsRaw] = await Promise.all([
            fetchJson(`/runs/${encodeURIComponent(runId)}`),
            fetchJson(`/runs/${encodeURIComponent(runId)}/metrics?after=${after}`),
            fetchJson(`/models?run_id=${encodeURIComponent(runId)}`),
            fetchJson(`/runs/${encodeURIComponent(runId)}/events?limit=100`),
            fetchJson(`/arena?limit=1&run_id=${encodeURIComponent(runId)}`).catch(() => []),
          ]);
          if (cancelled) return;
          const latestArenaRaw = Array.isArray(arenaJobsRaw) ? arenaJobsRaw[0] : null;
          const latestArena = latestArenaRaw ? normalizeArenaJob(latestArenaRaw) : null;
          setArenaResult(latestArena);
          if (latestArena) {
            setArenaProgress(latestArena.progress);
            const isActiveArena = ["queued", "running"].includes(latestArena.status.toLowerCase());
            setArenaRunning(isActiveArena);
            setArenaJobId(isActiveArena ? latestArena.id : null);
          } else {
            setArenaProgress(0);
            setArenaRunning(false);
            setArenaJobId(null);
          }
          const detail = isRecord(detailRaw) ? detailRaw : { run: detailRaw };
          const detailedRun = isRecord(detail.run) ? detail.run : {};
          const detailedConfig = isRecord(detailedRun.config) ? detailedRun.config : null;
          if (changedRun && detailedConfig) {
            setConfig((current) => configFromApi(detailedConfig, current));
            if (detailedConfig.preset === "quick") basePresetRef.current = "quick";
            else if (detailedConfig.preset === "m4_24h") basePresetRef.current = "m4_24h";
          }
          const combined: Record<string, unknown> = {
            ...detail,
            system: systemRaw,
            metrics: Array.isArray(metricsRaw) ? metricsRaw : [],
            models: Array.isArray(modelsRaw) ? modelsRaw : [],
            events: Array.isArray(eventsRaw) ? eventsRaw : [],
          };
          setSnapshot((current) => {
            const next = normalizeSnapshot(combined, current);
            metricsSeqRef.current = next.metrics.at(-1)?.seq ?? -1;
            return next;
          });
          activeRunIdRef.current = runId;
          setRemoteRunId(runId);
        }
        hasConnectedRef.current = true;
        setConnected(true);
        setLastSync(new Date());
      } catch {
        if (cancelled) return;
        setConnected(false);
        if (!hasConnectedRef.current) tickDemo();
      } finally {
        pollInFlight.current = false;
      }
    };

    poll();
    const interval = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!arenaRunning || (connected && arenaJobId)) return;
    const interval = window.setInterval(() => {
      setArenaProgress((current) => {
        const next = Math.min(100, current + 4 + Math.random() * 5);
        if (next >= 100) {
          setArenaRunning(false);
          showToast("Demo arena evaluation complete");
        }
        return next;
      });
    }, 500);
    return () => window.clearInterval(interval);
  }, [arenaRunning, connected, arenaJobId]);

  useEffect(() => {
    if (!connected || !arenaJobId || !arenaRunning) return;
    let cancelled = false;
    let inFlight = false;
    const pollArena = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const raw = await fetchJson(`/arena/${encodeURIComponent(arenaJobId)}`);
        if (cancelled || !isRecord(raw)) return;
        const normalized = normalizeArenaJob(raw);
        if (normalized) {
          setArenaResult(normalized);
          setArenaProgress(normalized.progress);
        }
        const status = asString(raw.status, "running").toLowerCase();
        if (["complete", "completed", "failed", "error"].includes(status)) {
          setArenaRunning(false);
          showToast(status.startsWith("complete") ? "Arena evaluation complete" : `Arena evaluation ${status}`);
        }
      } catch (error) {
        if (!cancelled) {
          setArenaRunning(false);
          showToast(`Arena polling stopped: ${error instanceof Error ? error.message : "unknown error"}`);
        }
      } finally {
        inFlight = false;
      }
    };
    pollArena();
    const interval = window.setInterval(pollArena, 1_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [connected, arenaJobId, arenaRunning, arenaPairs]);

  useEffect(() => {
    if (!snapshot.models.length) return;
    const champion = snapshot.models.find((model) => model.isChampion) ?? snapshot.models[0];
    const challenger = snapshot.models.find((model) => model.id !== champion.id);
    if (playModel !== "baseline" && !snapshot.models.some((model) => model.id === playModel)) setPlayModel(champion.id);
    if (!snapshot.models.some((model) => model.id === arenaA) && !arenaBaselines.some((model) => model.id === arenaA)) setArenaA(champion.id);
    if (!snapshot.models.some((model) => model.id === arenaB) && !arenaBaselines.some((model) => model.id === arenaB)) setArenaB(challenger?.id ?? "baseline:balanced");
    if (arenaA === arenaB) setArenaB(challenger?.id ?? "baseline:balanced");
  }, [snapshot.models, playModel, arenaA, arenaB]);

  useEffect(() => {
    if (!connected || !remoteGame?.id || remoteGame.status !== "model_thinking") return;
    let cancelled = false;
    let inFlight = false;
    const pollGame = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const raw = await fetchJson(`/games/${encodeURIComponent(remoteGame.id)}`);
        if (cancelled) return;
        const normalized = normalizeRemoteGame(raw, gameRef.current);
        if (!normalized) return;
        setRemoteGame(normalized.session);
        setGame(normalized.game);
      } catch (error) {
        if (!cancelled) {
          setRemoteGame((current) => current ? { ...current, status: "error", error: error instanceof Error ? error.message : "Game polling failed" } : current);
        }
      } finally {
        inFlight = false;
      }
    };
    pollGame();
    const interval = window.setInterval(pollGame, 700);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [connected, remoteGame?.id, remoteGame?.status]);

  const createConfiguredRun = async (start: boolean) => {
    const basePreset = config.preset === "quick" ? "quick" : config.preset === "m4_24h" ? "m4_24h" : basePresetRef.current;
    const overrides = configToApi(config);
    if (connected) {
      const result = await fetchJson("/runs", {
        method: "POST",
        body: JSON.stringify({
          preset: basePreset,
          name: config.name,
          overrides,
          start,
        }),
      });
      const item = isRecord(result) ? result : {};
      const runId = asString(item.id, "");
      if (runId) {
        activeRunIdRef.current = runId;
        metricsSeqRef.current = -1;
        setRemoteRunId(runId);
      }
      setSnapshot((current) => normalizeSnapshot({ run: result, system: current.hardware }, current));
      return;
    }
    setSnapshot((current) => ({
      ...current,
      run: {
        ...current.run,
        name: config.name,
        status: start ? "running" : "ready",
        phase: start ? "Initializing self-play" : "Recipe ready",
        durationSeconds: config.durationMinutes * 60,
        elapsedSeconds: start ? 0 : current.run.elapsedSeconds,
      },
    }));
  };

  const invokeControl = async (action: "start" | "pause" | "resume" | "stop" | "checkpoint") => {
    setCommandBusy(action);
    try {
      if (connected) {
        if (action === "start" && !remoteRunId) {
          await createConfiguredRun(true);
        } else if (remoteRunId) {
          const result = await fetchJson(`/runs/${encodeURIComponent(remoteRunId)}/${action}`, {
            method: "POST",
          });
          setSnapshot((current) => normalizeSnapshot({ run: result }, current));
        } else {
          throw new Error("Create or select a run first");
        }
      } else {
        setSnapshot((current) => {
          const nextStatus: RunStatus =
            action === "start" || action === "resume"
              ? "running"
              : action === "pause"
                ? "paused"
                : action === "stop"
                  ? "ready"
                  : current.run.status;
          const event: AuditEvent = {
            id: `local-${Date.now()}`,
            at: "now",
            kind: action === "stop" ? "warning" : "info",
            title: action === "checkpoint" ? "Manual checkpoint saved" : `${titleCase(action)} requested`,
            detail: connected ? "Trainer acknowledged" : "Demo session · connect the local trainer to persist this command",
          };
          return {
            ...current,
            run: { ...current.run, status: nextStatus, phase: nextStatus === "paused" ? "Paused at safe boundary" : current.run.phase },
            events: [event, ...current.events].slice(0, 20),
          };
        });
      }
      showToast(
        action === "checkpoint"
          ? "Checkpoint captured at a safe boundary"
          : `${titleCase(action)} command ${connected ? "accepted" : "previewed in demo mode"}`,
      );
    } catch (error) {
      showToast(`Trainer did not accept ${action}: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const choosePreset = (preset: TrainerConfig["preset"]) => {
    if (preset === "m4_24h") {
      basePresetRef.current = "m4_24h";
      setConfig(configFromApi(presetCatalogRef.current.m4_24h, initialConfig));
    } else if (preset === "quick") {
      basePresetRef.current = "quick";
      const fallbackQuick: TrainerConfig = {
        ...initialConfig,
        name: "Quick validation run",
        preset: "quick",
        durationMinutes: 5,
        actorProcesses: 4,
        gamesPerActorBatch: 2,
        hiddenSize: 128,
        residualBlocks: 2,
        batchSize: 256,
        replayCapacity: 25_000,
        replayWarmup: 500,
        heuristicBootstrapUpdates: 256,
        checkpointEveryGames: 1_000,
        evaluateEveryGames: 2_000,
        evaluationPairs: 16,
      };
      setConfig(configFromApi(presetCatalogRef.current.quick, fallbackQuick));
    } else {
      setConfig((current) => ({ ...current, preset: "custom" }));
    }
  };

  const updateConfig = <K extends keyof TrainerConfig>(key: K, value: TrainerConfig[K]) => {
    setConfig((current) => {
      const next = { ...current, [key]: value } as TrainerConfig;
      if (key !== "preset") next.preset = "custom";
      return next;
    });
  };

  const saveConfig = async (event: FormEvent) => {
    event.preventDefault();
    setCommandBusy("config");
    try {
      await createConfiguredRun(false);
      showToast(connected ? "Run created and ready to launch" : "Configuration validated in demo mode");
    } catch (error) {
      showToast(`Configuration rejected: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const saveLiveConfig = async (event: FormEvent) => {
    event.preventDefault();
    setCommandBusy("config");
    const changes = {
      duration_minutes: config.durationMinutes,
      epsilon_end: config.epsilonEnd,
      current_selfplay_fraction: config.currentSelfplayFraction,
      league_fraction: config.leagueFraction,
      baseline_fraction: config.baselineFraction,
      evaluation_pairs: config.evaluationPairs,
      evaluate_every_games: config.evaluateEveryGames,
      checkpoint_every_games: config.checkpointEveryGames,
      promotion_confidence: config.promotionConfidence,
      promotion_margin: config.promotionMargin,
    };
    try {
      if (connected) {
        if (!remoteRunId) throw new Error("No active run to update");
        const result = await fetchJson(`/runs/${encodeURIComponent(remoteRunId)}/config`, {
          method: "PATCH",
          body: JSON.stringify({ changes }),
        });
        setSnapshot((current) => normalizeSnapshot({ run: result }, current));
      }
      showToast(connected ? "Configuration queued for the next safe boundary" : "Live update previewed in demo mode");
    } catch (error) {
      showToast(`Configuration rejected: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const launchConfiguredRun = async () => {
    setCommandBusy("start");
    try {
      await createConfiguredRun(true);
      showToast(connected ? "New training run launched" : "Training launch previewed in demo mode");
    } catch (error) {
      showToast(`Run could not launch: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const runArena = async () => {
    if (arenaA === arenaB) {
      showToast("Choose two different models for a useful arena match");
      return;
    }
    setArenaProgress(3);
    setArenaRunning(true);
    setArenaJobId(null);
    if (connected) setArenaResult(null);
    if (connected) {
      try {
        const result = await fetchJson("/arena", {
          method: "POST",
          body: JSON.stringify({ model_a: arenaA, model_b: arenaB, pairs: arenaPairs }),
        });
        if (isRecord(result)) {
          setArenaJobId(asString(result.id ?? result.job_id, "") || null);
          const normalized = normalizeArenaJob(result);
          if (normalized) setArenaResult(normalized);
        }
      } catch (error) {
        setArenaRunning(false);
        showToast(`Arena job could not start: ${error instanceof Error ? error.message : "unknown error"}`);
      }
    }
  };

  const refreshModels = async () => {
    if (!connected) {
      showToast("Demo registry is already current");
      return;
    }
    setCommandBusy("models-refresh");
    try {
      const raw = await fetchJson(`/models${remoteRunId ? `?run_id=${encodeURIComponent(remoteRunId)}` : ""}`);
      const models = Array.isArray(raw)
        ? raw.map((model) => {
            const modelId = isRecord(model) ? asString(model.id, "") : "";
            return normalizeModel(model, snapshot.models.find((item) => item.id === modelId));
          })
        : [];
      setSnapshot((current) => ({ ...current, models }));
      showToast(`Registry refreshed · ${models.length} checkpoint${models.length === 1 ? "" : "s"}`);
    } catch (error) {
      showToast(`Registry refresh failed: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const togglePinned = async (modelId: string) => {
    const model = snapshot.models.find((item) => item.id === modelId);
    if (!model) return;
    const pinned = !model.isPinned;
    setCommandBusy(`pin-${modelId}`);
    try {
      if (connected) {
        const raw = await fetchJson(`/models/${encodeURIComponent(modelId)}`, {
          method: "PATCH",
          body: JSON.stringify({ pinned }),
        });
        setSnapshot((current) => ({
          ...current,
          models: current.models.map((item) => item.id === modelId ? normalizeModel(raw, item) : item),
        }));
      } else {
        setSnapshot((current) => ({
          ...current,
          models: current.models.map((item) => item.id === modelId ? { ...item, isPinned: pinned } : item),
        }));
      }
      showToast(`${model.label} ${pinned ? "pinned" : "unpinned"}${connected ? "" : " in demo mode"}`);
    } catch (error) {
      showToast(`Pin update failed: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const exportModel = async (modelId: string) => {
    if (!connected) {
      showToast("Actor downloads are available when the local service is connected");
      return;
    }
    setCommandBusy(`export-${modelId}`);
    try {
      const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}/actor`, { cache: "no-store" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `astrosynapse2-${modelId}.actor.npz`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      showToast("Actor snapshot downloaded");
    } catch (error) {
      showToast(`Actor download failed: ${error instanceof Error ? error.message : "unknown error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const exportAudit = () => {
    const blob = new Blob([JSON.stringify({ run_id: remoteRunId, events: snapshot.events }, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `astrosynapse2-${remoteRunId ?? "demo"}-audit.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    showToast("Audit JSON downloaded");
  };

  const newGame = async () => {
    setGame({ ...initialGame, log: [`New game · ${humanStarts ? "You" : "Orion"} move first`] });
    setSelectedCard(null);
    setRemoteGame(null);
    if (connected) {
      setCommandBusy("game-new");
      try {
        const modelId = snapshot.models.some((model) => model.id === playModel) ? playModel : null;
        const result = await fetchJson("/games", {
          method: "POST",
          body: JSON.stringify({ model_id: modelId, human_starts: humanStarts }),
        });
        const normalized = normalizeRemoteGame(result, initialGame);
        if (!normalized) throw new Error("The game service returned an invalid session");
        setRemoteGame(normalized.session);
        setGame(normalized.game);
        showToast("New game started");
      } catch (error) {
        showToast(`Game could not start: ${error instanceof Error ? error.message : "trainer unavailable"}`);
      } finally {
        setCommandBusy(null);
      }
      return;
    }
    showToast("New preview game started");
  };

  const submitRemoteChoice = async (actionId: number) => {
    if (!remoteGame) return;
    setCommandBusy("game-choice");
    try {
      const result = await fetchJson(`/games/${encodeURIComponent(remoteGame.id)}/choice`, {
        method: "POST",
        body: JSON.stringify({ action_id: actionId }),
      });
      const normalized = normalizeRemoteGame(result, game);
      if (!normalized) throw new Error("The game service returned an invalid position");
      setRemoteGame(normalized.session);
      setGame(normalized.game);
      setSelectedCard(null);
      if (normalized.session.result) showToast(normalized.session.result);
    } catch (error) {
      showToast(`Choice was not accepted: ${error instanceof Error ? error.message : "unknown game error"}`);
    } finally {
      setCommandBusy(null);
    }
  };

  const playHandCard = (card: GameCard) => {
    setGame((current) => ({
      ...current,
      trade: current.trade + (card.trade ?? 0),
      attack: current.attack + (card.attack ?? 0),
      humanAuthority: current.humanAuthority + (card.authority ?? 0),
      hand: current.hand.filter((item) => item.id !== card.id),
      log: [`You played ${card.name}${card.trade ? ` · +${card.trade} trade` : ""}${card.attack ? ` · +${card.attack} combat` : ""}`, ...current.log],
    }));
    setSelectedCard(null);
  };

  const acquireSelected = () => {
    const card = game.market.find((item) => item.id === selectedCard);
    if (!card || card.cost > game.trade) return;
    setGame((current) => ({
      ...current,
      trade: current.trade - card.cost,
      discardCount: current.discardCount + 1,
      market: current.market.filter((item) => item.id !== card.id),
      log: [`You acquired ${card.name} for ${card.cost} trade`, ...current.log],
    }));
    setSelectedCard(null);
  };

  const attackOpponent = () => {
    if (game.attack <= 0) return;
    setGame((current) => ({
      ...current,
      opponentAuthority: Math.max(0, current.opponentAuthority - current.attack),
      attack: 0,
      log: [`You dealt ${current.attack} damage to Orion`, ...current.log],
    }));
  };

  const endTurn = () => {
    setGame((current) => ({
      ...initialGame,
      turn: current.turn + 1,
      humanAuthority: current.humanAuthority,
      opponentAuthority: Math.max(0, current.opponentAuthority - 3),
      log: ["Orion completed its turn · 3 combat", "You drew 5 cards", ...current.log].slice(0, 18),
    }));
    setSelectedCard(null);
  };

  const phaseSteps = [
    { id: "collect", label: "Collect", detail: `${snapshot.hardware.actorProcesses} actors`, active: snapshot.run.phase.toLowerCase().includes("play") || snapshot.run.phase.toLowerCase().includes("collect") },
    { id: "learn", label: "Learn", detail: snapshot.hardware.learnerDevice, active: snapshot.run.phase.toLowerCase().includes("learn") },
    { id: "evaluate", label: "Evaluate", detail: `${numberFormatter.format(config.evaluationPairs)} pairs`, active: snapshot.run.phase.toLowerCase().includes("eval") },
  ];

  return (
    <main className="control-center">
      <div className="star-field" aria-hidden="true" />
      <header className="topbar">
        <button type="button" className="brand" onClick={() => setActiveTab("overview")} aria-label="Astrosynapse 2 overview">
          <span className="brand-mark"><i /><i /><i /></span>
          <span>
            <strong>Astrosynapse</strong>
            <small>Self-play control center</small>
          </span>
          <b>2</b>
        </button>

        <div className="topbar-context">
          <span className="run-monogram">M4</span>
          <span>
            <small>Active run</small>
            <strong>{snapshot.run.name}</strong>
          </span>
        </div>

        <div className="connection-cluster">
          <span className={`connection-pill ${connected ? "is-live" : "is-demo"}`}>
            <span className="connection-pulse" />
            {connected ? "Trainer live" : "Demo preview"}
          </span>
          <span className="sync-copy">
            {connected && lastSync ? `synced ${lastSync.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}` : "127.0.0.1:8765 offline"}
          </span>
        </div>
      </header>

      <nav className="primary-tabs" aria-label="Control center sections">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={activeTab === tab.id ? "is-active" : ""}
            onClick={() => setActiveTab(tab.id)}
            aria-current={activeTab === tab.id ? "page" : undefined}
          >
            <span>{tab.short}</span>
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="workspace">
        {activeTab === "overview" ? (
          <section className="tab-panel overview-panel" aria-labelledby="overview-title">
            <div className="mission-hero">
              <div className="mission-copy">
                <div className="eyebrow-row">
                  <span className={`run-state status-${snapshot.run.status}`}>
                    <StatusDot status={snapshot.run.status} />
                    {titleCase(snapshot.run.status)}
                  </span>
                  <span>{snapshot.run.id ? `Run ${snapshot.run.id.slice(0, 12)}` : "Run not created"}</span>
                  <span><Jargon term="seed" side="bottom">Seed</Jargon> {snapshot.run.id ? snapshot.run.seed : "—"}</span>
                </div>
                <h1 id="overview-title">Training the next champion.</h1>
                <p>
                  High-throughput <Jargon term="selfPlay">self-play</Jargon> on Apple silicon, evaluated with <Jargon term="pairedSeeds">paired seeds</Jargon> and confidence-aware promotion.
                </p>

                <div className="control-strip" aria-label="Training controls">
                  <button
                    type="button"
                    className="button button-primary"
                    onClick={() => invokeControl("start")}
                    disabled={!(["ready", "complete", "error", "interrupted"] as RunStatus[]).includes(snapshot.run.status) || commandBusy !== null}
                  >
                    <span className="button-symbol" aria-hidden="true">▶</span> Start
                  </button>
                  <button
                    type="button"
                    className="button"
                    onClick={() => invokeControl("pause")}
                    disabled={snapshot.run.status !== "running" || commandBusy !== null}
                  >
                    <span className="button-symbol" aria-hidden="true">Ⅱ</span> Pause
                  </button>
                  <button
                    type="button"
                    className="button"
                    onClick={() => invokeControl("resume")}
                    disabled={snapshot.run.status !== "paused" || commandBusy !== null}
                  >
                    <span className="button-symbol" aria-hidden="true">↗</span> Resume
                  </button>
                  <button
                    type="button"
                    className="button button-danger"
                    onClick={() => invokeControl("stop")}
                    disabled={!(["running", "pausing", "paused"] as RunStatus[]).includes(snapshot.run.status) || commandBusy !== null}
                  >
                    <span className="button-symbol" aria-hidden="true">■</span> Stop
                  </button>
                  <button
                    type="button"
                    className="button button-quiet"
                    onClick={() => invokeControl("checkpoint")}
                    disabled={!(["running", "paused"] as RunStatus[]).includes(snapshot.run.status) || commandBusy !== null}
                  >
                    <span className="button-symbol" aria-hidden="true">◇</span> Checkpoint
                  </button>
                </div>

                <div className="phase-ribbon">
                  <div className="phase-orbit" aria-hidden="true"><span /></div>
                  <div>
                    <small>Current phase</small>
                    <strong>{snapshot.run.phase}</strong>
                  </div>
                  <div className="phase-steps" aria-label="Training pipeline">
                    {phaseSteps.map((step, index) => (
                      <div key={step.id} className={step.active ? "is-active" : ""}>
                        <span>{String(index + 1).padStart(2, "0")}</span>
                        <p><strong>{step.label}</strong><small>{step.detail}</small></p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <aside className="mission-clock" aria-label="Run time budget">
                <div className="clock-topline"><span>Time budget</span><b>{Math.round(progress * 100)}%</b></div>
                <div className="clock-display">
                  <Gauge value={progress} label="elapsed" />
                  <div>
                    <small>Elapsed</small>
                    <strong>{formatDuration(snapshot.run.elapsedSeconds)}</strong>
                    <span>of {formatDuration(snapshot.run.durationSeconds, true)}</span>
                  </div>
                </div>
                <div className="remaining-block">
                  <span>Estimated remaining</span>
                  <strong>{formatDuration(remaining, true)}</strong>
                </div>
                <div className="mission-progress"><i style={{ width: `${progress * 100}%` }} /></div>
                <p>At current throughput, approximately <b>{compactFormatter.format(latestMetric.gamesPerSecond * remaining)}</b> more games.</p>
              </aside>

              <div className="hardware-rail">
                <div><small>Machine</small><strong>{snapshot.hardware.machine}</strong><span>{snapshot.hardware.chip}</span></div>
                <div><small>Learner</small><strong>{snapshot.hardware.backend}</strong><span>{snapshot.hardware.metalAvailable === null ? "awaiting live Metal telemetry" : snapshot.hardware.metalAvailable ? `${snapshot.hardware.learnerDevice} · ${snapshot.hardware.metalActiveGb.toFixed(2)} GB active` : "Metal unavailable"}</span></div>
                <div><small>Simulation</small><strong>{snapshot.hardware.actorProcesses} actor processes</strong><span>{latestMetric.gamesPerSecond.toFixed(0)} games / second</span></div>
                <div><small>Unified memory</small><strong>{snapshot.hardware.memoryUsedGb.toFixed(1)} / {snapshot.hardware.memoryTotalGb} GB</strong><span>{snapshot.hardware.cpuPercent.toFixed(0)}% CPU load</span></div>
              </div>
            </div>

            {snapshot.run.lastError ? (
              <div className="offline-note run-error-note" role="alert">
                <span aria-hidden="true">!</span>
                <p><strong>Training stopped with an error.</strong> {snapshot.run.lastError}</p>
              </div>
            ) : null}

            {!connected ? (
              <div className="offline-note" role="status">
                <span aria-hidden="true">⌁</span>
                <p><strong>The local trainer is not answering yet.</strong> You are seeing a fully interactive preview; live values will replace it automatically when <code>{API_BASE}</code> comes online.</p>
                <button type="button" onClick={() => setActiveTab("diagnostics")}>Connection details</button>
              </div>
            ) : null}

            <div className="overview-grid">
              <article className="panel performance-panel">
                <header className="panel-header">
                  <div><span className="panel-kicker">Velocity</span><h2><Jargon term="selfPlay">Self-play</Jargon> throughput</h2></div>
                  <div className="chart-legend"><span className="legend-mint">Games/s</span><span className="legend-blue">Decisions/s ÷ 240</span></div>
                </header>
                <div className="metric-headline">
                  <div><strong>{latestMetric.gamesPerSecond.toFixed(1)}</strong><span>games / sec</span></div>
                  <div><strong>{compactFormatter.format(latestMetric.decisionsPerSecond)}</strong><span>decisions / sec</span></div>
                  <div><strong>{compactFormatter.format(snapshot.run.games)}</strong><span>games total</span></div>
                </div>
                <MetricCanvas points={snapshot.metrics} mode="throughput" />
                <footer className="chart-axis"><span>Earlier</span><span>Last 72 samples</span><span>Now</span></footer>
              </article>

              <article className="panel quality-panel">
                <header className="panel-header">
                  <div><span className="panel-kicker">Signal through noise</span><h2><Jargon term="heldOutStrength">Held-out strength</Jargon></h2></div>
                  <span className="confidence-chip"><Jargon term="confidenceInterval">95% confidence</Jargon></span>
                </header>
                <div className="quality-score">
                  <strong>{arenaHasSamples || latestMetric.hasEvaluation ? formatPercent(heldOutScore) : "Not evaluated"}</strong>
                  <span>{arenaResult ? `${arenaResult.modelALabel} vs ${arenaResult.modelBLabel}` : "paired arena required"}</span>
                  <small>{arenaHasSamples || latestMetric.hasEvaluation ? `${formatPercent(heldOutLow)} – ${formatPercent(heldOutHigh)}` : "No confidence interval yet"}</small>
                </div>
                {evaluatedMetricPoints.length > 1 ? <MetricCanvas points={evaluatedMetricPoints} mode="quality" height={178} /> : <EmptyState title="Awaiting paired evidence" detail="Run a 5,000-pair arena to estimate strength through game variance." />}
                <div className="quality-footer">
                  <span><i className="diamond" /> {(arenaResult?.modelALabel ?? snapshot.run.championId.replaceAll("-", " ")) || "No champion yet"}</span>
                  <span>{arenaHasSamples ? <><Jargon term="elo">Elo equivalent</Jargon> {arenaResult!.elo >= 0 ? "+" : ""}{arenaResult!.elo.toFixed(0)}</> : "Not evaluated"}</span>
                </div>
              </article>

              <article className="panel live-stats-panel">
                <header className="panel-header"><div><span className="panel-kicker">Live instruments</span><h2>Learning pulse</h2></div></header>
                <div className="instrument-list">
                  <div><span><Jargon term="replayBuffer">Replay buffer</Jargon></span><strong>{formatPercent(latestMetric.replayFill, 0)}</strong><i><b style={{ width: `${latestMetric.replayFill * 100}%` }} /></i><small>{compactFormatter.format(config.replayCapacity * latestMetric.replayFill)} / {compactFormatter.format(config.replayCapacity)}</small></div>
                  <div><span><Jargon term="outcomeBce">Outcome BCE</Jargon></span><strong>{latestMetric.hasLearnerDiagnostics ? latestMetric.outcomeLoss.toFixed(3) : "—"}</strong><i><b style={{ width: `${latestMetric.hasLearnerDiagnostics ? Math.min(100, latestMetric.outcomeLoss * 100) : 0}%` }} /></i><small>{latestMetric.hasLearnerDiagnostics ? "prediction error · lower is better" : "replay warming up"}</small></div>
                  <div><span><Jargon term="brierScore">Brier score</Jargon></span><strong>{latestMetric.hasLearnerDiagnostics ? latestMetric.brier.toFixed(3) : "—"}</strong><i><b style={{ width: `${latestMetric.hasLearnerDiagnostics ? Math.min(100, latestMetric.brier * 300) : 0}%` }} /></i><small>{latestMetric.hasLearnerDiagnostics ? "probability error · lower is better" : "no learner update yet"}</small></div>
                  <div><span><Jargon term="uncertainty">Uncertainty</Jargon></span><strong>{latestMetric.hasLearnerDiagnostics ? latestMetric.uncertainty.toFixed(3) : "—"}</strong><i><b style={{ width: `${latestMetric.hasLearnerDiagnostics ? Math.min(100, latestMetric.uncertainty * 500) : 0}%` }} /></i><small>{latestMetric.hasLearnerDiagnostics ? "prediction-head disagreement" : "no learner update yet"}</small></div>
                  <div><span><Jargon term="safetyTruncations">Safety truncations</Jargon></span><strong>{formatPercent(latestMetric.truncationRate)}</strong><i><b style={{ width: `${Math.min(100, latestMetric.truncationRate * 500)}%` }} /></i><small>{latestMetric.meanTurns ? `${latestMetric.meanTurns.toFixed(1)} mean turns` : "awaiting completed games"}</small></div>
                </div>
              </article>

              <article className="panel event-panel">
                <header className="panel-header">
                  <div><span className="panel-kicker">Chronicle</span><h2>Recent events</h2></div>
                  <button type="button" className="text-button" onClick={() => setActiveTab("diagnostics")}>Full audit →</button>
                </header>
                <div className="event-list">
                  {snapshot.events.slice(0, 4).map((event) => (
                    <div className={`event event-${event.kind}`} key={event.id}>
                      <span className="event-mark" />
                      <div><strong>{event.title}</strong><p>{event.detail}</p></div>
                      <time>{event.at}</time>
                    </div>
                  ))}
                  {!snapshot.events.length ? <EmptyState title="Chronicle is quiet" detail="Run lifecycle and promotion events will appear here." /> : null}
                </div>
              </article>
            </div>
          </section>
        ) : null}

        {activeTab === "train" ? (
          <section className="tab-panel train-panel" aria-labelledby="train-title">
            <header className="section-heading">
              <div><span className="section-number">02 / TRAIN</span><h1 id="train-title">Shape the training mission.</h1><p>Start from an M4-calibrated recipe, then expose only the controls worth changing.</p></div>
              <div className="section-summary"><span>Projected games</span><strong>{compactFormatter.format(latestMetric.gamesPerSecond * config.durationMinutes * 60)}</strong><small>at {latestMetric.gamesPerSecond.toFixed(0)} games/s</small></div>
            </header>

            <div className="train-layout">
              <form className="recipe-form panel" onSubmit={saveConfig}>
                <div className="recipe-title"><div><span className="panel-kicker">Run recipe</span><h2>Choose a flight plan</h2></div><span className="recommended-label">M4 · 16 GB tuned</span></div>
                <div className="preset-grid" role="radiogroup" aria-label="Training preset">
                  <button type="button" role="radio" aria-checked={config.preset === "m4_24h"} className={config.preset === "m4_24h" ? "is-selected" : ""} onClick={() => choosePreset("m4_24h")}>
                    <span>Recommended</span><strong>24-hour champion</strong><p>Maximum useful self-play without exceeding unified memory.</p><small>8 actors · 192 hidden · 900k replay</small>
                  </button>
                  <button type="button" role="radio" aria-checked={config.preset === "quick"} className={config.preset === "quick" ? "is-selected" : ""} onClick={() => choosePreset("quick")}>
                    <span>Validation</span><strong>5-minute systems check</strong><p>Confirms engine, learner, checkpoints, and dashboard wiring.</p><small>4 actors · 128 hidden · 25k replay</small>
                  </button>
                  <button type="button" role="radio" aria-checked={config.preset === "custom"} className={config.preset === "custom" ? "is-selected" : ""} onClick={() => choosePreset("custom")}>
                    <span>Expert</span><strong>Custom mission</strong><p>Tune the core architecture, replay, league, and evaluation controls.</p><small>Guardrails remain active</small>
                  </button>
                </div>

                <div className="field-grid primary-fields">
                  <label><span>Run name</span><input value={config.name} onChange={(event) => updateConfig("name", event.target.value)} /></label>
                  <label><span>Time budget</span><div className="input-suffix"><input type="number" min="1" value={config.durationMinutes} onChange={(event) => updateConfig("durationMinutes", Number(event.target.value))} /><b>min</b></div></label>
                  <label><span><Jargon term="actors">Actor processes</Jargon></span><input type="number" min="1" max="16" value={config.actorProcesses} onChange={(event) => updateConfig("actorProcesses", Number(event.target.value))} /></label>
                  <label><span><Jargon term="evaluationPairs">Evaluation pairs</Jargon></span><input type="number" min="8" max="20000" value={config.evaluationPairs} onChange={(event) => updateConfig("evaluationPairs", Number(event.target.value))} /></label>
                </div>

                <button type="button" className="advanced-toggle" onClick={() => setAdvancedOpen((open) => !open)} aria-expanded={advancedOpen}>
                  <span><b>{advancedOpen ? "−" : "+"}</b> Advanced architecture & learning settings</span><small>{advancedOpen ? "Hide expert controls" : "22 validated fields"}</small>
                </button>

                {advancedOpen ? (
                  <div className="advanced-fields">
                    <div className="field-section"><h3>Network & <Jargon term="learner">learner</Jargon></h3><div className="field-grid">
                      <label><span><Jargon term="hiddenSize">Hidden size</Jargon></span><input type="number" value={config.hiddenSize} onChange={(event) => updateConfig("hiddenSize", Number(event.target.value))} /></label>
                      <label><span><Jargon term="residualBlocks">Residual blocks</Jargon></span><input type="number" value={config.residualBlocks} onChange={(event) => updateConfig("residualBlocks", Number(event.target.value))} /></label>
                      <label><span><Jargon term="bootstrapHeads">Bootstrap heads</Jargon></span><input type="number" value={config.bootstrapHeads} onChange={(event) => updateConfig("bootstrapHeads", Number(event.target.value))} /></label>
                      <label><span><Jargon term="batchSize">Batch size</Jargon></span><input type="number" value={config.batchSize} onChange={(event) => updateConfig("batchSize", Number(event.target.value))} /></label>
                      <label><span><Jargon term="learningRate">Learning rate</Jargon></span><input type="number" step="0.00001" value={config.learningRate} onChange={(event) => updateConfig("learningRate", Number(event.target.value))} /></label>
                      <label><span><Jargon term="actors">Games / actor batch</Jargon></span><input type="number" value={config.gamesPerActorBatch} onChange={(event) => updateConfig("gamesPerActorBatch", Number(event.target.value))} /></label>
                    </div></div>
                    <div className="field-section"><h3>Replay & exploration</h3><div className="field-grid">
                      <label><span><Jargon term="replayCapacity">Replay capacity</Jargon></span><input type="number" value={config.replayCapacity} onChange={(event) => updateConfig("replayCapacity", Number(event.target.value))} /></label>
                      <label><span><Jargon term="replayWarmup">Replay warmup</Jargon></span><input type="number" value={config.replayWarmup} onChange={(event) => updateConfig("replayWarmup", Number(event.target.value))} /></label>
                      <label><span><Jargon term="epsilon">Epsilon start</Jargon></span><input type="number" step="0.005" value={config.epsilonStart} onChange={(event) => updateConfig("epsilonStart", Number(event.target.value))} /></label>
                      <label><span><Jargon term="epsilon">Epsilon end</Jargon></span><input type="number" step="0.005" value={config.epsilonEnd} onChange={(event) => updateConfig("epsilonEnd", Number(event.target.value))} /></label>
                      <label><span><Jargon term="checkpoint">Checkpoint games</Jargon></span><input type="number" value={config.checkpointEveryGames} onChange={(event) => updateConfig("checkpointEveryGames", Number(event.target.value))} /></label>
                      <label><span>Evaluation games</span><input type="number" value={config.evaluateEveryGames} onChange={(event) => updateConfig("evaluateEveryGames", Number(event.target.value))} /></label>
                    </div></div>
                    <div className="field-section"><h3>Curriculum, league & promotion</h3><div className="field-grid">
                      <label><span><Jargon term="bootstrapUpdates">Bootstrap updates</Jargon></span><input type="number" min="0" value={config.heuristicBootstrapUpdates} onChange={(event) => updateConfig("heuristicBootstrapUpdates", Number(event.target.value))} /></label>
                      <label><span><Jargon term="selfPlay">Current self-play</Jargon></span><input type="number" min="0" max="1" step="0.01" value={config.currentSelfplayFraction} onChange={(event) => updateConfig("currentSelfplayFraction", Number(event.target.value))} /></label>
                      <label><span><Jargon term="league">League opponents</Jargon></span><input type="number" min="0" max="1" step="0.01" value={config.leagueFraction} onChange={(event) => updateConfig("leagueFraction", Number(event.target.value))} /></label>
                      <label><span><Jargon term="baseline">Baseline anchors</Jargon></span><input type="number" min="0" max="1" step="0.01" value={config.baselineFraction} onChange={(event) => updateConfig("baselineFraction", Number(event.target.value))} /></label>
                      <label><span><Jargon term="promotionConfidence">Promotion confidence</Jargon></span><input type="number" min="0.8" max="0.999" step="0.01" value={config.promotionConfidence} onChange={(event) => updateConfig("promotionConfidence", Number(event.target.value))} /></label>
                      <label><span><Jargon term="promotionMargin">Promotion margin</Jargon></span><input type="number" min="0" max="0.25" step="0.005" value={config.promotionMargin} onChange={(event) => updateConfig("promotionMargin", Number(event.target.value))} /></label>
                    </div></div>
                  </div>
                ) : null}

                <div className="form-actions">
                  <div><span className="validation-light" /> Server validates recipe bounds and schedule consistency on create</div>
                  <button type="submit" className="button" disabled={commandBusy !== null}>{commandBusy === "config" ? "Creating…" : "Create ready run"}</button>
                  <button type="button" className="button button-primary" onClick={launchConfiguredRun} disabled={(["running", "pausing", "paused", "stopping"] as RunStatus[]).includes(snapshot.run.status) || commandBusy !== null}>Launch new run</button>
                </div>
              </form>

              <aside className="train-sidebar">
                <article className="panel pipeline-panel">
                  <header className="panel-header"><div><span className="panel-kicker">Live pipeline</span><h2>One iteration</h2></div><span className={`run-state status-${snapshot.run.status}`}><StatusDot status={snapshot.run.status} /> {titleCase(snapshot.run.status)}</span></header>
                  <div className="pipeline-map">
                    {phaseSteps.map((step, index) => (
                      <div key={step.id} className={step.active ? "is-active" : ""}><b>{String(index + 1).padStart(2, "0")}</b><span><strong>{step.label}</strong><small>{step.detail}</small></span><i /></div>
                    ))}
                  </div>
                  <dl className="compact-dl"><div><dt>Iteration</dt><dd>{numberFormatter.format(snapshot.run.updates)}</dd></div><div><dt>Batch</dt><dd>{config.batchSize}</dd></div><div><dt>Replay</dt><dd>{formatPercent(latestMetric.replayFill, 0)}</dd></div><div><dt>Next evaluation</dt><dd>{compactFormatter.format(Math.max(0, config.evaluateEveryGames - (snapshot.run.games % config.evaluateEveryGames)))} games</dd></div></dl>
                </article>
                <article className="panel resource-panel"><header className="panel-header"><div><span className="panel-kicker">Resource envelope</span><h2>M4 headroom</h2></div></header><div className="resource-rings"><Gauge value={snapshot.hardware.cpuPercent / 100} label="CPU" /><Gauge value={snapshot.hardware.memoryTotalGb ? snapshot.hardware.memoryUsedGb / snapshot.hardware.memoryTotalGb : 0} label="memory" /></div><p>Metal learning and CPU actors share unified memory. Watch this live envelope when tuning actor count, batch size, and replay capacity.</p></article>
                <article className="panel estimate-panel"><span className="panel-kicker">Mission estimate</span><strong>{compactFormatter.format(latestMetric.gamesPerSecond * config.durationMinutes * 60)} games</strong><p>{compactFormatter.format(latestMetric.decisionsPerSecond * config.durationMinutes * 60)} decisions · approximately {Math.round((config.durationMinutes * 60) / (config.evaluateEveryGames / latestMetric.gamesPerSecond))} evaluation gates</p></article>
              </aside>
            </div>
          </section>
        ) : null}

        {activeTab === "models" ? (
          <section className="tab-panel models-panel" aria-labelledby="models-title">
            <header className="section-heading">
              <div><span className="section-number">03 / MODELS & ARENA</span><h1 id="models-title">Prove strength, don’t infer it.</h1><p>Every result uses <Jargon term="pairedSeeds">paired seeds</Jargon>, reversed seats, and a <Jargon term="confidenceInterval">confidence interval</Jargon>.</p></div>
              <div className="section-summary"><span>Current <Jargon term="champion">champion</Jargon></span><strong>{snapshot.models.find((model) => model.isChampion)?.label ?? "—"}</strong><small>{snapshot.models.find((model) => model.isChampion)?.evaluated ? <>{formatPercent(snapshot.models.find((model) => model.isChampion)!.score)} <Jargon term="heldOutStrength">held-out score</Jargon></> : "Not evaluated"}</small></div>
            </header>

            <div className="arena-layout">
              <article className="panel arena-console">
                <header className="panel-header"><div><span className="panel-kicker">Head-to-head laboratory</span><h2>Arena match</h2></div><span className="paired-chip"><Jargon term="pairedSeeds">Paired randomness</Jargon></span></header>
                <div className="versus-row">
                  <label><span>Model A</span><select value={snapshot.models.some((model) => model.id === arenaA) || arenaBaselines.some((model) => model.id === arenaA) ? arenaA : "baseline:balanced"} onChange={(event) => setArenaA(event.target.value)}>{snapshot.models.length ? <optgroup label="Checkpoints">{snapshot.models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</optgroup> : null}<optgroup label="Reference baselines">{arenaBaselines.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</optgroup></select><small>candidate / challenger</small></label>
                  <div className="versus-mark"><span>VS</span><i /></div>
                  <label><span>Model B</span><select value={snapshot.models.some((model) => model.id === arenaB) || arenaBaselines.some((model) => model.id === arenaB) ? arenaB : "baseline:balanced"} onChange={(event) => setArenaB(event.target.value)}>{snapshot.models.length ? <optgroup label="Checkpoints">{snapshot.models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</optgroup> : null}<optgroup label="Reference baselines">{arenaBaselines.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</optgroup></select><small>champion / baseline</small></label>
                </div>
                <div className="arena-settings"><label><span><Jargon term="pairedSeeds">Seed pairs</Jargon></span><input type="number" min="1" max="20000" value={arenaPairs} onChange={(event) => setArenaPairs(Math.min(20_000, Math.max(1, Number(event.target.value) || 1)))} /></label><div><span>Games</span><strong>{numberFormatter.format(arenaPairs * 2)}</strong></div><div><span><Jargon term="confidenceInterval">Interval confidence</Jargon></span><strong>95%</strong></div><button type="button" className="button button-primary" onClick={runArena} disabled={arenaRunning || snapshot.models.length < 1 || arenaA === arenaB}>{arenaRunning ? "Evaluating…" : snapshot.models.length < 1 ? "Need a checkpoint" : arenaA === arenaB ? "Choose two rivals" : "Run arena"}</button></div>
                {arenaRunning || arenaProgress > 0 ? <div className="arena-progress" aria-live="polite"><div><span>Evaluation progress</span><strong>{Math.round(arenaProgress)}%</strong></div><i><b style={{ width: `${arenaProgress}%` }} /></i><p>{arenaRunning ? `${numberFormatter.format((arenaResult?.pairsCompleted ?? Math.round(arenaPairs * arenaProgress / 100)) * 2)} of ${numberFormatter.format((arenaResult?.pairsRequested ?? arenaPairs) * 2)} games · exact seats reversed` : "Complete · paired result persisted with both checkpoints"}</p></div> : null}
                {arenaResult ? <div className="arena-result">
                  <div className="result-score"><small>{arenaResult.status === "complete" ? "Latest result" : titleCase(arenaResult.status)}</small><strong>{arenaResult.pairsCompleted ? formatPercent(arenaResult.score) : "Pending"}</strong><span>{arenaResult.modelALabel}</span></div>
                  {arenaResult.pairsCompleted ? <><div className="interval-track"><i className="threshold" /><span style={{ left: `${arenaResult.ciLow * 100}%`, width: `${Math.max(0, arenaResult.ciHigh - arenaResult.ciLow) * 100}%` }} /><b style={{ left: `${arenaResult.score * 100}%` }} /></div><div className="interval-labels"><span>50% tie</span><strong><Jargon term="confidenceInterval">95% CI</Jargon> {formatPercent(arenaResult.ciLow)}–{formatPercent(arenaResult.ciHigh)}</strong><span>{arenaResult.elo >= 0 ? "+" : ""}{arenaResult.elo.toFixed(0)} <Jargon term="elo">Elo</Jargon></span></div></> : <div className="arena-pending">Waiting for the first paired games…</div>}
                  <p className="arena-recommendation">{arenaResult.recommendation} · {numberFormatter.format(arenaResult.pairsCompleted)} / {numberFormatter.format(arenaResult.pairsRequested)} pairs · {numberFormatter.format(arenaResult.gamesCompleted)} games · seat A {formatPercent(arenaResult.firstSeatScore)} / seat B {formatPercent(arenaResult.secondSeatScore)} · {numberFormatter.format(arenaResult.truncatedGames)} truncations · {titleCase(arenaResult.intervalMethod)}</p>
                </div> : <EmptyState title={snapshot.models.length < 1 ? "Arena opens after the first checkpoint" : "No arena evidence yet"} detail={snapshot.models.length < 1 ? "Launch training; the first actor snapshot can immediately face a reference baseline." : "Compare a checkpoint with another model or baseline using paired seeds."} />}
              </article>

              <article className="panel lineage-panel">
                <header className="panel-header"><div><span className="panel-kicker"><Jargon term="lineage">Lineage</Jargon></span><h2><Jargon term="champion">Champion</Jargon> ascent</h2></div></header>
                <div className="lineage-list">
                  {snapshot.models.slice(0, 4).reverse().map((model) => <div key={model.id} className={model.isChampion ? "is-champion" : ""}><i /><span><small>{compactFormatter.format(model.games)} games</small><strong>{model.label}</strong></span><b>{model.hasElo ? `${model.eloDelta >= 0 ? "+" : ""}${model.eloDelta.toFixed(0)}` : "—"}</b></div>)}
                  {!snapshot.models.length ? <EmptyState title="No lineage yet" detail="The first checkpoint becomes the root of the champion tree." /> : null}
                </div>
              </article>
            </div>

            <article className="panel registry-panel">
              <header className="panel-header"><div><span className="panel-kicker">Model registry</span><h2><Jargon term="checkpoint">Checkpoints</Jargon></h2></div><div className="registry-actions"><button type="button" className="button button-quiet" onClick={refreshModels} disabled={commandBusy !== null}>Refresh</button><button type="button" className="button" onClick={() => {
                const champion = snapshot.models.find((model) => model.isChampion);
                if (champion) exportModel(champion.id);
                else showToast("No champion actor is available yet");
              }} disabled={commandBusy !== null || !snapshot.models.some((model) => model.isChampion)}>Export champion</button></div></header>
              <div className="model-table" role="table" aria-label="Model checkpoints">
                <div className="model-row model-header" role="row"><span role="columnheader">Model</span><span role="columnheader">Games</span><span role="columnheader"><Jargon term="heldOutStrength">Held-out score</Jargon></span><span role="columnheader"><Jargon term="confidenceInterval">Confidence</Jargon></span><span role="columnheader">Δ <Jargon term="elo">Elo</Jargon></span><span role="columnheader">Created</span><span role="columnheader"><Jargon term="modelActions" align="right">Actions</Jargon></span></div>
                {snapshot.models.map((model) => <div className="model-row" role="row" key={model.id}><span role="cell"><i className={model.isChampion ? "champion-gem" : "model-node"} /><span><strong>{model.label}</strong><small>{model.id} · {model.sizeMb === null ? "size —" : `${model.sizeMb.toFixed(1)} MB`}</small></span>{model.isChampion ? <b className="champion-label"><Jargon term="champion">Champion</Jargon></b> : null}</span><span role="cell">{compactFormatter.format(model.games)}</span><span role="cell"><strong>{model.evaluated ? formatPercent(model.score) : "Not evaluated"}</strong></span><span role="cell">{model.evaluated ? `${formatPercent(model.ciLow)}–${formatPercent(model.ciHigh)}` : "—"}</span><span role="cell" className={model.hasElo && model.eloDelta >= 0 ? "positive" : ""}>{model.hasElo ? `${model.eloDelta >= 0 ? "+" : ""}${model.eloDelta.toFixed(0)}` : "—"}</span><span role="cell">{model.created}</span><span role="cell"><button type="button" aria-label={`${model.isPinned ? "Unpin" : "Pin"} ${model.label}`} onClick={() => togglePinned(model.id)} className={model.isPinned ? "is-pinned" : ""} disabled={commandBusy !== null}>◇</button><button type="button" aria-label={`Download actor for ${model.label}`} onClick={() => exportModel(model.id)} disabled={commandBusy !== null}>↓</button></span></div>)}
                {!snapshot.models.length ? <EmptyState title="Registry is empty" detail="Checkpoints, actor exports, evaluations, and lineage metadata will appear here." /> : null}
              </div>
            </article>
          </section>
        ) : null}

        {activeTab === "play" ? (
          <section className="tab-panel play-panel" aria-labelledby="play-title">
            <header className="section-heading play-heading">
              <div><span className="section-number">04 / PLAY</span><h1 id="play-title">Enter the arena yourself.</h1><p>Challenge any checkpoint through the same legal-action interface used in self-play.</p></div>
              <div className="game-setup"><label><span>Opponent</span><select value={snapshot.models.length || playModel === "baseline" ? playModel : "baseline"} onChange={(event) => setPlayModel(event.target.value)}><option value="baseline">Balanced baseline</option>{snapshot.models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</select></label><label className="toggle-label"><input type="checkbox" checked={humanStarts} onChange={(event) => setHumanStarts(event.target.checked)} /><span />You start</label><button type="button" className="button card-visibility-button" onClick={() => setInventoryOpen(true)}>Hands & decks</button><button type="button" className="button button-primary" onClick={newGame} disabled={commandBusy !== null}>{commandBusy === "game-new" ? "Starting…" : "New game"}</button></div>
            </header>

            {connected && !remoteGame ? <div className="panel connected-game-empty"><EmptyState title="Start a live game" detail="Choose a checkpoint or the balanced baseline, then create a session. Every card and legal action will come from the engine." /></div> : <div className="game-shell">
              <div className="board-column">
                <section className="player-zone opponent-zone" aria-label="Opponent board">
                  <header><div><span className="player-avatar opponent-avatar">AI</span><p><strong>Orion</strong><small>{remoteGame?.modelLabel ?? snapshot.models.find((model) => model.id === playModel)?.label ?? "Balanced baseline"}</small></p></div><button type="button" className="zone-cards-button" onClick={() => setInventoryOpen(true)}><b>{game.opponentHand.length}</b><span>hand</span></button><div className="authority-display"><small>Authority</small><strong>{game.opponentAuthority}</strong></div><button type="button" className="deck-display" onClick={() => setInventoryOpen(true)} aria-label="View opponent hand and unordered deck"><i /><span>{game.opponentDeckCount}<small>deck</small></span></button></header>
                  <div className="base-row">{game.opponentBases.map((card) => <CardTile key={card.id} card={card} compact />)}<span className="zone-label">Opponent bases</span></div>
                </section>

                <section className="market-zone" aria-label="Trade row">
                  <header><span>Trade row</span><p>Click a card to inspect or acquire it</p><div className="explorer-stack"><b>{game.explorersRemaining}</b><span>Explorers left</span><small>cost 2 each</small></div></header>
                  <div className="market-row">{game.market.map((card) => <CardTile key={card.id} card={card} selected={selectedCard === card.id} disabled={card.cost > game.trade} onClick={() => setSelectedCard(card.id)} />)}{game.market.length === 0 ? <EmptyState title="Trade row depleted" detail="Start a new game to restore the market." /> : null}</div>
                </section>

                <section className="player-zone human-zone" aria-label="Your board">
                  <div className="base-row">{game.humanBases.map((card) => <CardTile key={card.id} card={card} compact />)}<span className="zone-label">Your bases</span></div>
                  <header><div><span className="player-avatar human-avatar">YOU</span><p><strong>Your fleet</strong><small>Turn {game.turn} · main phase</small></p></div><div className="resource-pips"><span className="trade-pip"><b>{game.trade}</b>Trade</span><span className="attack-pip"><b>{game.attack}</b>Combat</span></div><div className="authority-display"><small>Authority</small><strong>{game.humanAuthority}</strong></div><button type="button" className="deck-display" onClick={() => setInventoryOpen(true)} aria-label="View your hand and unordered deck"><i /><span>{game.deckCount}<small>deck</small></span><span>{game.discardCount}<small>discard</small></span></button></header>
                  <div className="hand-row">{game.hand.map((card) => <CardTile key={card.id} card={card} selected={selectedCard === card.id} onClick={() => setSelectedCard(card.id)} />)}{game.hand.length === 0 ? <EmptyState title="Hand played" detail="Spend remaining trade or combat, then end the turn." /> : null}</div>
                </section>
              </div>

              <aside className="action-console">
                <div className="decision-label"><span className="connection-pulse" /><p><small>{remoteGame?.status === "model_thinking" ? "Opponent thinking" : remoteGame?.status === "complete" ? "Game complete" : "Decision requested"}</small><strong>{remoteGame?.result ?? remoteGame?.prompt ?? "Your main phase"}</strong></p><b>Turn {game.turn}</b></div>
                <div className="legal-actions">
                  <span className="panel-kicker"><Jargon term="actions">Legal actions</Jargon></span>
                  {remoteGame ? remoteGame.actions.map((action, index) => <button key={action.id} type="button" className={`action-button${action.recommended ? " is-recommended" : ""}`} onClick={() => submitRemoteChoice(action.id)} disabled={commandBusy === "game-choice" || remoteGame.status !== "your_turn"}><b>{titleCase(action.label)}</b><span>{action.modelValue === null ? `${titleCase(remoteGame.family)} · legal engine action` : `${formatPercent(action.modelValue)} acting-player outcome value${action.recommended ? " · model choice" : ""}`}</span><i>{index + 1}</i></button>) : <>
                    {selectedCard && game.hand.some((card) => card.id === selectedCard) ? <button type="button" className="action-button is-recommended" onClick={() => playHandCard(game.hand.find((card) => card.id === selectedCard)!)}><b>Play selected card</b><span>Resolve its primary effect</span><i>↵</i></button> : null}
                    {selectedCard && game.market.some((card) => card.id === selectedCard) ? <button type="button" className="action-button is-recommended" onClick={acquireSelected} disabled={(game.market.find((card) => card.id === selectedCard)?.cost ?? 99) > game.trade}><b>Acquire selected</b><span>Cost {(game.market.find((card) => card.id === selectedCard)?.cost ?? 0)} · {game.trade} trade available</span><i>↵</i></button> : null}
                    <button type="button" className="action-button" onClick={attackOpponent} disabled={game.attack <= 0 || game.opponentBases.length > 0}><b>Attack opponent</b><span>{game.opponentBases.length ? "Destroy the outpost first" : `${game.attack} combat available`}</span><i>A</i></button>
                    <button type="button" className="action-button" onClick={endTurn}><b>End turn</b><span>Discard hand and draw five</span><i>E</i></button>
                  </>}
                  {remoteGame && remoteGame.actions.length === 0 ? <EmptyState title={remoteGame.status === "model_thinking" ? "Opponent is thinking" : remoteGame.result ?? "No action pending"} detail={remoteGame.error ?? "The board will update automatically."} /> : null}
                </div>
                <div className="choice-inspector"><span className="panel-kicker">Model lens</span>{remoteGame ? (() => {
                  const recommendation = remoteGame.actions.find((action) => action.recommended);
                  return recommendation && recommendation.modelValue !== null ? <><div><span>Recommended action</span><strong>{titleCase(recommendation.label)}</strong><b>{formatPercent(recommendation.modelValue)}</b></div><div><span><Jargon term="actionValue">Value semantics</Jargon></span><strong>Acting-player win outcome</strong><b>Q</b></div><p>Values come directly from the checkpoint actor’s bootstrapped heads. Your choice is sent unchanged.</p></> : <><div><span><Jargon term="actionValue">Action values</Jargon></span><strong>Not exposed for this opponent</strong><b>—</b></div><p>Baseline sessions provide legal actions but no model scores; no values are inferred or fabricated.</p></>;
                })() : <><div><span>Recommended action</span><strong>Play Federation Shuttle</strong><b>42%</b></div><div><span><Jargon term="outcomeEstimate">Outcome estimate</Jargon></span><strong>Acting-player win value</strong><b>56.4%</b></div><p>Illustrative demo values are replaced by live checkpoint scores when the local service connects.</p></>}</div>
                <div className="game-log"><header><span className="panel-kicker">Action log</span><button type="button" onClick={() => setGame((current) => ({ ...current, log: [] }))}>Clear</button></header>{game.log.length ? game.log.map((entry, index) => <p key={`${entry}-${index}`}><span>{String(game.turn - Math.min(index, 2)).padStart(2, "0")}</span>{entry}</p>) : <EmptyState title="No actions yet" detail="Play a card to begin the log." />}</div>
              </aside>
            </div>}
            {inventoryOpen ? <CardInventory game={game} onClose={() => setInventoryOpen(false)} /> : null}
          </section>
        ) : null}

        {activeTab === "diagnostics" ? (
          <section className="tab-panel diagnostics-panel" aria-labelledby="diagnostics-title">
            <header className="section-heading">
              <div><span className="section-number">05 / DIAGNOSTICS & SETTINGS</span><h1 id="diagnostics-title">Trust every signal.</h1><p>Hardware health, learning dynamics, decision balance, configuration, and audit history in one place.</p></div>
              <div className={`diagnostic-status ${connected ? "is-good" : "is-warning"}`}><span>{connected ? "Local service responding" : "Trainer connection offline"}</span><strong>{connected ? remoteRunId ? `${titleCase(snapshot.run.status)} · telemetry ${snapshot.metrics.length ? "available" : "pending"}` : "Ready · no run selected" : "Demo data active"}</strong><small>{connected && lastSync ? `Last sync ${lastSync.toLocaleTimeString()}` : API_BASE}</small></div>
            </header>

            <div className="diagnostic-grid">
              <article className="panel loss-panel"><header className="panel-header"><div><span className="panel-kicker">Optimization</span><h2><Jargon term="outcomeLearning">Outcome learning</Jargon></h2></div><div className="chart-legend"><span className="legend-blue"><Jargon term="outcomeBce">Outcome BCE</Jargon></span><span className="legend-pink"><Jargon term="brierScore">Brier</Jargon></span></div></header>{snapshot.metrics.some((point) => point.hasLearnerDiagnostics) ? <MetricCanvas points={snapshot.metrics.filter((point) => point.hasLearnerDiagnostics)} mode="loss" /> : <EmptyState title="Learner diagnostics pending" detail="Outcome metrics begin after replay warmup and the first MLX update." />}<dl className="compact-dl"><div><dt><Jargon term="outcomeBce">Outcome BCE</Jargon></dt><dd>{latestMetric.hasLearnerDiagnostics ? latestMetric.outcomeLoss.toFixed(4) : "—"}</dd></div><div><dt><Jargon term="brierScore">Brier score</Jargon></dt><dd>{latestMetric.hasLearnerDiagnostics ? latestMetric.brier.toFixed(4) : "—"}</dd></div><div><dt><Jargon term="explainedVariance">Explained variance</Jargon></dt><dd>{latestMetric.hasLearnerDiagnostics ? latestMetric.explainedVariance.toFixed(3) : "—"}</dd></div><div><dt><Jargon term="bootstrapUncertainty">Bootstrap uncertainty</Jargon></dt><dd>{latestMetric.hasLearnerDiagnostics ? latestMetric.uncertainty.toFixed(3) : "—"}</dd></div><div><dt><Jargon term="curriculum">Curriculum</Jargon></dt><dd>{titleCase(latestMetric.curriculumPhase)}{latestMetric.bootstrapUpdatesRemaining ? ` · ${numberFormatter.format(latestMetric.bootstrapUpdatesRemaining)} left` : ""}</dd></div><div><dt><Jargon term="safetyTruncations">Safety truncations</Jargon></dt><dd>{formatPercent(latestMetric.truncationRate)}</dd></div><div><dt><Jargon term="draws">Draws</Jargon></dt><dd>{formatPercent(latestMetric.drawRate)}</dd></div><div><dt><Jargon term="meanTurns">Mean turns</Jargon></dt><dd>{latestMetric.meanTurns ? latestMetric.meanTurns.toFixed(1) : "—"}</dd></div><div><dt><Jargon term="forcedChoices">Forced choices skipped</Jargon></dt><dd>{compactFormatter.format(latestMetric.forcedChoices)}</dd></div></dl></article>
              <article className="panel decision-panel"><header className="panel-header"><div><span className="panel-kicker"><Jargon term="stratifiedReplay">Stratified replay</Jargon></span><h2><Jargon term="decisionFamilies">Decision families</Jargon></h2></div></header><div className="decision-bars">{replayFamilyEntries.map((family, index) => {
                const percent = replayFamilyTotal ? (family.size / replayFamilyTotal) * 100 : 0;
                const colors = ["#e8bd64", "#6aa9ff", "#e58ad4", "#ff7d78", "#5fe6ca", "#92a0b2", "#9b8cff", "#f29f67"];
                return <div key={family.name}><span>{family.name}</span><i><b style={{ width: `${percent}%`, background: colors[index % colors.length] }} /></i><strong>{percent.toFixed(1)}%</strong></div>;
              })}</div>{!replayFamilyEntries.length ? <EmptyState title="Replay is warming up" detail="Per-family occupancy appears after the first self-play batches." /> : null}<p className="panel-note">Each semantic chooser family has its own replay ring, preventing common main-phase choices from drowning rare tactical decisions.</p></article>
              <article className="panel system-panel"><header className="panel-header"><div><span className="panel-kicker">Local hardware</span><h2>System health</h2></div></header><div className="system-readings"><div><Gauge value={snapshot.hardware.cpuPercent / 100} label="CPU" /><p><strong>{snapshot.hardware.actorProcesses} actors</strong><span>{snapshot.hardware.chip}</span></p></div><div><Gauge value={snapshot.hardware.memoryTotalGb ? snapshot.hardware.memoryUsedGb / snapshot.hardware.memoryTotalGb : 0} label="memory" /><p><strong>{snapshot.hardware.memoryUsedGb.toFixed(1)} / {snapshot.hardware.memoryTotalGb.toFixed(1)} GB</strong><span>Unified memory</span></p></div><div className="metal-reading"><span className="metal-glyph">M</span><p><strong>{snapshot.hardware.backend}</strong><span>{snapshot.hardware.metalAvailable === null ? "Live Metal sample pending" : snapshot.hardware.metalAvailable ? `${snapshot.hardware.learnerDevice} · ${snapshot.hardware.metalActiveGb.toFixed(2)} GB active · ${snapshot.hardware.metalPeakGb.toFixed(2)} GB peak` : "Metal unavailable"}</span></p><b>{snapshot.hardware.metalAvailable === null ? "Pending" : snapshot.hardware.metalAvailable ? "Ready" : "Offline"}</b></div></div></article>
              <article className="panel connection-panel"><header className="panel-header"><div><span className="panel-kicker">Service link</span><h2>Local API</h2></div></header><dl className="connection-details"><div><dt>Endpoint</dt><dd><code>{API_BASE}</code></dd></div><div><dt>State</dt><dd className={connected ? "positive" : "warning-text"}>{connected ? "Connected" : "Offline · retrying each second"}</dd></div><div><dt>Transport</dt><dd>JSON · 1s incremental polling</dd></div><div><dt>Last sequence</dt><dd>{latestMetric.seq >= 0 ? latestMetric.seq : "Waiting"}</dd></div><div><dt>Exposure</dt><dd>Loopback only</dd></div></dl><button type="button" className="button" onClick={() => window.location.reload()}>Reconnect now</button></article>
            </div>

            <div className="settings-audit-grid">
              <form className="panel live-settings" onSubmit={saveLiveConfig}>
                <header className="panel-header"><div><span className="panel-kicker">Safe live settings</span><h2>Next boundary update</h2></div><span className="boundary-chip">Applies between batches</span></header>
                <div className="field-grid">
                  <label><span>Time budget (minutes)</span><input type="number" value={config.durationMinutes} onChange={(event) => updateConfig("durationMinutes", Number(event.target.value))} /></label>
                  <label><span><Jargon term="evaluationPairs">Evaluation pairs</Jargon></span><input type="number" min="8" max="20000" value={config.evaluationPairs} onChange={(event) => updateConfig("evaluationPairs", Number(event.target.value))} /></label>
                  <label><span>Evaluate every games</span><input type="number" value={config.evaluateEveryGames} onChange={(event) => updateConfig("evaluateEveryGames", Number(event.target.value))} /></label>
                  <label><span><Jargon term="checkpoint">Checkpoint every games</Jargon></span><input type="number" value={config.checkpointEveryGames} onChange={(event) => updateConfig("checkpointEveryGames", Number(event.target.value))} /></label>
                  <label><span><Jargon term="selfPlay">Current self-play</Jargon></span><input type="number" step="0.01" value={config.currentSelfplayFraction} onChange={(event) => updateConfig("currentSelfplayFraction", Number(event.target.value))} /></label>
                  <label><span><Jargon term="league">League opponents</Jargon></span><input type="number" step="0.01" value={config.leagueFraction} onChange={(event) => updateConfig("leagueFraction", Number(event.target.value))} /></label>
                  <label><span><Jargon term="baseline">Baseline anchors</Jargon></span><input type="number" step="0.01" value={config.baselineFraction} onChange={(event) => updateConfig("baselineFraction", Number(event.target.value))} /></label>
                  <label><span><Jargon term="promotionConfidence">Promotion confidence</Jargon></span><input type="number" min="0.8" max="0.999" step="0.01" value={config.promotionConfidence} onChange={(event) => updateConfig("promotionConfidence", Number(event.target.value))} /></label>
                  <label><span><Jargon term="promotionMargin">Promotion margin</Jargon></span><input type="number" min="0" max="0.25" step="0.005" value={config.promotionMargin} onChange={(event) => updateConfig("promotionMargin", Number(event.target.value))} /></label>
                </div>
                <div className="mix-check"><span>Opponent mix</span><i><b style={{ width: `${config.currentSelfplayFraction * 100}%` }} /><b style={{ width: `${config.leagueFraction * 100}%` }} /><b style={{ width: `${config.baselineFraction * 100}%` }} /></i><strong>{Math.round((config.currentSelfplayFraction + config.leagueFraction + config.baselineFraction) * 100)}%</strong></div>
                <div className="form-actions"><p>Unsafe architecture changes require a new run.</p><button type="submit" className="button button-primary" disabled={commandBusy !== null}>Queue update</button></div>
              </form>

              <article className="panel audit-panel"><header className="panel-header"><div><span className="panel-kicker">Immutable record</span><h2>Audit trail</h2></div><button type="button" className="text-button" onClick={exportAudit} disabled={!snapshot.events.length}>Export JSON →</button></header><div className="audit-list">{snapshot.events.map((event) => <div key={event.id}><time>{event.at}</time><i className={`event-${event.kind}`} /><span><strong>{event.title}</strong><small>{event.detail}</small></span></div>)}{!snapshot.events.length ? <EmptyState title="No persisted events yet" detail="Training commands and checkpoint events will be recorded here." /> : null}</div></article>
            </div>
          </section>
        ) : null}
      </div>

      {toast ? <div className="toast" role="status"><span>✓</span>{toast}</div> : null}
      <footer className="app-footer"><span>Astrosynapse 2 · local-first</span><span>Engine <b>deterministic</b></span><span>Trainer <b>{connected ? "connected" : "preview"}</b></span><span>Metrics <b>1 Hz</b></span></footer>
    </main>
  );
}
