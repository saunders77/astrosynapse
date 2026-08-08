"""Astrosynapse 2: deterministic Star Realms self-play on Apple silicon."""

from .arena import ArenaConfig, ArenaManager
from .cards import ALL_CARDS, CARD_BY_ID, CARD_BY_NAME, Card, CardType, Faction
from .engine import (
    Action,
    ActionKind,
    Decision,
    DecisionFamily,
    Game,
    GameConfig,
    GameResult,
    Observation,
    RNGStreams,
    Seating,
    play_game,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_CARDS",
    "CARD_BY_ID",
    "CARD_BY_NAME",
    "Action",
    "ActionKind",
    "ArenaConfig",
    "ArenaManager",
    "Card",
    "CardType",
    "Decision",
    "DecisionFamily",
    "Faction",
    "Game",
    "GameConfig",
    "GameResult",
    "Observation",
    "RNGStreams",
    "Seating",
    "play_game",
]
