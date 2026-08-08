"""Canonical Star Realms Core Set card data.

The old simulator encoded cards as positional tuples.  The new engine keeps the
same physical deck, but makes every field named, typed, immutable, and stable
for model encoders and replay files.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Faction(StrEnum):
    UNALIGNED = "unaligned"
    BLOB = "blob"
    MACHINE_CULT = "machine_cult"
    STAR_EMPIRE = "star_empire"
    TRADE_FEDERATION = "trade_federation"


class CardType(StrEnum):
    SHIP = "ship"
    BASE = "base"
    OUTPOST = "outpost"


@dataclass(frozen=True)
class Card:
    """A card definition, not a mutable physical-card instance."""

    card_id: int
    name: str
    cost: int
    combat: int = 0
    authority: int = 0
    trade: int = 0
    faction: Faction = Faction.UNALIGNED
    card_type: CardType = CardType.SHIP
    primary: str = ""
    ally: str = ""
    ally_amount: int = 0
    scrap: str = ""
    scrap_amount: int = 0
    defense: int = 0
    copies: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["faction"] = self.faction.value
        value["card_type"] = self.card_type.value
        return value

    @property
    def is_base(self) -> bool:
        return self.card_type in (CardType.BASE, CardType.OUTPOST)

    @property
    def is_ship(self) -> bool:
        return self.card_type == CardType.SHIP


U = Faction.UNALIGNED
G = Faction.BLOB
R = Faction.MACHINE_CULT
Y = Faction.STAR_EMPIRE
B = Faction.TRADE_FEDERATION
SHIP = CardType.SHIP
BASE = CardType.BASE
OUTPOST = CardType.OUTPOST


# IDs deliberately follow sim.py's long-standing order, making old data dumps
# straightforward to migrate while avoiding its fragile tuple offsets.
ALL_CARDS: tuple[Card, ...] = (
    Card(0, "Scout", 0, trade=1),
    Card(1, "Viper", 0, combat=1),
    Card(2, "Explorer", 2, trade=2, scrap="gain_combat", scrap_amount=2),
    Card(
        3,
        "Battle Blob",
        6,
        combat=8,
        faction=G,
        ally="draw",
        scrap="gain_combat",
        scrap_amount=4,
        copies=1,
    ),
    Card(
        4,
        "Battle Pod",
        2,
        combat=4,
        faction=G,
        primary="scrap_trade_row",
        ally="gain_combat",
        ally_amount=2,
        copies=2,
    ),
    Card(5, "Blob Carrier", 6, combat=7, faction=G, ally="free_ship", copies=1),
    Card(6, "Blob Destroyer", 4, combat=6, faction=G, ally="destroy_and_scrap", copies=2),
    Card(7, "Blob Fighter", 1, combat=3, faction=G, ally="draw", copies=3),
    Card(
        8,
        "Blob Wheel",
        3,
        combat=1,
        faction=G,
        card_type=BASE,
        scrap="gain_trade",
        scrap_amount=3,
        defense=5,
        copies=3,
    ),
    Card(9, "Blob World", 8, faction=G, card_type=BASE, primary="blob_world", defense=7, copies=1),
    Card(10, "Mothership", 7, combat=6, faction=G, primary="draw", ally="draw", copies=1),
    Card(
        11,
        "Ram",
        3,
        combat=5,
        faction=G,
        ally="gain_combat",
        ally_amount=2,
        scrap="gain_trade",
        scrap_amount=3,
        copies=2,
    ),
    Card(12, "The Hive", 5, combat=3, faction=G, card_type=BASE, ally="draw", defense=5, copies=1),
    Card(13, "Trade Pod", 2, trade=3, faction=G, ally="gain_combat", ally_amount=2, copies=3),
    Card(14, "Battle Mech", 5, combat=4, faction=R, primary="scrap_any", ally="draw", copies=1),
    Card(
        15,
        "Battle Station",
        3,
        faction=R,
        card_type=OUTPOST,
        scrap="gain_combat",
        scrap_amount=5,
        defense=5,
        copies=2,
    ),
    Card(
        16,
        "Brain World",
        8,
        faction=R,
        card_type=OUTPOST,
        primary="scrap_two_draw",
        defense=6,
        copies=1,
    ),
    Card(17, "Junkyard", 6, faction=R, card_type=OUTPOST, primary="scrap_any", defense=5, copies=1),
    Card(
        18,
        "Machine Base",
        7,
        faction=R,
        card_type=OUTPOST,
        primary="draw_then_scrap",
        defense=6,
        copies=1,
    ),
    Card(
        19, "Mech World", 5, faction=R, card_type=OUTPOST, primary="all_ally", defense=6, copies=1
    ),
    Card(
        20,
        "Missile Bot",
        2,
        combat=2,
        faction=R,
        primary="scrap_any",
        ally="gain_combat",
        ally_amount=2,
        copies=3,
    ),
    Card(21, "Missile Mech", 6, combat=6, faction=R, primary="destroy_base", ally="draw", copies=1),
    Card(22, "Patrol Mech", 4, faction=R, primary="patrol_mech", ally="scrap_any", copies=2),
    Card(23, "Stealth Needle", 4, faction=R, primary="copy_ship", copies=1),
    Card(
        24,
        "Supply Bot",
        3,
        trade=2,
        faction=R,
        primary="scrap_any",
        ally="gain_combat",
        ally_amount=2,
        copies=3,
    ),
    Card(
        25,
        "Trade Bot",
        1,
        trade=1,
        faction=R,
        primary="scrap_any",
        ally="gain_combat",
        ally_amount=2,
        copies=3,
    ),
    Card(
        26,
        "Battlecruiser",
        6,
        combat=5,
        faction=Y,
        primary="draw",
        ally="opponent_discard",
        scrap="draw_destroy",
        copies=1,
    ),
    Card(
        27,
        "Corvette",
        2,
        combat=1,
        faction=Y,
        primary="draw",
        ally="gain_combat",
        ally_amount=2,
        copies=2,
    ),
    Card(
        28,
        "Dreadnaught",
        7,
        combat=7,
        faction=Y,
        primary="draw",
        scrap="gain_combat",
        scrap_amount=5,
        copies=1,
    ),
    Card(29, "Fleet HQ", 8, faction=Y, card_type=BASE, primary="fleet_hq", defense=8, copies=1),
    Card(
        30,
        "Imperial Fighter",
        1,
        combat=2,
        faction=Y,
        primary="opponent_discard",
        ally="gain_combat",
        ally_amount=2,
        copies=3,
    ),
    Card(
        31,
        "Imperial Frigate",
        3,
        combat=4,
        faction=Y,
        primary="opponent_discard",
        ally="gain_combat",
        ally_amount=2,
        scrap="draw",
        copies=3,
    ),
    Card(
        32,
        "Recycling Station",
        4,
        faction=Y,
        card_type=OUTPOST,
        primary="recycle",
        defense=4,
        copies=2,
    ),
    Card(
        33,
        "Royal Redoubt",
        6,
        combat=3,
        faction=Y,
        card_type=OUTPOST,
        ally="opponent_discard",
        defense=6,
        copies=1,
    ),
    Card(
        34,
        "Space Station",
        4,
        combat=2,
        faction=Y,
        card_type=OUTPOST,
        ally="gain_combat",
        ally_amount=2,
        scrap="gain_trade",
        scrap_amount=4,
        defense=4,
        copies=2,
    ),
    Card(
        35,
        "Survey Ship",
        3,
        trade=1,
        faction=Y,
        primary="draw",
        scrap="opponent_discard",
        defense=0,
        copies=3,
    ),
    Card(
        36,
        "War World",
        5,
        combat=3,
        faction=Y,
        card_type=OUTPOST,
        ally="gain_combat",
        ally_amount=4,
        defense=4,
        copies=1,
    ),
    Card(
        37,
        "Barter World",
        4,
        faction=B,
        card_type=BASE,
        primary="barter_world",
        scrap="gain_combat",
        scrap_amount=5,
        defense=4,
        copies=2,
    ),
    Card(
        38,
        "Central Office",
        7,
        trade=2,
        faction=B,
        card_type=BASE,
        primary="ship_top",
        ally="draw",
        defense=6,
        copies=1,
    ),
    Card(
        39,
        "Command Ship",
        8,
        combat=5,
        authority=4,
        faction=B,
        primary="draw_two",
        ally="destroy_base",
        copies=1,
    ),
    Card(
        40,
        "Cutter",
        2,
        authority=4,
        trade=2,
        faction=B,
        ally="gain_combat",
        ally_amount=4,
        copies=3,
    ),
    Card(
        41,
        "Defense Center",
        5,
        faction=B,
        card_type=OUTPOST,
        primary="defense_center",
        ally="gain_combat",
        ally_amount=2,
        defense=5,
        copies=1,
    ),
    Card(
        42, "Embassy Yacht", 3, authority=3, trade=2, faction=B, primary="embassy_yacht", copies=2
    ),
    Card(
        43,
        "Federation Shuttle",
        1,
        trade=2,
        faction=B,
        ally="gain_authority",
        ally_amount=4,
        copies=3,
    ),
    Card(
        44,
        "Flagship",
        6,
        combat=5,
        faction=B,
        primary="draw",
        ally="gain_authority",
        ally_amount=5,
        copies=1,
    ),
    Card(45, "Freighter", 4, trade=4, faction=B, ally="ship_top", copies=2),
    Card(
        46,
        "Port of Call",
        6,
        trade=3,
        faction=B,
        card_type=OUTPOST,
        scrap="draw_destroy",
        defense=6,
        copies=1,
    ),
    Card(47, "Trade Escort", 5, combat=4, authority=4, faction=B, ally="draw", copies=1),
    Card(
        48,
        "Trading Post",
        3,
        faction=B,
        card_type=OUTPOST,
        primary="trading_post",
        scrap="gain_combat",
        scrap_amount=3,
        defense=4,
        copies=2,
    ),
)


CARD_BY_ID: dict[int, Card] = {card.card_id: card for card in ALL_CARDS}
CARD_BY_NAME: dict[str, Card] = {card.name: card for card in ALL_CARDS}
SCOUT = CARD_BY_NAME["Scout"]
VIPER = CARD_BY_NAME["Viper"]
EXPLORER = CARD_BY_NAME["Explorer"]
TRADE_DECK_CARDS: tuple[Card, ...] = tuple(card for card in ALL_CARDS if card.copies)


def build_trade_deck() -> list[Card]:
    """Return all 80 physical Core Set trade-deck cards."""

    return [card for card in TRADE_DECK_CARDS for _ in range(card.copies)]


def card_counts(cards: Iterable[Card]) -> tuple[tuple[int, int], ...]:
    """Canonical immutable multiset used in observations and save files."""

    counts: dict[int, int] = {}
    for card in cards:
        counts[card.card_id] = counts.get(card.card_id, 0) + 1
    return tuple(sorted(counts.items()))


assert len(ALL_CARDS) == 49
assert len(build_trade_deck()) == 80
