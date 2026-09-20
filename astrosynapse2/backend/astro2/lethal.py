"""Small rules-checked lethal search shared by every learned actor.

Only cards already in hand/play may fund draws. Drawn cards become usable by
this search only once the original deck is exhausted, so neither hidden order
nor the discard pile can supply speculative combat. The real engine executes
every selected action, preserving replays, card conservation and human rules.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .cards import CARD_BY_ID, CardType
from .engine import ActionKind as K
from .engine import Decision, Game
from .engine import DecisionFamily as F

Plan = list[tuple[F, tuple[Any, ...]]]


def _combat(card):
    return (
        card.combat
        + (card.ally_amount if card.ally == "gain_combat" else 0)
        + (card.scrap_amount if card.scrap == "gain_combat" else 0)
        + {"blob_world": 5, "patrol_mech": 5, "defense_center": 2}.get(card.primary, 0)
    )


def _possible(decision: Decision) -> bool:
    """Cheap optimistic bound; all timing and costs are checked by the engine."""
    o = decision.observation
    cards = list(o.hand) + [item.card for item in o.own_in_play]
    blob_count = o.blob_cards_played + sum(c.faction.value == "blob" for c in o.hand)

    def draws(effect):
        return {
            "draw": 1,
            "draw_two": 2,
            "draw_destroy": 1,
            "scrap_two_draw": 2,
            "draw_then_scrap": 1,
            "recycle": 2,
            "embassy_yacht": 2,
            "blob_world": blob_count,
        }.get(effect, 0)

    draw_count = sum(draws(c.primary) + draws(c.ally) + draws(c.scrap) for c in o.hand)
    draw_count += sum(
        (0 if i.activated else draws(i.card.primary))
        + (0 if i.ally_triggered else draws(i.card.ally))
        + draws(i.card.scrap)
        for i in o.own_in_play
    )
    if any(c.primary == "copy_ship" for c in o.hand):
        draw_count += max(
            (draws(c.primary) + draws(c.ally) + draws(c.scrap) for c in cards if c.is_ship),
            default=0,
        )
    if o.own_deck_count and draw_count >= o.own_deck_count:
        cards += list(o.own_deck) + list(o.own_known_top)
    total = o.combat + sum(_combat(c) for c in cards)
    # Resources already credited to cards in play must not be counted twice.
    total -= sum(i.card.combat for i in o.own_in_play)
    total -= sum(
        i.card.ally_amount
        for i in o.own_in_play
        if i.ally_triggered and i.card.ally == "gain_combat"
    )
    total -= sum(
        {"blob_world": 5, "patrol_mech": 5, "defense_center": 2}.get(i.card.primary, 0)
        for i in o.own_in_play
        if i.activated
    )
    if any(c.card_id == 29 for c in cards):
        total += sum(c.is_ship for c in cards)
    total += sum(c.primary == "copy_ship" for c in cards) * max(
        (_combat(c) + 1 for c in cards if c.is_ship), default=0
    )
    destroy_effects = {"destroy_base", "destroy_and_scrap", "draw_destroy"}
    destroy_count = sum(
        sum(effect in destroy_effects for effect in (c.primary, c.ally, c.scrap)) for c in cards
    )
    destroy_count += sum(c.primary == "copy_ship" for c in cards) * 2
    defenses = sorted(
        (i.card.defense for i in o.opponent_in_play if i.card.card_type == CardType.OUTPOST),
        reverse=True,
    )
    return total >= o.opponent_authority + sum(defenses[destroy_count:])


class _Branch(Exception):
    def __init__(self, choices):
        self.choices = choices


class _DeadEnd(Exception):
    pass


def find_lethal_plan(game: Game, decision: Decision, *, budget: int = 256) -> Plan:
    """Return a verified legal winning sequence, or leave the actor in control.

    Search is bounded because it also runs in high-throughput training. A bound
    can miss a complicated lethal but can never manufacture a winning result.
    """
    if not _possible(decision):
        return []
    player_id = decision.observation.player_id
    prefixes: list[list[int]] = [[]]
    visited = set()
    for _ in range(budget):
        if not prefixes:
            break
        prefix = prefixes.pop()
        branch = game.fork()
        branch.cancel_hook = None
        branch.decision_hook = None
        own = branch.players[player_id]
        # Canonical public multiset, never the simulator's hidden deck order.
        own.deck.sort(key=lambda c: c.card_id)
        own.known_top.clear()
        own.revealed_hand.clear()
        original_hand = Counter(c.card_id for c in own.hand)
        plan: Plan = []
        choices_used = 0

        def draw(player, count):
            # No discard reshuffles. Extra real draws can only add unused cards.
            for _ in range(min(count, len(player.deck))):
                player.hand.append(player.deck.pop())

        branch._draw = draw

        def choose(
            player,
            family,
            actions,
            prompt,
            *,
            branch=branch,
            own=own,
            plan=plan,
            prefix=prefix,
            original_hand=original_hand,
        ):
            nonlocal choices_used
            options = list(branch._deduplicate(actions))
            if len(plan) >= game.config.max_actions_per_turn - game._turn_actions:
                raise _DeadEnd
            # Until every deck card is drawn, the contents of partial draws
            # cannot influence a choice, including mandatory scrap/discard.
            if own.deck:
                options = [
                    a
                    for a in options
                    if not (
                        a.kind == K.PLAY_CARD
                        or (a.kind in {K.SCRAP_CARD, K.DISCARD_CARD} and a.source_zone == "hand")
                    )
                    or original_hand[a.card_id] > 0
                ]
            if family == F.MAIN:
                attacks = [
                    a
                    for a in options
                    if a.kind == K.ATTACK_PLAYER
                    and own.combat >= branch.players[1 - player_id].authority
                ]
                if attacks:
                    options = attacks
                else:
                    options = [
                        a
                        for a in options
                        if a.kind
                        in {
                            K.PLAY_CARD,
                            K.ACTIVATE_BASE,
                            K.ACTIVATE_ALLY,
                            K.SCRAP_FOR_ABILITY,
                            K.ATTACK_BASE,
                        }
                    ]
                    options = [
                        a
                        for a in options
                        if a.kind != K.ATTACK_BASE
                        or CARD_BY_ID[a.target_card_id].card_type == CardType.OUTPOST
                    ]
                    options = [
                        a
                        for a in options
                        if a.kind != K.SCRAP_FOR_ABILITY
                        or a.ability in {"gain_combat", "draw", "draw_destroy"}
                    ]
                    # Prefer bases and draw ships, but retain other orders:
                    # drawing Fleet HQ before playing a Viper can be decisive.
                    plays = [a for a in options if a.kind == K.PLAY_CARD]
                    if plays:
                        plays.sort(
                            key=lambda a: (
                                CARD_BY_ID[a.card_id].primary == "copy_ship",
                                CARD_BY_ID[a.card_id].is_ship,
                                CARD_BY_ID[a.card_id].primary not in {"draw", "draw_two"},
                                CARD_BY_ID[a.card_id].primary == "embassy_yacht",
                                -_combat(CARD_BY_ID[a.card_id]),
                                a.card_id,
                            )
                        )
                        options = plays + [a for a in options if a.kind != K.PLAY_CARD]
                    options.sort(
                        key=lambda a: (
                            {
                                K.PLAY_CARD: 0,
                                K.ACTIVATE_ALLY: 1,
                                K.ACTIVATE_BASE: 2,
                                K.SCRAP_FOR_ABILITY: 3,
                                K.ATTACK_BASE: 4,
                            }.get(a.kind, 5),
                            -(a.amount if a.kind == K.ATTACK_BASE else 0),
                        )
                    )
                state = (
                    tuple(sorted(c.card_id for c in own.hand)),
                    tuple(
                        sorted(
                            (i.card.card_id, i.original_card.card_id, i.activated, i.ally_triggered)
                            for i in own.in_play
                        )
                    ),
                    own.combat,
                    own.blob_cards_played,
                    len(own.deck),
                    tuple(sorted(original_hand.items())),
                    tuple(sorted(i.card.card_id for i in branch.players[1 - player_id].in_play)),
                    tuple(sorted(c.card_id for c in own.discard)),
                )
                if choices_used == len(prefix):
                    if state in visited:
                        raise _DeadEnd
                    visited.add(state)
            elif family in {F.SCRAP_TRADE_ROW, F.FREE_ACQUIRE}:
                options = [a for a in options if a.kind == K.DECLINE]
            elif family == F.DESTROY_BASE:
                targets = [a for a in options if a.kind == K.DESTROY_BASE]
                options = (
                    sorted(targets, key=lambda a: -CARD_BY_ID[a.target_card_id].defense)[:1]
                    or options
                )
            elif family == F.ABILITY_MODE:
                options.sort(
                    key=lambda a: (a.ability != "gain_combat", a.ability not in {"draw", "cycle"})
                )
            elif family == F.COPY_SHIP:
                options.sort(key=lambda a: -_combat(CARD_BY_ID[a.target_card_id]))
            elif family in {F.SCRAP, F.DISCARD}:
                options.sort(
                    key=lambda a: (
                        a.kind != K.DECLINE,
                        a.source_zone != "discard",
                        _combat(CARD_BY_ID[a.card_id]) if a.card_id >= 0 else 0,
                    )
                )
            if not options:
                raise _DeadEnd
            if len(options) > 1:
                if choices_used == len(prefix):
                    raise _Branch(len(options))
                selected = options[prefix[choices_used]]
                choices_used += 1
            else:
                selected = options[0]
            plan.append((family, selected.semantic_key))
            if own.deck and (
                selected.kind == K.PLAY_CARD
                or (
                    selected.kind in {K.SCRAP_CARD, K.DISCARD_CARD}
                    and selected.source_zone == "hand"
                )
            ):
                original_hand[selected.card_id] -= 1
            return selected

        branch._choose = choose
        try:
            while branch._winner is None:
                action = choose(own, F.MAIN, branch._main_actions(own), "Main phase")
                branch._apply_main_action(own, action)
            return plan
        except _Branch as split:
            prefixes.extend(prefix + [i] for i in reversed(range(split.choices)))
        except _DeadEnd:
            pass
    return []
