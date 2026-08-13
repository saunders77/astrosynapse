# Legacy simulator audit and corrected rules

Astrosynapse 2 treats the original `sim.py` as a card-catalog and differential reference, not as its production hot loop. The new engine was written independently with injected random streams, immutable observations, semantic legal actions, explicit safety truncation, and focused tests for every base-set card effect.

The primary rules references are the official [Star Realms Box Set rulebook](https://files.wisewizardgames.com/wwg/rules/SRBOX_Rulebook_1.1E_WEB.pdf), the official [Frontiers rulebook](https://files.wisewizardgames.com/starrealms/StarRealms_Frontiers_RuleBook2.pdf), and the publisher's [Star Realms FAQ](https://www.starrealms.com/faq/).

## Material legacy issues

### Setup order and starting hands

Legacy behavior drew three cards for one list position and five for the other, then shuffled the player list. That could make the five-card player move first.

Corrected behavior determines the starting player first, then draws three cards for that player and five for the second player. The starting role is exposed to the policy as public information and self-play balances it explicitly.

### Explorer versus Trade Row

Legacy behavior stored Explorer at `tradeRow[0]` alongside five actual market cards. Effects could scrap or skip the wrong target, and Blob Carrier iterated only five entries of the six-entry structure.

Corrected behavior has exactly five Trade Row slots and a separate ten-card Explorer supply. Trade-row scrap never targets Explorer. Blob Carrier considers every actual Trade Row slot.

### Blob Carrier is optional

Legacy handling effectively treated the free acquisition as an unconditional follow-up when a ship was available.

Corrected behavior offers every eligible ship plus a semantic decline action and, when accepted, places the acquired ship on top of the player's deck.

### Hand-authored dominance pruning

Legacy `isFirstEqualOrBetter` logic removed legal discard and scrap choices using static card comparisons. That silently encoded the old bot's strategy into the environment and could eliminate the strategically correct faction/synergy choice.

Corrected behavior removes only provably identical physical-card options. It never prunes a distinct legal card because a heuristic considers it weaker.

### Starting-turn discard and base timing

Legacy ordering could allow card/base effects before required discard pressure resolved, and some base abilities fired immediately instead of remaining available at the legal point in the main phase.

Corrected behavior resolves forced discard before main-phase actions. Newly played and surviving bases expose once-per-turn activations as legal main-phase actions, allowing effects such as Blob World, Central Office, and Recycling Station to be timed. Continuous effects such as Mech World and Fleet HQ remain continuous.

### Stealth Needle and faction state

Legacy logic could count Stealth Needle as a Blob independently of the copied ship and mishandle copied-card state.

Corrected behavior represents the copied ship explicitly while retaining the original physical card for conservation and scrap/discard handling. Faction and ability behavior come from the copied ship; the observation marks the copy state.

### Effect order

Legacy effect sequencing had cases where destruction/scrap choices happened before their associated draw, changing the information available to the player.

Corrected effect dispatch follows card text order. Tests cover draw-before-destroy and all base-set effect dispatch paths.

### End turn and action caps

Legacy code could prohibit end turn while cards remained in hand and silently force a turn/game outcome after an action cap. Capped games could receive a fabricated heuristic winner.

Corrected behavior always permits a legal end turn; unplayed hand cards are discarded normally. A turn/game safety cap produces an explicit truncation, never an invented winner. Truncated training trajectories are excluded from replay and league rewards. In an automatic arena, truncations are conservatively scored as candidate losses for promotion, and the adjusted paired confidence interval must still clear every normal gate.

### Randomness and observation safety

Legacy code used module-global randomness and passed mutable live object references through chooser observations. Evaluation “balanced seeds” could still use unrelated randomness and player order.

Corrected behavior derives isolated seating, market, player-shuffle, and policy streams from an injected seed. Observations and decisions are frozen dataclasses. Unknown deck order is never exposed; order-invariant zones are immutable sorted multisets. Arena evaluation reuses one game seed and model-role policy seeds, then swaps model seats exactly.

## Engine invariants and tests

The focused suite checks:

- all 49 base-set definitions and the 80-card trade deck;
- five-card Trade Row and separate Explorer supply;
- deterministic replay under the same seed and policies;
- hidden-order observation invariance;
- no duplicate semantic actions and no strategic dominance pruning;
- forced single-option decisions bypassing the chooser;
- all-card effect dispatch and activation timing;
- Blob Carrier's fifth slot and optional decline;
- Stealth Needle copies and continuous faction effects;
- discard, scrap, top-deck, base destruction, and trade-row scrap zones;
- cancellation, action/turn truncation, and neutral results;
- conservation of all 110 physical base-set/starter/Explorer cards through randomized games;
- hidden hands and deck order omitted from human/API state unless explicitly requested by a test/debug caller.

## Compatibility policy

Astrosynapse 2 intentionally does not preserve legacy bugs as training behavior. If exact legacy replay is needed for comparison, it should run in a named compatibility mode and never share evaluation results with the corrected ruleset.

Before adding an expansion or native engine, record seed plus semantic action log in the Python reference, replay it in the candidate engine, and compare every public decision, terminal result, and card-conservation snapshot. Speed is accepted only after rule equivalence passes.
