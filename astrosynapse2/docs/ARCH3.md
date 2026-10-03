# arch3 inputs and generation10 continuation

`arch3` means encoder version 3, independent of the training algorithm or champion generation. Generation10 remains unchanged. Its port is a separate lineage under `data/arch3/`, discoverable in Models & Arena, Play, Elo probes, and the critic policy selector. Disable **Champions only** to see unpromoted ports and training candidates.

## Public information contract

Arch3 appends 49 per-card counts for `opponent_inferred_hand` to the 1,292 arch2 state inputs (1,341 total). Action inputs remain 219. The original state prefix is unchanged, including its historical known-hand and hidden-pool definitions. The appended counts are a subset of that public pool, not additional owned cards.

At cleanup, a remaining deck of one through five cards is entirely drawn into the next hand. Its multiset is public because the old hand has been discarded face up. The engine remembers these cards across a discard reshuffle, including duplicate copies. Drawn public ship-to-top cards are also included. When the opponent has no unknown cards left in their deck, their hand is completely known. Playing, discarding, and scrapping remove the corresponding known copy. Forks and public-belief rollouts preserve these constraints. The identities of draws from a shuffled discard are not exposed.

The existing `TRADE_DECK` block already provides the exact count of each card still unrevealed in the market. It is retained in arch3; it contains no draw-order information. Acquiring or scrapping a trade-row card and revealing its replacement reduces the appropriate remaining count.

The generic and cached Python encoders support versions 1, 2, and 3. Each arena, evaluation, and opponent actor uses its own encoder. The learner's replay uses its selected architecture. Existing checkpoint inputs retain their old meanings.

## Porting

On **Independent critic**, choose a playing policy and click **Port selected policy to arch3**. The converter copies all weights and extends `state_in.weight` with 49 zero columns. This preserves the initial policy and embedded value predictions. It writes both `.safetensors` and `.actor.npz`, a versioned model sidecar, and a lineage file with the immutable source hash. It does not copy optimizer/replay state or grant champion status.

The installed generation10 port is indexed in `data/arch3/generation10.json`. A four-game mixed-architecture check compared 1,168 decisions and found identical policy logits and choices; results are in `data/arch3/validation.json`. This establishes migration equivalence, not improved playing strength.

## Critic, then policy

1. Open **Independent critic** (`/critics`). Select the original generation10 playing policy and **arch3** as the critic architecture. Playing and baseline-value prediction continue to use the source policy's original encoder; sampled critic states use arch3.
2. The requested run uses **50,000 games**, up to **2,000 epochs**, width **128**, **64 sampled decisions/game**, **six collection workers**, and **16 games/update**. Each complete game has equal weight. Whole games are split 70% training, 15% validation, and 15% test. Truncated games are excluded.
3. Two thousand epochs is a ceiling. Stop after 20 epochs without improved validation log loss, keep the best validation checkpoint, and evaluate the held-out test split once. Data are streamed from disk; the entire dataset is not loaded into RAM. Collection and completed epochs survive pause/resume. An interrupted epoch restarts from its last durable checkpoint.
4. Once complete, select the generation10 arch3 port as the playing policy and the completed critic run below. Optionally select older checkpoint opponents, then click **Train with selected arch3 critic**. The UI starts a fresh arch3 lineage for 80 iterations × 256 games, learning rate 0.000002, with six workers. The immutable independent critic supplies the policy-advantage baseline; the actor's separate embedded value head continues to fit fresh trajectories. The external critic itself stays frozen.
5. With historical opponents selected, 75% of rollout games face the critic's source policy and 25% face the selected history. Every opponent retains its own architecture. Use the policy pause/resume controls and log link on the critic page. Saved candidates appear in Models & Arena for paired evaluations and Elo probes; training does not automatically promote them.

A frozen self-play critic may become less calibrated as the policy or opponent mix changes. It remains an action-independent baseline; train a fresh critic on a later policy when needed. Architecture and game-rules mismatches are rejected before policy updates. The critic hash and full configuration are recorded in the policy manifest.

For custom policy budgets, use `scripts/onpolicy_experiment.py` with `--model <arch3.safetensors> --opponent <old.actor.npz> --independent-critic <critic.npz> --rules-version 2 --output <new-folder>`. Add `--opponent-pool <JSON-list-of-actor-paths>` for history. Resume with the same output and `--resume`; code and critic identity are verified. The normal Train page also exposes input architecture for new from-scratch runs.

The 20-game/two-epoch critic and four-game policy smoke checks are stored separately in `data/arch3/smoke-critic` and `data/arch3/smoke-policy`; they are validation artifacts, not production critics or promoted models.
