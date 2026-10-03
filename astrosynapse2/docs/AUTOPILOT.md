# Champion Autopilot

Open `/autopilot` in the local control center. The campaign supervises policy training, independent critic development, fresh evaluations, and automatic promotions. The existing Arena and Play selectors discover its retained policy checkpoints. The Autopilot page also provides Arena and Self-play matches with independently selected critics; each policy and critic uses its own architecture encoder. Critics report win-probability diagnostics and never replace the policy as move selector.

## Training contract

The default campaign uses two training seeds, 20,000-game policy blocks, 1,000 games per update, six workers, and the tested policy learning rate of 0.000002. Each block freezes its critic, current champion opponent, and historical opponent list. The learner faces the champion in 75% of games and historical opponents in 25%. Both arch2 and arch3 opponents work without migration. Learners are ported to arch3 with zero-initialized added input columns when necessary.

The 80/20 target applies to active task wall time, not physical energy consumption. Policy collection and learning count toward policy time; independent critic fitting and matched policy-training probes count toward critic time. All promotion and diagnostic evaluation time is counted separately. The scheduler accumulates critic allowance as `policy_time * 0.2 / 0.8 - critic_time`, waits until it can afford a probe round, and checks for useful critic work between blocks. Early stopping returns unused capacity to policy work. Complete probe rounds can temporarily overshoot the target. Resource limits are soft at checkpoint/batch boundaries.

Policy continuations retain their optimizer between blocks in the same lane. A new playing champion rebases subsequent lanes. A newly accepted critic is used at the next block boundary; it is never replaced mid-update.

## Critic qualification

Each policy rollout exports up to 32 sampled public positions per complete game, including forced decisions. Truncated games are excluded. Critic fitting selects up to 2,500 games, emphasizing the most recent data while retaining historical games when available. Stable, hashed whole-game partitions keep training, validation, and test data disjoint across rounds. Games receive equal weight. Validation selects the epoch; recent held-out Brier/log loss and historical log loss qualify the candidate. Because retained holdouts can be revisited in later rounds, these prediction diagnostics are qualification checks, not independent strength certificates.

A qualified candidate faces the existing critic in matched policy-training probes (2,000 games per critic per seed, up to two seeds). Both arms start from the same policy with the same game seeds and opponents. Fresh paired evaluations compare their resulting playing policies. Better prediction alone never promotes a critic. Its playing-strength lower bound must exceed zero; inconclusive probes retain the accepted critic. The bound is conditional on these trained probes, not a guarantee across future training seeds.

## Policy promotions

512 fresh seed pairs screen each candidate. Promising candidates enter a separate fresh gate with up to 12,000 seat-swapped pairs, permitting early acceptance after 2,000 pairs. Time-uniform bounded betting confidence sequences account for repeated looks. Across policy attempts, the incumbent gate spends `0.025 / (k*(k+1))`; historical panel tests share another equal-sized allocation. The attempt counter never resets after promotions. A candidate must beat the incumbent and establish no regression larger than two percentage points against every configured panel opponent, using matched candidate/incumbent cases (4,000 pairs per opponent by default). Any truncated gate game prevents promotion.

Critic probe gates separately spend `0.05 / (k*(k+1))` across critic attempts. Ties and insufficient evidence are reported as inconclusive. They never force a promotion. Frozen benchmark opponents stay fixed throughout a campaign, even as training opponents gain previous campaign champions.

## Management and recovery

Start/resume, pause at a durable checkpoint, or stop after the complete current block and its evaluations. The latter is the appropriate boundary for restoring retained champions. Paused campaigns can change total active hours, storage allowance, worker count, and policy/critic allocation. The initial defaults are 24 active hours and 30 GB; exhaustion pauses the campaign. Increase the limit and resume to continue. Less than 2 GB free disk also pauses work.

Workers run independently of the browser/API process and hold a single-owner filesystem lock. Each campaign snapshots code, input models, critics, configurations, and input hashes. Policy checkpoints include optimizer state; evaluations persist a contiguous sequence of paired cases. Resume uses the frozen runtime and existing evidence, rather than rerolling unfavorable games. A finished promotion decision is marked before advancing the scheduler, preventing duplicate promotion after interruption. Failed workers retain logs and can be resumed after the underlying issue is resolved. Orphan learner processes must finish or pause before another supervisor can start.

The GUI shows active jobs, training progress, measured allocation, resource limits, champion identities, associated critics, and a decision timeline with exact evidence. Policies and accepted critics retain their own version histories. Restore selects only previously accepted champions and requires a stopped campaign at a block boundary. It records an explicit restoration event; it does not manufacture promotion evidence.

## Storage and endpoints

Campaigns live under `data/autopilot/<id>/`: `config.json`, `inputs.json`, `code.json`, `state.json`, `jobs.json`, `worker.log`, immutable `inputs/` and `runtime/`, and resumable `tasks/`. All retained candidates are discoverable through the normal model registry; current champions have the champion flag. Nothing overwrites generation10.

- `GET/POST /api/autopilot`: inspect/create campaigns.
- `GET /api/autopilot/<id>`: state, resource accounting, progress, configuration and jobs.
- `POST /api/autopilot/<id>/{start,resume,pause,drain}`: lifecycle controls.
- `PATCH /api/autopilot/<id>/settings`: paused resource changes.
- `POST /api/autopilot/<id>/champions/restore`: restore retained champions at a boundary.
- `GET /api/autopilot/critics`: independent and embedded critic inventory, including architecture.
- `GET/POST /api/autopilot/matches`: policy/critic arena and self-play jobs.
- `POST /api/autopilot/matches/<id>/{pause,resume}`: match controls.

Match diagnostics average prediction error equally over completed games. They are diagnostics, not promotion evidence. Normal arenas remain policy-only, and the GUI links to the policy/critic match controls when those measurements are desired.
