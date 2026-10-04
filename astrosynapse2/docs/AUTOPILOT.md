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

## October 3 recovery revision

Campaign `828813dc45d24cf086e019c6537d1b02` stalled after `policy-00004` at
01:31 local time. The next 26 policy blocks produced 11 screen rejections and
15 inconclusive gates. Those gates averaged 50.236% against the champion
(range 49.863–50.717%). Policy entropy fell from about 0.117 at the champion
block to 0.070 in the latest lane. Continuing both lanes indefinitely with
zero entropy regularization was not yielding stronger candidates.

Critic fitting used only 2,500 games; most fits selected epoch zero or one.
Since the last promotion, 13 critics failed prediction qualification and 11
passed prediction checks but failed playing-strength qualification. The latter
averaged a −0.054 percentage-point policy-probe difference. Each probe trained
for only two updates. Better predictions had not demonstrated stronger play.

The installed revision applies the following settings to future tasks:

| Setting | Previous | Recovery |
| --- | --- | --- |
| Lane continuation | Unlimited | Rebase on champion after three blocks |
| Sampling temperature | 0.10 in both lanes | 0.15 / 0.25 |
| Entropy weight | 0 | 0.01 / 0.02 |
| Critic replay | 2,500 games | 20,000 games |
| Recent replay | Last 2,000 games | Sample 16,000 from last 100,000; 4,000 older games |
| Critic learning rate | 0.0003 | 0.00003 |
| Critic probe training | 2,000 games per arm/seed | 10,000 games per arm/seed |
| Critic probe evaluation | 4,000 pairs per seed | 12,000 pairs per seed |
| Policy nomination screen | 512 pairs, score >50% | 2,048 pairs, score >50.5% |
| Policy gate ceiling | 12,000 pairs | 50,000 pairs |
| Historical panel | 4,000 pairs per opponent | 12,000 pairs per opponent |

Learning rate remains 0.000002, with 20,000-game policy blocks, two epochs per
update, six workers and the 80/20 training allocation. Both critic probe arms
use identical exploration settings for their training seed. Legacy lane tips
restart from the verified champion on their next turn. Gates with a nonpositive
observed gain stop after at least 8,192 pairs; acceptance still requires the
original confidence bound and historical non-regression checks. Screening is
exploratory and cannot certify a promotion.

Prediction-only critic rejections no longer spend a playing-strength attempt.
The separate round counter preserves unique job/seed identities. All 27 prior
critic attempts and 20 prior policy attempts remain spent; neither alpha schedule
resets. Future qualified critics spend their attempt before probe training.

An isolated fit on existing replay selected epoch five and reduced held-out log
loss from 0.491344 to 0.481736 and Brier error from 0.166175 to 0.162494 across
2,014 games. Historical log loss improved from 0.475177 to 0.461099. These are
prediction diagnostics on reusable replay, not independent strength evidence.
The pilot critic was not installed as champion. A real 20-game policy smoke run
with the exploratory recipe completed without truncations, followed by an
eight-pair evaluation. It verifies execution, not playing strength.

`scripts/maintain_autopilot_runtime.py` requires a paused block boundary and the
worker lock, checks the prior runtime identity, backs up code/config/state, and
updates only the three Autopilot implementation files. Installation verified
261 protected state, champion and evaluation artifacts stayed byte identical.
The preserved runtime and configuration support rollback. Recovery settings
remain opt-in; old configurations retain their original defaults.

The same campaign resumes with 24 additional active hours (46.445994 cumulative)
and a 50 GiB campaign storage ceiling. Diagnosis, settings, installation audit,
critic pilot and policy smoke results are in `data/autopilot-recovery-20261003/`.
New promotions remain conditional on fresh evidence; none is manufactured by
the migration.
