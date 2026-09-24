"""Greedy policy mutations, selection leakage, durable search, and gate budgets."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from astro2.evolution import OPERATORS, choose_survivor, mutate, read_actor, recombine, write_actor
from astro2.experiment_control import atomic_json
from safetensors.numpy import load_file


@pytest.fixture
def weights():
    rng = np.random.default_rng(12)
    spec = dict(bootstrap_heads=2, families=3)
    result = {"__spec_json__": np.frombuffer(json.dumps(spec).encode(), dtype=np.uint8)}
    for head in range(2):
        result[f"head_outputs.{head}.weight"] = rng.normal(size=(3, 4)).astype(np.float32)
        result[f"head_outputs.{head}.bias"] = rng.normal(size=3).astype(np.float32)
    result["action_in.weight"] = rng.normal(size=(4, 6)).astype(np.float32)
    result["state_in.weight"] = rng.normal(size=(4, 6)).astype(np.float32)
    result["value_output.weight"] = rng.normal(size=(6, 4)).astype(np.float32)
    return result


@pytest.mark.parametrize("operator", OPERATORS)
def test_mutations_are_reproducible_portable_and_isolate_value(weights, tmp_path, operator):
    left = mutate(weights, seed=12, operator=operator, scale=0.01)
    repeat = mutate(weights, seed=12, operator=operator, scale=0.01)
    right = mutate(weights, seed=12, operator=operator, scale=0.01, sign=-1)
    assert all(np.array_equal(left[k], repeat[k]) for k in weights)
    for key in ("value_output.weight", "state_in.weight"):
        assert np.array_equal(left[key], weights[key])
    if operator != "head_mixture":
        for key in weights:
            assert np.allclose((left[key].astype(float) + right[key]) / 2, weights[key], atol=1e-6)
    if operator == "main_output":
        assert np.array_equal(
            left["head_outputs.0.weight"][1:], weights["head_outputs.0.weight"][1:]
        )
    assert any(not np.array_equal(left[k], weights[k]) for k in weights)
    path = tmp_path / "candidate.actor.npz"
    model = write_actor(left, path)
    loaded = read_actor(path)
    tensors = load_file(model)
    for key in weights:
        assert np.array_equal(left[key], loaded[key])
        if key != "__spec_json__":
            assert np.array_equal(left[key], tensors[key])


def rows(values):
    return [dict(pair=i, scores=[float(value), 0.0]) for i, value in enumerate(values)]


def test_recombination_uses_paired_signs_and_stays_within_proposals(weights):
    directions = [
        tuple(
            mutate(weights, seed=i, operator="all_outputs", scale=0.1, sign=sign)
            for sign in (1, -1)
        )
        for i in range(3)
    ]
    combined, recipe = recombine(weights, directions, [0.1, -0.3, 0], max_directions=2)
    assert [item["sign"] for item in recipe] == [-1, 1]
    assert [item["weight"] for item in recipe] == pytest.approx([0.75, 0.25])
    for key in ("state_in.weight", "value_output.weight", "__spec_json__"):
        assert np.array_equal(combined[key], weights[key])
    key = "head_outputs.0.weight"
    expected = 0.25 * directions[0][0][key] + 0.75 * directions[1][1][key]
    assert np.allclose(combined[key], expected)
    assert recombine(weights, directions, [0, 0, 0]) == (None, [])
    with pytest.raises(ValueError, match="finite"):
        recombine(weights, directions, [0, float("nan"), 0])


def test_selection_requires_fresh_matched_support_and_keeps_parent_on_ties():
    parent = rows([1] * 50 + [0] * 50)
    tie = rows([1] * 50 + [0] * 50)
    stronger = rows([1] * 75 + [0] * 25)
    assert choose_survivor([parent, tie])[0] == 0
    winner, comparisons = choose_survivor([parent, tie, stronger])
    assert winner == 2
    assert comparisons[2]["gain"] == 0.125
    with pytest.raises(ValueError, match="matching"):
        choose_survivor([parent, stronger[::-1]])


def test_small_precise_steps_have_no_arbitrary_effect_size_floor():
    parent = rows([0] * 5000)
    candidate = rows([1] * 10 + [0] * 4990)
    winner, comparisons = choose_survivor([parent, candidate], standard_errors=2)
    assert comparisons[1]["gain"] == 0.001
    assert winner == 1


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("evolution_training")


@pytest.fixture
def campaign(module, tmp_path, weights):
    actor = tmp_path / "source.actor.npz"
    model = write_actor(weights, actor)
    c = object.__new__(module.EvolutionCampaign)
    c.out = tmp_path
    c.settings = dict(
        seed=91,
        population=5,
        screen_pairs=4,
        selection_pairs=8,
        adoption_pairs=16,
        max_rounds=4,
        confirm_pairs=16,
        max_gate_pairs=131072,
    )
    c.state = dict(
        generation=0,
        stage=9,
        attempt=310,
        champion=str(actor),
        model=model,
        learner=str(actor),
        learner_model=model,
        champion_label="champion 9",
        branches=[],
        active_branch=None,
        events=[],
        history=[],
        gates=[],
        promotions=[],
        accepted_steps=0,
        stalled_generations=0,
        pending_gate=None,
        inherited_games=1000,
        games=1000,
        evaluation_counts={},
        search_games=0,
        verification_games=0,
        evaluated_games=0,
    )
    c.persist = lambda: atomic_json(tmp_path / "state.json", c.state)
    c.halt = lambda: False
    return c


def test_search_parent_and_selection_survive_restart_without_promoting(campaign, module):
    c = campaign
    branch = c.new_branch()
    c.propose(branch)
    assert len(branch["candidates"]) == 5
    identities = [item["sha256"] for item in branch["candidates"]]
    c.propose(branch)
    assert identities == [item["sha256"] for item in branch["candidates"]]
    branch["finalists"] = [0, 1]
    c.evaluate_set = lambda *a: (
        [dict(score=0.5), dict(score=0.7)],
        [rows([1, 0] * 50), rows([1] * 90 + [0] * 10)],
    )
    c.select(branch)
    assert c.state["accepted_steps"] == 0
    assert c.state["stage"] == 9 and c.state["attempt"] == 310
    assert c.state["champion"] == c.state["learner"]
    saved = json.loads((c.out / "state.json").read_text())
    assert saved["branches"][0]["status"] == "adopting"
    assert saved["learner"] == c.state["learner"]
    # Resume validates the same frozen nominee before touching the learner.
    c.state = saved
    c.adopt(c.state["branches"][0])
    assert c.state["accepted_steps"] == 1
    assert c.state["champion"] != c.state["learner"]
    saved = json.loads((c.out / "state.json").read_text())
    assert saved["branches"][0]["status"] == "confirming"
    c.state = saved
    c.matches = lambda *a, **k: dict(paused=False, score=0.52, paired_standard_error=0.002)
    c.confirm(c.state["branches"][0])
    assert c.state["pending_gate"]["attempt"] == 311
    assert c.state["accepted_steps"] == 1 and c.state["stage"] == 9
    assert c.state["pending_gate"]["seed"] != branch["seed"] + 10**9


def test_optimistic_selection_cannot_change_learner_if_fresh_step_fails(campaign):
    c = campaign
    branch = c.new_branch()
    c.propose(branch)
    branch["finalists"] = [0, 1]
    c.evaluate_set = lambda *a: (
        [dict(score=0.5), dict(score=0.7)],
        [rows([0, 1] * 50), rows([1] * 90 + [0] * 10)],
    )
    c.select(branch)
    assert branch["status"] == "adopting" and not branch["accepted"]
    c.evaluate_set = lambda *a: (
        [dict(score=0.5), dict(score=0.48)],
        [rows([0, 1] * 50), rows([0] * 55 + [1] * 45)],
    )
    c.adopt(branch)
    assert branch["status"] == "complete" and not branch["accepted"]
    assert c.state["learner"] == branch["parent"]
    assert c.state["accepted_steps"] == 0 and c.state["stalled_generations"] == 1
    assert c.state["pending_gate"] is None and c.state["attempt"] == 310


def test_small_confirmed_positive_score_can_nominate_without_weakening_gate(campaign):
    c = campaign
    branch = c.new_branch()
    c.propose(branch)
    branch.update(actor=branch["candidates"][1]["actor"], model=branch["candidates"][1]["model"])
    c.matches = lambda *a, **k: dict(paused=False, score=0.502, paired_standard_error=0.0005)
    c.confirm(branch)
    assert c.state["pending_gate"]["attempt"] == 311
    assert c.state["stage"] == 9


def test_recombined_proposal_goes_to_fresh_selection(campaign):
    c = campaign
    branch = c.new_branch()
    c.propose(branch)
    c.evaluate_set = lambda *a: ([dict(score=value) for value in [0.5, 0.55, 0.45, 0.48, 0.52]], [])
    c.screen(branch)
    candidate = branch["candidates"][-1]
    assert candidate["kind"] == "recombined"
    assert len(branch["candidates"]) - 1 in branch["finalists"]
    assert branch["status"] == "selecting"
    assert len(branch["screen_results"]) == 5
    assert Path(candidate["actor"]).exists() and Path(candidate["model"]).exists()
    assert c.state["learner"] == branch["parent"]


def test_partial_evaluation_counting_is_idempotent(campaign, module, monkeypatch):
    report = dict(pairs=128, paused=True)
    monkeypatch.setattr(module.Campaign, "matches", lambda *a, **k: dict(report))
    c = campaign
    folder = c.out / "branches/b0001/screen-00"
    for _ in range(2):
        c.matches("actor", "opponent", folder, 10, pairs=512)
    assert c.state["search_games"] == 256
    report.update(pairs=512, paused=False)
    c.matches("actor", "opponent", folder, 10, pairs=512)
    assert c.state["search_games"] == 1024
    assert c.state["games"] == 2024
    c.matches("actor", "opponent", c.out / "gate-009-0311", 11, attempt=311)
    assert c.state["verification_games"] == 1024
    assert c.state["games"] == 2024


def test_small_positive_gate_can_use_full_budget_without_relaxing_acceptance(campaign):
    c = campaign
    assert not c.gate_complete(dict(pairs=32768, score=0.5043, passed=False), 131072)
    assert not c.gate_complete(dict(pairs=8192, score=0.5001, passed=False), 131072)
    assert c.gate_complete(dict(pairs=8192, score=0.49, passed=False), 131072)
    assert c.gate_complete(dict(pairs=131072, score=0.51, passed=False), 131072)
    assert c.gate_complete(dict(pairs=4096, score=0.54, passed=True), 131072)


def test_finish_preserves_winner_and_evidence_prunes_only_own_rejects(campaign):
    c = campaign
    branch = c.new_branch()
    c.propose(branch)
    winner = branch["candidates"][1]
    c.state["learner"] = winner["actor"]
    branch.update(actor=winner["actor"], accepted=True)
    evidence = Path(branch["folder"]) / "evidence.json"
    evidence.write_text("{}")
    c.finish(branch)
    c.finish(branch)  # Crash after pruning before saving is safe to repeat.
    assert Path(winner["actor"]).exists() and Path(winner["model"]).exists()
    assert Path(branch["parent"]).exists() and evidence.exists()
    assert not Path(branch["candidates"][2]["actor"]).exists()
    assert c.state["generation"] == 1 and c.state["active_branch"] is None


def test_continuation_keeps_exact_eprocess_and_spent_attempts(campaign, module):
    from astro2.experiment_control import sha256
    from astro2.sequential import promotion_evidence

    c = campaign
    inherited = c.out / "old"
    folder = inherited / "gate-009-0310"
    folder.mkdir(parents=True)
    evidence_rows = [
        dict(pair=i, scores=([1.0, 1.0] if i < 4000 else [0.0, 0.0] if i < 7744 else [1.0, 0.0]))
        for i in range(32768)
    ]
    result = promotion_evidence(evidence_rows, 310)
    assert not result["passed"] and 0.5 < result["score"] < 0.505
    identity = dict(
        candidate_sha256=sha256(c.state["champion"]),
        opponent_sha256=sha256(c.state["champion"]),
        seed=77,
        attempt=310,
        maximum_pairs=131072,
        rules_version=1,
    )
    atomic_json(folder / "manifest.json", identity)
    raw = "".join(json.dumps(row) + "\n" for row in evidence_rows)
    (folder / "pairs.jsonl").write_text(raw)
    c.state["gates"] = [
        dict(**result, stage=9, seed=77, actor=c.state["champion"], model=c.state["model"])
    ]
    manifest = dict(settings=c.settings)
    module.continue_positive_gate(c.out, inherited, c.state, manifest)
    assert c.state["attempt"] == c.state["pending_gate"]["attempt"] == 310
    assert c.state["pending_gate"]["seed"] == 77
    assert (c.out / folder.name / "pairs.jsonl").read_text() == raw
    assert c.state["evaluation_counts"][folder.name] == 65536
    assert manifest["continued_gate"]["identity"] == identity
    assert result["alpha"] == 0.05 / (310 * 311)


def test_population_uses_separate_screen_selection_and_confirmation_streams(campaign):
    c = campaign
    branch = c.new_branch()
    c.propose(branch)
    calls = []

    def matches(actor, opponent, folder, seed, **kwargs):
        calls.append((str(folder), seed))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "pairs.jsonl").write_text(json.dumps(dict(pair=0, scores=[1.0, 0.0])) + "\n")
        return dict(paused=False, pairs=1, score=0.5)

    c.matches = matches
    c.evaluate_set(branch, [0, 1], "screen", 4)
    c.evaluate_set(branch, [0, 1], "select", 8)
    c.evaluate_set(branch, [0, 1], "adopt", 16)
    assert calls[0][1] == calls[1][1]
    assert calls[2][1] == calls[3][1]
    assert calls[4][1] == calls[5][1]
    assert len({call[1] for call in calls}) == 3
    assert branch["seed"] + 10**9 not in {call[1] for call in calls}
