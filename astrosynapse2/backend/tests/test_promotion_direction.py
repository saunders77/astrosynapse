import json

import numpy as np
from astro2.config import RunConfig
from astro2.promotion_direction import (
    build_promotion_direction,
    load_promotion_direction,
    save_promotion_direction,
)
from astro2.storage import Store
from safetensors.numpy import save_file


def _checkpoint(store, run_id, tmp_path, label, games, weights, *, champion=False):
    path = tmp_path / f"{label}.safetensors"
    save_file({"layer.weight": np.asarray(weights, dtype=np.float32)}, path)
    path.with_suffix(".safetensors.json").write_text(json.dumps({}))
    return store.add_checkpoint(
        run_id=run_id,
        label=label,
        path=str(path),
        actor_path=None,
        games=games,
        champion=champion,
    )


def _promotion(store, candidate, champion, score):
    job = store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": True,
            "trainer_scheduled": True,
            "promotion_tier": "full",
        },
        result={
            "model_a_score": score,
            "promotion": {"promoted": True},
        },
    )
    store.update_arena_job(job["id"], status="complete", result=job["result"])


def test_direction_keeps_only_coordinates_that_agree_across_promotions(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.astro5_search())
    first = _checkpoint(store, run["id"], tmp_path, "first", 0, [0, 0, 0, 0], champion=True)
    second = _checkpoint(store, run["id"], tmp_path, "second", 10, [1, -1, 1, -1])
    third = _checkpoint(store, run["id"], tmp_path, "third", 20, [3, -3, 0, 0])
    _promotion(store, second, first, 0.55)
    _promotion(store, third, second, 0.53)

    direction, metadata = build_promotion_direction(
        store,
        third["id"],
        maximum_transitions=2,
        minimum_sign_agreement=1.0,
        recent_decay=1.0,
    )

    np.testing.assert_array_equal(direction["layer.weight"][2:], np.zeros(2))
    assert direction["layer.weight"][0] > 0
    assert direction["layer.weight"][1] < 0
    assert metadata["transition_count"] == 2
    assert metadata["retained_coordinate_fraction"] == 0.5

    path, saved_metadata = save_promotion_direction(
        store,
        third["id"],
        tmp_path / "direction.npz",
        maximum_transitions=2,
        minimum_sign_agreement=1.0,
        recent_decay=1.0,
    )
    loaded, loaded_metadata = load_promotion_direction(path)
    np.testing.assert_allclose(loaded["layer.weight"], direction["layer.weight"])
    assert loaded_metadata == saved_metadata

