from starrealms_selfplay import choose_with_saved_policy


def choose(playerName, options, knownGameState):
    return choose_with_saved_policy(
        playerName,
        options,
        knownGameState,
        run_name="default",
        checkpoint="latest",
        deterministic=True,
    )
