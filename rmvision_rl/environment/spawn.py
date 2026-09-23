"""Identical deterministic geometry retries for Gym and checkpoint evaluation."""


def reset_spawn(reset, rng, scenario, attempts):
    for _ in range(32):
        seed = int(rng.integers(0, 2**63))
        attempt = {"seed": seed}
        attempts.append(attempt)
        try:
            return reset(seed, scenario)
        except RuntimeError as error:
            if not str(error).startswith("invalid seed "):
                raise
            attempt["error"] = str(error)
    raise RuntimeError(f"no legal static spawn in 32 attempts: {attempts}")
