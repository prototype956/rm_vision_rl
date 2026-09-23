"""为 Gym 和检查点评估提供一致、可复现的出生几何重试。"""


def reset_spawn(reset, rng, scenario, attempts):
    """为非法出生几何最多尝试 32 个派生种子，其他运行错误直接上抛。

    Args:
        reset: 接收种子和场景的重置回调。
        rng: 调用方的随机数生成器，用于派生仿真种子。
        scenario: 本次重置的场景参数。
        attempts: 原地追加各次尝试的种子及几何拒绝原因。

    Returns:
        首次成功的重置结果。
    """
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
    raise RuntimeError(f"no legal spawn in 32 attempts: {attempts}")
