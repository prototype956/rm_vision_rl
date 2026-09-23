"""RM control adapters; Gym registration is available when Gymnasium is installed."""
try:
    from gymnasium.envs.registration import register
except ModuleNotFoundError as error:
    if error.name != "gymnasium":
        raise
else:
    register(id="RMStaticFire-v0",
             entry_point="rmvision_rl.environment.static_fire:StaticFireEnv")
