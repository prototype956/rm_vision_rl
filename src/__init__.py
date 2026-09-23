"""提供 RoboMaster 控制适配接口，并在 Gymnasium 可用时注册环境。"""
try:
    from gymnasium.envs.registration import register
except ModuleNotFoundError as error:
    if error.name != "gymnasium":
        raise
else:
    register(id="RMStaticFire-v0",
             entry_point="src.environment.static_fire:StaticFireEnv")
    register(id="RMRotationFire-v0",
             entry_point="src.environment.rotation_fire:RotationFireEnv")
