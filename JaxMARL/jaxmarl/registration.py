from .environments import SimpleMPE, SimpleSpreadMPE, AssemblyEnv


def make(env_id: str, **env_kwargs):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    if env_id == "MPE_simple_v3":
        env = SimpleMPE(**env_kwargs)
    elif env_id == "MPE_simple_spread_v3":
        env = SimpleSpreadMPE(**env_kwargs)
    elif env_id == "assembly_v0":
        env = AssemblyEnv(**env_kwargs)

    return env


registered_envs = [
    "MPE_simple_v3",
    "MPE_simple_spread_v3",
    "assembly_v0",
]
