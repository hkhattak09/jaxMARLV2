from .environments import (
    SimpleMPE,
    AssemblyEnv,
    SUBMODULE_ENVIRONMENTS
)


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    if env_id == "MPE_simple_v3":
        env = SimpleMPE(**env_kwargs)
    elif env_id == "assembly":
        env = AssemblyEnv(**env_kwargs)

    return env

registered_envs = [
    "MPE_simple_v3",
    "assembly",
]
