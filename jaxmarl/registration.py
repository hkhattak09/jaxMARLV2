from .environments import (
    SMAX,
    HeuristicEnemySMAX,
    LearnedPolicyEnemySMAX,
)

SUBMODULE_ENVIRONMENTS = False

def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    # 3. SMAX
    if env_id == "SMAX":
        env = SMAX(**env_kwargs)
    elif env_id == "HeuristicEnemySMAX":
        env = HeuristicEnemySMAX(**env_kwargs)
    elif env_id == "LearnedPolicyEnemySMAX":
        env = LearnedPolicyEnemySMAX(**env_kwargs)

    return env

registered_envs = [
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
]
