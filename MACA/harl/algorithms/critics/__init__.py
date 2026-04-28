"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.trans_v_critic import TransVCritic
from harl.algorithms.critics.coma_q_critic import ComaQCritic
from harl.algorithms.critics.value_decomp_critic import ValueDecompQCritic

CRITIC_REGISTRY = {
    "happo": VCritic,
    "mappo": VCritic,
    "mappo_t": TransVCritic,
    "ippo": VCritic,
    "coma": ComaQCritic,
    "mappo_vd": ValueDecompQCritic,
}
