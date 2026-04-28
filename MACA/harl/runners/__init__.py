"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.on_policy_ta_runner import OnPolicyTARunner
from harl.runners.on_policy_ia_runner import OnPolicyIARunner
from harl.runners.on_policy_coma_runner import OnPolicyComaRunner
from harl.runners.on_policy_vd_runner import OnPolicyVdRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "mappo": OnPolicyMARunner,
    "mappo_t": OnPolicyTARunner,
    "ippo": OnPolicyIARunner,
    "coma": OnPolicyComaRunner,
    "mappo_vd": OnPolicyVdRunner,
}
