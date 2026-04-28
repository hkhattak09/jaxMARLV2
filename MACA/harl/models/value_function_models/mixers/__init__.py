from harl.models.value_function_models.mixers.qmix import QMixer
from harl.models.value_function_models.mixers.vdn import VDNMixer

MIXER_REGISTRY = {
    "qmix": QMixer,
    "vdn": VDNMixer,
}