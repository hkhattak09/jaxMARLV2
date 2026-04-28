"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.mappo_t import MAPPOTrans

ALGO_REGISTRY = {
    "happo": HAPPO,
    "mappo": MAPPO,
    "mappo_t": MAPPOTrans,
    "ippo": MAPPO,
    "coma": MAPPOTrans,
    "mappo_vd": MAPPOTrans,
}
