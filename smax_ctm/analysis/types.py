from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AnalysisCheckpoint:
    path: str
    config: Dict[str, Any]
    actor_params: Dict[str, Any]


@dataclass
class EpisodeCollection:
    metadata: Dict[str, Any]
    episodes: List[Dict[str, Any]]
