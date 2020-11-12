from dataclasses import dataclass, field
from enum import Enum

from typing import List


@dataclass
class CoMatchConfig:
    match_threshold: float = 5
    location_attr: str = "location"


@dataclass
class ReconstructionConfig:
    comatch: CoMatchConfig = CoMatchConfig()
    new_edge_len: float = 1

