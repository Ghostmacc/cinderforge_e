from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class LevelSpec:
    units: int
    neurons_per: int

@dataclass(frozen=True)
class UnitAddr:
    level: int
    index: int

@dataclass
class Edge:
    src: UnitAddr
    dst: UnitAddr
    kind: str = "lateral"

@dataclass
class Futures:
    vertical: List[Edge] = field(default_factory=list)
    lateral: List[Edge] = field(default_factory=list)

@dataclass
class GraphSpec:
    id: str
    seed: int = 0
    levels: List[LevelSpec] = field(default_factory=list)
    max_vertical_skip: int = 1
    max_lateral_right: int = 2
    lateral_gate_after: int = 1
    resources: Optional[Dict] = field(default_factory=dict)