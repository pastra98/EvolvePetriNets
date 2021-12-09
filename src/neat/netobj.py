from dataclasses import dataclass
from copy import deepcopy

@dataclass(frozen=False) # think about if we can make it true -> would require copying shit
class GArc:
    id: int
    source_id: str
    target_id: str
    n_arcs: int = 1

    def get_copy(self):
        return deepcopy(self)

@dataclass(frozen=True)
class GTrans:
    id: str
    is_task: bool

    def get_copy(self):
        return deepcopy(self)


@dataclass(frozen=True)
class GPlace:
    id: str
    is_start: bool = False
    is_end: bool = False

    def get_copy(self):
        return deepcopy(self)
