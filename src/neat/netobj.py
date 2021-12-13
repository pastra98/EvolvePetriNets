from dataclasses import dataclass
from copy import copy

@dataclass(frozen=True)
class GArc:
    id: int
    source_id: str
    target_id: str

    def get_copy(self):
        return copy(self)

@dataclass(frozen=True)
class GTrans:
    id: str
    is_task: bool

    def get_copy(self):
        return copy(self)


@dataclass(frozen=True)
class GPlace:
    id: str
    is_start: bool = False
    is_end: bool = False

    def get_copy(self):
        return copy(self)
