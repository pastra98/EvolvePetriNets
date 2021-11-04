class GArc:
    def __init__(self, id, source_id, target_id, n_arcs=1):
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.n_arcs = n_arcs
        self.pm4py_obj = None

class GTrans:
    def __init__(self, id, is_task):
        self.id = id
        self.is_task = is_task
        self.pm4py_obj = None

class GPlace:
    def __init__(self, id, is_end=False, is_start=False):
        self.id = id
        self.is_start = is_start
        self.is_end = is_end
        self.pm4py_obj = None
