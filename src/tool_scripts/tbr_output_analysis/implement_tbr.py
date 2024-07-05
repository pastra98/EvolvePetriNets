# %%
from tool_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_fitness

from neat import params, genome, initial_population
from pprint import pprint

# %%
alpha_g = load_genome("./tool_scripts/tbr_output_analysis/alpha_bootstrap.pkl")
show_genome(alpha_g)

# %%
eval_and_print_fitness(alpha_g, log)

# %%
from typing import Tuple, Dict, List
from enum import Enum

class Place:
    def __init__(self):
        self.n_tokens = 0

    def remove_t(self):
        self.n_tokens -= 1

    def insert_t(self):
        self.n_tokens += 1

    def has_tokens(self) -> bool:
        return self.n_tokens > 0


class Transition:
    def __init__(self, is_task: bool):
        self.is_task = is_task
        self.inputs: List[Place] = []
        self.outputs: List[Place] = []
    
    def add_place(self, place: Place, is_input: bool):
        if is_input:
            self.inputs.append(place)
        else:
            self.outputs.append(place)
    
    def fire_and_get_quality(self):
        # enable trans if necessary, assign missing tokens
        missing = 0 if not self._is_enabled() else self._enable()
        quality = (len(self.inputs), len(self.outputs), missing)
        self._fire()
        return quality


    def _fire(self):
        for p in self.inputs:
            p.remove_t()
        for p in self.outputs:
            p.insert_t()
    
    def _is_enabled(self):
        return all([p.has_tokens() for p in self.inputs])

    def _enable(self):
        # maybe bad design to have a fruitful function that mutates state, but this is practical
        produced = 0
        for p in self.inputs:
            if not p.has_tokens():
                p.insert_t()
                produced += 1
        return produced


class Petri:
    def __init__(self, places: Dict[str, Place], transitions: Dict[str, Transition]):
        self.places = places
        self.transitions = transitions

    def replay_trace(self, trace: Tuple[str]):
        self.set_initial_marking()
        replay: List[Tuple[str, List[int]]] = []
        c, p, m, r = 0, 0, 0, 0 # consumed, produced, missing, remaining
        for task in trace:
            # check if trans exists
            if task not in self.transitions:
                replay.append((task, None)) # Quality = None if nonexsistent
                continue
            # if exists, fire it (enable if necessary), save quality
            trans = self.transitions[task]
            quality = trans.fire_and_get_quality()
            # update cpm
            c += quality[0]; p += quality[1]; m += quality[2]
            # append quality
            replay.append((task, quality))

        # at the end of the replay, count the missing
        r = sum([self.places[p].n_tokens for p in self.places if p != "end"])

        return {"replay": replay, "consumed": c, "produced": p,
                "missing": m, "remaining": r}

    def replay_log(self, variants: List[List[str]]):
        # TODO: factor in cardinalities of how many traces per variant
        log_replay: List[Dict] = []
        for trace in variants:
            log_replay.append(self.replay_trace(trace))
        return log_replay


    def set_initial_marking(self):
        for p in self.places.values():
            p.n_tokens = 0
        self.places["start"].n_tokens = 1


"""this will later be a method in the genome, still worth considering if
I maybe just extend the genome class to support this functionality
"""
def get_petri(g: genome.GeneticNet):
    # add places
    p_dict: Dict[str, Place] = {}
    for p in g.places.values():
        p_dict[p.id] = Place()
    # add trans
    t_dict: Dict[str, Transition] = {}
    for t in g.transitions.values():
        t_dict[t.id] = Transition(t.is_task)
    # connect them
    for a in g.arcs.values():
        if a.source_id in g.transitions: # t -> p
            p = p_dict[a.target_id]
            t_dict[a.source_id].add_place(p, is_input=False)
        else: # p -> t
            p = p_dict[a.source_id]
            t_dict[a.target_id].add_place(p, is_input=True)
    return Petri(p_dict, t_dict)


# %%
variants = get_log_variants(log)

pnet = get_petri(alpha_g)
my_replay = pnet.replay_log(variants)
pm4py_replay = get_aligned_traces(alpha_g, log)

for mr, pr in zip(my_replay, pm4py_replay):
    pprint(mr)
    print()
    pprint(pr)
    print(80*"-", "\n")

# %%
