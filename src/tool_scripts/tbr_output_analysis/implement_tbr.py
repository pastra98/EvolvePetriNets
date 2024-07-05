# %%
from tool_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_fitness

from neat import params, genome, initial_population
from pprint import pprint

# %%
alpha_g = load_genome("./tool_scripts/tbr_output_analysis/alpha_bootstrap.pkl")
inductive_g = load_genome("./tool_scripts/tbr_output_analysis/inductive_bootstrap.pkl")
ilp_g = load_genome("./tool_scripts/tbr_output_analysis/inductive_bootstrap.pkl")
show_genome(alpha_g)

# %%
eval_and_print_fitness(alpha_g, log)

# %%
from typing import Tuple, Dict, List
import random as rd

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
        self.inputs: Dict[str, Place] = dict()
        self.outputs: Dict[str, Place] = dict()
    
    def add_place(self, place_id: str, place: Place, is_input: bool):
        if is_input:
            self.inputs[place_id] = place
        else:
            self.outputs[place_id] = place
    
    def fire_and_get_quality(self):
        # enable trans if necessary, assign missing tokens
        missing = 0 if self.is_enabled() else self._enable()
        quality = (len(self.inputs), len(self.outputs), missing)
        self._fire()
        return quality


    def _fire(self):
        for p in self.inputs.values():
            p.remove_t()
        for p in self.outputs.values():
            p.insert_t()
    
    def is_enabled(self):
        return all([p.has_tokens() for p in self.inputs.values()])

    def _enable(self):
        # maybe bad design to have a fruitful function that mutates state, but this is practical
        produced = 0
        for p in self.inputs.values():
            if not p.has_tokens():
                p.insert_t()
                produced += 1
        return produced


class Petri:
    def __init__(self, places: Dict[str, Place], transitions: Dict[str, Transition]):
        self.places = places
        self.transitions = transitions
        self.hidden_transitions = {
            id: t for id, t in transitions.items() if not t.is_task
            }

    def replay_trace(self, trace: Tuple[str]):
        self.set_initial_marking()
        replay: List[Tuple[str, List[int]]] = []
        c, p, m, r = 0, 0, 0, 0 # consumed, produced, missing, remaining
        for task in trace:
            if task not in self.transitions:
                replay.append((task, None)) # Quality = None if nonexsistent
                continue
            # if trans exists, fire it (enable if necessary), save quality
            trans = self.transitions[task]
            if not trans.is_enabled():
                fired_hiddens, ht_c, ht_p = self.try_enable_trans_through_hidden(trans)
                if fired_hiddens:
                    replay += fired_hiddens
                    c += ht_c; p += ht_p
            quality = trans.fire_and_get_quality()
            c += quality[0]; p += quality[1]; m += quality[2]
            replay.append((task, quality))

        # at the end of the replay, count the missing
        r = sum([p.n_tokens for id, p in self.places.items() if id != "end"])

        return {"replay": replay, "consumed": c, "produced": p,
                "missing": m, "remaining": r}

    def try_enable_trans_through_hidden(self, trans: Transition):
        """In the current version of this method, I do not recurse into ht further back.
        Returns the list [(ht_id, quality), ...], as well as total tokens
        consumed and produced by firing all hidden trans required to enable trans.
        May return an empty list if enabling through hidden not possible.
        """
        fired_hiddens = []
        c, p = 0, 0 # consumed, produced (no missing bc. ht only fires if enabled)
        # find the places that miss tokens, then find potential hidden trans
        places_missing_token = {id for id, p in trans.inputs.items() if not p.has_tokens()}
        hidden_trans = list(self.hidden_transitions.items()); rd.shuffle(hidden_trans)
        for id, ht in hidden_trans: # shuffled to not give pref to any ht
            overlap = set(ht.outputs.keys()).intersection(places_missing_token)
            if overlap and ht.is_enabled(): # overlap contains the place(s) that connect to ht
                quality = ht.fire_and_get_quality()
                c += quality[0]; p += quality[1]
                fired_hiddens.append((id, quality))
                places_missing_token -= overlap
                if not places_missing_token:
                    break # stop if there are no more places missing a token
        return fired_hiddens, c, p

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
            t_dict[a.source_id].add_place(a.target_id, p, is_input=False)
        else: # p -> t
            p = p_dict[a.source_id]
            t_dict[a.target_id].add_place(a.source_id, p, is_input=True)
    return Petri(p_dict, t_dict)


# %%

variants = get_log_variants(log)

replay_g = inductive_g

replay_g.clear_cache()
show_genome(replay_g)

pnet = get_petri(replay_g)
my_replay = pnet.replay_log(variants)
pm4py_replay = get_aligned_traces(replay_g, log)

for mr, pr in zip(my_replay, pm4py_replay):
    pprint(mr); print()
    pprint(pr)
    print(80*"-", "\n")


# %%
_ = get_log_variants(log, debug=True)
pnet = show_genome(inductive_g)
# %%
# remove the arc that points to the hidden trans
for a_id, a in inductive_g.arcs.items():
    if a.target_id in inductive_g.transitions:
        if not inductive_g.transitions[a.target_id].is_task:
            # print(a.source_id)
            print(a_id)

inductive_g.remove_arcs(["7559713d-2310-42f9-ad43-561ce122759f"])
# %%
inductive_g.clear_cache()
# show_genome(inductive_g)
pnet = get_petri(inductive_g)
pnet.replay_log(variants)