from neat import params

from functools import cache
from typing import Tuple, Dict, List
from statistics import mean
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
    
    def __str__(self):
        inputs_str = ', '.join(self.inputs.keys())
        outputs_str = ', '.join(self.outputs.keys())
        return f"Transition(is_task={self.is_task}, inputs=[{inputs_str}], outputs=[{outputs_str}])"

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
        """Checks if all places have tokens, trans are also enabled if they have no inputs,
        but a missing penalty will be added
        """
        return all([p.has_tokens() for p in self.inputs.values()])


    def _enable(self):
        # maybe bad design to have a fruitful function that mutates state, but this is practical
        produced = 0
        for p in self.inputs.values():
            if not p.has_tokens():
                p.insert_t()
                produced += 1
        return produced

# constants for replay
MULT = 1.5
MAX_PTS = 1
NO_INPUTS_PENAL = 0.5
NO_OUTPUTS_PENAL = 0.5
MISSING_PENAL = 0.4
REMAINING_PENAL = 1

class Petri:
    def __init__(
            self,
            places: Dict[str, Place],
            transitions: Dict[str, Transition],
            log: Dict[str, dict]):
        
        self.transitions = transitions
        self.places = places
        self.hidden_transitions = {
            t_id: t for t_id, t in transitions.items() if not t.is_task
            }
        self.log = log

# -------------------- REPLAY -------------------- 

    def replay_log(self) -> dict:
        """Replay every trace of the log
        """
        log_replay: List[dict] = []
        for trace in self.log["variants"]:
            trace_replay = self._replay_trace(trace)
            trace_fitness = self._get_trace_fitness(trace_replay)
            log_replay.append(trace_replay | {"fitness": trace_fitness})
        return log_replay


    def _replay_trace(self, trace: Tuple[str]):
        """Replay a trace, return cpmr counts
        """
        self.set_initial_marking()
        replay: List[Tuple[str, List[int]]] = []
        c, p, m, r = 0, 0, 0, 0 # consumed, produced, missing, remaining
        for task in trace:
            if task not in self.transitions:
                replay.append((task, (0, 0, 0), []))
                continue
            trans = self.transitions[task]
            # if trans exists, check first if it needs to be enabled through hiddens
            if not trans.is_enabled():
                fired_hiddens = self._try_enable_trans_through_hidden(trans)
                if fired_hiddens:
                    for fh in fired_hiddens:
                        quality = fh[1]
                        c += quality[0]; p += quality[1] # no missings, ht is only fired if is enabled
                        replay.append(fh)
            # fire trans (if it cannot be enabled, add missing tokens
            quality = trans.fire_and_get_quality()
            c += quality[0]; p += quality[1]; m += quality[2]
            enabled_t = [t_id for t_id, t in self.transitions.items() if t.is_enabled()]
            replay.append((task, quality, enabled_t))
        # at the end of the replay, count the missing
        r = sum([p.n_tokens for p_id, p in self.places.items() if p_id != "end"])
        return {"replay": replay, "consumed": c, "produced": p,
                "missing": m, "remaining": r}


    def _try_enable_trans_through_hidden(self, trans: Transition):
        """In the current version of this method, I do not recurse into ht further back.
        Returns the list [(ht_id, quality), ...], as well as total tokens
        consumed and produced by firing all hidden trans required to enable trans.
        May return an empty list if enabling through hidden not possible.
        """
        fired_hiddens = []
        # find the places that miss tokens, then find potential hidden trans
        places_missing_token = {id for id, p in trans.inputs.items() if not p.has_tokens()}
        hidden_trans = list(self.hidden_transitions.items())
        rd.shuffle(hidden_trans) # shuffled to not give pref to any ht
        for t_id, ht in hidden_trans:
            overlap = set(ht.outputs.keys()).intersection(places_missing_token)
            if overlap and ht.is_enabled(): # overlap contains the place(s) that connect to ht
                quality = ht.fire_and_get_quality()
                enabled_t = [t_id for t_id, t in self.transitions.items() if t.is_enabled()]
                fired_hiddens.append((t_id, quality, enabled_t))
                places_missing_token -= overlap
                if not places_missing_token:
                    break # stop if there are no more places missing a token
        return fired_hiddens


    def set_initial_marking(self):
        """Clear net of tokens, set 1 in token source
        """
        for p in self.places.values():
            p.n_tokens = 0
        self.places["start"].n_tokens = 1


    def _get_trace_fitness(self, trace_replay: dict):
        """Use replay stats to calculate the fitness of trace, reward successive
        good executions (none missing, trans has output places)
        """
        fitness, n_flawless = 0, 0

        for firing_info in trace_replay["replay"]:
            trans, quality = firing_info[0], firing_info[1]
            # do not add pts for firing hidden trans
            if trans in self.transitions and not self.transitions[firing_info[0]].is_task:
                continue
            # if a task trans was fired, evaluate how many pts it generated
            pts = MAX_PTS
            consumed, produced, missing = quality[0], quality[1], quality[2]
            # subtract input/output penalties
            if not consumed: pts -= NO_INPUTS_PENAL
            if not produced: pts -= NO_OUTPUTS_PENAL
            # missing token penalty
            if missing:
                pts -= MISSING_PENAL * (consumed/missing)
            # update multiplier
            if pts == MAX_PTS:
                n_flawless += 1
            else:
                n_flawless = 0
            fitness += pts * max(1, MULT * (n_flawless-1))
        # penalize for remaining tokens
        fitness -= REMAINING_PENAL * trace_replay["remaining"]
        return fitness

# -------------------- METRICS -------------------- 

    def _aggregate_trace_fitness(self, log_replay: list):
        """Aggregate fitness from all traces of replay, divides by max achievable fitness.
        """
        agg_fitness = 0
        for replay, cardinality in zip(log_replay, self.log["variants"].values()):
            agg_fitness += replay["fitness"] * cardinality
        max_fit = max_replay_fitness( # tuples len of trace with cardinality
            tuple([(len(t), cardinality) for t, cardinality in self.log["variants"].items()])
            )
        return agg_fitness / max_fit


    def _get_node_degrees(self):
        """Helper func to calculate degrees of nodes and places
        """
        p_degrees = {p: [0, 0] for p in self.places}
        t_degrees = {t: [0, 0] for t in self.transitions}
        for t in self.transitions.values():
            t_degrees[t] = [len(t.inputs), len(t.outputs)]
            for p in list(t.inputs.keys()):
                p_degrees[p][0] += 1
            for p in list(t.outputs.keys()):
                p_degrees[p][1] += 1
        return {"places": p_degrees, "transitions": t_degrees}
    

    def _mean_by_max_simplicity(self, node_degrees):
        """Idea is to penalize the max degree being further away from the mean
        """
        # TODO: could also use variance maybe?
        p_degrees = [sum(p) for p in list(node_degrees["places"].values())]
        t_degrees = [sum(t) for t in list(node_degrees["transitions"].values())]
        all_degrees = [d for d in t_degrees + t_degrees if d > 0]
        return mean(all_degrees) / max(all_degrees)
    

    def _io_connectedness_simplicity(self, node_degrees):
        """Fraction of places that have both inputs and outputs
        """
        # could do similar thing for transitions, but for now just disable
        io_connected = [p for p in node_degrees["places"].values() if p[0]>0 and p[1]>0]
        return len(io_connected) / len(self.places)


    def _precision(self, replay):
        """Precision formula from ProDiGen miner
        """
        all_enabled = 0
        for trace in replay:
            for q in trace["replay"]:
                all_enabled += len(q[2])
        return 1 / max(all_enabled, 1)


    def _transitions_by_tokens(self, replay):
        """Penalize the model for producing lots of tokens
        """
        all_produced = 0
        for trace in replay:
            for q in trace["replay"]:
                all_produced += q[1][1]
        tbt = len(self.transitions) / max(all_produced, 1)
        return min(tbt, 1)


    def _fraction_task_trans(self):
        """get fraction of task trans from overall trans, i.e. penalize for every hidden trans
        """
        my_task_trans = [t for t in self.transitions.values() if t.is_task]
        if my_task_trans:
            return len(my_task_trans) / len(self.transitions)
        else:
            return 0 # TODO: this is bullshit, but basically if a model has 0 task trans its fitness is 0


    def _num_arcs(self):
        """just 1 divided by number of arcs
        """
        return 1 / sum([len(t.inputs) + len(t.outputs) for t in self.transitions.values()])


    def _trans_place_ratio(self):
        """ratio of transitions to places, capped to [0-1]
        """
        return min(len(self.transitions) / len(self.places), 1)


    def _over_enabled_transitions(self, log_replay):
        """Whenever more trans are enabled than indicated in dfg, increase denominator
        """
        s_dict = {t: [] for t in self.log["footprints"]["activities"]}
        for s in self.log["footprints"]["dfg"]:
            s_dict[s[0]].append(s[1])

        enabled_too_much = 0
        for trace_replay in log_replay:
            # if a fired trans is hidden, count its enabled trans towards next task
            enabled_by_hiddens = []
            for firing_info in trace_replay["replay"]:
                trans = firing_info[0]
                enables = firing_info[2]
                # hidden trans
                if trans not in s_dict:
                    enabled_by_hiddens += enables
                # task trans
                else:
                    should_enable = s_dict[trans]
                    enabled_by_task = enables + enabled_by_hiddens
                    if len(enabled_by_task) > len(should_enable):
                        enabled_too_much += len(enabled_by_task) - len(should_enable)
                        # print(f"{trans} enabled too much:\n{enabled_by_task}\n")
                    enabled_by_hiddens.clear() # reset enabled_by_hiddens

        return 1 / max(enabled_too_much, 1) # if none were enabled too much, perfect score


    def evaluate(self):
        """Get all fitness metrics
        """
        # get replay & node degrees
        replay = self.replay_log()
        node_degrees = self._get_node_degrees()
        # simplicity / generalization metrics
        metrics = {
            "aggregated_replay_fitnesss": self._aggregate_trace_fitness(replay),
            "io_connectedness": self._io_connectedness_simplicity(node_degrees),
            "mean_by_max": self._mean_by_max_simplicity(node_degrees),
            "trans_by_tasks": self._fraction_task_trans(),
            "precision": self._precision(replay),
            "trans_by_tokens": self._transitions_by_tokens(replay),
            "over_enabled_trans": self._over_enabled_transitions(replay),
            "num_arcs": self._num_arcs(),
            "trans_by_places": self._trans_place_ratio()
        }
        return {
            "replay": replay,
            "metrics": metrics
        }

@cache
def max_replay_fitness(len_and_card: tuple):
    """Uses closed form for sum of multipliers, distributes multiplier across them
    works under the assumption that MAX_PTS == 1
    """
    if MAX_PTS != 1:
        raise Exception("this function was built under the assumptio MAX_PTS == 1")
    fit = 0
    for tlen, cardinality in len_and_card:
        trace_fit = 1 + MULT * ((tlen * (tlen-1)) / 2)
        fit += trace_fit * cardinality
    return fit
