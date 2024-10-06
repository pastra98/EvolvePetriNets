import numpy as np
from typing import Dict, List, Tuple
from functools import cache
from statistics import mean
import random as rd

class PetriNetNP:
    def __init__(self, places: List[str], transitions: List[str], log: Dict[str, dict]):
        self.places = places
        self.transitions = transitions
        self.log = log
        
        self.num_places = len(places)
        self.num_transitions = len(transitions)
        
        # Initialize matrices
        self.input_matrix = np.zeros((self.num_places, self.num_transitions), dtype=int)
        self.output_matrix = np.zeros((self.num_places, self.num_transitions), dtype=int)
        self.marking = np.zeros(self.num_places, dtype=int)
        
        # Map place and transition names to indices
        self.place_to_index = {place: i for i, place in enumerate(places)}
        self.transition_to_index = {trans: i for i, trans in enumerate(transitions)}
        
        # Set of task transitions (to be populated when adding arcs)
        self.task_transitions = set()

    def add_arc(self, place: str, transition: str, is_input: bool, is_task: bool = True):
        p_idx = self.place_to_index[place]
        t_idx = self.transition_to_index[transition]
        
        if is_input:
            self.input_matrix[p_idx, t_idx] = 1
        else:
            self.output_matrix[p_idx, t_idx] = 1
        
        if is_task:
            self.task_transitions.add(transition)

    def set_initial_marking(self, initial_place: str):
        self.marking = np.zeros(self.num_places, dtype=int)
        self.marking[self.place_to_index[initial_place]] = 1

    def is_enabled(self, transition: str) -> bool:
        t_idx = self.transition_to_index[transition]
        return np.all(self.marking >= self.input_matrix[:, t_idx])

    def fire_transition(self, transition: str) -> Tuple[int, int, int]:
        t_idx = self.transition_to_index[transition]
        
        consumed = np.sum(self.input_matrix[:, t_idx])
        produced = np.sum(self.output_matrix[:, t_idx])
        
        if not self.is_enabled(transition):
            missing = np.sum(np.maximum(self.input_matrix[:, t_idx] - self.marking, 0))
            self.marking = np.maximum(self.marking, self.input_matrix[:, t_idx])
        else:
            missing = 0
        
        self.marking += self.output_matrix[:, t_idx] - self.input_matrix[:, t_idx]
        
        return consumed, produced, missing

    def get_enabled_transitions(self) -> List[str]:
        enabled = np.all(self.marking[:, np.newaxis] >= self.input_matrix, axis=0)
        return [trans for trans, idx in self.transition_to_index.items() if enabled[idx]]

    def _try_enable_trans_through_hidden(self, transition: str) -> List[Tuple[str, Tuple[int, int, int], List[str]]]:
        fired_hiddens = []
        t_idx = self.transition_to_index[transition]
        places_missing_token = set(np.where(self.marking < self.input_matrix[:, t_idx])[0])
        
        hidden_trans = [t for t in self.transitions if t not in self.task_transitions]
        rd.shuffle(hidden_trans)
        
        for ht in hidden_trans:
            ht_idx = self.transition_to_index[ht]
            overlap = set(np.where(self.output_matrix[:, ht_idx] > 0)[0]).intersection(places_missing_token)
            
            if overlap and self.is_enabled(ht):
                quality = self.fire_transition(ht)
                enabled_t = self.get_enabled_transitions()
                fired_hiddens.append((ht, quality, enabled_t))
                places_missing_token -= overlap
                if not places_missing_token:
                    break

        return fired_hiddens

    def replay_log(self) -> List[dict]:
        log_replay = []
        for trace in self.log["variants"]:
            trace_replay = self.replay_trace(trace)
            trace_fitness = self._get_trace_fitness(trace_replay)
            log_replay.append({**trace_replay, "fitness": trace_fitness})
        return log_replay

    def replay_trace(self, trace: Tuple[str]) -> dict:
        self.set_initial_marking("start")
        replay = []
        c, p, m, r = 0, 0, 0, 0  # consumed, produced, missing, remaining

        for task in trace:
            if task not in self.transitions:
                replay.append((task, (0, 0, 0), []))
                continue

            if not self.is_enabled(task):
                fired_hiddens = self._try_enable_trans_through_hidden(task)
                for fh in fired_hiddens:
                    quality = fh[1]
                    c += quality[0]
                    p += quality[1]
                    replay.append(fh)

            quality = self.fire_transition(task)
            c += quality[0]
            p += quality[1]
            m += quality[2]
            enabled_t = self.get_enabled_transitions()
            replay.append((task, quality, enabled_t))

        r = np.sum(self.marking) - self.marking[self.place_to_index["end"]]
        return {"replay": replay, "consumed": c, "produced": p, "missing": m, "remaining": r}

    def _get_trace_fitness(self, trace_replay: dict) -> float:
        # Constants
        MULT = 1.5
        MAX_PTS = 1
        NO_INPUTS_PENAL = 0.5
        NO_OUTPUTS_PENAL = 0.5
        MISSING_PENAL = 0.4
        REMAINING_PENAL = 1

        fitness, n_flawless = 0, 0

        for firing_info in trace_replay["replay"]:
            trans, quality = firing_info[0], firing_info[1]
            if trans not in self.task_transitions:
                continue

            pts = MAX_PTS
            consumed, produced, missing = quality

            if not consumed:
                pts -= NO_INPUTS_PENAL
            if not produced:
                pts -= NO_OUTPUTS_PENAL
            if missing:
                pts -= MISSING_PENAL * (consumed / missing)

            if pts == MAX_PTS:
                n_flawless += 1
            else:
                n_flawless = 0
            fitness += pts * max(1, MULT * (n_flawless - 1))

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


    def _get_node_degrees(self) -> Dict[str, Dict[str, List[int]]]:
        p_degrees = {p: [np.sum(self.input_matrix[i, :]), np.sum(self.output_matrix[i, :])] 
                     for i, p in enumerate(self.places)}
        t_degrees = {t: [np.sum(self.input_matrix[:, i]), np.sum(self.output_matrix[:, i])] 
                     for i, t in enumerate(self.transitions)}
        return {"places": p_degrees, "transitions": t_degrees}


    def _mean_by_max_simplicity(self, node_degrees: Dict[str, Dict[str, List[int]]]) -> float:
        p_degrees = [sum(p) for p in node_degrees["places"].values()]
        t_degrees = [sum(t) for t in node_degrees["transitions"].values()]
        all_degrees = [d for d in p_degrees + t_degrees if d > 0]
        return mean(all_degrees) / max(all_degrees) if all_degrees else 0


    def _io_connectedness_simplicity(self, node_degrees: Dict[str, Dict[str, List[int]]]) -> float:
        io_connected = [p for p in node_degrees["places"].values() if p[0] > 0 and p[1] > 0]
        return len(io_connected) / self.num_places if self.num_places > 0 else 0


    def _precision(self, replay: List[Dict]) -> float:
        all_enabled = sum(len(q[2]) for trace in replay for q in trace["replay"])
        return 1 / max(all_enabled, 1)


    def _transitions_by_tokens(self, replay: List[Dict]) -> float:
        all_produced = sum(q[1][1] for trace in replay for q in trace["replay"])
        tbt = self.num_transitions / max(all_produced, 1)
        return min(tbt, 1)


    def _fraction_task_trans(self) -> float:
        if self.task_transitions:
            return len(self.task_transitions) / self.num_transitions
        else:
            return 0


    def _num_arcs(self) -> float:
        total_arcs = np.sum(self.input_matrix) + np.sum(self.output_matrix)
        return 1 / max(total_arcs, 1)


    def _trans_place_ratio(self) -> float:
        return min(self.num_transitions / self.num_places, 1) if self.num_places > 0 else 0


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


def max_replay_fitness(len_and_card: tuple):
    """Uses closed form for sum of multipliers, distributes multiplier across them
    works under the assumption that MAX_PTS == 1
    """
    # TODO: import the constants again
    MULT = 1.5
    fit = 0
    for tlen, cardinality in len_and_card:
        trace_fit = 1 + MULT * ((tlen * (tlen-1)) / 2)
        fit += trace_fit * cardinality
    return fit
