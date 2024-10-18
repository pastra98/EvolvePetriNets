from __future__ import annotations
from typing import Tuple, Dict, List
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

from pm4py.visualization.petri_net.common.visualize import graphviz_visualization

from pm4py.algo.conformance.tokenreplay.variants.token_replay import apply as get_replayed_traces
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import evaluate as get_fitness_dict
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as get_precision
from pm4py.algo.evaluation.generalization.variants.token_based import get_generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as get_simplicity
from pm4py.algo.analysis.woflan.algorithm import apply as get_soundness

import neatutils.fitnesscalc as fc
# import neatutils.fitnesscalc_np as fc_np # <- numpy implementation, abandoned POC
from neat import params

import random as rd
import numpy as np
import traceback
import itertools

from copy import copy
from functools import cache
from collections import Counter
from math import sqrt
from graphviz import Digraph
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class GArc:
    source_id: str
    target_id: str
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class GTrans:
    id: str = field(default_factory=lambda: str(uuid4()))
    is_task: bool = False


@dataclass(frozen=True)
class GPlace:
    id: str = field(default_factory=lambda: str(uuid4()))


class GeneticNet:
    def __init__(
            self,
            transitions: dict,
            places: dict,
            arcs: dict,
            parent_id=None,
            task_list=[],
            pop_component_tracker=None) -> None:
        """transitions, places and arcs must either be dicts containing valid id: netobj
        key-value pairings, or an empty dict.
        - Adds task transitions and start/end place automatically
        - make sure that argument dicts contain fresh genes
        Reasoning: Cannot use mutable default args, and didn't want to use *args or **kwargs
        """
        self.id = str(uuid4())
        self.parent_id: int = parent_id
        self.species_id: str = None # gets assigned by species.add_member()
        self.fitness: float = 0
        # fitness measures
        self.fitness_metrics: dict = {}
        # make Transition genes for every task saved in innovs and add to genome
        self.task_list = task_list
        task_trans = {t: GTrans(t, True) for t in self.task_list}
        self.transitions = transitions | task_trans
        # make place genes for start and end places
        self.places = places | {"start":GPlace("start"), "end":GPlace("end")}
        self.arcs = arcs
        # track mutations of that genome
        self.my_mutation = ""
        self.my_components = []
        # reference to the global component tracker
        self.pop_component_tracker = pop_component_tracker

# ------------------------------------------------------------------------------
# MUTATIONS --------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def mutate(self, mutation_rate):
        """Ensures a mutation happens, clears cache afterwards
        """
        self.atomic_mutation(mutation_rate)
        # remove nodes that are no longer connected
        self.remove_unused_nodes()
        # if a mutation failed (e.g. no arcs to remove/everything connected, call recursive)
        if not self.my_mutation:
            self.mutate(mutation_rate)
        # clear the cache of methods depend on the genome structure
        self.clear_cache()


    def clear_cache(self):
        """GeneticNet uses function caching to improve speed, after mutations this
        needs to be cleared.
        """
        self.get_arc_t_values.cache_clear()
        self.build_petri.cache_clear()
        self.get_component_list.cache_clear()
        self.get_unique_component_set.cache_clear()


    def atomic_mutation(self, mutation_rate):
        """only one mutation can occur
        """
        mutations = [
            self.remove_arcs,
            self.trans_place_arc,
            self.place_trans_arc,
            self.trans_trans_conn,
            self.extend_new_place,
            self.extend_new_trans,
            self.split_arc,
            self.prune_leaf,
            self.flip_arc
        ]
        probabilities = [
            params.prob_remove_arc[mutation_rate],
            params.prob_t_p_arc[mutation_rate],
            params.prob_p_t_arc[mutation_rate],
            params.prob_t_t_conn[mutation_rate],
            params.prob_new_p[mutation_rate],
            params.prob_new_empty_t[mutation_rate],
            params.prob_split_arc[mutation_rate],
            params.prob_prune_leafs[mutation_rate],
            params.prob_flip_arc[mutation_rate],
        ]
        mutation = rd.choices(mutations, weights=probabilities, k=1)[0]
        mutation()


    def pick_target_node(self, source) -> str:
        """ given a source node, return a random node that
         - isn't opposite node type
         - isn't already connected
         - doesn't already have a connection in the other direction
        """
        connected = set()
        for a in self.arcs.values():
            if source.id in [a.source_id, a.target_id]:
                connected = connected.union({a.source_id, a.target_id})

        if type(source) == GTrans:
            suitable_places = []
            for p in self.places.values():
                if p.id not in connected.union({'start'}):
                    suitable_places.append(p.id)
            if not suitable_places:
                return None
            return rd.choice(suitable_places)

        elif type(source) == GPlace:
            # handle case where place is already connected to all trans
            # might return none
            return self.pick_trans_with_preference(filter_out=list(connected))


    def pick_trans_with_preference(self, filter_out=None) -> str:
        """Returns transition id according to preferences set in params
        """
        # set of task trans and empty trans
        task_trans = set(self.task_list)
        empty_trans = set(self.transitions).difference(task_trans)
        all_trans = task_trans.union(empty_trans)
        # if there are nodes to filter out (because already connected to them)
        if filter_out:
            task_trans = task_trans.difference(filter_out)
            empty_trans = empty_trans.difference(filter_out)
            all_trans = all_trans.difference(filter_out)
        # pick a trans
        if params.is_no_preference_for_tasks and all_trans: # choose from all trans
            return rd.choice(list(all_trans))
        elif rd.random() < params.prob_pick_empty_trans and empty_trans: # choose from empty trans (provided there are any)
            return rd.choice(list(empty_trans)) 
        elif task_trans: # choose from tasks
            return rd.choice(list(task_trans))
        else:
            return None # this should only happen when filtering out, called by get_target()


    def pick_arc(self, filter_out=set()) -> str:
        """Returns a arc id, if use t-vals, return arcs with lower t-values
        with higher probability i.e. arcs that might correlate with negative fitness
        """
        choose_from = sorted(list(set(self.arcs).difference(filter_out)))
        # if just one element, no need to do choice computation
        if len(choose_from) == 1: return choose_from.pop()
        if params.use_t_vals:
            a_weights_dict = self.get_arc_t_values()
            # multiply weights by -1, to give higher prob for choosing bad arcs
            arc_weights = [-1*a_weights_dict[a] for a in choose_from] # most arc weights are negative
            # if weightsum < 0, make weights positive by multiplying (negative) sum with *-2
            arc_weights_sum = sum(arc_weights)
            if arc_weights_sum  < 0:
                arc_weights = [a - arc_weights_sum*2 for a in arc_weights]
            # weighted choice
            arc_id = rd.choices(choose_from, weights=arc_weights, k=1)[0]
        else:
            arc_id = rd.choices(choose_from, k=1)[0]
        return arc_id


    @cache
    def get_arc_t_values(self) -> dict:
        """Returns a dict, mapping the t-value (calculated from population fitness
        in the ga.PopulationComponentTracker.update_t_vals() method) of the component
        of which an arc is part of to the arc id. Used for targetted mutations
        """
        # extend this method for whatever info we need about arcs, places, transitions
        # from innovs during mutations
        arc_values = {}
        all_c = self.get_component_list()
        for c_dict in all_c:
            c_info = self.pop_component_tracker.component_dict[c_dict['comp']]
            # FUTUREIMPROVEMENT: this is just a hack to just get the biggest t_val,
            # other approaches may be more justified
            t_val = max(c_info["t_val"].values())
            for arc_id in c_dict['arcs']:
                arc_values[arc_id] = t_val
        return arc_values


    def add_new_place(self, id=""):
        """Adds a new place to the genome
        """
        new_place = GPlace() if not id else GPlace(id)
        self.places[new_place.id] = new_place
        return new_place.id


    def add_new_trans(self, id="", is_task=False):
        """Adds a new transition to the genome
        """
        new_trans = GTrans() if not id else GTrans(id, is_task)
        self.transitions[new_trans.id] = new_trans
        return new_trans.id


    def add_new_arc(self, source_id, target_id):
        """Adds a new arc to the genome
        """
        new_arc = GArc(source_id, target_id)
        self.arcs[new_arc.id] = new_arc
        return new_arc.id

# ------------------------------------------------------------------------------
# MUTATION METHODS -------------------------------------------------------------
# ------------------------------------------------------------------------------

    def place_trans_arc(self, place_id=None, trans_id=None) -> None:
        """connects a place with a transition, if no args are specified, two random
        are chosen
        """
        if not place_id and not trans_id: # no trans/place specified in arguments
            # pick a place that is not the end place, pick a trans
            place_id = rd.choice(list(set(self.places).difference({'end'})))
            trans_id = self.pick_target_node(self.places[place_id])
            if not trans_id:
                return # place already connected to all available transitions
        self.add_new_arc(place_id, trans_id)
        self.my_mutation = 'place_trans_arc'
        return


    def trans_place_arc(self, trans_id=None, place_id=None) -> None:
        """connects transition a with a place, if no args are specified, two random
        are chosen
        """
        if not trans_id and not place_id: # no trans/place specified in arguments
            # pick a trans, pick a place that is not the start place
            trans_id = self.pick_trans_with_preference()
            place_id = self.pick_target_node(self.transitions[trans_id])
            if not place_id:
                return # the only available places are already connected
        self.add_new_arc(trans_id, place_id)
        self.my_mutation = 'trans_place_arc'
        return


    def extend_new_place(self, trans_id=None) -> None:
        """Chooses random transition, adds an input or output place (50% prob)
        """
        if not trans_id: # FUTUREIMPROVEMENT: could also filter out trans that have leaf extensions?
            trans_id = self.pick_trans_with_preference()
        new_place_id = self.add_new_place()
        if rd.random() < 0.5: # t -> p
            self.add_new_arc(trans_id, new_place_id)
        else: # p -> t
            self.add_new_arc(new_place_id, trans_id)
        self.my_mutation = 'extend_new_place'
        return


    def extend_new_trans(self, place_id=None, is_output=False) -> str:
        """Chooses random place, adds an input or output hidden trans (50% prob)
        """
        new_trans_id = self.add_new_trans()
        if not place_id: # FUTUREIMPROVEMENT: could also filter out place that have leaf extensions?
            place_id = rd.choice(list(set(self.places).difference({'end'})))
        else:
            if is_output:
                self.add_new_arc(place_id, new_trans_id)
            else:
                self.add_new_arc(new_trans_id, place_id)
            return # do not add that as a mutation, because this extension was probably crossover

        if rd.random() < 0.5 and place_id != "start": # t -> p
            self.add_new_arc(new_trans_id, place_id)
        else: # p -> t
            self.add_new_arc(place_id, new_trans_id)
        self.my_mutation = 'extend_new_trans'
        return 


    def trans_trans_conn(self, source_id=None, target_id=None):
        """Chooses 2 random transitions, adds 2 arcs and a place in the middle between them
        """
        if not source_id and not target_id:
            source_id = self.pick_trans_with_preference()
            target_id = rd.choice([t for t in self.transitions.keys() if t != source_id])
        new_place_id = self.add_new_place()
        self.add_new_arc(source_id, new_place_id)
        self.add_new_arc(new_place_id, target_id)
        self.my_mutation = 'trans_trans_conn'
        return 


    def split_arc(self):
        """Chooses random arc, splits it by adding a hidden trans and place
        """
        if not self.arcs:
            return

        arc_to_split = self.arcs[self.pick_arc()]
        all_nodes = self.places | self.transitions
        source = all_nodes[arc_to_split.source_id]
        target = all_nodes[arc_to_split.target_id]

        is_t_p = isinstance(source, GTrans)

        new_place_id = self.add_new_place()
        new_trans_id = self.add_new_trans()

        if is_t_p:
            self.add_new_arc(source.id, new_place_id)
            self.add_new_arc(new_place_id, new_trans_id)
            self.add_new_arc(new_trans_id, target.id)
        else:
            self.add_new_arc(source.id, new_trans_id)
            self.add_new_arc(new_trans_id, new_place_id)
            self.add_new_arc(new_place_id, target.id)
        # insert new arcs into genome, delete old one
        del self.arcs[arc_to_split.id]
        self.my_mutation = 'split_arc'
        return


    def prune_leaf(self) -> None:
        """This mutation randomly chooses a place or hidden transition that only has an
        input or output arc, but not both - and removes it from the genome.
        """
        all_nodes = self.places | {n:t for n,t in self.transitions.items() if not t.is_task}
        del all_nodes["start"], all_nodes["end"] # exclude start and end

        has_inputs, has_outputs = set(), set()
        for a in self.arcs.values():
            has_inputs.add(a.target_id)
            has_outputs.add(a.source_id)

        leaves = all_nodes.copy()
        for n in all_nodes:
            if n in has_inputs and n in has_outputs:
                leaves.pop(n, None)

        # prevent this mutation from deleting the last nodes
        if len(leaves) == len(all_nodes) or not leaves:
            return

        # choose leaf, delete all arcs connecting to the leaf
        leaf_id = rd.choice(list(leaves.keys()))
        arcs_to_del = [a.id for a in self.arcs.values() if leaf_id in (a.source_id, a.target_id) ]
        for a_id in arcs_to_del:
            del self.arcs[a_id]

        if type(leaves[leaf_id]) == GPlace:
            del self.places[leaf_id]
        else:
            del self.transitions[leaf_id]

        self.my_mutation = "pruned_leaf"
        return


    def remove_arcs(self, arcs_to_remove=None) -> None:
        """chooses a random arc and removes it from the genome
        """
        if len(self.arcs) <= ((len(self.places) + len(self.transitions)) / 3):
            return # if the number of arcs is less than third of all nodes, do not remove
        # no arcs to remove specified
        if not arcs_to_remove: 
            arcs_to_remove = [self.pick_arc()]
        # remove arcs
        for a_id in arcs_to_remove:
            del self.arcs[a_id]
        self.my_mutation = 'removed_an_arc'


    def flip_arc(self, arc_to_flip=None):
        """flips the direction of a randomly chosen arc
        """
        if len(self.arcs) <= ((len(self.places) + len(self.transitions)) / 3):
            return # no arcs left to flip
        if not arc_to_flip:
            # filter out arcs that connect to start or end
            fo_list = []
            for a in self.arcs.values():
                if a.source_id == "start" or a.target_id == "end":
                    fo_list.append(a.id)
            # if the only arcs available are connected to start or end, don't flip
            if len(fo_list) == len(self.arcs):
                return
            arc_to_flip = self.arcs[self.pick_arc(filter_out=fo_list)]
        self.add_new_arc(arc_to_flip.target_id, arc_to_flip.source_id)
        del self.arcs[arc_to_flip.id]
        self.my_mutation = 'flipped_an_arc'

# ------------------------------------------------------------------------------
# REPRODUCTION RELATED STUFF ---------------------------------------------------
# ------------------------------------------------------------------------------

    def clone(self, self_is_parent=True):
        """returns a deepcopy
        """
        return GeneticNet(
            transitions = copy(self.transitions),
            places = copy(self.places),
            arcs = copy(self.arcs),
            parent_id = self.id if self_is_parent else None,
            task_list = self.task_list,
            pop_component_tracker = self.pop_component_tracker
            )


    def crossover(self, mate: GeneticNet):
        """Finds components that share transitions, assigns such components to a
        chromosomes list, chooses a pair from the chromosome list and inserts the
        component from the mate in its place. Retains old connections.
        """
        my_components = self.get_component_list()
        other_components = mate.get_component_list()

        # assigns matching component pairs to chromosomes
        chromosomes = []
        for my_c in my_components:
            for other_c in other_components:
                in_overlap = my_c["inputs"].intersection(other_c["inputs"])
                out_overlap = my_c["outputs"].intersection(other_c["outputs"])
                different_inputs = my_c["inputs"] != other_c["inputs"]
                different_outputs = my_c["outputs"] != other_c["outputs"]
                my_p, other_p = my_c["place"], other_c["place"]
                # if both are start or end and have diffferent outputs/inputs
                if bool({my_p, other_p} & {"start", "end"}):
                    if my_p == other_p and (different_inputs or different_outputs):
                        chromosomes.append((my_c, other_c))
                # if shared inputs or outputs but different outputs/inputs
                elif (out_overlap and different_inputs) or (in_overlap and different_outputs):
                    chromosomes.append((my_c, other_c))
        # if there are no matching chromosomes, return None
        if not chromosomes:
            return
        # select random chromosome for swapping
        swap = rd.choice(chromosomes)
        old_comp, new_comp = swap[0], swap[1]

        # spawn new child for modifications
        baby = self.clone()
        # delete all arcs and non task transitions connected to place in baby
        baby.remove_arcs(old_comp["arcs"])
        connected = baby.get_connected()
        for t_id in old_comp["inputs"].union(old_comp["outputs"]):
            if not (self.transitions[t_id].is_task or t_id in connected):
                baby.transitions.pop(t_id, None)

        # add the new component, by connecting to the now isolated place in the old component
        p_id = old_comp["place"]
        for t_id in new_comp["inputs"]:
            if t_id in self.task_list:
                baby.trans_place_arc(t_id, p_id)
            else:
                baby.extend_new_trans(p_id, is_output=False)
        for t_id in new_comp["outputs"]:
            if t_id in self.task_list:
                baby.place_trans_arc(p_id, t_id)
            else:
                baby.extend_new_trans(p_id, is_output=True)
        # lastly, check if the offspring has no arcs (edgecase when self is single(start/end) component genome)
        if not baby.arcs:
            return 
        baby.my_mutation = "crossover" # replace prev mutation with just crossover
        baby.clear_cache()
        return baby


# ----- component distance
    @cache
    def get_component_list(self) -> list:
        """Returns the list of components in the genome
        """

        def format_tname(t): # all hidden transitions are named "t"
            return t if t in self.task_list else "t"

        p_components = dict()
        for p in self.places.values():
            p_components[p.id] = {
                'comp': Counter(),
                'place': p.id,
                'inputs': set(),
                'outputs': set(),
                'arcs': []
                }
        for a in self.arcs.values():
            if a.source_id in p_components:   # p->t
                p_components[a.source_id]['outputs'].add(a.target_id)
                p_components[a.source_id]['arcs'].append(a.id)
                p_components[a.source_id]['comp'].update([("p", format_tname(a.target_id))])
            elif a.target_id in p_components: # t->p
                p_components[a.target_id]['inputs'].add(a.source_id)
                p_components[a.target_id]['arcs'].append(a.id)
                p_components[a.target_id]['comp'].update([(format_tname(a.source_id), "p")])
        
        # now format components as sorted tuple to make comparisons with innov possible
        for p in p_components:
            p_components[p]['comp'] = tuple(sorted(p_components[p]['comp'].items()))

        return list(p_components.values())
    

    @cache
    def get_unique_component_set(self) -> set:
        """get a set of the components that will be used for distance calc
        """
        unique_components = set() 
        for c_dict in self.get_component_list():
            unique_components.add(c_dict['comp'])
        return unique_components


    def get_genetic_distance(self, other_components: set) -> float:
        """Distance metric based on percentage of components that are not shared
        """
        my_components = self.get_unique_component_set()
        union = len(my_components | other_components)
        intersect = len(my_components & other_components)
        if union == 0:
            return 1 # they are assumed to be equal, but this should normally not happen
        compat = 1 - sqrt(intersect / union)
        return compat


    def save_component_ids(self) -> None:
        for c in self.get_unique_component_set():
            self.my_components.append(self.pop_component_tracker.get_comp_info(c)["id"])

# ------------------------------------------------------------------------------
# FITNESS RELATED STUFF --------------------------------------------------------
# ------------------------------------------------------------------------------
    def build_fc_petri(self, log) -> fc.Petri:
        """Returns an OOP-based petri net for token replay. For very small models & logs
        it performs better than numpy
        """
        connected = self.get_connected()
        # add connected places
        p_dict: Dict[str, fc.Place] = {}
        for p in self.places.values():
            p_dict[p.id] = fc.Place()
        # add connected trans
        t_dict: Dict[str, fc.Transition] = {}
        for t in self.transitions.values():
            if t.id in connected:
                t_dict[t.id] = fc.Transition(t.is_task)
        # connect them
        for a in self.arcs.values():
            if a.source_id in self.transitions: # t -> p
                p = p_dict[a.target_id]
                t_dict[a.source_id].add_place(a.target_id, p, is_input=False)
            else: # p -> t
                p = p_dict[a.source_id]
                t_dict[a.target_id].add_place(a.source_id, p, is_input=True)
        return fc.Petri(p_dict, t_dict, log)


    def build_fc_np_petri(self, log) -> fc_np.PetriNetNP:
        """Returns a numpy petri net for token replay. Code for that is not really
        verified however, written with great haste using ClaudeAI. Performance is not
        that great for small models. I'm sure there are lot's of possible improvments
        but this was just a Proof of concept implementation.
        """
        connected = self.get_connected()
        # Create lists of place and transition IDs
        place_ids = list(self.places.keys())
        transition_ids = [t_id for t_id, t in self.transitions.items() if t_id in connected]
        # Create the PetriNet object
        petri_net = fc_np.PetriNetNP(place_ids, transition_ids, log)
        # Add arcs to the PetriNet
        for arc in self.arcs.values():
            if arc.source_id in self.transitions:  # t -> p
                petri_net.add_arc(arc.target_id, arc.source_id, is_input=False, 
                                  is_task=self.transitions[arc.source_id].is_task)
            else:  # p -> t
                petri_net.add_arc(arc.source_id, arc.target_id, is_input=True, 
                                  is_task=self.transitions[arc.target_id].is_task)
        return petri_net


    @cache
    def build_petri(self):
        """Returns a pm4py PetriNet object. Deprecated for fitness calc at this moment
        """
        net = PetriNet(f"{self.id}-Net")
        temp_obj_d = {} # stores both trans and place pm4py objs in the scope of this method
        # genome contains all tasks, but not all are connected necessarily
        connected = self.get_connected()
        for place_id in self.places:
            if place_id in list(connected) + ["start", "end"]:
                temp_obj_d[place_id] = PetriNet.Place(place_id)
                net.places.add(temp_obj_d[place_id])
        for trans_id in self.transitions:
            if trans_id in connected:
                temp_obj_d[trans_id] = PetriNet.Transition(trans_id, label=trans_id)
                net.transitions.add(temp_obj_d[trans_id])
        for arc_id in self.arcs:
            arc = self.arcs[arc_id]
            source_obj = temp_obj_d[arc.source_id]
            target_obj = temp_obj_d[arc.target_id]
            add_arc_from_to(source_obj, target_obj, net)
        # initial marking
        im = Marking()
        im[temp_obj_d["start"]] = 1
        # final marking
        fm = Marking()
        fm[temp_obj_d["end"]] = 1
        return net, im, fm


    def evaluate_fitness(self, log, curr_gen=1):
        """builds petri net for fitness calculations, aggregates/combines it's fitness
        metrics based on the log. Optional curr_gen argument is currently unused.

        for efficiency, I could calculate the max achievable fitness only once and
        save it, but the perf is ok for me so far.
        """
        # model_eval = self.build_fc_np_petri(log).evaluate() # <- for using numpy, not recommended
        model_eval = self.build_fc_petri(log).evaluate()

        self.fitness_metrics = model_eval["metrics"]

        fit, max_fit = 0, 0
        for m, val in self.fitness_metrics.items():
            # skip metrics that are not listed in the fitness dict
            metric_params = params.metric_dict.get(m)
            if not metric_params or metric_params["weight"]==0:
                continue
            # skip metric if only active at later gen
            if curr_gen < metric_params["active_gen"]:
                continue
            # this is the max achievable fitness value, it gets the same transformations
            max_val = 1
            # check if the fitness value should be anchored to another fitness value
            anchor_metric, anchor_threshold = metric_params["anchor_to"]
            if anchor_metric in self.fitness_metrics:
                anchor_value = self.fitness_metrics[anchor_metric]
                val = min(anchor_value, val) if anchor_value > anchor_threshold else 0
            # transform the variable (default = 1, so no change)
            pow = metric_params.get("raise_by")
            val, max_val = val ** pow, max_val ** pow
            # apply weight
            weight = metric_params.get("weight")
            val, max_val = val * weight, max_val * weight
            # update the actual and maximum fitness
            fit += val; max_fit += max_val
        # assign fitness
        self.fitness = fit / max_fit
        return model_eval

# ------------------------------------------------------------------------------
# MISC STUFF -------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def get_gviz(self) -> set:
        """removes labels from dead transitions, uses start param for deterministic layout
        """
        net, im, fm = self.build_petri()
        for t in net.transitions:
            if t.label not in self.task_list:
                t.label = None
        viz = graphviz_visualization(net, initial_marking=im, final_marking=fm)
        # viz.graph_attr.update({"layout": 'neato', "start": "1"})
        return viz


    def get_connected(self) -> set:
        """Get set of all nodes that are connected to the network via arcs
        """
        connected = [(a.source_id, a.target_id) for a in self.arcs.values()]
        return set(itertools.chain.from_iterable(connected))


    def get_curr_info(self) -> dict:
        """Used for serialization when not wanting to save the entire object
        """
        discard = ["transitions", "places", "arcs", "pop_component_tracker", "task_list"]
        return {var: val for var, val in vars(self).items() if var not in discard}


    def remove_unused_nodes(self) -> None:
        """removes orphan hidden transitions and places
        """
        connected = self.get_connected()
        t_to_del = []
        for t in self.transitions:
            if t not in connected and t not in self.task_list:
                t_to_del.append(t)
        for t in t_to_del:
            del self.transitions[t]

        p_to_del = []
        for p in self.places:
            if p not in connected and p not in ["start", "end"]:
                p_to_del.append(p)
        for p in p_to_del:
            del self.places[p]