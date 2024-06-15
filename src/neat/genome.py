from __future__ import annotations
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

from pm4py.algo.conformance.tokenreplay.variants.token_replay import apply as get_replayed_traces
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import evaluate as get_fitness_dict
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as get_precision
from pm4py.algo.evaluation.generalization.variants.token_based import get_generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as get_simplicity
from pm4py.algo.analysis.woflan.algorithm import apply as get_soundness

from neatutils.fitnesscalc import transition_execution_quality
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
        self.fitness: float = None
        # fitness measures
        self.perc_fit_traces: float = None
        self.average_trace_fitness: float = None
        self.log_fitness: float = None
        self.is_sound: bool = None
        self.precision: float = None
        self.generalization: float = None
        self.simplicity: float = None
        self.fraction_used_trans: float = None
        self.fraction_tasks: float = None
        self.execution_score: float = None
        # make Transition genes for every task saved in innovs and add to genome
        self.task_list = task_list
        task_trans = {t: GTrans(t, True) for t in self.task_list}
        self.transitions = transitions | task_trans
        # make place genes for start and end places
        self.places = places | {"start":GPlace("start"), "end":GPlace("end")}
        self.arcs = arcs
        # track mutations of that genome
        self.my_mutations = []
        # reference to the global component tracker
        self.pop_component_tracker = pop_component_tracker

# ------------------------------------------------------------------------------
# MUTATIONS --------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def mutate(self, mutation_rate):
        if params.mutation_type == "multi":
            self.multi_mutation(mutation_rate)
        elif params.mutation_type == "atomic":
            self.atomic_mutation(mutation_rate)
        # remove nodes that are no longer connected
        self.remove_unused_nodes()
        # clear the cache of methods depend on the genome structure
        self.clear_cache()
        # if a mutation failed (e.g. no arcs to remove/everything connected, call recursive)
        if not self.my_mutations:
            self.mutate(mutation_rate)


    def clear_cache(self):
        self.get_arc_t_values.cache_clear()
        self.build_petri.cache_clear()
        self.get_component_list.cache_clear()
        self.get_unique_component_set.cache_clear()

    def multi_mutation(self, mutation_rate):
        """multiple mutations can occur
        """
        try:
            if rd.random() < params.prob_remove_arc[mutation_rate]:
                self.remove_arcs(mutation_rate)
            if rd.random() < params.prob_t_p_arc[mutation_rate]:
                self.trans_place_arc()
            if rd.random() < params.prob_p_t_arc[mutation_rate]:
                self.place_trans_arc()
            if rd.random() < params.prob_t_t_conn[mutation_rate]:
                self.trans_trans_conn()
            if rd.random() < params.prob_new_p[mutation_rate] and len(self.transitions) > 2:
                self.extend_new_place()
            if rd.random() < params.prob_new_empty_t[mutation_rate] and len(self.places) > 2:
                self.extend_new_trans()
            if rd.random() < params.prob_split_arc[mutation_rate]:
                self.split_arc()
            if rd.random() < params.prob_prune_leafs[mutation_rate]:
                self.prune_leaves()
            if rd.random() < params.prob_prune_leafs[mutation_rate]:
                self.flip_arc()
        except:
            print(traceback.format_exc()) # TODO: meh, aint got not time to do logging here


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
            self.prune_leaves,
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
        # TODO: check if no mutation occured (e.g. no extensions to prune, no places to connect)
        # and call itself again


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
        if params.use_t_vals:
            a_weights_dict = self.get_arc_t_values()
            arc_weights = [a_weights_dict[a] for a in choose_from]
            arc_id = rd.choices(choose_from, weights=arc_weights, k=1)[0]
        else:
            arc_id = rd.choices(choose_from, k=1)[0]
        return arc_id


    @cache
    def get_arc_t_values(self) -> dict:
        # extend this method for whatever info we need about arcs, places, transitions        # arc_values = {a.id: 1 for a in self.arcs.values()}
        # from innovs during mutations
        arc_values = {}
        all_c = self.get_component_list()
        for c_dict in all_c:
            pop_fit_val = self.pop_component_tracker.component_dict[c_dict['comp']]['t_val']
            for arc_id in c_dict['arcs']:
                arc_values[arc_id] = pop_fit_val
        return arc_values


    def add_new_place(self, id=""):
        new_place = GPlace() if not id else GPlace(id)
        self.places[new_place.id] = new_place
        return new_place.id


    def add_new_trans(self, id="", is_task=False):
        new_trans = GTrans() if not id else GTrans(id, is_task)
        self.transitions[new_trans.id] = new_trans
        return new_trans.id


    def add_new_arc(self, source_id, target_id):
        new_arc = GArc(source_id, target_id)
        self.arcs[new_arc.id] = new_arc
        return new_arc.id

# ------------------------------------------------------------------------------
# MUTATION METHODS -------------------------------------------------------------
# ------------------------------------------------------------------------------

    def place_trans_arc(self, place_id=None, trans_id=None) -> None:
        if not place_id and not trans_id: # no trans/place specified in arguments
            # pick a place that is not the end place, pick a trans
            place_id = rd.choice(list(set(self.places).difference({'end'})))
            trans_id = self.pick_target_node(self.places[place_id])
            if not trans_id:
                return # place already connected to all available transitions
        self.add_new_arc(place_id, trans_id)
        self.my_mutations.append('place_trans_arc')
        return


    def trans_place_arc(self, trans_id=None, place_id=None) -> None:
        if not trans_id and not place_id: # no trans/place specified in arguments
            # pick a trans, pick a place that is not the start place
            trans_id = self.pick_trans_with_preference()
            place_id = self.pick_target_node(self.transitions[trans_id])
            if not place_id:
                return # the only available places are already connected
        self.add_new_arc(trans_id, place_id)
        self.my_mutations.append('trans_place_arc')
        return


    def extend_new_place(self, trans_id=None) -> None:
        if not trans_id: # TODO: could also filter out trans that have leaf extensions?
            trans_id = self.pick_trans_with_preference()
        new_place_id = self.add_new_place()
        if rd.random() < 0.5: # t -> p
            self.add_new_arc(trans_id, new_place_id)
        else: # p -> t
            self.add_new_arc(new_place_id, trans_id)
        self.my_mutations.append('extend_new_place')
        return


    def extend_new_trans(self, place_id=None, is_output=False) -> str:
        new_trans_id = self.add_new_trans()
        if not place_id: # TODO: could also filter out place that have leaf extensions?
            place_id = rd.choice(list(set(self.places).difference({'end'})))
        else:
            if is_output:
                self.add_new_arc(place_id, new_trans_id)
            else:
                self.add_new_arc(new_trans_id, place_id)
            return # do not add that as a mutation, because this extension was probably crossover

        if rd.random() < 0.5: # p -> t
            self.add_new_arc(place_id, new_trans_id)
        else: # t -> p
            self.add_new_arc(new_trans_id, place_id)
        self.my_mutations.append('extend_new_trans')
        return 


    def trans_trans_conn(self, source_id=None, target_id=None):
        if not source_id and not target_id:
            source_id = self.pick_trans_with_preference()
            target_id = rd.choice([t for t in self.transitions.keys() if t != source_id])
        new_place_id = self.add_new_place()
        self.add_new_arc(source_id, new_place_id)
        self.add_new_arc(new_place_id, target_id)
        self.my_mutations.append('trans_trans_conn')
        return 


    def split_arc(self):
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
        self.my_mutations.append('split_arc')
        return


    def prune_leaves(self) -> None: # TODO: name this prune leafs later
        all_nodes = self.places | self.transitions
        del all_nodes["start"], all_nodes["end"] # exclude start and end

        for a in self.arcs.values(): # remove all nodes that have outgoing conns
           all_nodes.pop(a.source_id, None)
        
        if not all_nodes:
            return # no suitable leaf nodes

        leaf_id = rd.choice(list(all_nodes.keys()))
        # delete all arcs pointing to the leaf
        arcs_to_del = [a.id for a in self.arcs.values() if a.target_id == leaf_id]
        for a_id in arcs_to_del:
            del self.arcs[a_id]

        if type(all_nodes[leaf_id]) == GPlace:
            del self.places[leaf_id]
        else:
            del self.transitions[leaf_id]
        self.my_mutations.append("pruned_leaf")
        return


    def remove_arcs(self, arcs_to_remove=None) -> None:
        if len(self.arcs) <= ((len(self.places) + len(self.transitions)) / 3):
            return # if the number of arcs is less than third of all nodes, do not remove
        if not arcs_to_remove: # no arcs to remove specified
            arcs_to_remove = set()
            for _ in range(params.max_arcs_removed):
                # use a set to prevent duplicates
                arcs_to_remove.add(self.pick_arc())
            arcs_to_remove = list(arcs_to_remove)
        # delete arcs in arcs to remove
        for a_id in arcs_to_remove:
            del self.arcs[a_id]
            self.my_mutations.append('removed_an_arc')


    def flip_arc(self, arc_to_flip=None):
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
        self.my_mutations.append('flipped_an_arc')

# ------------------------------------------------------------------------------
# REPRODUCTION RELATED STUFF ---------------------------------------------------
# ------------------------------------------------------------------------------

    def clone(self):
        """returns a deepcopy
        """
        return GeneticNet(
            transitions = copy(self.transitions),
            places = copy(self.places),
            arcs = copy(self.arcs),
            parent_id = self.id,
            task_list = self.task_list,
            pop_component_tracker = self.pop_component_tracker
            )


    def crossover(self, mate: GeneticNet):
        """Finds components that share transitions, assigns such components to a
        chromosomes list, chooses a pair from the chromosome list and inserts the
        component from the mate in its place. Retains old connections.
        """
        def get_input_output_t(c: tuple):
            inputs, outputs = set(), set()
            for a in c:
                if a[0][0] in self.task_list:
                    inputs.add(a[0][0])
                elif a[0][1] in self.task_list:
                    outputs.add(a[0][1])
            return {"in": inputs, "out": outputs}

        # gets the inputs and outputs of all components
        my_c = self.get_unique_component_set()
        my_c_dict = [(c, get_input_output_t(c)) for c in list(my_c)]
        mate_c = mate.get_unique_component_set()
        mate_c_dict = [(c, get_input_output_t(c)) for c in list(mate_c)]
        # assigns matching component pairs to chromosomes
        chromosomes = []
        for my_c, my_in_out in my_c_dict:
            for mate_c, mate_in_out in mate_c_dict:
                in_overlap = my_in_out["in"].intersection(mate_in_out["in"])
                out_overlap = my_in_out["out"].intersection(mate_in_out["out"])
                different_outputs = my_in_out["out"] != mate_in_out["out"]
                different_inputs = my_in_out["in"] != mate_in_out["in"]
                # if shared task-t in input and outputs are different or
                # shared task-t in output and inputs are different add to chromosomes
                if (in_overlap and different_outputs) or (out_overlap and different_inputs):
                    chromosomes.append((my_c, mate_c))
        # if there are no matching chromosomes, return None
        if not chromosomes:
            return
        # TODO: could add probabilites here for selecting bad c and replace with good c
        swap = rd.choice(chromosomes)
        my_comp, new_comp = swap[0], swap[1]
        # get the child genome and modify it
        baby = self.clone()
        # find the component to be replaced
        for c_dict in baby.get_component_list():
            if my_comp == c_dict["comp"]:
                comp_to_del = c_dict
                break
        # check if we are deleting start or end place
        p_id = comp_to_del["place"]
        del baby.places[p_id]
        p_id = p_id if p_id in ["start", "end"] else None
        # delete all empty transitions
        arcs_to_del = []
        for t_id in comp_to_del["transitions"]:
            # delete all arcs that pointed to dead transitions
            if not baby.transitions[t_id].is_task:
                del baby.transitions[t_id]
                for a in baby.arcs.values():
                    if t_id in [a.source_id, a.target_id]:
                        arcs_to_del.append(a.id)
        # delete all arcs within the old component + empty_trans conn
        for a_id in comp_to_del["arcs"] + arcs_to_del:
            baby.arcs.pop(a_id, None)
        # add the new component
        new_p_id = baby.add_new_place(p_id) if p_id else baby.add_new_place()
        for a in new_comp:
            source, target = a[0][0], a[0][1]
            if source == "p":
                if target == "t":
                    for _ in range(a[1]): # if there are multiple empty output trans
                        baby.extend_new_trans(new_p_id, is_output=True)
                else:
                    baby.add_new_arc(new_p_id, target)
            else:
                if source == "t":
                    for _ in range(a[1]): # if there are multiple empty input trans
                        baby.extend_new_trans(new_p_id, is_output=False)
                else:
                    baby.add_new_arc(source, new_p_id)

        baby.my_mutations = ["crossover"] # replace all mutations with just crossover
        baby.clear_cache()
        return baby


# ----- component compatibility
    @cache
    def get_component_list(self) -> list:

        def format_tname(t): # all hidden transitions are named "t"
            return t if t in self.task_list else "t"

        p_components = dict()
        for p in self.places.values():
            p_components[p.id] = {
                'comp': Counter(),
                'place': p.id,
                'transitions': [],
                'arcs': []
                }
        for a in self.arcs.values():
            if a.source_id in p_components:   # p->t
                p_components[a.source_id]['transitions'].append(a.target_id)
                p_components[a.source_id]['arcs'].append(a.id)
                p_components[a.source_id]['comp'].update([("p", format_tname(a.target_id))])
            elif a.target_id in p_components: # t->p
                p_components[a.target_id]['transitions'].append(a.source_id)
                p_components[a.target_id]['arcs'].append(a.id)
                p_components[a.target_id]['comp'].update([(format_tname(a.source_id), "p")])
        
        # now format components as sorted tuple to make comparisons with innov possible
        for p in p_components:
            p_components[p]['comp'] = tuple(sorted(p_components[p]['comp'].items()))

        return list(p_components.values())
    
    @cache
    def get_unique_component_set(self) -> set:
        # TODO: add cache
        # point of this is to just get a set of the components
        # this will be used for compatibility calc
        unique_components = set() 
        for c_dict in self.get_component_list():
            unique_components.add(c_dict['comp'])
        return unique_components


    def component_compatibility(self, other_components: set) -> float:
        """Distance metric based on percentage of components that are not shared
        """
        my_components = self.get_unique_component_set()
        union = len(my_components | other_components)
        intersect = len(my_components & other_components)
        if union == 0:
            return 1 # they are assumed to be equal, but this should normally not happen
        compat = 1 - sqrt(intersect / union)
        return compat

# ------------------------------------------------------------------------------
# FITNESS RELATED STUFF --------------------------------------------------------
# ------------------------------------------------------------------------------

    @cache
    def build_petri(self):
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


    def evaluate_fitness(self, log) -> None:
        # remove nodes that are no longer connected (again, just to make sure)
        self.remove_unused_nodes()
        net, im, fm = self.build_petri()
        # fitness eval
        default_params = {"show_progress_bar": False}
        aligned_traces = get_replayed_traces(log, net, im, fm, default_params)
        trace_fitness = get_fitness_dict(aligned_traces)
        self.perc_fit_traces = trace_fitness["perc_fit_traces"] / 100
        self.average_trace_fitness = trace_fitness["average_trace_fitness"]
        self.log_fitness = trace_fitness["log_fitness"]
        # get fraction of task trans represented in genome
        my_task_trans = [t for t in self.transitions.values() if t.is_task]
        if my_task_trans:
            self.fraction_used_trans = len(my_task_trans) / len(self.task_list)
            self.fraction_tasks = len(my_task_trans) / len(self.transitions)
        else:
            self.fraction_used_trans = 0
            self.fraction_tasks = 0
        # soundness check
        soundness_params = {"return_asap_when_not_sound": True, "print_diagnostics": False}
        self.is_sound = get_soundness(net, im, fm, soundness_params)
        # precision, generalization, simplicity, execution score
        self.precision = get_precision(log, net, im, fm, default_params)
        self.generalization = get_generalization(net, aligned_traces)
        self.simplicity = get_simplicity(net)
        self.execution_score = transition_execution_quality(aligned_traces)

        self.fitness = (
            + params.perc_fit_traces_weight * (self.perc_fit_traces / 100)
            + params.average_trace_fitness_weight * (self.average_trace_fitness**2)
            + params.log_fitness_weight * self.log_fitness
            + params.soundness_weight * int(self.is_sound)
            + params.precision_weight * (self.precision**2)
            + params.generalization_weight * (self.generalization**2)
            + params.simplicity_weight * (self.simplicity**2)
            + params.fraction_used_trans_weight * self.fraction_used_trans
            + params.fraction_tasks_weight * self.fraction_tasks
            + self.execution_score
        )
        

        if self.fitness < 0:
            raise Exception("Fitness below 0 should not be possible!!!")
        return

# ------------------------------------------------------------------------------
# MISC STUFF -------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def get_connected(self) -> set:
        # get set of all nodes that are connected to the network via arcs
        connected = [(a.source_id, a.target_id) for a in self.arcs.values()]
        return set(itertools.chain.from_iterable(connected))


    def get_curr_info(self) -> dict:
        """Used for serialization when not wanting to save the entire object
        """
        discard = ["transitions", "places", "arcs", "pop_component_tracker"]
        return {var: val for var, val in vars(self).items() if var not in discard}


    def remove_unused_nodes(self) -> None:
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