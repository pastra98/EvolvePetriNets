from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

from pm4py.algo.conformance.tokenreplay.variants.token_replay import apply as get_replayed_traces
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import evaluate as get_fitness_dict
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as get_precision
from pm4py.algo.evaluation.generalization.variants.token_based import get_generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as get_simplicity
from pm4py.algo.analysis.woflan.algorithm import apply as get_soundness
from pm4py.algo.simulation.playout.petri_net.variants.extensive import apply as extensive_playout
from pm4py.stats import get_variants
from pm4py.analysis import maximal_decomposition


from neatutils.fitnesscalc import transition_execution_quality
from neat.netobj import GArc, GPlace, GTrans
from neat import params, innovs

import random as rd
import numpy as np
import traceback
import itertools
from functools import cache
from collections import Counter

from graphviz import Digraph


class GeneticNet:
    def __init__(self, transitions: dict, places: dict, arcs: dict, parent_id=None) -> None:
        """transitions, places and arcs must either be dicts containing valid id: netobj
        key-value pairings, or an empty dict.
        - Adds task transitions and start/end place automatically
        - make sure that argument dicts contain fresh genes
        Reasoning: Cannot use mutable default args, and didn't want to use *args or **kwargs
        """
        self.id = innovs.get_new_genome_id()
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
        task_trans = {t: GTrans(t, True) for t in innovs.get_task_list()}
        self.transitions = transitions | task_trans
        # make place genes for start and end places
        self.places = places | {"start":GPlace("start", is_start=True), "end":GPlace("end", is_end=True)}
        self.arcs = arcs
        # track mutations of that genome
        self.my_mutations = []

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
            self.prune_leaves
        ]
        probabilities = [
            params.prob_remove_arc[mutation_rate],
            params.prob_t_p_arc[mutation_rate],
            params.prob_p_t_arc[mutation_rate],
            params.prob_t_t_conn[mutation_rate],
            params.prob_new_p[mutation_rate],
            params.prob_new_empty_t[mutation_rate],
            params.prob_split_arc[mutation_rate],
            params.prob_prune_leafs[mutation_rate]
        ]
        mutation = rd.choices(mutations, weights=probabilities, k=1)[0]
        mutation()
        # TODO: check if no mutation occured (e.g. no extensions to prune, no places to connect)
        # and call itself again


    def get_target(self, source) -> str:
        """ given a source node, return a random node that
         - isn't opposite node type
         - isn't already connected
         - doesn't already have a connection in the other direction
        """
        connected = set()
        for a in self.arcs.values():
            if source.id in [a.source_id, a.target_id]:
                connected.add(a.target_id)

        if type(source) == GTrans:
            suitable_places = []
            for p in self.places.values():
                if p.id not in connected and not p.is_start:
                    suitable_places.append(p.id)
                if not suitable_places:
                    return None
            return rd.choice(suitable_places)

        elif type(source) == GPlace:
            # handle case where place is already connected to all trans
            # might return none
            return self.pick_trans_with_preference(filter_out=list(connected))

        else:
            raise Exception("This should only be called for places or transitions")


    def get_new_id(self, obj_type) -> int:
        # id is simply an increment of the current num of places/transitions/arcs
        if obj_type == GPlace:
            return f"p{len(self.places) + 1}"
        elif obj_type == GTrans:
            return f"t{len(self.transitions) + 1}"
        elif obj_type == GArc:
            return len(self.arcs) + 1


    def place_trans_arc(self, place_id=None, trans_id=None) -> None:
        if not place_id and not trans_id: # no trans/place specified in arguments
            # pick a place that is not the end place, pick a trans
            place_id = rd.choice([p for p in self.places if p != "end"])
            trans_id = self.get_target(self.places[place_id])
            if not trans_id:
                return # place already connected to all available transitions
            arc_id = self.get_new_id(GArc)
        else: # TODO: no checks happen here
            arc_id = self.get_new_id(GArc)
        new_arc = GArc(arc_id, place_id, trans_id)
        self.arcs[arc_id] = new_arc
        self.my_mutations.append('place_trans_arc')
        return


    def trans_place_arc(self, trans_id=None, place_id=None) -> None:
        if not trans_id and not place_id: # no trans/place specified in arguments
            # pick a trans, pick a place that is not the start place
            trans_id = self.pick_trans_with_preference()
            place_id = self.get_target(self.transitions[trans_id])
            if not place_id:
                return # the only available places are already connected
            arc_id = self.get_new_id(GArc)
        else: # TODO: no checks happen here
            arc_id = self.get_new_id(GArc)
        new_arc = GArc(arc_id, trans_id, place_id)
        self.arcs[arc_id] = new_arc
        self.my_mutations.append('trans_place_arc')
        return


    def extend_new_place(self, trans_id=None) -> None:
        if not trans_id: # TODO: could also filter out trans that have leaf extensions?
            trans_id = self.pick_trans_with_preference()
        new_place_id = self.get_new_id(GPlace)
        new_arc_id = self.get_new_id(GArc)
        self.places[new_place_id] = GPlace(new_place_id)
        self.arcs[new_arc_id] = GArc(new_arc_id, trans_id, new_place_id)
        self.my_mutations.append('extend_new_place')
        return


    def extend_new_trans(self, place_id=None) -> str:
        if not place_id: # TODO: could also filter out place that have leaf extensions?
            place_id = rd.choice([p for p in self.places.values() if not p.is_end]).id
        new_trans_id = self.get_new_id(GTrans)
        new_arc_id = self.get_new_id(GArc)
        self.transitions[new_trans_id] = GTrans(new_trans_id, is_task=False)
        self.arcs[new_arc_id] = GArc(new_arc_id, place_id, new_trans_id)
        self.my_mutations.append('extend_new_trans')
        return 


    def trans_trans_conn(self, source_id=None, target_id=None):
        if not source_id and not target_id:
            source_id = self.pick_trans_with_preference()
            target_id = rd.choice([t for t in self.transitions.keys() if t != source_id])
        a1_id = self.get_new_id(GArc)
        a2_id = self.get_new_id(GArc)
        p_id = self.get_new_id(GPlace)
        self.arcs[a1_id] = GArc(a1_id, source_id, p_id)
        self.places[p_id] = GPlace(p_id)
        self.arcs[a2_id] = GArc(a2_id, p_id, target_id)
        self.my_mutations.append('trans_trans_conn')
        return 


    def split_arc(self):
        if not self.arcs:
            return

        # TODO: should also consider arc t-values here
        arc_to_split = rd.choice(list(self.arcs.values()))
        all_nodes = self.places | self.transitions
        source = all_nodes[arc_to_split.source_id]
        target = all_nodes[arc_to_split.target_id]

        is_t_p = isinstance(source, GTrans)

        new_place_id = self.get_new_id(GPlace)
        new_place = GPlace(new_place_id)
        self.places[new_place_id] = new_place

        new_trans_id = self.get_new_id(GTrans)
        new_trans = GTrans(new_trans_id, is_task=False)
        self.transitions[new_trans_id] = new_trans

        if is_t_p:
            a1 = GArc(self.get_new_id(GArc), source.id, new_place.id)
            a2 = GArc(self.get_new_id(GArc), new_place.id, new_trans.id)
            a3 = GArc(self.get_new_id(GArc), new_trans.id, target.id)
        else:
            a1 = GArc(self.get_new_id(GArc), source.id, new_trans.id)
            a2 = GArc(self.get_new_id(GArc), new_trans.id, new_place.id)
            a3 = GArc(self.get_new_id(GArc), new_place.id, target.id)
        # insert new arcs into genome, delete old one
        self.arcs.update({a1.id: a1, a2.id: a2, a3.id: a3})
        del self.arcs[arc_to_split.id]
        self.my_mutations.append('split_arc')
        return


    def pick_trans_with_preference(self, filter_out=None) -> str:
        """Returns transition id according to preferences set in params
        """
        # set of task trans and empty trans
        task_trans = set(innovs.get_task_list())
        empty_trans = set(self.transitions.keys()).difference(task_trans)
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


    def prune_leaves(self) -> None: # TODO: name this prune leafs later
        filtered_places = {k: v for k, v in self.places.items() if not (v.is_start or v.is_end)}
        all_nodes = filtered_places | self.transitions # exclude start and end
        for a in self.arcs.values(): # remove all nodes that have outgoing conns
           try: del all_nodes[a.source_id]
           except: pass
        
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
        if len(self.arcs) <= 1:
            # TODO: could add parameter here for min number of arcs before this mutation triggers
            return # don't delete the last arc
        if not arcs_to_remove: # no arcs to remove specified
            arcs_to_remove = set()
            for _ in range(params.max_arcs_removed):
                if params.use_t_vals:
                    arcdict = self.get_arc_t_values()
                    # since higher t is better, multiply ts with -1 for removal
                    arc_weights = np.array(list(arcdict.values()))
                    arc_weights = arc_weights * -1 + abs(arc_weights.sum())
                    arc = rd.choices(list(arcdict.keys()), weights=arc_weights, k=1)[0]
                else:
                    arc = rd.choices(list(self.arcs.keys()), k=1)[0]
                arcs_to_remove.add(arc) # use a set to prevent duplicates
            arcs_to_remove = list(arcs_to_remove)
        # delete arcs in arcs to remove
        for a_id in arcs_to_remove:
            del self.arcs[a_id]
            self.my_mutations.append('removed_an_arc')


    @cache
    def get_arc_t_values(self) -> dict:
        # extend this method for whatever info we need about arcs, places, transitions        # arc_values = {a.id: 1 for a in self.arcs.values()}
        # from innovs during mutations
        arc_values = {}
        all_c = self.get_component_list()
        for c_dict in all_c:
            pop_fit_val = innovs.component_dict[c_dict['comp']]['t_val']
            for arc_id in c_dict['arcs']:
                arc_values[arc_id] = pop_fit_val
        return arc_values

# ------------------------------------------------------------------------------
# REPRODUCTION RELATED STUFF ---------------------------------------------------
# ------------------------------------------------------------------------------

    def clone(self):
        """returns a deepcopy
        """
        new_transitions = {k: v.get_copy() for k, v in self.transitions.items()}
        new_places = {k: v.get_copy() for k, v in self.places.items()}
        new_arcs = {k: v.get_copy() for k, v in self.arcs.items()}
        return GeneticNet(new_transitions, new_places, new_arcs, int(self.id))


# ----- component compatibility
    @cache
    def get_component_list(self) -> list:

        def format_tname(t): # all hidden transitions are named "t"
            return t if t in innovs.get_task_list() else "t"

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


    def component_compatibility(self, other_genome) -> float:
        """Distance metric based on percentage of components that are not shared
        """
        my_c = self.get_unique_component_set()
        other_c = other_genome.get_unique_component_set()   
        union = len(my_c | other_c)
        intersect = len(my_c & other_c)
        if union == 0:
            return 1 # they are assumed to be equal, but this should normally not happen
        return 1 - (intersect / union) * params.component_mult

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
            self.fraction_used_trans = len(my_task_trans) / len(innovs.get_task_list())
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


    def get_graphviz(self) -> Digraph:
        # parameter stuff, TODO: think about where to put this
        fsize = "12"
        tcol = "yellow"
        pcol = "lightblue"
        ahead = "normal"
        viz = Digraph(format="png")
        viz.graph_attr["rankdir"] = "LR"
        viz.attr("node", shape="box")
        viz.attr(overlap="false")
        # viz.attr(size="21, 21")
        # viz.attr(size="11, 11")
        connected = self.get_connected()
        # transitions
        for t in self.transitions:
            if t in connected:
                if self.transitions[t].is_task: # task
                    viz.node(t, t, style='filled', fillcolor=tcol, border='1', fontsize=fsize)
                else: # empty trans
                    viz.node(t, t, style='filled', fontcolor="white", fillcolor="black", fontsize=fsize)
        # places
        for p in self.places:
            if p in connected:
                if p == "start":
                    viz.node("start", style='filled', fillcolor="green", fontsize=fsize,
                                shape='circle', fixedsize='true', width='0.75')
                elif p == "end":
                    viz.node("end", style='filled', fillcolor="orange", fontsize=fsize,
                                shape='circle', fixedsize='true', width='0.75')
                else:
                    viz.node(p, p, style='filled', fillcolor=pcol, fontsize=fsize,
                                shape='circle', fixedsize='true', width='0.75')
        # arcs
        for name, a in self.arcs.items():
            viz.edge(a.source_id, a.target_id, label=str(name), fontsize=fsize, arrowhead=ahead)
        return viz


    def get_curr_info(self) -> dict:
        """Used for serialization when not wanting to save the entire object
        """
        discard = ["transitions", "places", "arcs"]
        return {var: val for var, val in vars(self).items() if var not in discard}

    def show_nb_graphviz(self) -> None:
        from IPython.core.display import display, Image
        gviz = self.get_graphviz()
        display(Image(data=gviz.pipe(format="png"), unconfined=True, retina=True))


    def remove_unused_nodes(self) -> None:
        connected = self.get_connected()
        t_to_del = []
        for t in self.transitions:
            if t not in connected and t not in innovs.get_task_list():
                t_to_del.append(t)
        for t in t_to_del:
            del self.transitions[t]

        p_to_del = []
        for p in self.places:
            if p not in connected and p not in ["start", "end"]:
                p_to_del.append(p)
        for p in p_to_del:
            del self.places[p]