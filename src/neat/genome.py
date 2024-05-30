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
        self.get_extensive_variants.cache_clear()
        self.get_component_set.cache_clear()


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
            if rd.random() < params.prob_prune_extensions[mutation_rate]:
                self.prune_extensions()
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
            self.prune_extensions
        ]
        probabilities = [
            params.prob_remove_arc[mutation_rate],
            params.prob_t_p_arc[mutation_rate],
            params.prob_p_t_arc[mutation_rate],
            params.prob_t_t_conn[mutation_rate],
            params.prob_new_p[mutation_rate],
            params.prob_new_empty_t[mutation_rate],
            params.prob_split_arc[mutation_rate],
            params.prob_prune_extensions[mutation_rate]
        ]
        mutation = rd.choices(mutations, weights=probabilities, k=1)[0]
        mutation()


    def place_trans_arc(self, place_id=None, trans_id=None) -> None:
        if not place_id and not trans_id: # no trans/place specified in arguments
            # search until a place/trans combi is found that is not already connected
            for _try in range(params.num_trys_make_conn):
                # pick a place that is not the end place, pick a trans
                place_id = rd.choice([p for p in self.places if p != "end"])
                trans_id = self.pick_trans_with_preference()
                # check in innovs
                arc_id = innovs.get_arc(place_id, trans_id)
                if arc_id not in self.arcs:
                    break
        else: # if place/trans specified in arguments, just get that innov number
            arc_id = innovs.get_arc(place_id, trans_id)
            if arc_id in self.arcs:
                return # no connection is made
        new_arc = GArc(arc_id, place_id, trans_id)
        self.arcs[arc_id] = new_arc
        self.my_mutations.append('place_trans_arc')
        return


    def trans_place_arc(self, trans_id=None, place_id=None) -> None:
        if not trans_id and not place_id: # no trans/place specified in arguments
            # search until a place/trans combi is found that is not already connected
            for _try in range(params.num_trys_make_conn):
                # pick a trans, pick a place that is not the start place
                trans_id = self.pick_trans_with_preference()
                place_id = rd.choice([p for p in self.places if p != "start"])
                # check in innovs
                arc_id = innovs.get_arc(trans_id, place_id)
                if arc_id not in self.arcs:
                    break
        else: # if place/trans specified in arguments, just get that innov number
            arc_id = innovs.get_arc(trans_id, place_id)
            if arc_id in self.arcs:
                return # no connection is made
        new_arc = GArc(arc_id, trans_id, place_id)
        self.arcs[arc_id] = new_arc
        self.my_mutations.append('trans_place_arc')
        return


    def extend_new_place(self, trans_id=None) -> str:
        if not trans_id:
            for _try in range(params.num_trys_make_conn):
                trans_id = self.pick_trans_with_preference()
                ext_info = innovs.get_extension(trans_id)
                if ext_info["node"] not in self.places: # check if place not already exist
                    break
                else: # place already exists, reset ext_info
                    ext_info = None
        else:
            ext_info = innovs.get_extension(trans_id)
            if ext_info["node"] in self.places:
                print(f"trans {trans_id} has already been ext to {ext_info['node']}!")
                return
        if ext_info:
            self.places[ext_info["node"]] = GPlace(ext_info["node"])
            self.arcs[ext_info["arc"]] = GArc(ext_info["arc"], trans_id, ext_info["node"])
            self.my_mutations.append('extend_new_place')
            return ext_info["node"] # return id of new place
        return # nothing found


    def extend_new_trans(self, place_id=None) -> str:
        if len(self.places) <= 2:
            return
        if not place_id:
            for _try in range(params.num_trys_make_conn):
                place_id = rd.choice([p for p in self.places if p not in ["start", "end"]])
                ext_info = innovs.get_extension(place_id)
                if ext_info["node"] not in self.transitions: # check if transition not already exist
                    break
                else: # place already exists, reset ext_info
                    ext_info = None
        else:
            ext_info = innovs.get_extension(place_id)
            if ext_info["node"] in self.transitions:
                print(f"place {place_id} has already been ext to {ext_info['node']}!")
                return
        if ext_info:
            self.transitions[ext_info["node"]] = GTrans(ext_info["node"], is_task=False)
            self.arcs[ext_info["arc"]] = GArc(ext_info["arc"], place_id, ext_info["node"])
            self.my_mutations.append('extend_new_trans')
            return ext_info["node"] # return id of new trans
        return # nothing found


    def trans_trans_conn(self, source_id=None, target_id=None):
        # this pos function should check if the two are really transitions
        if not source_id and not target_id:
            for _try in range(params.num_trys_make_conn):
                source_id = self.pick_trans_with_preference()
                target_id = self.pick_trans_with_preference()
                a1_id, p_id, a2_id = innovs.get_trans_to_trans(source_id, target_id)
                if p_id not in self.places and source_id != target_id: # check if valid
                    break
                else:
                    p_id = None
        else:
            a1_id, p_id, a2_id = innovs.get_trans_to_trans(source_id, target_id)
            if p_id in self.places:
                print("trans-trans-conn already made")
                return
        if p_id:
            self.arcs[a1_id] = GArc(a1_id, source_id, p_id)
            self.places[p_id] = GPlace(p_id)
            self.arcs[a2_id] = GArc(a2_id, p_id, target_id)
            self.my_mutations.append('trans_trans_conn')
            return 


    def split_arc(self):
        if not self.arcs:
            return
        for _try in range(params.num_trys_split_arc):
            arc_to_split = rd.choice(list(self.arcs.values()))
            all_nodes = self.places | self.transitions
            source = all_nodes[arc_to_split.source_id]
            target = all_nodes[arc_to_split.target_id]
            # check if arc is trans -> place, and if this mutation should occur
            is_t_p = isinstance(source, GTrans)
            if (is_t_p and not target.is_start) or (not is_t_p and not source.is_end):
                sp_d = innovs.get_split(source, target)
                # check if mutation has already occured via place (could also be t or a)
                if sp_d["p"] not in self.places:
                    break
                else:
                    sp_d = None
        if sp_d:
            new_place = GPlace(sp_d["p"])
            new_trans = GTrans(sp_d["t"], False)
            if is_t_p:
                a1 = GArc(sp_d["a1"], source.id, new_place.id)
                a2 = GArc(sp_d["a2"], new_place.id, new_trans.id)
                a3 = GArc(sp_d["a3"], new_trans.id, target.id)
            else:
                a1 = GArc(sp_d["a1"], source.id, new_trans.id)
                a2 = GArc(sp_d["a2"], new_trans.id, new_place.id)
                a3 = GArc(sp_d["a3"], new_place.id, target.id)
            # insert new stuff to genome
            self.places[sp_d["p"]] = new_place
            self.transitions[sp_d["t"]] = new_trans
            self.arcs.update({sp_d["a1"]:a1, sp_d["a2"]:a2, sp_d["a3"]:a3})
            # remove old arc
            del self.arcs[arc_to_split.id]
            self.my_mutations.append('split_arc')
            return # mutation success
        return # nothing found


    def pick_trans_with_preference(self) -> str:
        """Returns transition id according to preferences set in params
        """
        # set of task trans and empty trans
        task_trans = set(innovs.get_task_list())
        empty_trans = set(self.transitions.keys()).difference(task_trans)
        # pick a trans
        if params.is_no_preference_for_tasks: # choose from all trans
            trans_id = rd.choice(list(self.transitions.keys()))
        elif rd.random() < params.prob_pick_empty_trans and empty_trans: # choose from empty trans (provided there are any)
            trans_id = rd.choice(list(empty_trans)) 
        else: # choose from tasks
            trans_id = rd.choice(list(task_trans))
        return trans_id


    def prune_extensions(self) -> None:
        source_nodes = map(lambda a: a.source_id, self.arcs.values())
        for ext_info in innovs.extensions.values():
            arc_id, node_id, ntype = ext_info["arc"], ext_info["node"], ext_info["ntype"]
            if arc_id in self.arcs:
                if node_id not in source_nodes:
                    # delete node, delete arc
                    if ntype == GTrans: del self.transitions[node_id]
                    elif ntype == GPlace: del self.places[node_id]
                    del self.arcs[arc_id]
                    # delete all arcs pointing to extension (they shouldnt exist, don't know why still here) 
                    arcs_to_ext = [a.id for a in self.arcs.values() if a.target_id == node_id]
                    for arc_id in arcs_to_ext:
                        del self.arcs[arc_id]
                    self.my_mutations.append('pruned_an_extension')


    def remove_arcs(self, arcs_to_remove=None) -> None:
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
        arc_values = {}
        associated_arcs = self.get_component_set()
        for comp in associated_arcs.keys():
            pop_fit_val = innovs.component_dict[comp]['t_val']
            for arc_id in associated_arcs[comp]:
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

    def get_compatibility_score(self, other_genome, debug=False) -> float:
        if params.distance_metric == "innovs":
            return self.innov_compatibility(other_genome)
        elif params.distance_metric == "behavior":
            return self.behavior_compatibility(other_genome)
        elif params.distance_metric == "components":
            return self.component_compatibility(other_genome)

# ----- innovs compatibility

    def innov_compatibility(self, other_genome, debug=False) -> float:
        """Calculates how similar this genome is to another genome according to the
        formula proposed in the original NEAT Paper. See distance variable to see
        how the formula works. It's parameters can be adjusted.
        """
        num_matched, num_disjoint, num_excess = 0, 0, 0
        # get sorted arrays of both genomes innovation historys 
        my_innovs = sorted(list(self.arcs.keys()))
        other_innovs = sorted(list(other_genome.arcs.keys()))
        # if both genomes don't have enabled links, they are compatible. stop calculations here
        if not (my_innovs and other_innovs):
            return params.species_boundary - 1
        # if either of the two genomes has no innovs, all of the other genes are excess genes
        # --> the first comparison for excess genes will evaluate true, and a score is calc.
        if not (my_innovs or other_innovs):
            older_innovs = [-1]
        # if both have innovs, find the lower last innovation (genome with older innovs)
        else:
            older_innovs = my_innovs if my_innovs[-1] < other_innovs[-1] else other_innovs
        # get a list of every shared innovation from both genomes, without duplicates
        all_Innovations = sorted(set(my_innovs + other_innovs))
        # the compat. score formula uses the num of enabled links in larger genome
        longest = max(len(my_innovs), len(other_innovs))
        # analyze every innovation and tally up matching, disjoint and excess scores
        innov_count = 0
        for innov in all_Innovations:
            # match: both genomes have invented this arc, calculate difference in number of links
            if innov in self.arcs and innov in other_genome.arcs:
                num_matched += 1
            # excess: elif innov_id exceeds last innov_id of older_Innovations genome, the
            # remaining Innovations in all_Innovations are excess genes. stop the search.
            elif innov > older_innovs[-1]:
                num_excess = len(all_Innovations) - innov_count
                break
            # disjoint: if we are sure, that both genomes still have Innovations,
            # (=not excess), and just one of them has the innov (xor check) -> disjoint
            elif innov in self.arcs or innov in other_genome.arcs:
                num_disjoint += 1
            innov_count += 1
        # calculate the distance TODO: this is still not perfect - maybe consider matched ones?
        distance = ((params.coeff_disjoint * num_disjoint) / longest + # disjoint increase dist
                    (params.coeff_excess * num_excess) / longest) # excess increase dist
        if distance < 0:
            raise Exception("Distance should not be < 0")
        if debug:
            print(f"""num_matched: {num_matched}\nnum_disjoint: {num_disjoint}
                num_excess: {num_excess}\ncomputed distance: {distance}""")
        return distance

# ----- behavior compatibility - TODO: this is not working yet
    @cache
    def get_extensive_variants(self, maxlen=None):
        net, im, fm = self.build_petri()
        if maxlen:
            res = extensive_playout(net, im, fm, parameters={"maxTraceLength": maxlen})
        else:
            res = extensive_playout(net, im, fm)
        return set(get_variants(res).keys()) # return only the variants


    def behavior_compatibility(self, other) -> float:
        # this needs to be fetched fresh, bc. the genome might have changed
        my_variants = self.get_extensive_variants()
        other_variants = other.get_extensive_variants()
        # calculate the fraction of overlapping traces
        overlap = len(my_variants.intersection(other_variants))
        union = len(my_variants.union(other_variants))
        fraction = overlap / union if union else 0
        return 1 - fraction # 0 means identical, 1 means completely different

# ----- component compatibility
    @cache
    def get_component_set(self) -> set:
        comp_dict = {}
        net, im, fm = self.build_petri()

        def format_tname(t): # all hidden transitions are named "t"
            return t.label if t.label in innovs.get_task_list() else "t"

        for md in maximal_decomposition(net, im, fm): # loop the components
            a_multi_set = Counter() # multiset of arcs in the component
            arc_ids = []
            for a in md[0].arcs:
                arc_ids.append(innovs.get_arc(a.source.name, a.target.name)) # get the innov id of arc in component
                if type(a.source) == PetriNet.Transition: # target must be a place
                    # pack into iterable (list) to avoid unpacking
                    a_multi_set.update([(format_tname(a.source), "p")]) # only one place per component
                else: # source must be a place, target must be a transition
                    a_multi_set.update([("p", format_tname(a.target))])

            # convert multiset to tuple to make it hashable, order of tuples must be the same
            if res := tuple(sorted(a_multi_set.items())): # only add non-empty components
                comp_dict[res] = arc_ids

        return comp_dict


    def component_compatibility(self, other_genome) -> float:
        """Distance metric based on percentage of components that are not shared
        """
        my_c = set(self.get_component_set().keys())
        other_c = set(other_genome.get_component_set().keys())
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
        # wow this is a piece of shit
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