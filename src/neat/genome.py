import random as rd
import itertools
from copy import deepcopy
from typing import Tuple

from graphviz import Digraph

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

from . import params, innovs, fitnesscalc
from .netobj import GArc, GPlace, GTrans


class GeneticNet:
    def __init__(self, transitions: dict, places: dict, arcs: dict) -> None:
        """transitions, places and arcs must either be dicts containing valid id: netobj
        key-value pairings, or an empty dict.
        - Adds task transitions and start/end place automatically
        - make sure that argument dicts contain fresh genes
        Reasoning: Cannot use mutable default args, and didn't want to use *args or **kwargs
        """
        self.id = innovs.get_new_genome_id()
        self.net: PetriNet = None
        self.im: Marking = None
        self.fm: Marking = None
        self.fitness: float = None
        # make Transition genes for every task saved in innovs and add to genome
        task_trans = {t: GTrans(t, True) for t in innovs.tasks}
        self.transitions = transitions | task_trans
        # make place genes for start and end places
        se_places = {"start":GPlace("start", is_start=True), "end":GPlace("end", is_end=True)}
        self.places = places | se_places
        # make arcs
        self.arcs = arcs

# ------------------------------------------------------------------------------
# MUTATIONS --------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def mutate(self, mutation_rate):
        if rd.random() < params.prob_t_p_arc[mutation_rate]:
            self.trans_place_arc()
        if rd.random() < params.prob_p_t_arc[mutation_rate]:
            self.place_trans_arc()
        if rd.random() < params.prob_t_t_conn[mutation_rate]:
            pass
        if rd.random() < params.prob_new_p[mutation_rate]:
            pass
        if rd.random() < params.prob_new_empty_t[mutation_rate]:
            pass
        if rd.random() < params.prob_split_arc[mutation_rate]:
            pass
        # for arc in self.arcs:
        #     if rd.random() < params.prob_increase_arcs[mutation_rate]:
        #         pass
        #     if rd.random() < params.prob_disable_arc[mutation_rate]:
        #         pass
        return


    def place_trans_arc(self, place_id=None, trans_id=None) -> None:
        if not place_id and not trans_id: # no trans/place specified in arguments
            # search until a place/trans combi is found that is not already connected
            for _try in range(params.num_trys_make_conn):
                # pick a place that is not the end place, pick a trans
                place_id = rd.choice([p for p in self.places if p != "end"])
                trans_id = self.pick_trans_with_preference()
                # check in innovs
                arc_id = innovs.check_arc(place_id, trans_id)
                if arc_id not in self.arcs:
                    break
        else: # if place/trans specified in arguments, just get that innov number
            arc_id = innovs.check_arc(place_id, trans_id)
        new_arc = GArc(arc_id, place_id, trans_id)
        self.arcs[arc_id] = new_arc
        return


    def trans_place_arc(self, trans_id=None, place_id=None):
        if not trans_id and not place_id: # no trans/place specified in arguments
            # search until a place/trans combi is found that is not already connected
            for _try in range(params.num_trys_make_conn):
                # pick a trans, pick a place that is not the start place
                trans_id = self.pick_trans_with_preference()
                place_id = rd.choice([p for p in self.places if p != "start"])
                # check in innovs
                arc_id = innovs.check_arc(trans_id, place_id)
                if arc_id not in self.arcs:
                    break
        else: # if place/trans specified in arguments, just get that innov number
            arc_id = innovs.check_arc(trans_id, place_id)
        new_arc = GArc(arc_id, trans_id, place_id)
        self.arcs[arc_id] = new_arc
        return


    def pick_trans_with_preference(self) -> str:
        """Just pick a transition according to preferences set in params
        """
        # set of task trans and empty trans
        task_trans = set(innovs.tasks)
        empty_trans = set(self.transitions.keys()).difference(task_trans)
        # pick a trans
        if params.is_no_preference_for_tasks: # choose from all trans
            trans_id = rd.choice(list(self.transitions.keys()))
        elif rd.random() < params.prob_pick_task_trans: # choose from tasks
            trans_id = rd.choice(list(task_trans))
        else: # choose from empty trans
            trans_id = rd.choice(list(empty_trans)) 
        return trans_id


    def new_place(self, trans_id=None) -> str:
        if trans_id:
            trans = self.transitions[trans_id]
        else:
            for _try in range(params.num_trys_make_conn):
                trans = rd.choice(list(self.transitions.values()))
                # this can also be more fancy, e.g. consider number of dead trans
                if rd.random() < params.prob_connect_nontask_trans and not trans.is_task:
                    break
        # !!!!!!!!!!!!!!!!!!!
        # IMPORTANT WILL HAVE TO CHECK IF TRANS IS ALREADY CONNECTED TO EMPTY PLACE!!!!
        # !!!!!!!!!!!!!!!!!!!
        place_id = innovs.store_new_node(GPlace)
        arc_id = innovs.check_arc(trans.id, place_id)
        new_place = GPlace(place_id)
        new_arc = GArc(arc_id, trans.id, place_id)
        self.places[place_id] = new_place
        self.arcs[arc_id] = new_arc
        return place_id


    def new_empty_trans(self, place_id=None):
        # would need to check innovations to prevent creating duplicate empty transitions
        new_trans_id = innovs.store_new_node(GTrans)
        new_trans = GTrans(new_trans_id, is_task=False)
        self.transitions[new_trans_id] = new_trans
        # kinda pointless because new_trans_id was just generated without check in innovs
        arc_id = innovs.check_arc(place_id, new_trans_id) 
        new_arc = GArc(arc_id, place_id, new_trans_id)
        self.arcs[arc_id] = new_arc
        return new_trans_id


    def trans_trans_conn(self, source_id=None, target_id=None):
        # this pos function should check if the two are really transitions
        if not source_id and not target_id:
            # code this later
            pass # try to find two transitions that can be connected
        place_id = innovs.check_trans_to_trans(source_id, target_id)
        if place_id not in self.places:
            # this will always make a new innov
            self.places[place_id] = GPlace(place_id)
            arc1_id = innovs.check_arc(source_id, place_id)
            self.arcs[arc1_id] = GArc(arc1_id, source_id, place_id)
            arc2_id = innovs.check_arc(place_id, target_id)
            self.arcs[arc2_id] = GArc(arc2_id, place_id, target_id)
        
    def split_arc(self, arc_id=None):
        for _try in range(params.num_trys_split_arc):
            arc_to_split = rd.choice(list(self.arcs.values()))
################################################################################
            # super shitty hack immediately delete this later
            joined = self.places | self.transitions
            source = joined[arc_to_split.source_id]
            target = joined[arc_to_split.target_id]
################################################################################
            # check if arc is trans -> place, and if this mutation should occur
            is_t_p = isinstance(source, GTrans)
            if is_t_p and not (rd.random() < params.prob_t_p):
                break
            if (is_t_p and not target.is_start) or (not is_t_p and not source.is_end):
                sp_d = innovs.check_split(source, target)
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
                # disable old arc
                arc_to_split.n_arcs=0
                return


    def increase_arcs(self, arc_id=None):
        pass


    def disable_arc(self, arc_id=None):
        pass


    def disable_place(self, place_id=None):
        pass

# ------------------------------------------------------------------------------
# REPRODUCTION RELATED STUFF ---------------------------------------------------
# ------------------------------------------------------------------------------

    def get_compatibility_score(self, other_genome) -> float:
        """Calculates how similar this genome is to another genome according to the
        formula proposed in the original NEAT Paper. See distance variable to see
        how the formula works. It's parameters can be adjusted.
        """
        # numbers needed for compatibility score formula
        num_matched = 0
        num_disjoint = 0
        num_excess = 0
        arc_count_diff = 0.0
        # get sorted arrays of both genomes innovation historys 
        my_innovs = sorted([a_id for a_id in self.arcs if self.arcs[a_id].n_arcs > 0])
        other_innovs = sorted([a_id for a_id in other_genome.arcs if other_genome.arcs[a_id].n_arcs > 0])
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
                arc_count_diff += abs(self.arcs[innov].n_arcs - other_genome.arcs[innov].n_arcs)
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
        # If none match, set to 1 to prevent division by zero if no genes match. The
        # match score is still gives zero because the weight difference is zero.
        if num_matched == 0:
            num_matched = 1
        # calculate the distance
        distance = ((params.coeff_matched * arc_count_diff) / num_matched +
                    (params.coeff_disjoint * num_disjoint) / longest +
                    (params.coeff_excess * num_excess) / longest)
        # print(f"num_matched : {num_matched}")
        # print(f"num_disjoint : {num_disjoint}")
        # print(f"num_excess : {num_excess}")
        # print(f"arc_count_diff : {arc_count_diff}")
        return distance


    def crossover(self, other_genome):
        """TODO: need to implement
        """
        return self.copy()


    def clone(self):
        """returns a deepcopy
        """
        new_transitions = {k: v.get_copy() for k, v in self.transitions.items()}
        new_places = {k: v.get_copy() for k, v in self.places.items()}
        new_arcs = {k: v.get_copy() for k, v in self.arcs.items()}
        return GeneticNet(new_transitions, new_places, new_arcs)

# ------------------------------------------------------------------------------
# FITNESS RELATED STUFF --------------------------------------------------------
# ------------------------------------------------------------------------------

    def build_petri(self) -> None:
        try:
            del self.net
            del self.im
            del self.fm
            print("Genome already has net, rebuilding it")
        except:
            pass
        self.net = PetriNet(f"{self.id}-Net")
        merged_nodes = self.places | self.transitions
        # only add transitions that are actually connected (since all tasks are in genome)
        connected_t = self.get_connected_trans()
        for place_id in self.places:
            place = self.places[place_id]
            place.pm4py_obj = PetriNet.Place(place_id)
            self.net.places.add(place.pm4py_obj)
        for trans_id in self.transitions:
            if trans_id in connected_t:
                trans = self.transitions[trans_id]
                trans.pm4py_obj = PetriNet.Transition(trans_id, label=trans_id)
                self.net.transitions.add(trans.pm4py_obj)
        for arc_id in self.arcs:
            arc = self.arcs[arc_id]
            if arc.n_arcs > 0:
                source_obj = merged_nodes[arc.source_id].pm4py_obj
                target_obj = merged_nodes[arc.target_id].pm4py_obj
                arc.pm4py_obj = petri_utils.add_arc_from_to(source_obj, target_obj, self.net)
        # initial marking
        self.im = Marking()
        start = self.places["start"].pm4py_obj
        self.im[start] = 1
        # final marking
        self.fm = Marking()
        end = self.places["end"].pm4py_obj
        self.fm[end] = 1
        return


    def evaluate_fitness(self, log) -> None:
        if not self.net:
            self.build_petri()
        else:
            print("net has already been built!!, not building a new one")
        # fitness eval
        aligned_traces = fitnesscalc.get_aligned_traces(log, self.net, self.im, self.fm)
        trace_fitness = fitnesscalc.get_replay_fitness(aligned_traces)
        # soundness check
        is_sound = woflan.apply(self.net, self.im, self.fm, parameters={
            woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
            woflan.Parameters.PRINT_DIAGNOSTICS: False,
            woflan.Parameters.RETURN_DIAGNOSTICS: False
            })
        # precision
        prec = fitnesscalc.get_precision(log, self.net, self.im, self.fm)
        # generealization
        gen = fitnesscalc.get_generalization(self.net, aligned_traces)
        # simplicity
        simp = simplicity_evaluator.apply(self.net)
        # some preliminary fitness measure
        self.fitness = (
            + params.perc_fit_traces_weight * (trace_fitness["perc_fit_traces"] / 100)
            + params.soundness_weight * int(is_sound)
            + params.precision_weight * prec
            + params.generalization_weight * gen
            + params.simplicity_weight * simp
        )
        if self.fitness <= 0:
            raise Exception("Fitness below 0 should not be possible!!!")
        return

# ------------------------------------------------------------------------------
# MISC STUFF -------------------------------------------------------------------
# ------------------------------------------------------------------------------
    def get_connected_trans(self) -> set:
        # get set of all transitions that are connected to the network via arcs
        connected = [(a.source_id, a.target_id) for a in self.arcs.values()]
        connected = set(itertools.chain.from_iterable(connected))
        return set(self.transitions.keys()).intersection(connected)


    def get_graphviz(self) -> Digraph:
        # parameter stuff, TODO: think about where to put this
        fsize = "20"
        tcol = "yellow"
        pcol = "lightblue"
        ahead = "normal"
        viz = Digraph()
        viz.graph_attr['rankdir'] = 'LR'
        viz.attr('node', shape='box')
        viz.attr(overlap='false')
        # viz.attr(size="21, 21")
        # viz.attr(size="11, 11")
        connected_t = self.get_connected_trans()
        # transitions
        for t in self.transitions:
            if t in connected_t:
                if self.transitions[t].is_task: # task
                    viz.node(t, t, style='filled', fillcolor=tcol, border='1', fontsize=fsize)
                else: # empty trans
                    viz.node(t, t, style='filled', fontcolor="white", fillcolor="black", fontsize=fsize)
        # places
        for p in self.places:
            if p == "start":
                viz.node("start", style='filled', fillcolor="green", fontsize=fsize, shape='circle', fixedsize='true', width='0.75')
            elif p == "start":
                viz.node("end", style='filled', fillcolor="orange", fontsize=fsize, shape='circle', fixedsize='true', width='0.75')
            else:
                viz.node(p, p, style='filled', fillcolor=pcol, fontsize=fsize, shape='circle', fixedsize='true', width='0.75')
        # arcs
        for name, a in self.arcs.items():
            for _ in range(a.n_arcs):
                viz.edge(a.source_id, a.target_id, label=str(name), fontsize=fsize, arrowhead=ahead)
        return viz
