from typing import Tuple
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
import random as rd

import params
import innovs
from netobj import GArc, GPlace, GTrans

class GeneticNet:
    def __init__(self, id, transitions, places, arcs) -> None:
        self.id = id
        self.net: PetriNet
        self.transitions = transitions
        self.places = places | {"start":GPlace("start", is_start=True),
                                "end":GPlace("end", is_end=True)}
        self.arcs = arcs
        self.initial_marking: Marking
        self.final_marking: Marking

    def mutate(self):
        pass

    def trans_place_arc(self):
        for _try in range(params.num_trys_make_conn):
            place = rd.choice(list(self.places.values()))
            trans = rd.choice(list(self.transitions.values()))
            # this can also be more fancy, e.g. consider number of dead trans
            if rd.random() < params.prob_connect_nontask_trans and not trans.is_task:
                break
            trans_to_place = rd.random() < params.prob_t_p
            arc_id = ""
            # trans -> place
            if trans_to_place and not place.is_start:
                arc_id = innovs.check_arc(trans.id, place.id)
            # place -> trans
            elif (not trans_to_place) and (not place.is_end):
                arc_id = innovs.check_arc(place.id, trans.id)
            # finally check if we haven't already picked that arc
            if arc_id != "" and arc_id not in self.arcs:
                if trans_to_place:
                    new_arc = GArc(arc_id, trans.id, place.id)
                else:
                    new_arc = GArc(arc_id, place.id, trans.id)
                self.arcs[arc_id] = new_arc
                return

    def new_place(self):
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
            return

    # def new_trans(self):
    #     pass

    def trans_trans_conn(self, source_id=None, target_id=None):
        if not source_id and not target_id:
            # code this later
            pass # try to find two transitions that can be connected
        place_id = innovs.check_trans_to_trans(source_id, target_id)
        if place_id not in self.places:
            self.places[place_id] = GPlace(place_id)
            arc1_id = innovs.check_arc(source_id, place_id)
            self.arcs[arc1_id] = GArc(arc1_id, source_id, place_id)
            arc2_id = innovs.check_arc(place_id, target_id)
            self.arcs[arc2_id] = GArc(arc2_id, place_id, target_id)
        

    # def place_place_conn(self):
    #     pass

    def split_arc(self):
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

    def increase_arcs(self, is_t_to_p: bool):
        pass

    def disable_arc(self, is_t_to_p: bool):
        pass

    def build_petri(self) -> Tuple:
        self.net = PetriNet(f"{self.id}-Net")
        merged_nodes = self.places | self.transitions
        for place_id in self.places:
            place = self.places[place_id]
            place.pm4py_obj = PetriNet.Place(place_id)
            self.net.places.add(place.pm4py_obj)
        for trans_id in self.transitions:
            trans = self.transitions[trans_id]
            # check if it is a task, else make it a dead transition
            if trans.is_task:
                trans.pm4py_obj = PetriNet.Transition(trans_id, label=trans_id)
            else:
                trans.pm4py_obj = PetriNet.Transition(trans_id)
            self.net.transitions.add(trans.pm4py_obj)
        for arc_id in self.arcs:
            arc = self.arcs[arc_id]
            if arc.n_arcs > 0:
                source_obj = merged_nodes[arc.source_id].pm4py_obj
                target_obj = merged_nodes[arc.target_id].pm4py_obj
                arc.pm4py_obj = petri_utils.add_arc_from_to(source_obj, target_obj, self.net)
        # initial marking
        self.initial_marking = Marking()
        start = self.places["start"].pm4py_obj
        self.initial_marking[start] = 1
        # final marking
        self.final_marking = Marking()
        end = self.places["end"].pm4py_obj
        self.final_marking[end] = 1
        return self.net, self.initial_marking, self.final_marking

    def evaluate_fitness():
        pass
