from typing import List
from .netobj import GArc, GTrans, GPlace

tasks = []
nodes = {"start":GPlace, "end":GPlace}
arcs = {}
splits = {}

trans = {}

curr_genome_id = 0

curr_arc_id = 0
curr_node_id = 0

def reset():
    # implement this function, to be called when setting up a new ga
    print("Reset innovs!")
    return

def get_new_genome_id():
    global curr_genome_id
    curr_genome_id += 1
    return curr_genome_id

def check_trans_to_trans(source_id, target_id):
    if (source_id, target_id) in trans:
        return trans[(source_id, target_id)]
    else:
        new_id = store_new_node(GPlace)
        trans[(source_id, target_id)] = new_id
        return new_id

def check_split(source, target):
    check_tasks_set()
    split_name = f"{source.id}-x->{target.id}"
    if split_name in splits:
        new_place_id, new_trans_id, arc1_id, arc2_id, arc3_id = splits[split_name]
    else:
        new_place_id = store_new_node(GPlace)
        new_trans_id = store_new_node(GTrans)
        # trans -> place: TRANS -ARC1-> NEW_PLACE -ARC2-> NEW_TRANS -ARC3-> PLACE
        if isinstance(source, GTrans) and isinstance(target, GPlace):
            arc1_id = store_new_arc(source.id, new_place_id)
            arc2_id = store_new_arc(new_place_id, new_trans_id)
            arc3_id = store_new_arc(new_trans_id, target.id)
        # place -> trans: PLACE -ARC1-> NEW_TRANS -ARC2-> NEW_PLACE -ARC3-> TRANS
        if isinstance(source, GPlace) and isinstance(target, GTrans):
            arc1_id = store_new_arc(source.id, new_trans_id)
            arc2_id = store_new_arc(new_trans_id, new_place_id)
            arc3_id = store_new_arc(new_place_id, target.id)
        # store new ids
        split_d = {"p":new_place_id, "t":new_trans_id, "a1":arc1_id, "a2":arc2_id, "a3":arc3_id}
        splits[split_name] = split_d
        return split_d

def check_arc(source_id: int, target_id: int) -> int:
    check_tasks_set()
    arc_name = f"{source_id}--->{target_id}"
    if arc_name in arcs:
        return arcs[arc_name]
    else:
        new_id = store_new_arc(source_id, target_id)
        return new_id

def store_new_arc(source_id: int, target_id: int)-> int:
    check_tasks_set()
    arc_name = f"{source_id}--->{target_id}"
    global curr_arc_id
    curr_arc_id += 1
    arcs[arc_name] = curr_arc_id
    return curr_arc_id

def store_new_node(node_type: type, *task_name: str)-> str:
    if task_name: # it is a task
        node_name = task_name
        tasks.append(task_name)
        is_task = True
    else: # it is NOT a task
        check_tasks_set()
        global curr_node_id
        curr_node_id += 1
        prefix = "t" if node_type == GTrans else "p"
        node_name = prefix + str(curr_node_id)
        is_task = False
    nodes[node_name] = (node_type, is_task)
    return node_name

def set_tasks(task_list: list):
    for name in task_list:
        store_new_node(GTrans, name)
    check_tasks_set()

def check_tasks_set():
    if not tasks:
        raise Exception("Initial task list must be set!!")