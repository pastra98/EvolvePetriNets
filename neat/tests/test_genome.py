"""
current problem:
Need to somehow tell innovations about the fixed set of tasks, and we want to give
them the proper names...
"""
import innovs
import genome
import netobj
from pm4py.visualization.petri_net import visualizer
# from pm4py import view_petri_net


def main_loop():
    tasks = ["A", "B", "C", "D", "E"]
    innovs.set_tasks(tasks)
    gen_net = get_test_net()
    for i in range(10):
        f_name = ""
        gen_net.trans_place_arc()
        if i % 2 == 0:
            # gen_net.new_place()
            gen_net.split_arc()
            f_name = "arc_split"
        net, im, fm = gen_net.build_petri()
        gviz = visualizer.apply(net, im, fm)
        visualizer.save(gviz, f"./vis/{i}_{f_name}.png")
    return


def get_test_net():
    a_t = netobj.GTrans("A", True)
    b_t = netobj.GTrans("B", True)
    c_t = netobj.GTrans("C", True)
    d_t = netobj.GTrans("D", True)
    e_t = netobj.GTrans("E", True)
    t_d = {"A":a_t, "B":b_t, "C":c_t, "D":d_t, "E":e_t}
    ##########
    a1_id = innovs.store_new_arc("start", "A")
    a1 = netobj.GArc(a1_id, "start", "A")
    a2_id = innovs.store_new_arc("E", "end")
    a2 = netobj.GArc(a2_id, "E", "end")
    a_d = {a1_id:a1, a2_id:a2}
    ##########
    gen_net = genome.GeneticNet("test", t_d, dict(), a_d)
    return gen_net

main_loop()