# %%
import os
os.chdir('..')

# %%
import pm4py
log = pm4py.read_xes("../pm_data/running_example.xes")
log

# %%
from neatutils.splicing import balanced_splice
bs = balanced_splice(log, 4)

# %%
from pm4py.algo.discovery.alpha import algorithm as alpha
from pm4py.algo.discovery.inductive import algorithm as inductive
from pm4py.algo.discovery.heuristics import algorithm as heuristics
from pm4py.algo.discovery.ilp import algorithm as ilp

# -------------------- applying mining algos

def mine_wf_net(log, algo, *algo_params):
    # WIP: this is not finished
    net, im, fm = algo.apply(log, parameters=algo_params)
    return net, im, fm

# this is just for testing the layout of the new params layout for bootstrap
configdesign = {
    "bootstrap": {
        "alpha" : {
            "n_splices": 10,
            "splicing_method": "balanced"
        }
    }
}

# %%
for miner in [alpha, inductive, heuristics, ilp]:
    print(miner)
    for log in bs:
        net, im, fm = mine_wf_net(bs[0], miner)
        pm4py.view_petri_net(net)
        print()

# %%
# %%
type(inductive.apply(bs[0]))