
# -------------------- applying mining algos

def mine_wf_net(log, algo, algo_params):
    # WIP: this is not finished
    net, im, fm = algo.apply(log, parameters=algo_params)
    return net, im, fm