from pm4py import view_petri_net
from pm4py.visualization.petri_net import visualizer

def vis_genome(g, view=True, save=False):
    g.build_petri()
    if save:
        net_gviz = visualizer.apply(g.net, g.im, g.fm)
        savepath = f"vis/t1/{g.id}_petrinet.png"
        visualizer.save(net_gviz, savepath)
        print(f"saved under {savepath}")
    if view:
        print(g.id)
        view_petri_net(g.net, g.im, g.fm)