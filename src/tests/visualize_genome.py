from pm4py import view_petri_net
from pm4py.visualization.petri_net import visualizer
from IPython.core.display import display, HTML, Image

def vis_genome(g, view=True, save=False):
    net, im, fm = g.build_petri()
    if save:
        net_gviz = visualizer.apply(net, im, fm)
        savepath = f"vis/t1/{g.id}_petrinet.png"
        visualizer.save(net_gviz, savepath)
        print(f"saved under {savepath}")
    if view:
        print(g.id)
        view_petri_net(net, im, fm)



def show_graphviz(g):
    gviz = g.get_graphviz()
    display(Image(data=gviz.pipe(format="png"), unconfined=True, retina=True))