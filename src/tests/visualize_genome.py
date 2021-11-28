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


# src: graphviz_visualization (c:\Users\pauls\AppData\Local\Programs\Python\Python39\Lib\site-packages\pm4py\visualization\petri_net\common\visualize.py:112)

import tempfile

from graphviz import Digraph

from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.util import exec_utils
from enum import Enum
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_TIMESTAMP_KEY

def graphviz_visualization(net, image_format="png", initial_marking=None, final_marking=None, decorations=None,
                           debug=False, set_rankdir=None, font_size="12", bgcolor="transparent"):
    """
    Provides visualization for the petrinet

    Parameters
    ----------
    net: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    image_format
        Format that should be associated to the image
    initial_marking
        Initial marking of the Petri net
    final_marking
        Final marking of the Petri net
    decorations
        Decorations of the Petri net (says how element must be presented)
    debug
        Enables debug mode
    set_rankdir
        Sets the rankdir to LR (horizontal layout)

    Returns
    -------
    viz :
        Returns a graph object
    """
    if initial_marking is None:
        initial_marking = Marking()
    if final_marking is None:
        final_marking = Marking()
    if decorations is None:
        decorations = {}

    font_size = str(font_size)

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(net.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': bgcolor})
    if set_rankdir:
        viz.graph_attr['rankdir'] = set_rankdir
    else:
        viz.graph_attr['rankdir'] = 'LR'

    # transitions
    viz.attr('node', shape='box')
    for t in net.transitions:
        if t.label is not None:
            if t in decorations and "label" in decorations[t] and "color" in decorations[t]:
                viz.node(str(id(t)), decorations[t]["label"], style='filled', fillcolor=decorations[t]["color"],
                         border='1', fontsize=font_size)
            else:
                viz.node(str(id(t)), str(t.label), fontsize=font_size)
        else:
            if debug:
                viz.node(str(id(t)), str(t.name), fontsize=font_size)
            elif t in decorations and "color" in decorations[t] and "label" in decorations[t]:
                viz.node(str(id(t)), decorations[t]["label"], style='filled', fillcolor=decorations[t]["color"],
                         fontsize=font_size)
            else:
                viz.node(str(id(t)), "", style='filled', fillcolor="black", fontsize=font_size)

        if petri_properties.TRANS_GUARD in t.properties:
            guard = t.properties[petri_properties.TRANS_GUARD]
            viz.node(str(id(t))+"guard", style="dotted", label=guard)
            viz.edge(str(id(t))+"guard", str(id(t)), arrowhead="none", style="dotted")

    # places
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted([x for x in list(net.places) if x in initial_marking], key=lambda x: x.name)
    places_sort_list_fm = sorted([x for x in list(net.places) if x in final_marking and not x in initial_marking],
                                 key=lambda x: x.name)
    places_sort_list_not_im_fm = sorted(
        [x for x in list(net.places) if x not in initial_marking and x not in final_marking], key=lambda x: x.name)
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places not belonging neither to the initial marking and the final marking
    # - at last, the places belonging to the final marking (but not to the initial marking)
    # in this way, is more probable that the initial marking is on the left and the final on the right
    places_sort_list = places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm

    for p in places_sort_list:
        if p in initial_marking:
            viz.node(str(id(p)), str(initial_marking[p]), style='filled', fillcolor="green", fontsize=font_size, shape='circle', fixedsize='true', width='0.75')
        elif p in final_marking:
            viz.node(str(id(p)), "", style='filled', fillcolor="orange", fontsize=font_size, shape='circle', fixedsize='true', width='0.75')
        else:
            if debug:
                viz.node(str(id(p)), str(p.name), fontsize=font_size, shape="ellipse")
            else:
                if p in decorations and "color" in decorations[p] and "label" in decorations[p]:
                    viz.node(str(id(p)), decorations[p]["label"], style='filled', fillcolor=decorations[p]["color"],
                             fontsize=font_size, shape="ellipse")
                else:
                    viz.node(str(id(p)), "", shape='circle', fixedsize='true', width='0.75')

    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))
    for a in arcs_sort_list:
        arrowhead = "normal"
        if petri_properties.ARCTYPE in a.properties:
            if a.properties[petri_properties.ARCTYPE] == petri_properties.RESET_ARC:
                arrowhead = "vee"
            elif a.properties[petri_properties.ARCTYPE] == petri_properties.INHIBITOR_ARC:
                arrowhead = "dot"
        if a in decorations and "label" in decorations[a] and "penwidth" in decorations[a]:
            viz.edge(str(id(a.source)), str(id(a.target)), label=decorations[a]["label"],
                     penwidth=decorations[a]["penwidth"], fontsize=font_size, arrowhead=arrowhead)
        elif a in decorations and "color" in decorations[a]:
            viz.edge(str(id(a.source)), str(id(a.target)), color=decorations[a]["color"], fontsize=font_size, arrowhead=arrowhead)
        else:
            if a.weight > 1:
                viz.edge(str(id(a.source)), str(id(a.target)), fontsize=font_size, arrowhead=arrowhead, label=str(a.weight))
            else:
                viz.edge(str(id(a.source)), str(id(a.target)), fontsize=font_size, arrowhead=arrowhead)
    viz.attr(overlap='false')

    viz.format = image_format

    return viz
