from pm4py.visualization.petri_net import visualizer
from . import params, startconfigs, innovs

class GeneticAlgorithm:
    def __init__(self, params_name:str, log)-> None:
        self.history = {}
        self.params_name = params_name
        self.log = log

        params.read_file(params_name)
        innovs.reset()

        self.population = self.get_initial_pop()

        print(self.population)
        print(len(self.population))
        for g, l in self.population: # why is dis a nested list? lots of stuff to fix here
            # implement genome.copy() method
            print(g.id)
            net, im, fm = g.build_petri()
            net_gviz = visualizer.apply(net, im, fm)
            savepath = f"vis/t1/{g.id}_petrinet.png"
            visualizer.save(net_gviz, savepath)
            print(f"saved under {savepath}")

    
    def next_generation(self) -> dict:
        """ Makes new generation, evaluates it, and returns info that can be used
        as stopping criteria.
        """
        return

    def get_evaluation(self) -> dict:
        """ Should only be called at end of this ga instance, returns bunch of info
        """
        return

    def get_initial_pop(self) -> list:
        """
        """ 
        if params.start_config == "concurrent_traces":
            return startconfigs.generate_n_traces_with_concurrency(params.popsize, self.log)