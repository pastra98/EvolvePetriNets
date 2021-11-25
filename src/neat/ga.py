from . import params, startconfigs, innovs

class GeneticAlgorithm:
    def __init__(self, params_name:str, log)-> None:
        self.history = {}
        self.params_name = params_name
        self.log = log

        params.read_file(params_name)
        innovs.reset()

        self.population = self.get_initial_pop()

    
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