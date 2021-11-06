from os import terminal_size
import params

class GeneticAlgorithm:
    def __init__(self, params_name:str)-> None:
        self.history = {}
        self.population = {}
        self.params_name = params_name

        params.read_file(params_name)
    
    def next_generation(self) -> dict:
        """ Makes new generation, evaluates it, and returns info that can be used
        as stopping criteria.
        """
        return

    def get_evaluation(self) -> dict:
        """ Should only be called at end of this ga instance, returns bunch of info
        """
        return