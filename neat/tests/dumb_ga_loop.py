import params
import loginterface

class DumbGeneticAlgorithm:
    def __init__(self, parampath:str, logpath:str)-> None:
        self.history = {}
        self.population = {}
        self.params_name = parampath

        params.read_file(parampath)
        self.log = loginterface(logpath)
    
    def next_generation(self) -> dict:
        """Makes new generation, evaluates it, and returns info that can be used
        as stopping criteria.
        """
        return
    
    def create_initial_population(self) -> dict:
        return

    def run_n_generations(self, n):
        self.population = self.create_initial_population(popsize)
        for _n in range(n):
            self.history[n] = self.next_generation()
        return self.history
        


################################################################################
# log_path = "./pm_data/BPI_Challenge_2012.xes"
parampath = "./neat/param_files/"
logpath = "./pm_data/m1_log.xes"

test_ga = DumbGeneticAlgorithm(parampath, logpath)