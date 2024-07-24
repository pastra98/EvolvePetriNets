import numpy as np
from copy import copy
from datetime import datetime

from neatutils import timer

from neat import params, initial_population
from neat.genome import GeneticNet
from neat.species import Species

# ------------------------------------------------------------------------------
# GeneticAlgorithm class -------------------------------------------------------
# ------------------------------------------------------------------------------

class GeneticAlgorithm:
    def __init__(self, params_name:str, log)-> None:

        self.start_time = datetime.now()
        self.history = {}
        self.improvements = {} # store every best genome that improved upon the previous one
        self.params_name = params_name
        self.log = log
        self.curr_gen = 1

        self.pop_component_tracker = PopulationComponentTracker()

        # not sure if I will use this, can set mutation rate context for non-neat
        self.global_mutation_rate = 0 # 0 -> normal or 1 -> high

        # measurement stuff general
        self.timer = timer.Timer()
        self.curr_best_genome = None
        self.total_pop_fitness = None
        self.avg_pop_fitness = None
        self.old_comp_num = 0
        self.new_comp_num = 0

        # makeup of the new generation
        self.num_crossover = 0
        self.num_asex = 0
        self.num_elite = 0
        
        # measurements specific to speciation
        self.num_new_species = 0 # these are set by calling get_initial_pop (only used if strat speciation)
        self.species = [] 
        self.surviving_species = []
        self.population = []
        self.best_species = None

        params.load(params_name)
        if params.mutation_type == "atomic":
            params.max_arcs_removed = 1 # bad practice, but I want to ensure this

        self.set_initial_pop()

    
    def next_generation(self) -> dict:
        """Makes new generation, evaluates it, and returns info that can be used
        as stopping criteria.
        """
        # evaluate old generation and save results in history
        self.evaluate_curr_generation()

        # writes info into gen info dict
        self.log_gen_info()

        # increment generation
        self.curr_gen += 1

        # make a new population
        self.pop_update()
        self.is_curr_gen_evaluated = False

        # return info about curr generation
        return self.get_printable_gen_info(self.curr_gen - 1)
        

    def evaluate_curr_generation(self) -> None:
        """Evaluates genomes (and species if necessary)
        """
        # evaluate current genomes and species
        self.timer.start("evaluate_curr_generation", self.curr_gen)
        self.evaluate_curr_genomes()
        if params.selection_strategy == "speciation":
            self.evaluate_curr_species()
        self.timer.stop("evaluate_curr_generation", self.curr_gen)

        # component_tracker updates the component fitnesses
        self.pop_component_tracker.update_global_components(self.population, self.curr_gen)
        self.old_comp_num = self.new_comp_num
        self.new_comp_num = len(self.pop_component_tracker.component_history)
        self.is_curr_gen_evaluated = True
        return


    def evaluate_curr_genomes(self) -> None:
        # calc fitness for every genome
        self.total_pop_fitness = 0
        for g in self.population:
            g.evaluate_fitness(self.log, self.curr_gen)
            self.total_pop_fitness += g.fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        # check if fitness improvement happened
        if self.curr_gen > 1 and self.population[0].fitness > self.curr_best_genome.fitness:
            self.improvements[self.curr_gen] = self.population[0]
        # update best genome
        self.curr_best_genome = self.population[0]
        return


    def log_gen_info(self) -> None:
        """Writes info about current gen into history. Careful not to add too much
        info that can be a-posteriori calculated in get_info_about_gen()
        """
        if self.is_curr_gen_evaluated:
            # dict for evaluation of current gen
            gen_info = self.history[self.curr_gen] = {}

            # save current species & info about them
            if params.selection_strategy == "speciation":
                gen_info["species"] = [s.get_curr_info() for s in self.species]
                gen_info["best_species"] = self.best_species.name
                gen_info["num_total_species"] = len(self.species)
                gen_info["num_new_species"] = self.num_new_species
                gen_info["best_species_avg_fitness"] = self.best_species.avg_fitness

            # save current population
            gen_info["population"] = [g.get_curr_info() for g in self.population]

            # save info about generation in general
            gen_info["best_genome"] = self.curr_best_genome.id
            gen_info["num_total_components"] = self.new_comp_num
            gen_info["num_new_components"] = self.new_comp_num - self.old_comp_num
            gen_info["num_crossover"] = self.num_crossover
            gen_info["num_elite"] = self.num_elite
            gen_info["num_asex"] = self.num_asex
            gen_info["best_genome_fitness"] = self.curr_best_genome.fitness
            gen_info["avg_pop_fitness"] = self.total_pop_fitness / params.popsize
            gen_info["total_pop_fitness"] = self.total_pop_fitness
            gen_info["times"] = self.timer.get_gen_times(self.curr_gen)
            if self.curr_gen == 0:
                gen_info["times"]["pop_update"] = 0
        else:
            raise Exception("Tried to log gen before evaluating")
        return


    def get_printable_gen_info(self, gen) -> dict:
        """returns a dict of info about a generation. Uses info from self.history,
        and writes nothing to it. Intended to print info during evolution runs.
        Can calculate new info that I don't want to save in history.
        """
        gen_info  = self.history[gen] # stuff to take info from
        print_info = {"gen": gen, "best_fitness": self.curr_best_genome.fitness}
        keep = [
            "avg_pop_fitness", "num_total_components", "num_new_components",
            "total_pop_fitness", "times"
            ]
        if params.selection_strategy == "speciation":
            keep += [
                "num_total_species", "num_new_species", "best_species_avg_fitness",
                "num_crossover", "num_elite", "num_asex"
            ]
        print_info = print_info | {k: gen_info[k] for k in keep}
        return print_info


    def set_initial_pop(self) -> None:
        """
        """ 
        # TODO: consider moving this stuff into one function and start_config as arg
        if params.start_config == "random":
            initial_pop = initial_population.generate_n_random_genomes(
                params.popsize,
                self.log,
                self.pop_component_tracker
                )
        elif params.start_config == "bootstrap":
            initial_pop = initial_population.get_bootstrapped_population(
                params.popsize,
                self.log,
                self.pop_component_tracker
                )
        else:
            raise NotImplementedError()
        # if using speciation, generate initial set of spec, place genomes there
        if params.selection_strategy == "speciation":
            for g in initial_pop:
                found_species = self.find_and_add_to_species(g)
                self.best_species = found_species # just to initialize best species to a species for allowing comparison
        # set initial pop
        self.population = initial_pop
        return
    

    def get_ga_final_info(self) -> dict:
        """Returns dict of history along with dict of param_values and max_fitnesss
        """ 
        info = {
            "history": self.history,
            "best_genome": self.curr_best_genome,
            "improvements": self.improvements,
            "total_components": len(self.pop_component_tracker.component_history),
        }
        if params.selection_strategy == "speciation":
            info["species_leaders"] = [s.leader for s in self.species]
        return info

# ------------------------------------------------------------------------------
# POPULATION UPDATES -----------------------------------------------------------
# ------------------------------------------------------------------------------

    def pop_update(self) -> None:
        self.timer.start("pop_update", self.curr_gen)

        if params.selection_strategy == "speciation":
            self.speciation_pop_update()
        elif params.selection_strategy == "roulette":
            self.roulette_pop_update()
        elif params.selection_strategy == "truncation": # https://www.researchgate.net/publication/259461147_Selection_Methods_for_Genetic_Algorithms
            self.truncation_pop_update()

        self.timer.stop("pop_update", self.curr_gen)

# SPECIATION -------------------------------------------------------------------

    def speciation_pop_update(self) -> None:
        """Get spawns from species, and add them to the population.
        """ 
        # remove all species that won't go into next gen after logging
        self.species = self.surviving_species
        # first get the crossover spawns
        if self.curr_gen >= params.start_crossover:
            n_crossover = int(params.popsize * params.pop_perc_crossover)
            new_genomes = self.get_crossover_spawns(n_crossover)
        else:
            new_genomes = []
        self.num_crossover = len(new_genomes)
        # then get the remaining asex spawns
        self.num_asex = params.popsize - self.num_crossover
        self.num_new_species = 0
        num_spawned = 0
        for s in self.species: # species already sorted by fitness due to eval
            # reduce num_to_spawn if it would exceed population size
            if num_spawned == self.num_asex:
                break
            elif num_spawned + s.num_to_spawn > self.num_asex:
                s.num_to_spawn = self.num_asex - num_spawned
            spawned_elite = False
            # spawn all the new members of a species
            for _ in range(s.num_to_spawn):
                baby: GeneticNet = None
                # if elitism, spawn clone of the species leader
                if not spawned_elite and params.elitism:
                    baby = s.elite_spawn()
                    spawned_elite = True
                # spawn asex baby
                else:
                    baby = s.asex_spawn()
                # check if baby should speciate away from it's current species
                if params.compat_to_multiple: # get species component set
                    cset = s.component_set
                else:
                    cset = s.representative.get_unique_component_set()
                if baby.get_genetic_distance(cset) > params.species_boundary:
                    # if the baby is too different, find an existing species to change
                    # into. If no compatible species is found, a new one is made and returned
                    found_species = self.find_and_add_to_species(baby)
                else:
                    # If the baby is still within the species of it's parents, add it as member
                    s.add_member(baby)
                num_spawned += 1
                new_genomes.append(baby)
        # if all the current species didn't provide enough offspring, get some more
        self.num_asex = num_spawned # update num_asex to correct value
        self.num_elite = params.popsize - len(new_genomes)
        new_genomes += self.get_more_mutated_leaders(self.num_elite)
        self.population = new_genomes
        return


    def evaluate_curr_species(self) -> None:
        """update species and update spawn amounts
        """
        self.surviving_species = []
        total_adjusted_species_avg_fitness = 0
        total_species_avg_fitness = 0
        num_dead_species = 0
        # first update all, determine best species
        for s in self.species:
            s.update()
        # order the updated species by fitness, select the current best species
        self.species.sort(key=lambda s: s.avg_fitness, reverse=True)
        self.best_species = self.species[0]
        # now that best species is determined, kill off stale species and update spawn amount
        for s in self.species:
            # don't kill off best species or species containing curr best genome
            if (not s.obliterate) or (s == self.best_species) or (self.curr_best_genome.species_id == s.name):
                self.surviving_species.append(s)
                total_species_avg_fitness += s.avg_fitness
                total_adjusted_species_avg_fitness += s.avg_fitness_adjusted 
            else:
                num_dead_species += 1 # dont add it to updated species
        if not self.surviving_species or total_adjusted_species_avg_fitness == 0:
            raise Exception("mass extinction")
        # calculate offspring amt based on fitness relative to the total_adjusted_species_avg_fitness
        for s in self.surviving_species:
            s.calculate_offspring_amount(total_adjusted_species_avg_fitness)
        return


    def find_and_add_to_species(self, new_genome: GeneticNet) -> Species:
        """Tries to find a species to which the given genome is similar enough to be
        added as a member. If no compatible species is found, a new one is made. Returns
        the species (but the genome still needs to be added as a member).
        """
        found_species: Species = None
        # try to find an existing species to which the genome is close enough to be a member
        distance = params.species_boundary
        for s in self.species:
            if params.compat_to_multiple:
                cset = s.component_set
            else:
                cset = s.representative.get_unique_component_set()
            if new_genome.get_genetic_distance(cset) < distance:
                distance = new_genome.get_genetic_distance(cset)
                found_species = s
        # new genome matches no current species -> make a new one
        if not found_species:
            found_species = self.make_new_species(new_genome)
        found_species.add_member(new_genome)
        return found_species


    def get_crossover_spawns(self, num_to_spawn: int) -> list:
        # select species that will cross over already sorted by fitness due to eval
        cs = self.species[:int(len(self.species)*params.species_perc_crossover)]
        # crossover spawns depend on relative fitness of each species
        cs_fitnesses = [s.avg_fitness for s in cs]
        cs_total_fit = sum(cs_fitnesses)
        cs_spawn_counts = [int((f/cs_total_fit)*num_to_spawn) for f in cs_fitnesses]
        # ensure that exactly num_to_spawn genomes get spawned
        cs_spawn_counts[0] += num_to_spawn - sum(cs_spawn_counts)
        new_genomes = []

        for i, species in enumerate(cs): 
            mom_species = cs[i]
            for j in range(cs_spawn_counts[i]):
                dad_species = cs[(i + j + 1) % len(cs)]
                dad = dad_species.leader
                mom = mom_species.leader
                baby = mom.crossover(dad)
                # if mom and dad are not able to reproduce, find a new dad lol
                if not baby:
                    for new_dad in dad_species.members:
                        baby = mom.crossover(new_dad)
                        if baby: break
                if baby:
                    mom_species.add_member(baby)
                    new_genomes.append(baby)
        return new_genomes


    def get_more_mutated_leaders(self, num_to_spawn) -> list:
        # iterate over species leaders, but mutate them
        new_genomes = []
        for i in range(num_to_spawn):
            if i + 1 > len(self.species):
                s = self.species[i % len(self.species)]
            else:
                s = self.species[i]
            baby: GeneticNet = s.elite_spawn_with_mutations()
            if params.compat_to_multiple:
                cset = s.component_set
            else:
                cset = s.representative.get_unique_component_set()
            if baby.get_genetic_distance(cset) > params.species_boundary:
                found_species = self.find_and_add_to_species(baby)
            else:
                s.add_member(baby)
            new_genomes.append(baby)
        return new_genomes


    def make_new_species(self, founding_member: GeneticNet) -> Species:
        """Generates a new species with a unique id, assigns the founding member as
        representative, and adds the new species to curr_species and returns it.
        """
        new_species_id = f"{self.curr_gen}_{founding_member.id}"
        new_species = Species(new_species_id, founding_member)
        self.species.append(new_species)
        self.num_new_species += 1
        return new_species

# ROULETTE ---------------------------------------------------------------------

    def roulette_pop_update(self) -> None:
        """perform roulette wheel selection
        """ 
        def roulette_select(pop, probs) -> GeneticNet:
            chosen_genome = np.random.choice(pop, p=probs)
            return chosen_genome.clone()

        fitnesses = [g.fitness for g in self.population]
        fit_sum = sum(fitnesses)
        probabilities = [fit / fit_sum for fit in fitnesses]

        n_elites = 100
        new_genomes = []
        for _ in range(params.popsize - n_elites): # elitism: keep a slot for best g
            new_g = roulette_select(self.population, probabilities)
            new_g.mutate(1)
            new_genomes.append(new_g)

        for i, g in enumerate(self.population):
            new_elite = g.clone() # append unmutated top g
            if i > 0: # mutate all other tops
                new_elite.mutate(1)
            new_genomes.append(new_elite)
            if i == n_elites - 1:
                break
        self.population = new_genomes

# TRUNCATION -------------------------------------------------------------------

    def truncation_pop_update(self) -> None:
        """
        """ 
        new_genomes = []
        pool = self.population[:int(params.popsize*params.spawn_cutoff)]

        for i in range(params.popsize - 1): # elitism: keep a slot for best g
            if i < len(pool):
                new_g = pool[i].clone()
            else:
                new_g = pool[i % len(pool)].clone()
            new_g.mutate(1)
            new_genomes.append(new_g)

        new_genomes.append(self.curr_best_genome.clone()) # add best g w.o. mutation
        self.population = new_genomes

# ------------------------------------------------------------------------------
# ComponentTracker class -------------------------------------------------------
# ------------------------------------------------------------------------------

class PopulationComponentTracker:
    """This class tracks the components of the overall population
    """
    def __init__(self)-> None:
        self.component_dict = dict()
        self.component_history = dict()


    def update_global_components(self, population: list, generation: int):
        """Registers the unique components in the history, also pairs up components
        of all genomes with their fitness, if use_t_vals this will be used to calculate
        mean difference significance for all components in the population.
        """
        # pair the components of the new population with their fitness values
        fitness_components = []
        for g in population:
            c_set = g.get_unique_component_set()
            for c in list(c_set):
                if c not in self.component_history:
                    self.component_history[c] = generation
            fitness_components.append((g.fitness, c_set))

        # TODO: can do other stuff that influences the probability map here
        # like check the best improvements
        # TODO: maybe use info from more generations later
        if params.use_t_vals:
            self.component_dict = self.calculate_t_vals(fitness_components)

            

    # @njit(parallel=True)
    def calculate_t_vals(self, fitness_components: list) -> dict:
        comp_fitness_dict = {}

        # TODO: this logic could also be handled in caller, especially if other
        # keys are added to the dict for other prob_map influences
        pop_fit, comp_fitness_dict  = [], {}
        for fc in fitness_components: 
            pop_fit.append(fc[0])
            for component in fc[1]: # assort fitness val with every component
                    comp_fitness_dict.setdefault(component,
                                            {"all_fitnesses": [],
                                            "t_val": None}
                                        ).get("all_fitnesses").append(fc[0])
        pop_fit = np.array(pop_fit)
        pop_sum, pop_len = pop_fit.sum(), len(pop_fit)
        pop_df = pop_len - 2

        for component in comp_fitness_dict:
            included = np.array(comp_fitness_dict[component]["all_fitnesses"])
            comp_fitness_dict[component]["t_val"] = self.compute_t(included, pop_fit, pop_len, pop_sum, pop_df)
            # this should only happen in the first gen, when the start and end connections
            # yield components, shared by everyone, making t value comparison impossible
            if len(included) == params.popsize:
                comp_fitness_dict[component]["t_val"] = 1 # TODO: revisit this number maybe later
        return comp_fitness_dict

    # @njit(parallel=True)
    def compute_t(inc, pop, pop_len, pop_sum, pop_df):
        inc_sum, inc_len = inc.sum(), len(inc)
        inc_avg_fit = inc_sum / inc_len
        exc_len = pop_len - inc_len
        exc_avg_fit = (pop_sum - inc_sum) / exc_len
        mean_diff = inc_avg_fit - exc_avg_fit

        inc_df, exc_df = inc_len-1, exc_len-1

        inc_ss = ((inc - inc_avg_fit)**2).sum()
        inc_var = inc_ss / inc_len
        
        exc_ss = ((pop - exc_avg_fit)**2).sum() - inc_ss - inc_len*mean_diff**2
        exc_var = exc_ss / exc_len

        pool_var = (inc_var*inc_df + exc_var*exc_df) / pop_df
        se = (pool_var/inc_len + pool_var/exc_len)**0.5
        return float(mean_diff/se)
