import numpy as np
import random as rd
from copy import copy

from neatutils import timer
from neatutils.splicing import get_spliced_log_on_gen

from neat import params, innovs, initial_population
from neat.genome import GeneticNet
from neat.species import Species

class GeneticAlgorithm:
    def __init__(
            self,
            params_name:str,
            log,
            is_minimal_serialization=False,
            is_pop_serialized=True,
            is_timed=True)-> None:

        self.history = {}
        self.improvements = {} # store every best genome that improved upon the previous one
        self.params_name = params_name
        self.log = log
        self.curr_gen = 1

        # not sure if I will use this, can set mutation rate context for non-neat
        self.global_mutation_rate = 0 # 0 -> normal or 1 -> high

        # measurement stuff general
        if is_timed:
            self.timer = timer.Timer()
            self.is_timed = True
        self.is_minimal_serialization = is_minimal_serialization
        self.is_pop_serialized = is_pop_serialized
        self.best_genome = None
        self.total_pop_fitness = None
        self.avg_pop_fitness = None
        self.old_innovnum = 0
        self.new_innovnum = 0
        
        # measurements specific to speciation
        self.num_new_species = 0 # these are set by calling get_initial_pop (only used if strat speciation)
        self.species = [] 
        self.population = []
        self.best_species = None

        params.load(params_name)
        innovs.reset()
        innovs.set_tasks(log)

        self.set_initial_pop()

    
    def next_generation(self) -> dict:
        """Makes new generation, evaluates it, and returns info that can be used
        as stopping criteria.
        """
        # evaluate old generation and save results in history
        self.evaluate_curr_generation()
        self.is_curr_gen_evaluated = True

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
        if self.is_timed: self.timer.start("evaluate_curr_generation", self.curr_gen)
        self.evaluate_curr_genomes()
        if params.selection_strategy == "speciation":
            self.evaluate_curr_species()
        if self.is_timed: self.timer.stop("evaluate_curr_generation", self.curr_gen)
        return


    def evaluate_curr_genomes(self) -> None:
        # if log is spliced according to params, pass spliced log for curr gen here
        if params.log_splices:
            log = get_spliced_log_on_gen(self.curr_gen, self.log)
        # calc fitness for every genome
        self.total_pop_fitness = 0
        for g in self.population:
            g.evaluate_fitness(log)
            self.total_pop_fitness += g.fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        # check if fitness improvement happened
        if self.curr_gen > 1 and self.population[0].fitness > self.best_genome.fitness:
            self.improvements[self.curr_gen] = self.population[0]
        # update best genome
        self.best_genome = self.population[0]
        return


    def log_gen_info(self) -> None:
        """Writes info about current gen into history. Careful not to add too much
        info that can be a-posteriori calculated in get_info_about_gen()
        """
        if self.is_curr_gen_evaluated:
            # dict for evaluation of current gen
            gen_info = self.history[self.curr_gen] = {}

            # save info about species
            if params.selection_strategy == "speciation":
                if self.is_pop_serialized:
                    if self.is_minimal_serialization:
                        gen_info["species"] = [s.get_curr_info() for s in self.species]
                        gen_info["best species"] = self.best_species.get_curr_info()
                    else:
                        gen_info["species"] = [copy(s) for s in self.species]
                        gen_info["best species"] = copy(self.best_species)
                gen_info["num total species"] = len(self.species)
                gen_info["num new species"] = self.num_new_species
                gen_info["best species avg fitness"] = self.best_species.avg_fitness

            # save info about generation in general
            if self.is_pop_serialized:
                if self.is_minimal_serialization:
                    gen_info["best genome"] = self.best_genome.get_curr_info()
                    gen_info["population"] = [g.get_curr_info() for g in self.population]
                else:
                    gen_info["best genome"] = copy(self.best_genome)
                    gen_info["population"] = [copy(g) for g in self.population]

            gen_info["num total innovations"] = self.new_innovnum
            gen_info["num new innovations"] = self.new_innovnum - self.old_innovnum
            gen_info["best genome fitness"] = self.population[0].fitness
            gen_info["avg pop fitness"] = self.total_pop_fitness / params.popsize
            gen_info["total pop fitness"] = self.total_pop_fitness
            if self.is_timed:
                gen_info["times"] = self.timer.get_gen_times(self.curr_gen)
                if self.curr_gen == 0:
                    gen_info["times"]["pop_update"] = 0 # TODO total hack lol
        else:
            raise Exception("Tried to log gen before evaluating")
        return


    def get_printable_gen_info(self, gen) -> dict:
        """returns a dict of info about a generation. Uses info from self.history,
        and writes nothing to it. Intended to print info during evolution runs.
        Can calculate new info that I don't want to save in history.
        """
        gen_info  = self.history[gen] # stuff to take info from
        print_info = {"gen": gen} # other stuff to put info into?
        keep = [
            "avg pop fitness", "num total innovations", "num new innovations",
            "total pop fitness", "best genome fitness", "times"
            ]
        if params.selection_strategy == "speciation":
            keep += [
                "num total species", "num new species", "best species avg fitness"
            ]
        print_info = print_info | {k: gen_info[k] for k in keep}
        return print_info


    def set_initial_pop(self) -> None:
        """
        """ 
        if params.start_config == "concurrent_traces": # DEPRECATED
            initial_pop = initial_population.generate_n_traces_with_concurrency(params.popsize, self.log)
        elif params.start_config == "random":
            initial_pop = initial_population.generate_n_random_genomes(params.popsize, self.log)
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
        results = {
            "history": self.history,
            "param_values": params.get_curr_curr_dict(),
            "best_genome": self.best_genome,
            "improvements": self.improvements,
            "max_fitness": self.best_genome.fitness
        }
        return results

# ------------------------------------------------------------------------------
# POPULATION UPDATES -----------------------------------------------------------
# ------------------------------------------------------------------------------

    def pop_update(self) -> None:
        if self.is_timed: self.timer.start("pop_update", self.curr_gen)
        self.old_innovnum = innovs.curr_arc_id

        if params.selection_strategy == "speciation":
            self.speciation_pop_update()
        elif params.selection_strategy == "roulette":
            self.roulette_pop_update()
        elif params.selection_strategy == "truncation": # https://www.researchgate.net/publication/259461147_Selection_Methods_for_Genetic_Algorithms
            self.truncation_pop_update()

        self.new_innovnum = innovs.curr_arc_id

        if self.is_timed: self.timer.stop("pop_update", self.curr_gen)

# SPECIATION -------------------------------------------------------------------

    def speciation_pop_update(self) -> None:
        """
        """ 
        self.num_new_species = 0
        new_genomes = []
        num_spawned = 0
        for s in self.species:
            # reduce num_to_spawn if it would exceed population size
            if num_spawned == params.popsize:
                break
            elif num_spawned + s.num_to_spawn > params.popsize:
                s.num_to_spawn = params.popsize - num_spawned
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
                if baby.get_compatibility_score(s.representative) > params.species_boundary:
                    # if the baby is too different, find an existing species to change
                    # into. If no compatible species is found, a new one is made and returned
                    found_species = self.find_and_add_to_species(baby)
                else:
                    # If the baby is still within the species of it's parents, add it as member
                    s.add_member(baby)
                num_spawned += 1
                new_genomes.append(baby)
        # if all the current species didn't provide enough offspring, get some more
        if params.popsize - num_spawned > 0:
            new_genomes += self.get_more_mutated_leaders(params.popsize - num_spawned)
        self.population = new_genomes
        return


    def evaluate_curr_species(self) -> None:
        """
        """
        updated_species = []
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
            if (not s.obliterate) or (s == self.best_species) or (self.best_genome.species_id == s.name):
                updated_species.append(s)
                total_species_avg_fitness += s.avg_fitness
                total_adjusted_species_avg_fitness += s.avg_fitness_adjusted 
            else:
                num_dead_species += 1 # dont add it to updated species
        if not updated_species or total_adjusted_species_avg_fitness == 0:
            raise Exception("mass extinction")
        # calculate offspring amt based on fitness relative to the total_adjusted_species_avg_fitness
        for s in updated_species:
            s.calculate_offspring_amount(total_adjusted_species_avg_fitness)
        self.species = updated_species


    def find_and_add_to_species(self, new_genome) -> Species:
        """Tries to find a species to which the given genome is similar enough to be
        added as a member. If no compatible species is found, a new one is made. Returns
        the species (but the genome still needs to be added as a member).
        """
        found_species: Species = None
        # try to find an existing species to which the genome is close enough to be a member
        comp_score = params.species_boundary
        for s in self.species:
            if new_genome.get_compatibility_score(s.representative) < comp_score:
                comp_score = new_genome.get_compatibility_score(s.representative)
                found_species = s
        # new genome matches no current species -> make a new one
        if not found_species:
            found_species = self.make_new_species(new_genome)
        found_species.add_member(new_genome)
        return found_species


    def get_more_mutated_leaders(self, num) -> list:
        # iterate over species leaders, but mutate them
        new_genomes = []
        for i in range(num):
            if i + 1 > len(self.species):
                s = self.species[i % len(self.species)]
            else:
                s = self.species[i]
            baby = s.elite_spawn_with_mutations()
            if baby.get_compatibility_score(s.representative) > params.species_boundary:
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
        new_species = Species(new_species_id)
        new_species.representative = founding_member
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

        new_genomes.append(self.best_genome.clone()) # add best g w.o. mutation
        self.population = new_genomes