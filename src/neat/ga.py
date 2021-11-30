from os import popen
import random as rd

from . import params, innovs, startconfigs
from .genome import GeneticNet
from .species import Species

class GeneticAlgorithm:
    def __init__(self, params_name:str, log)-> None:
        self.history = {}
        self.params_name = params_name
        self.log = log

        params.load(params_name)
        innovs.reset()

        self.curr_generation = 0 # TODO: need to increment that /deal with it

        # these are set by calling get_initial_pop (only used if strat speciation)
        self.num_new_species = 0
        self.curr_species = [] 
        self.population = []
        self.set_initial_pop()

        self.best_genome = None
        self.best_species = None

        self.total_pop_fitness = None
        self.avg_pop_fitness = None

        # not sure if I will use this, can set mutation rate context for non-neat
        self.global_mutation_rate = 0 # 0 -> normal or 1 -> high
    
    def next_generation(self) -> dict:
        """Makes new generation, evaluates it, and returns info that can be used
        as stopping criteria.
        """
        self.evaluate_curr_generation()
        # TODO increment generation counter, store shit in a dictionary
        if params.selection_strategy == "speciation":
            self.speciation_pop_update()
        elif params.selection_strategy == "roulette":
            self.roulette_pop_update()
        elif params.selection_strategy == "truncation": # https://www.researchgate.net/publication/259461147_Selection_Methods_for_Genetic_Algorithms
            self.truncation_pop_update()
        # increment generation
        self.curr_generation += 1
        # return info about curr generation
        return {
            "best_genome": self.best_genome,
            "best_species": self.best_species,
            "total_pop_fitness": self.total_pop_fitness,
            "avg_pop_fitness": self.avg_pop_fitness,
            "gen": self.curr_generation,
            "other stuff": "",
            "some condition": "xyz"
        }

    def evaluate_curr_generation(self) -> None:
        """
        """
        # this is where multithreading magic will happen - i.e. that will need a rewrite
        self.total_pop_fitness = 0
        for g in self.population:
            g.evaluate_fitness(self.log)
            self.total_pop_fitness += g.fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        # determine best genome, avg pop fitness
        self.best_genome = self.population[0]
        self.avg_pop_fitness = self.total_pop_fitness / params.popsize
        return

    def get_ga_info(self) -> dict:
        """ Should only be called at end of this ga instance, returns bunch of info
        """
        return

    def set_initial_pop(self) -> None:
        """
        """ 
        if params.start_config == "concurrent_traces":
            initial_pop = startconfigs.generate_n_traces_with_concurrency(params.popsize, self.log)
        elif params.start_config == "blabla":
            pass
            # make other type of startconfig
        # if using speciation, generate initial set of spec, place genomes there
        if params.selection_strategy == "speciation":
            for g in initial_pop:
                found_species = self.find_species(g)
                found_species.add_member(g)
        # set initial pop
        self.population = initial_pop
        return

# ------------------------------------------------------------------------------
# POPULATION UPDATES -----------------------------------------------------------
# ------------------------------------------------------------------------------

# SPECIATION -------------------------------------------------------------------

    def speciation_pop_update(self) -> None:
        """
        """ 
        self.update_curr_species()
        self.num_new_species = 0
        new_genomes = []
        num_spawned = 0
        for s in self.curr_species:
            # reduce num_to_spawn if it would exceed population size
            if num_spawned == params.popsize:
                break
            elif num_spawned + s.num_to_spawn > params.popsize:
                s.num_to_spawn = params.popsize - num_spawned
            spawned_elite = False
            # spawn all the new members of a species
            for _ in s.num_to_spawn:
                baby: GeneticNet = None
                # first clone the species leader for elitism
                if not spawned_elite and params.elitism:
                    baby = s.elite_spawn()
                    spawned_elite = True
                # if at least 2 member and prob_asex, spawn asex baby
                elif len(s.pool) < 2 or rd.random() < params.prob_asex:
                    baby = s.asex_spawn()
                # else produce produce a crossed-over genome
                else:
                    baby = s.mate_spawn()
                # check if baby should speciate away from it's current species
                if baby.get_compatibility_score(s.representative) > params.species_boundary:
                    # if the baby is too different, find an existing species to change
                    # into. If no compatible species is found, a new one is made and returned
                    found_species = self.find_species(baby)
                    found_species.add_member(baby)
                else:
                    # If the baby is still within the species of it's parents, add it as member
                    s.add_member(baby)
                num_spawned += 1
                new_genomes.append(baby)
        # if all the current species didn't provide enough offspring, get some more
        if params.popsize - num_spawned > 0:
            new_genomes += self.make_hybrids(params.popsize - num_spawned)
        self.population = new_genomes
        return

    def update_curr_species(self) -> None:
        """
        """
        updated_species = []
        total_adjusted_species_avg_fitness = 0
        total_species_avg_fitness = 0
        num_dead_species = 0
        for s in self.curr_species:
            s.update()
            if not s.obliterate:
                updated_species.append(s)
                total_species_avg_fitness += s.avg_fitness
                total_adjusted_species_avg_fitness += s.avg_fitness_adjusted 
            else:
                num_dead_species += 1 # dont add it to pool, no more s.purge()
        if not updated_species or total_adjusted_species_avg_fitness == 0:
            raise Exception("mass extinction")
        # calculate offspring amt based on fitness relative to the total_adjusted_species_avg_fitness
        for s in updated_species:
            s.calculate_offspring_amount(total_adjusted_species_avg_fitness)
        # order the updated species by fitness, select the current best species, return
        updated_species.sort(key=lambda s: s.avg_fitness, reverse=True)
        self.best_species = updated_species[0]
        self.curr_species = updated_species
        return

    def find_species(self, new_genome) -> Species:
        """Tries to find a species to which the given genome is similar enough to be
        added as a member. If no compatible species is found, a new one is made. Returns
        the species (but the genome still needs to be added as a member).
        """
        found_species: Species = None
        # try to find an existing species to which the genome is close enough to be a member
        comp_score = params.species_boundary
        for s in self.curr_species:
            if new_genome.get_compatibility_score(s.representative) < comp_score:
                comp_score = new_genome.get_compatibility_score(s.representative)
                found_species = s
        # new genome matches no current species -> make a new one
        if not found_species:
            found_species = self.make_new_species(new_genome)
        return found_species

    def make_new_species(self, founding_member: GeneticNet) -> Species:
        """Generates a new species with a unique id, assigns the founding member as
        representative, and adds the new species to curr_species and returns it.
        """
        new_species_id = f"{self.curr_generation}_{founding_member.id}"
        new_species = Species(new_species_id)
        new_species.representative = founding_member
        self.curr_species.append(new_species)
        self.num_new_species += 1
        return new_species



# ROULETTE ---------------------------------------------------------------------

    def roulette_pop_update(self) -> None:
        """
        """ 
        return

# TRUNCATION -------------------------------------------------------------------

    def truncation_pop_update(self) -> None:
        """
        """ 
        return