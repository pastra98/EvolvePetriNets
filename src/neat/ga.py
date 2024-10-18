import numpy as np
import random as rd
from datetime import datetime
from uuid import uuid4
from typing import List

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
        self.population: List[GeneticNet] = []
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
        self.species: List[Species] = []
        self.surviving_species: List[Species] = []
        self.best_species: Species = None

        params.load(params_name)

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
        self.new_comp_num = len(self.pop_component_tracker.component_dict)
        # lastly save the components ids to the genomes
        for g in self.population:
            g.save_component_ids()
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
        # create initial pop
        initial_pop = initial_population.create_initial_pop(
            self.log,
            self.pop_component_tracker
            )
        # if using speciation, generate initial set of spec, place genomes there
        if params.selection_strategy == "speciation":
            # initialize species and add a first one based on first genome
            initial_species: List[Species] = []
            initial_species.append(self.get_fresh_species(initial_pop[0]))
            for g in initial_pop[1:]: # skip the first genome
                found_species = self.find_species(g, initial_species)
                if found_species:
                    found_species.add_member(g)
                else: # new genome matches no current species -> make a fresh one
                    fresh_species = self.get_fresh_species(g)
                    fresh_species.add_member(g)
                    initial_species.append(fresh_species)
                self.best_species = found_species # just to initialize best species to a species for allowing comparison
            self.species = initial_species
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
            "total_components": len(self.pop_component_tracker.component_dict),
            "component_dict": self.pop_component_tracker.get_inverted_comp_dict(),
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
        else:
            raise NotImplementedError()

        self.timer.stop("pop_update", self.curr_gen)

# SPECIATION -------------------------------------------------------------------

    def speciation_pop_update(self) -> None:
        """Get spawns from species, and add them to the population.
        """ 
        # remove all species that won't go into next gen after evaluation
        self.species = self.surviving_species

        # get the crossover spawns
        crossover_g = []
        if self.curr_gen >= params.start_crossover:
            n_crossover = int(params.popsize * params.pop_perc_crossover)
            crossover_g = self.get_crossover_spawns(n_crossover)
        self.num_crossover = len(crossover_g)

        # get elite spawns
        elite_g = []
        if params.elitism:
            for s in self.species:
                l = s.leader.clone()
                s.add_member(l)
                elite_g.append(l)
        self.num_elite = len(elite_g)
        
        # get the remaining asex spawns
        self.num_asex = params.popsize - self.num_crossover - self.num_elite
        new_species: List[Species] = []
        asex_g: List[GeneticNet] = []
        for s in self.species: # species already sorted by fitness due to eval
            # calc remaining spawns, break if no spawns left
            remaining_spawns = self.num_asex - len(asex_g)
            if not remaining_spawns: break
            # calc and reduce num_to_spawn if it would exceed remaining spawns
            num_to_spawn = int(s.fitness_share * self.num_asex) + 1 # add one to mitigate rounding down errors
            num_to_spawn = min(num_to_spawn, remaining_spawns)
            # spawn all the new members of a species
            for _ in range(num_to_spawn):
                baby = s.asex_spawn()
                # check if baby still simillar enough to current species
                cset = s.component_set if params.compat_to_multiple else s.representative.get_unique_component_set()
                if baby.get_genetic_distance(cset) < params.species_boundary:
                    s.add_member(baby)
                else: # try to adopt
                    found_species = self.find_species(baby, self.species + new_species)
                    if found_species:
                        found_species.add_member(baby)
                    else: # new genome matches no current species -> make a fresh one
                        fresh_species = self.get_fresh_species(baby)
                        fresh_species.add_member(baby)
                        new_species.append(fresh_species)
                asex_g.append(baby)
        # add new species
        self.species += new_species
        self.num_new_species = len(new_species)

        # set new population
        self.population = crossover_g + elite_g + asex_g
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
            s.calculate_fitness_share(total_adjusted_species_avg_fitness)
        return


    def find_species(self, new_genome: GeneticNet, species_to_search: List[GeneticNet]):
        """Tries to find a species to which the given genome is similar enough to be
        added as a member. If no compatible species is found, None is returned
        """
        found_species: Species = None
        # try to find an existing species to which the genome is close enough to be a member
        distance = params.species_boundary
        for s in species_to_search:
            cset = s.component_set if params.compat_to_multiple else s.representative.get_unique_component_set()
            if new_genome.get_genetic_distance(cset) < distance:
                distance = new_genome.get_genetic_distance(cset)
                found_species = s
        return found_species


    def get_crossover_spawns(self, num_to_spawn: int) -> list:
        # tournament selection approach
        new_genomes = []

        while len(new_genomes) < num_to_spawn:
            tournament = rd.sample(self.population, params.tournament_size)
            tournament.sort(key=lambda g: g.fitness)
            mom, dad = tournament[0], tournament[1]
            baby = mom.crossover(dad)
            if baby:
                new_genomes.append(baby)
                if params.selection_strategy == "speciation":
                    mom_species = next(s for s in self.species if s.name == mom.species_id)
                    mom_species.add_member(baby)

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


    def get_fresh_species(self, founding_member: GeneticNet) -> Species:
        """Generates a new species with a unique id, assigns the founding member as
        representative, and returns it.
        """
        return Species(f"{self.curr_gen}_{founding_member.id}", founding_member)

# ROULETTE ---------------------------------------------------------------------

    def roulette_pop_update(self) -> None:
        """perform roulette wheel selection
        """ 
        def roulette_select(pop, probs) -> GeneticNet:
            return np.random.choice(pop, p=probs)

        fitnesses = [g.fitness for g in self.population]
        fit_sum = sum(fitnesses)
        probabilities = [fit / fit_sum for fit in fitnesses]

        # determine number of spawns
        if self.curr_gen >= params.start_crossover:
            n_crossover = int(params.popsize * params.pop_perc_crossover)
        else:
            n_crossover = 0
        n_elites = int(params.popsize * params.pop_perc_elite)
        n_asex = params.popsize - n_elites - n_crossover

        # elite spawns - without mutation
        elite_spawns = []
        for i, g in enumerate(self.population):
            new_elite = g.clone() # append unmutated top g
            elite_spawns.append(new_elite)
            if i == n_elites - 1: break

        # crossover spawns - without mutation
        crossover_spawns = []
        while len(crossover_spawns) < n_crossover:
            # those should be from the previous gen - which they are??
            p1 = roulette_select(self.population, probabilities)
            p2 = roulette_select(self.population, probabilities)
            new_g = p1.crossover(p2)
            if new_g:
                crossover_spawns.append(new_g)

        # asex spawns - with mutation
        asex_spawns = []
        for _ in range(n_asex):
            p = roulette_select(self.population, probabilities)
            new_g = p.clone()
            new_g.mutate(0)
            asex_spawns.append(new_g)

        self.population = elite_spawns + crossover_spawns + asex_spawns

# TRUNCATION -------------------------------------------------------------------

    def truncation_pop_update(self) -> None:
        """Simply chops away the sorted population falling below spawn_cutoff
        Makes no effort at maintaining diversity
        """ 
        if self.curr_gen >= params.start_crossover:
            n_crossover = int(params.popsize * params.pop_perc_crossover)
        else:
            n_crossover = 0
        n_elites = int(params.popsize * params.pop_perc_elite)
        n_asex = params.popsize - n_elites - n_crossover

        pool = self.population[:int(params.popsize*params.spawn_cutoff)]

        # elite spawns - without mutation
        elite_spawns = []
        for i, g in enumerate(self.population):
            new_elite = g.clone() # append unmutated top g
            elite_spawns.append(new_elite)
            if i == n_elites - 1: break

        # crossover spawns, using speciation method (so truncation not active here)
        crossover_spawns = self.get_crossover_spawns(n_crossover)

        # asex spawns - with mutation
        asex_spawns = []
        for i in range(n_asex):
            i = i % len(pool)
            new_g = pool[i].clone()
            new_g.mutate(0)
            asex_spawns.append(new_g)

        self.population = elite_spawns + crossover_spawns + asex_spawns

# ------------------------------------------------------------------------------
# ComponentTracker class -------------------------------------------------------
# ------------------------------------------------------------------------------

class PopulationComponentTracker:
    """This class tracks the components of the overall population
    """
    def __init__(self)-> None:
        self.component_dict = dict()


    def update_global_components(self, population: list[GeneticNet], gen: int):
        """Registers the unique components in the history, also pairs up components
        of all genomes with their fitness, if use_t_vals this will be used to calculate
        mean difference significance for all components in the population.
        """
        # all_components = set(c["comp"] for c in self.component_dict.values())
        pop_fit_vals = []
        for g in population:
            pop_fit_vals.append(g.fitness)
            c_set = g.get_unique_component_set()
            for c in list(c_set):
                if c not in self.component_dict:
                    self.component_dict[c] = {
                        "id": str(uuid4()),
                        "fitnesses": {gen: [g.fitness]},
                        "t_val": {gen: None}
                        }
                else:
                    self.component_dict[c]["fitnesses"].setdefault(gen, []).append(g.fitness)


        # FUTUREIMPROVEMENT: can do other stuff that influences the probability map here
        # like check the best improvements
        # FUTUREIMPROVEMENT: maybe use multiple gens later, with a discount factor
        if params.use_t_vals:
            self.update_t_vals(gen, pop_fit_vals)

            
    def get_comp_info(self, comp: tuple) -> dict:
        return self.component_dict[comp]

    def update_t_vals(self, gen, pop_fit_vals) -> dict:
        # FUTUREIMPROVEMENT: this logic could also be handled in caller, especially if other
        # keys are added to the dict for other prob_map influences
        pop_fit_vals = np.array(pop_fit_vals)
        pop_sum, pop_len = pop_fit_vals.sum(), len(pop_fit_vals)
        pop_df = pop_len - 2

        for data in self.component_dict.values():
            # skip components that are not existent in this gen
            if gen not in data["fitnesses"]:
                continue
            included = np.array(data["fitnesses"][gen])
            # this should only happen in the first gen, when the start and end connections
            # yield components, shared by everyone, making t value comparison impossible
            if gen == 1:
                data["t_val"][gen] = -1
            else:
                data["t_val"][gen] = compute_t(included, pop_fit_vals, pop_len, pop_sum, pop_df)


    def get_inverted_comp_dict(self):
        inverted_comp_dict = dict()
        for c, data in self.component_dict.items():
            inverted_comp_dict[data["id"]] = {
                "component": str(c),
                "fitnesses": {gen: sum(all_fit) for gen, all_fit in data["fitnesses"].items()},
                "t_val": data["t_val"]
            }
        return inverted_comp_dict

    
def compute_t(inc, pop_fit_vals, pop_len, pop_sum, pop_df):
    inc_sum, inc_len = inc.sum(), len(inc)
    inc_avg_fit = inc_sum / inc_len
    exc_len = pop_len - inc_len
    exc_avg_fit = (pop_sum - inc_sum) / exc_len
    mean_diff = inc_avg_fit - exc_avg_fit

    inc_df, exc_df = inc_len-1, exc_len-1

    inc_ss = ((inc - inc_avg_fit)**2).sum()
    inc_var = inc_ss / inc_len
    
    exc_ss = ((pop_fit_vals - exc_avg_fit)**2).sum() - inc_ss - inc_len*mean_diff**2
    exc_var = exc_ss / exc_len

    pool_var = (inc_var*inc_df + exc_var*exc_df) / pop_df
    se = (pool_var/inc_len + pool_var/exc_len)**0.5
    return float(mean_diff/se)
