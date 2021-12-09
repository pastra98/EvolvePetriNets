import random as rd
from . import params
from .genome import GeneticNet

class Species:
    def __init__(self, name) -> None:
        self.name = name
        self.alive_members = []

        self.age = 0
        self.representative = None
        self.leader = None
        self.alive_members = []
        self.num_members: int
        self.pool = []
        self.expected_offspring = 0
        self.spawn_count = 0 # number of spawns the species got during the last generation it was active
        self.avg_fitness = 0 # average fitness of all alive members
        self.avg_fitness_adjusted = 0 # average fitness adjusted by the age modifier
        self.best_ever_fitness = 0 # best ever fitness witnessed in this species
        self.obliterate = False
        self.num_to_spawn = 0 # The amount of offspring to be spawned in the next generation
        self.num_gens_no_improvement = 0
        self.curr_mutation_rate = 0 # 0 -> normal or 1 -> high

    
    def add_member(self, new_genome) -> None:
        """Just appends a new genome to the alive_members and updates the new_members count.
        """
        self.alive_members.append(new_genome)
        new_genome.species_id = self.name
        return

    
    def update(self) -> None:
        """Checks if the species continues to survive into the next generation. If so,
        the total fitness of the species is calculated and adjusted according to the age
        bonus of the species. It's members are ranked according to their fitness, and
        a certain percentage of them is placed into the pool that gets to produce offspring.
        """
        # first check if the species hasn't spawned new members in the last gen or if it
        # survived for too many generations without improving, in which case it is marked
        # for obliteration.
        if not self.alive_members or self.num_gens_no_improvement > params.allowed_gens_no_improvement:
            self.obliterate = True
        else:
            # the species survives into the next generation
            self.spawn_count = 0
            self.num_to_spawn = 0
            self.age += 1
            # first sort the alive members, and determine the fittest member
            self.alive_members.sort(key=lambda m: m.fitness, reverse=True)
            self.leader = self.alive_members[0]
            self.num_members = len(self.alive_members)
            # check if current best member is fitter than previous best
            if self.leader.fitness > self.best_ever_fitness:
                # this means the species is improving -> normal mutation rate
                self.best_ever_fitness = self.leader.fitness
                self.num_gens_no_improvement = 0
                self.curr_mutation_rate = 0 # normal mutation rate
            else:
                self.num_gens_no_improvement += 1
                if self.num_gens_no_improvement > params.enough_gens_to_change_things:
                    self.curr_mutation_rate = 1 # high mutation rate
            # if the representative should be updated, do so now
            if params.update_species_rep:
                self.representative = self.leader if params.leader_is_rep else rd.choice(self.alive_members)
            # pool is a reference to the alive members of the last gen
            # If a species reaches selection_threshold, not every member gets in the pool
            if len(self.alive_members) > params.selection_threshold:
                # self.pool = alive_members.slice(0, int(alive_members.size()*Params.spawn_cutoff))
                self.pool = self.alive_members[:int(len(self.alive_members)*params.spawn_cutoff)]
            else:
                self.pool = self.alive_members
            # calculate the average fitness and adjusted fitness of this species
            self.avg_fitness = sum(map(lambda m: m.fitness, self.alive_members)) / len(self.alive_members)
            fit_modif = params.youth_bonus if self.age < params.old_age else params.old_penalty
            self.avg_fitness_adjusted = self.avg_fitness * fit_modif
            # reassign alive members to a new empty array, so new agents can be placed
            # in the next gen. Clearing it also clear the pool, since pool is a reference.
            self.alive_members = [] # gotta test this for python
            return
            
    
    def calculate_offspring_amount(self, total_avg_species_fitness) -> None:
        """This func does not care about the fitness of individual members. It
        calculates the total spawn tickets allotted to this species by comparing
        how fit this species is relative to all other species, and multiplying this
        result with the total population size.
        """
        # prevent species added in the current gen from producing offspring
        if self.age > 0:
            self.num_to_spawn = round((self.avg_fitness_adjusted / total_avg_species_fitness) * params.popsize)
    
    
    def elite_spawn(self) -> GeneticNet:
        """Returns a copy of the species leader WITHOUT INCREASING SPAWN COUNT
        """
        return self.leader.clone()


    def mate_spawn(self) -> GeneticNet:
        """Chooses to members from the pool and produces a baby via crossover. Baby
        then gets mutated and returned.
        """
        # if random mating, pick 2 random unique parent genomes for crossing over.
        if params.random_mating:
            mom, dad = rd.sample(self.pool, k=2)
        # else just go through every member of the pool, possibly multiple times and
        # breed genomes sorted by their fitness. Genomes with fitness scores next to each
        # other are therefore picked as mates, the exception being the first and last one.
        else:
            pool_index = self.spawn_count % (len(self.pool) - 1)
            mom = self.pool[pool_index]
            # ensure that second parent is not out of pool bounds
            dad = self.pool[-1] if pool_index == 0 else self.pool[pool_index + 1]
        if mom == dad: raise Exception("woops, this shouldn't happen!!!!")
        # now that the parents are determined, produce a baby and mutate it
        baby = dad.crossover(mom)
        baby.mutate(self.curr_mutation_rate)
        self.spawn_count += 1
        return baby


    def asex_spawn(self) -> GeneticNet:
        """Copy a member from the pool, mutates it, and returns it.
        """
        # As long as not every pool member as been spawned, pick next one from pool
        if self.spawn_count < len(self.pool):
            baby = self.pool[self.spawn_count].clone()
        # if more spawns than pool size, start again
        else:
            baby = self.pool[self.spawn_count % len(self.pool)].clone()
        baby.mutate(self.curr_mutation_rate)
        self.spawn_count += 1
        return baby


    def get_curr_info(self) -> dict:
        """Used for serialization when not wanting to save the entire object
        """
        info_d = {}
        info_d["name"] = self.name
        info_d["alive_member_ids"] = [g.id for g in self.alive_members]
        info_d["age"] = self.age
        info_d["representative_id"] = self.representative.id
        info_d["leader_id"] = self.leader.id
        info_d["num_members"] = self.num_members
        info_d["pool"] = [g.id for g in self.pool]
        info_d["expected_offspring"] = self.expected_offspring
        info_d["spawn_count"] = self.spawn_count
        info_d["avg_fitness"] = self.avg_fitness
        info_d["avg_fitness_adjusted"] = self.avg_fitness_adjusted
        info_d["best_ever_fitness"] = self.best_ever_fitness
        info_d["obliterate"] = self.obliterate
        info_d["num_to_spawn"] = self.num_to_spawn
        info_d["num_gens_no_improvement"] = self.num_gens_no_improvement
        info_d["curr_mutation_rate"] = self.curr_mutation_rate
        return info_d
