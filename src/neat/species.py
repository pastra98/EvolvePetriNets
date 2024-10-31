import random as rd
from neat import params
from neat.genome import GeneticNet
from typing import List
from math import ceil

class Species:
    def __init__(self, name: str, founder: GeneticNet) -> None:
        self.name = name
        self.members: List[GeneticNet] = []
        self.age = 0
        self.representative: GeneticNet = founder
        self.leader: GeneticNet = founder
        self.num_members: int = 0
        self.pool: List[GeneticNet] = []
        self.expected_offspring = 0
        self.spawn_count = 0 # number of spawns the species got during the last generation it was active
        self.asex_spawn_count = 0 # counter used for asex spawns, iterating through the pool
        self.avg_fitness = 0 # average fitness of all new members
        self.avg_fitness_adjusted = 0 # average fitness adjusted by the age modifier
        self.fitness_share = 0 # proportion of own (adjusted) fitness relative to total
        self.best_ever_fitness = 0 # best ever fitness witnessed in this species
        self.obliterate = False
        self.num_gens_no_improvement = 0
        self.curr_mutation_rate = 0 # 0 -> normal or 1 -> high
        self.component_set = founder.get_unique_component_set() # for first gen just use founder
        # automatically add the founder as a member
        self.add_member(founder)

    
    def add_member(self, new_genome) -> None:
        """Just appends a new genome to the members and updates the members count.
        """
        self.members.append(new_genome)
        new_genome.species_id = self.name

    
    def update(self, has_best_genome) -> None:
        """Checks if the species continues to survive into the next generation. If so,
        the total fitness of the species is calculated and adjusted according to the age
        bonus of the species. It's members are ranked according to their fitness, and
        a certain percentage of them is placed into the pool that gets to produce offspring.
        """
        # first check if the species hasn't spawned new members in the last gen or if it
        # survived for too many generations without improving, in which case it is marked
        # for obliteration.
        if not self.members or self.num_gens_no_improvement > params.allowed_gens_no_improvement:
            if not has_best_genome:
                self.obliterate = True
        else:
            # the species survives into the next generation
            self.spawn_count, self.asex_spawn_count = 0, 0
            self.num_to_spawn = 0
            self.age += 1
            # first sort the new members, and determine the fittest member
            self.members.sort(key=lambda m: m.fitness, reverse=True)
            self.leader = self.members[0]
            self.num_members = len(self.members)
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
                self.representative = self.leader if params.leader_is_rep else rd.choice(self.members)
            # pool is a reference to the members of the last gen that will spawn offspring
            self.pool = self.members[:ceil(len(self.members)*params.spawn_cutoff)] # ensure that even super-small species get spawns when using threshold
            # calculate the average fitness and adjusted fitness of this species
            self.avg_fitness = sum(map(lambda m: m.fitness, self.members)) / len(self.members)
            fit_modif = params.youth_bonus if self.age < params.old_age else params.old_penalty
            self.avg_fitness_adjusted = self.avg_fitness * fit_modif
            # save the components of the last gen
            self.component_set = self.update_components()
            # reassign new members to a new empty array, so new agents can be placed
            # in the next gen. Clearing it also clear the pool, since pool is a reference.
            self.members = []
            

    def update_components(self) -> set:
        """Adds species_component_pool_size component sets to a shared set of components
        """
        n_representatives = min(params.species_component_pool_size, len(self.members))
        # add the species representative + n_representatives (excl. rep)
        representatives = rd.choices(self.members[1:], k=n_representatives-1)
        representatives.append(self.representative)
        components = set()
        for rep in representatives:
            components.update(rep.get_unique_component_set())
        return components
    

    def calculate_fitness_share(self, total_avg_species_fitness) -> None:
        """Assigns the fitness share to the species
        FUTUREIMPROVEMENT: could test alternatives to mean here, maybe mode?
        """
        self.fitness_share = self.avg_fitness_adjusted / total_avg_species_fitness


    def elite_spawn(self) -> GeneticNet:
        """Returns a copy of the species leader
        """
        self.spawn_count += 1
        return self.leader.clone()


    def asex_spawn(self) -> GeneticNet:
        """Copy a member from the pool, mutates it, and returns it.
        """
        # As long as not every pool member as been spawned, pick next one from pool
        baby = self.pool[self.asex_spawn_count % len(self.pool)].clone()
        baby.mutate(self.curr_mutation_rate)
        self.spawn_count += 1
        self.asex_spawn_count += 1
        return baby


    def elite_spawn_with_mutations(self) -> GeneticNet:
        """Returns a copy of the species leader, mutates it and increases spawn count
        """
        baby = self.elite_spawn()
        baby.mutate(self.curr_mutation_rate)
        self.spawn_count += 1
        return baby


    def get_curr_info(self) -> dict:
        """Used for serialization when not wanting to save the entire object
        """
        info_d = {}
        info_d["name"] = self.name
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
