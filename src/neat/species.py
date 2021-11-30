from . import params

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
        self.curr_mutation_rate = params.mutation_rate["normal"]
        self.num_gens_no_improvement = 0


    def add_member(self, new_genome) -> None:
        """Just appends a new genome to the alive_members and updates the new_members count.
        """
        self.alive_members.append(new_genome)
        new_genome.species_id = self.name
        return