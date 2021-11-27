class Species:
    def __init__(self, name) -> None:
        self.name = name
        self.alive_members = []

    def add_member(self, new_genome) -> None:
        """Just appends a new genome to the alive_members and updates the new_members count.
        """
        self.alive_members.append(new_genome)
        new_genome.species_id = self.name
        return