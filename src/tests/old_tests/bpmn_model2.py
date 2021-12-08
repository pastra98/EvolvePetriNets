class Node:
    def __init__(self, name) -> None:
        self.name = name

class Task(Node):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.prev_node: Node = None
        self.next_node: Node = None

class Gate(Node):
    def __init__(self, name, gtype) -> None:
        super().__init__(name)
        self.gtype = gtype
        self.incoming_conns = {}
        self.outgoing_conns = {}

class DirectedEdge:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

################################################################################

class GeneticBPMN:
    def __init__(self) -> None:
        self.tasks = {}
        self.connections = {}
        self.gates = {}
    
    def mutate():
        pass

    def add_connection(in_node, out_node):
        pass

    def insert_node(self, connection, node_type):
        name = innovs.check_node(connection.start, connection.end, node_type)
        prev_node = connection.start
        next_node = connection.end

        if node_type != "task":
            node = Gate(name, node_type)
        else:
            node = Task()
            
        self.add_connection(prev_node, node)
        self.add_connection(node, next_node)

    # def move_task(connection_name, node_type):
    #     pass

    def get_compatibility(other_genome):
        pass
    
    def crossover(other_genome):
        pass
    
    def get_fitness(log):
        pass

