# About
This is the WIP repository in which I implement a genetic process mining algorithm for my Bachelor & Master Thesis.
The repository is still in active development and is not yet documented.

## What is a genetic process mining algorithm?
**Process mining algorithms** seek to discover a model that describes a given process, e.g. processing insurance claims, registering patients in a hospital or handling the cash-to-order process for a webshop.
Processes generally involve tasks that need to be done *sequentially* (e.g. stock availability needs to be checked before an order can be confirmed), in *parallel* (e.g. invoicing the customer and shipping the goods), in *repetition* (e.g. constant updating of website prices) or tasks may be *exclusive* (e.g. an order cannot be accepted and then rejected).
A process model describes the relations between tasks, defining which tasks are performed in which order and what the dependencies between tasks are.

There are many different notations for process models (wf-nets, bpmn, flow charts...), but at the core most use a graph notation that establishes the relation among tasks in the process (e.g. `a -> b`).
The modelling notation used in my algorithm are [petri nets](https://en.wikipedia.org/wiki/Petri_net), referred to as workflow-nets in the context of process mining.
Petri nets are capable of representing the different relations among tasks such as sequential order, parallelism, (exclusive) choice or repetition with a minimal syntax.

In order to determine if a model accurately represents a process, it is compared against a execution log in a process called conformance checking (the inverse can also be done, checking if a process conforms to a specified model).
An execution log is a recording of instances of a processes execution, e.g. all insurance claim tickets that were handled in a month.
The central idea behind process mining is to then use this execution log as input to a process mining algorithm which outputs a model describing the underlying process in the log.
In other words, process mining helps to make visible the structure of a process when there is sufficient data recorded about its execution.

**Genetic Algorithms** are inspired by the principles of evolution and implement random mutations, crossover and selection digitally.
In the case of my genetic process mining algorithm, the process models can be thought of as organisms whose fitness depends on their conformance to a chosen log.
Concretely, the algorithm first reads all the tasks that exist in the execution log and creates an initial population of petri nets where those tasks are connected randomly.
It then evaluates the conformance of each process model and assigns a (multi-metric) fitness value to each model.
In the next step, a selection of the 'best' process models is made, and those models are subjected to random mutations that change their connections slightly.
These changed process models then form the population next generation, which will again be evaluated, selected and mutated for a set number of generations.

In this repository I have implemented three different selection strategies, various mutations, the code necessary to perform conformance checking as well as several metrics that are used to assess the fitness of process models.
The 'meat' of the algorithm can be found in the `src/neat/ga.py` module.

This readme is also still WIP.