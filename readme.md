# About
This repository contains the code of the *Genetic Workflow Miner (GWFM)* which I implemented for my Master's Thesis at the [Vienna University of Economics and Business](https://www.wu.ac.at/en/) (2025).
The GWFM is a **genetic process mining algorithm**, implemented in python and this repository contains everything required to execute the algorithm (refer to Section How to build & run).

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

## How to build & run

### Prerequisites
- **Python ≥ 3.9**
- **Graphviz** system package (`dot` must be on your PATH for Petri net visualization)
  - Linux (Debian/Ubuntu): `sudo apt install graphviz`
  - macOS: `brew install graphviz`
  - Windows: `winget install graphviz --interactive` or download from [graphviz.org](https://graphviz.org/download/)
    - **NOTE:** by default winget does not add graphviz to the PATH, therefore it is recommended to  install using the interactive mode and checking the option to add gviz to the PATH. If gviz is already installed, then ensure the /bin folder is added to the PATH. Restart your terminal session for the changes to take effect.

### Installation

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\activate
   # Windows (cmd)
   .venv\Scripts\activate.bat
   ```

   > **Windows note:** If PowerShell gives you an `UnauthorizedAccess` error, its execution
   > policy is blocking the activation script. Run this once to fix it permanently for your user:
   > ```powershell
   > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   > ```

2. Install the project — pick **one** of the following:

   ```bash
   # Option A – pip install (recommended, creates the `gwfm` command)
   # Hint: if you want to change the code, use pip install -e .
   pip install .

   # Option B – requirements.txt only
   pip install -r requirements.txt
   ```

   > **Optional extras:**
   > `pip install ".[gui]"` adds PyQt6 for the config-maker GUI.
   > `pip install ".[dev]"` adds IPython for interactive exploration.

### Running the algorithm

The algorithm requires a JSON config file that specifies parameter files, the event log, stopping conditions and how many runs to execute. A ready-to-use config that mines the included running example is shipped in the repo:

```bash
# if installed with pip install .
gwfm configs/default_config.json

# if installed with pip install -r requirements.txt
python src/main.py configs/default_config.json
```

Results are written to `results/data/<config-name>_<timestamp>/`. Each run produces:
- **Feather files** with population, species and fitness data
- **SVG renders** of the best discovered Petri nets
- **Plots** (fitness progression, species evolution, mutation analysis, …) when `save_plots` is enabled
- A text **execution report** with timing and summary statistics

**NOTE:** By default the program is quite chatty because it prints the evaluation results of every generation. This behavior can be disabled by setting `send_gen_info_to_console": true`.

### Configuration

A config file has this structure (see `configs/default_config.json`):

```json
{
    "name": "mine_running_example",
    "setups": [
        {
            "setupname": "default_params",
            "parampath": "./params/default_params.json",
            "logpath": "pm_data/running_example.xes",
            "stop_cond": {"var": "gen", "val": 300},
            "n_runs": 1,
            "send_gen_info_to_console": true,
            "is_profiled": false,
            "save_plots": true
        }
    ]
}
```

| Field | Description |
|---|---|
| `name` | Name for this execution (used in the results folder name) |
| `setups` | Array of experiment setups to run |
| `parampath` | Path to a JSON file with GA parameters (population size, mutation rates, fitness weights, …) |
| `logpath` | Path to an XES event log |
| `stop_cond` | Stopping condition — currently `{"var": "gen", "val": N}` to run for N generations |
| `n_runs` | Number of parallel repetitions for this setup |
| `send_gen_info_to_console` | Print per-generation stats to the terminal |
| `is_profiled` | Enable cProfile profiling for performance analysis |
| `save_plots` | Generate matplotlib plots at the end of a run |

Multiple setups can be listed in one config to compare parameter configurations. All runs across all setups are parallelized using Python multiprocessing.

The default parameter configuration should yield the target process model for the running example log (i.e. the model that was used to generate the execution log). My thesis provides a detailed explanation of the parameters and how the current set of parameters was determined.

### Config maker GUI (optional)

A PyQt6-based GUI for constructing configs with parameter sweeps is available:

```bash
pip install ".[gui]"
python src/scripts/config_maker.py
```

## Navigating this repository

```
src/
├── main.py                      # CLI entry point – config loading, multiprocessing orchestration
├── neat/                        # Core genetic algorithm
│   ├── ga.py                    # GeneticAlgorithm class – main evolutionary loop
│   ├── genome.py                # GeneticNet – Petri net genome with mutations & crossover
│   ├── species.py               # Species management for NEAT-style speciation
│   ├── initial_population.py    # Bootstrap population from pm4py miners or random generation
│   └── params.py                # Parameter loading from JSON
├── neatutils/                   # Supporting utilities
│   ├── setuprunner.py           # Per-run orchestration (logging, profiling, report saving)
│   ├── fitnesscalc.py           # Token replay engine & fitness metrics (11 metrics)
│   ├── endreports.py            # Result serialization (Feather) & plot generation (16+ plots)
│   ├── log.py                   # XES event log loading & prefix optimization
│   ├── neatlogger.py            # Logger setup
│   └── timer.py                 # Execution time profiling
└── scripts/
    └── config_maker.py          # PyQt6 config generator GUI

configs/                         # Example configuration files
params/                          # Example GA parameter files
pm_data/                         # Sample event logs and Petri net models
```

The core algorithm lives in `src/neat/ga.py`. It implements three selection strategies:
1. **Speciation** (default) – NEAT-inspired species-based selection with fitness sharing
2. **Roulette** – fitness-proportionate selection
3. **Truncation** – only the top fraction of the population breeds

Fitness evaluation happens in `src/neatutils/fitnesscalc.py`, which implements a custom token-replay engine with prefix optimization and 11 configurable fitness metrics.

## Attributions
### NEAT (NeuroEvolution of Augmenting Topologies)
The genetic encoding with dynamically growing structures and the speciated selection mechanism implemented in this algorithm are heavily inspired by the NEAT algorithm, originally developed by Kenneth O. Stanley and Risto Miikkulainen. 
* DOI: https://doi.org/10.1162/106365602320169811

### Libraries and Frameworks
This project relies on several open-source libraries:
* **pm4py**: Used for processing XES event logs, generating initial populations via conventional mining algorithms (Alpha, Inductive, Heuristics, ILP), evaluation functionalities as well as general experimentation and visualization. 
  * Repository: https://github.com/pm4py/pm4py
  * Paper: *Process Mining for Python (PM4Py): Bridging the Gap Between Process- and Data Science* (Berti et al., 2019).
* **Data Processing**: 
  * **Polars** & **Pandas**: Utilized for dataframe processing and fitness evaluations.
  * **PyArrow**: Used for serializing population data into the Apache Feather format.
* **Visualization**: 
  * **Graphviz**: Used for rendering the phenotypes (Petri nets) in combination with pm4py.
  * **Matplotlib**: Used for plotting fitness progressions and component distributions.

### Running Example Event Log
The running example event log and model included in this repository (`/pm_data`) are the canonical example of a simple event log from W.M.P. van der Aalsts *Process Mining: Data Science in Action*.

DOI (2nd edition): https://doi.org/10.1007/978-3-662-49851-4_1

The event log is also included in the pm4py repository:
https://github.com/process-intelligence-solutions/pm4py/blob/release/notebooks/data/running_example.xes

Other event logs used in the Thesis have been taken from Process Discovery Contest (PDC) 2024 (doi: https://doi.org/10.4121/3CFCDBB7-C909-4F60-8BEC-62C780598047.V1) and PDC 2022 (doi: https://doi.org/10.4121/21261402.v2). Cited here are the specific versions used in my Thesis.