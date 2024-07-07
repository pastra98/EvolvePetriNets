# %%
from tool_scripts.useful_functions import \
    load_genome, show_genome, get_aligned_traces, get_log_variants, log, reset_ga, \
    eval_and_print_fitness

from neat import params, genome, initial_population
from pprint import pprint
from importlib import reload

# %%
path = "./tool_scripts/model_analysis/"
alpha_g = load_genome(path + "alpha_bootstrap.pkl")
inductive_g = load_genome(path + "inductive_bootstrap.pkl")
ilp_g = load_genome(path + "inductive_bootstrap.pkl")
spaghetti_g2 = load_genome(path + "inductive_bootstrap.pkl")
show_genome(alpha_g)

