"""
This analyzes the size and order of components, need to know if components contain
more than one unique place.
The dataframe needs to be generated with minimal_serialization = False
"""
# %%
import pm4py
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

population = "C:/Users/pauls/Documents/GitHubRepos/EvolvePetriNets/results/data/test_sa_ea_10-09-2024_12-27-17/whatever/4_10-09-2024_12-27-23/data/population.feather"
df = pd.read_feather(population)

# %%
# i = 0
num_places = []

for g in tqdm(df["all_genomes"]):
    net, im, fm = g.build_petri()
    for c in pm4py.analysis.maximal_decomposition(net, im, fm):
        num_places.append(len(c[0].places))


# %%
import collections

counter = collections.Counter(num_places)
for value, count in counter.items():
    print(f"Value: {value}, Count: {count}")
# %%
