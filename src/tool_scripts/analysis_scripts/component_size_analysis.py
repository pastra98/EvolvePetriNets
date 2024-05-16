"""
This analyzes the size and order of components, need to know if components contain
more than one unique place.
The dataframe needs to be generated with minimal_serialization = False
"""
# %%
from neat import params, innovs, genome, netobj, initial_population
import pm4py
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

population = "../results/data/test_components_05-16-2024_12-36-11/test_components/1_05-16-2024_12-36-20/reports/population.pkl"
df = pd.read_pickle(population)

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
