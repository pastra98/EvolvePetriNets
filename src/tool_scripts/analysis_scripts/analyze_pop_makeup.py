"""
Stackplot of population makeup (num crossover, etc)
TODO
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

population_path = "../results/data/test_crossover_06-15-2024_18-15-10/whatever/1_06-15-2024_18-15-16/reports/population.feather"
df = pd.read_feather(population_path)

# %%
df
# %%
