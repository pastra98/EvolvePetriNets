# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

history_path = "../results/data/test_mutation_lineage_stats_05-09-2024_16-49-25/equal_probabilities/1_05-09-2024_16-49-34/reports/population.feather"
df = pd.read_feather(history_path)
# %%
df
list_of_mutations = [
    'removed_an_arc', 'trans_place_arc', 'place_trans_arc', 'trans_trans_conn',
    'extend_new_place', 'extend_new_trans', 'split_arc', 'pruned_an_extension']


# Step 1: Filter out genomes without parents and ensure it's a copy for safe modifications
df_with_parents = df.dropna(subset=['parent_id']).copy()

# Convert parent_id to int for easy lookup
df_with_parents['parent_id'] = df_with_parents['parent_id'].astype(int)

# Step 2: Calculate fitness difference
# Create a dictionary from id to fitness for fast lookup
fitness_dict = df.set_index('id')['fitness'].to_dict()

# Use .loc to avoid SettingWithCopyWarning when modifying the DataFrame
df_with_parents.loc[:, 'fitness_difference'] = df_with_parents.apply(lambda row: row['fitness'] - fitness_dict[row['parent_id']], axis=1)

# Step 3: Parse the mutations and associate with fitness difference
# Initialize a dictionary to hold mutation effects and frequency
mutation_effects = {}
mutation_frequency = {}

for _, row in df_with_parents.iterrows():
    mutations = row['my_mutations']
    fitness_diff = row['fitness_difference']
    for mutation in mutations:
        if mutation not in mutation_effects:
            mutation_effects[mutation] = []
        mutation_effects[mutation].append(fitness_diff)
        mutation_frequency[mutation] = mutation_frequency.get(mutation, 0) + 1

# Step 4: Aggregate and analyze the data
# Calculate average fitness impact per mutation
average_impacts = {mutation: sum(effects) / len(effects) for mutation, effects in mutation_effects.items()}
overall_average_impact = sum(df_with_parents['fitness_difference']) / len(df_with_parents)

# Calculate relative frequency
total_mutations = sum(mutation_frequency.values())
relative_frequencies = {mutation: count / total_mutations for mutation, count in mutation_frequency.items()}

# Plot the results
mutations = list(average_impacts.keys())
average_fitness_impacts = list(average_impacts.values())
frequencies = [relative_frequencies[mutation] for mutation in mutations]

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for average fitness impact
color = 'tab:blue'
ax1.set_xlabel('Mutation')
ax1.set_ylabel('Average Fitness Impact', color=color)
ax1.bar(mutations, average_fitness_impacts, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(len(mutations)))
ax1.set_xticklabels(mutations, rotation=45, ha='right')

# Create a second y-axis to show relative frequency
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Relative Frequency', color=color)
ax2.plot(mutations, frequencies, color=color, marker='o', linestyle='-')
ax2.tick_params(axis='y', labelcolor=color)

# Final touches
plt.title('Average Fitness Impact and Relative Frequency of Each Mutation')
fig.tight_layout()
plt.show()

# Print overall average fitness impact
print("Overall Average Fitness Impact:", overall_average_impact)

#%%
mutation_frequency