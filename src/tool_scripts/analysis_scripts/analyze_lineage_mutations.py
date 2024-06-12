"""
This analysis file plots the relative frequency of mutations along with their
fitness impact.
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

population_path = "../results/data/test_if_pickling_works_again_05-30-2024_18-12-37/whatever/1_05-30-2024_18-12-45/reports/population.pkl"
# df = pd.read_pickle(population_path)
df = pd.read_feather(population_path)
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
# average_impacts = {mutation: sum(effects) / len(effects) for mutation, effects in mutation_effects.items()}
# average_impacts = {k: average_impacts[k] for k in sorted(average_impacts)}
# overall_average_impact = sum(df_with_parents['fitness_difference']) / len(df_with_parents)
mutation_effects = {m: mutation_effects[m] for m in sorted(mutation_effects)}
total_mutations = sum(mutation_frequency.values())
relative_frequencies = {mutation: count / total_mutations for mutation, count in mutation_frequency.items()}

# Plot the results
mutations = mutation_effects.keys()
data_for_boxplot = mutation_effects.values()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Boxplot for fitness impact distribution per mutation
color = 'tab:blue'
ax1.set_xlabel('Mutation')
ax1.set_ylabel('Fitness Impact Distribution', color=color)
bp = ax1.boxplot(data_for_boxplot, patch_artist=True, meanline=True, showmeans=True)

# Coloring and styling
for box in bp['boxes']:
    box.set(color=color, linewidth=2)
    box.set(facecolor='lightblue')

# Set x-ticks to mutation names
ax1.set_xticks(np.arange(1, len(mutations) + 1))
ax1.set_xticklabels(mutations, rotation=45, ha='right')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(y=0, color='lightgray', linestyle='--')

# Create a second y-axis to show relative frequency
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Relative Frequency', color=color)
ax2.plot(np.arange(1, len(mutations) + 1), [relative_frequencies[mutation] for mutation in mutations], color=color, marker='o', linestyle='-')
ax2.tick_params(axis='y', labelcolor=color)

# Final touches
plt.title('Fitness Impact Distribution and Relative Frequency of Each Mutation')
fig.tight_layout()
plt.show()
# %%
summary_df = pd.DataFrame(columns=['Min', '25%', 'Median', '75%', 'Max', 'Mean', 'Frequency'])

for mutation, effects in mutation_effects.items():
    df_temp = pd.DataFrame(effects, columns=['Effects'])
    summary = df_temp['Effects'].describe(percentiles=[.25, .5, .75])
    # Add 'Mean' and 'Frequency' (count) to the summary
    summary_df.loc[mutation] = [
        summary['min'], summary['25%'], summary['50%'], summary['75%'], summary['max'],
        summary['mean'], summary['count']
    ]

summary_df.to_markdown('mutation_effects.txt')
# %%
