"""
This analyzes which metric is suitable for identifying good components.
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

population = "../results/data/component_fitness_analysis_200gen_05-14-2024_13-03-59/no_log_splices/1_05-14-2024_13-04-08/reports/population.pkl"
df = pd.read_pickle(population)
# %%
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
components_expanded = mlb.fit_transform(df['my_components'])
components_df = pd.DataFrame(components_expanded, columns=mlb.classes_)

# Concatenate this with the 'fitness' column
df_expanded = pd.concat([df['fitness'], components_df], axis=1)

# %%
################################################################################
#################### CORRELATION ANALYSIS ######################################
################################################################################
# Step 2: Compute the correlation between each component and fitness
correlations = df_expanded.drop('fitness', axis=1).apply(lambda x: x.corr(df_expanded['fitness']))

# Step 3: Sort the correlations to find the 10 best and 10 worst components
sorted_correlations = correlations.sort_values(ascending=False)

# Display the 10 best and 10 worst components
print("10 Best Components:")
print(sorted_correlations.head(10))

print("\n10 Worst Components:")
print(sorted_correlations.tail(10))

# %%
pprint(list(sorted_correlations.items())[:10])
# %%
################################################################################
#################### AVG FITNESS OF COMPONENTS #################################
################################################################################
from tqdm import tqdm

component_fitness = {}

for component in tqdm(df_expanded.columns[1:]):
    component_fitness[component] = df_expanded[df_expanded[component] == 1]['fitness'].mean()

# %%
# sort the components by average fitness
from pprint import pprint
sorted_component_fitness = {k: v for k, v in sorted(component_fitness.items(), key=lambda item: item[1], reverse=True)}

print("10 Best Components by Average Fitness:")
pprint(list(sorted_component_fitness.items())[:10])

# %%
################################################################################
#################### UPDATE CORRELATIONS FOR EVERY GEN #########################
################################################################################
import time

def update_and_calculate_correlations(df_expanded, all_possible_components, previous_scores, generation):
    # Initialize sums if first generation
    if generation == 0:
        num_components = len(all_possible_components)
        previous_scores['sum_xy'] = np.zeros(num_components)
        previous_scores['sum_x'] = np.zeros(num_components)
        previous_scores['sum_y'] = 0
        previous_scores['sum_x2'] = np.zeros(num_components)
        previous_scores['sum_y2'] = 0
        previous_scores['n'] = 0
    
    # Filter data for the current generation
    current_gen_data = df_expanded[df_expanded['gen'] == generation]
    
    # Update sums
    start_time = time.time()
    for index, row in current_gen_data.iterrows():
        fitness = row['fitness']
        for i, component in enumerate(all_possible_components):
            x = row[component]
            y = fitness
            previous_scores['sum_xy'][i] += x * y
            previous_scores['sum_x'][i] += x
            previous_scores['sum_y'] += y
            previous_scores['sum_x2'][i] += x * x
            previous_scores['sum_y2'] += y * y
        previous_scores['n'] += 1
    
    # Calculate correlations
    mean_x = previous_scores['sum_x'] / previous_scores['n']
    mean_y = previous_scores['sum_y'] / previous_scores['n']
    numerator = previous_scores['sum_xy'] - previous_scores['n'] * mean_x * mean_y
    denominator = np.sqrt((previous_scores['sum_x2'] - previous_scores['n'] * mean_x**2) * (previous_scores['sum_y2'] - previous_scores['n'] * mean_y**2))
    correlations = numerator / denominator

    # Find the 5 best components
    best_indices = np.argsort(correlations)[-5:]
    best_components = [all_possible_components[i] for i in best_indices]

    # Print the 5 best components and the time taken
    print(f"Generation {generation}: 5 Best Components - {best_components}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    return correlations

# Example usage
df_expanded = pd.DataFrame({
    'fitness': np.random.rand(100000),
    'gen': np.repeat(np.arange(200), 500),
    # Dummy data for components, replace with actual component names
    **{f'component_{i}': np.random.randint(0, 2, 100000) for i in range(10)}
})

all_possible_components = [f'component_{i}' for i in range(10)]
previous_scores = {}

for generation in range(200):
    correlations = update_and_calculate_correlations(df_expanded, all_possible_components, previous_scores, generation)


# %%
################################################################################
#################### LINEAR REGRESSION #########################################
################################################################################
import statsmodels.api as sm

X = sm.add_constant(components_df)
y = df['fitness']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(model.summary())

# Step 3: Interpret the results
# Get the coefficients and their p-values
coefficients = model.params
p_values = model.pvalues

# Create a DataFrame with the results
regression_results = pd.DataFrame({
    'Component': coefficients.index,
    'Coefficient': coefficients.values,
    'p-value': p_values.values
})

# Sort by the coefficient to find the 10 most and least impactful components
sorted_results = regression_results.sort_values(by='Coefficient', ascending=False)

print("10 Best Components:")
print(sorted_results.head(10))

print("\n10 Worst Components:")
print(sorted_results.tail(10))

# %%
################################################################################
#################### RANDOM FOREST #############################################
################################################################################
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Fit a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(components_df, y)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for i in range(10):
    print(f"{i + 1}. Component {components_df.columns[indices[i]]} ({importances[indices[i]]})")

# %%
################################################################################
#################### PERMUTATION IMPORTANCE ####################################
################################################################################
from sklearn.inspection import permutation_importance

# Compute permutation feature importance
results = permutation_importance(rf, components_df, y, n_repeats=10, random_state=42, n_jobs=2)

# Print the feature ranking
print("Permutation Feature Importances:")
for i in results.importances_mean.argsort()[::-1][:10]:
    print(f"{components_df.columns[i]}: {results.importances_mean[i]:.3f} +/- {results.importances_std[i]:.3f}")

# %%
################################################################################
#################### MUTUAL INFORMATION ########################################
################################################################################
from sklearn.feature_selection import mutual_info_regression

# Compute mutual information
mi = mutual_info_regression(components_df, y)

# Sort the components by mutual information
mi_series = pd.Series(mi, index=components_df.columns)
mi_sorted = mi_series.sort_values(ascending=False)

print("Top 10 Components by Mutual Information:")
print(mi_sorted.head(10))

# %%
################################################################################
#################### ASSOCIATION RULES #########################################
################################################################################
from mlxtend.frequent_patterns import apriori, association_rules

# Convert the components DataFrame to boolean
components_bool_df = components_df.astype(bool)

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(components_bool_df, min_support=0.1, use_colnames=True)

# Derive association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Filter rules that imply high fitness
high_fitness_rules = rules[rules['consequents'] == {True}]
print(high_fitness_rules)

# %%
