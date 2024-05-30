"""
This analyzes which metric is suitable for identifying good components.
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import pickle
from tqdm import tqdm
from numba import njit

population = "../results/data/component_fitness_analysis_200gen_05-14-2024_13-03-59/no_log_splices/1_05-14-2024_13-04-08/reports/population.pkl"
df_expanded = pd.read_pickle(population)
# %%
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
components_expanded = mlb.fit_transform(df_expanded['my_components'])
components_df = pd.DataFrame(components_expanded, columns=mlb.classes_)

# Concatenate this with the 'fitness' column
df_expanded = pd.concat([df_expanded['fitness'], components_df], axis=1)

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
#################### OPTIMIZED T TEST COMPARISON ###############################
################################################################################

def optimized_t(expanded_df: pd.DataFrame):
    comp_dict = {}

    pop = expanded_df['fitness'].to_numpy()
    pop_sum, pop_len = pop.sum(), len(pop)
    pop_df = pop_len - 2

    for component in tqdm(df_expanded.columns[1:]):
        inc = df_expanded[df_expanded[component] == 1]['fitness'].to_numpy()
        comp_dict[component] = compute_t(inc, pop, pop_len, pop_sum, pop_df)
    return comp_dict

@njit(parallel=True)
def compute_t(inc, pop, pop_len, pop_sum, pop_df):
    inc_sum, inc_len = inc.sum(), len(inc)
    inc_avg_fit = inc_sum / inc_len
    exc_len = pop_len - inc_len
    exc_avg_fit = (pop_sum - inc_sum) / exc_len
    mean_diff = inc_avg_fit - exc_avg_fit

    inc_df, exc_df = inc_len-1, exc_len-1

    inc_ss = ((inc - inc_avg_fit)**2).sum()
    inc_var = inc_ss / inc_len
    
    exc_ss = ((pop - exc_avg_fit)**2).sum() - inc_ss - inc_len*mean_diff**2
    exc_var = exc_ss / exc_len

    pool_var = (inc_var*inc_df + exc_var*exc_df) / pop_df
    se = (pool_var/inc_len + pool_var/exc_len)**0.5
    return mean_diff/se


opt_comp_dict = optimized_t(df_expanded)

with open('opt_comp_dict.pkl', 'wb') as f:
    pickle.dump(opt_comp_dict, f)


# %%
################################################################################
#################### T TEST COMPARISON #########################################
################################################################################

def slow_t(expanded_df: pd.DataFrame):
    unop_comp_dict = {}
    for component in tqdm(expanded_df.columns[1:]):
        # Split data into two groups based on the presence of the gene
        present = expanded_df[expanded_df[component] == 1]['fitness']
        absent = expanded_df[expanded_df[component] == 0]['fitness']
        
        # Perform independent two-sample t-test (one-sided)
        t_stat, p_val = ttest_ind(present, absent, alternative='greater', equal_var=True)
        
        # Append the result
        unop_comp_dict[component] = t_stat

    return unop_comp_dict

unop_comp_dict = slow_t(df_expanded)

with open('unop_comp_dict.pkl', 'wb') as f:
    pickle.dump(unop_comp_dict, f)
# %%
results_df['P-Value'].value_counts()
# plot the p-values
# plt.hist(results_df['P-Value'], bins=50)
# plt.hist(results_df['T-Statistic'], bins=50)
# results_df.sort_values(by='T-Statistic', ascending=False)

for g in results_df.sort_values(by='T-Statistic', ascending=True)['Gene'][:10]:
    print(g)
    print()

# %%
################################################################################
#################### UPDATE CORRELATIONS FOR EVERY GEN #########################
################################################################################
# think about this later
# %%
################################################################################
#################### LINEAR REGRESSION #########################################
################################################################################
import statsmodels.api as sm

X = sm.add_constant(components_df)
y = df_expanded['fitness']

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
################################################################################
#################### EXPERIMENTING WITH T-VAL RESULTS ##########################
################################################################################

# plot the t-values
ts = unop_comp_dict.values()
# ts = opt_comp_dict.values()

plt.hist(ts, bins=50)
# scale histogram y-axis to log scale
plt.yscale('log')

# %%
# print top 10 highest t-values and their corresponding components
from pprint import pprint

sorted_unop_comp_dict = {k: v for k, v in sorted(unop_comp_dict.items(), key=lambda item: item[1], reverse=True)}
pprint(list(sorted_unop_comp_dict.items())[:10])
