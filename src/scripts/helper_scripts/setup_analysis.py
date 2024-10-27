import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import gzip
import pickle
from tqdm import tqdm

################################################################################
#################### PROCESSING AND COMBINING DATAFRAMES #######################
################################################################################

# -------------------- PICKLED FILES (E.G. COMPONENT DICT)

def load_compressed_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

# -------------------- GEN INFO & MUTATION DF

def aggregate_dataframes(dataframes, grouper, exclude_cols, sortbygen=False):
    try:
        combined_df = pl.concat(dataframes)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("this was most likely because there are missing columns in one of the dataframes, will use diagonal join")
        combined_df = pl.concat(dataframes, how="diagonal")
    aggregated = combined_df.group_by(grouper).agg([
        pl.exclude(exclude_cols).mean()
    ])
    if sortbygen: aggregated = aggregated.sort("gen")
    return aggregated

def aggregate_geninfo_dataframes(dataframes):
    return aggregate_dataframes(dataframes, "gen", ["best_genome"], True)

def aggregate_mutation_dataframes(dataframes):
    return aggregate_dataframes(dataframes, "my_mutation", ["generations"])

# -------------------- POP DF
def analyze_spawns_by_fitness_rank(pop_df, popsize, maxgen):
    """
    Analyzes how many offspring each fitness rank produces across generations.
    
    Args:
        pop_df: Polars DataFrame containing genetic algorithm data
        popsize: Size of population per generation (default: 500)
    
    Returns:
        dict: Mapping of fitness ranks to total number of offspring spawned
    """
    rank_spawns = {rank: 0 for rank in range(1, popsize + 1)}
    
    for gen in range(2, maxgen + 1):
        prev_gen_df = pop_df.filter(pl.col('gen') == gen - 1)
        sorted_prev = prev_gen_df.sort('fitness', descending=True)
        previous_parents = dict(zip(sorted_prev['id'], range(1, len(sorted_prev) + 1)))
        
        # Get current generation's data and count offspring per parent
        parent_counts = (
            pop_df
            .filter(pl.col('gen') == gen)
            .group_by('parent_id')
            .agg(pl.len().alias('spawn_count'))
        )
        
        # Map parent ranks to spawn counts
        for row in parent_counts.iter_rows():
            parent_id, spawn_count = row
            parent_rank = previous_parents[parent_id]
            rank_spawns[parent_rank] += spawn_count
    
    return rank_spawns


def aggregate_spawn_ranks(rank_dicts_list):
    """
    Calculates the average spawn count for each rank across multiple rank dictionaries.
    """
    # Initialize dictionary to store sums
    sum_dict = {}
    # Sum up values for each rank
    for rank_dict in rank_dicts_list:
        for rank, count in rank_dict.items():
            if rank not in sum_dict:
                sum_dict[rank] = []
            sum_dict[rank].append(count)
    # Calculate averages
    return {rank: np.mean(counts) for rank, counts in sum_dict.items()}


def get_mutation_stats_expanded(pop_df):
    return pop_df.filter(pl.col("my_mutation")!="").group_by(
        ["gen", "my_mutation"]).agg([
            pl.col("fitness_difference").mean().alias("mean_fitness_difference"),
            pl.col("fitness_difference").max().alias("max_fitness_difference"),
            pl.col("my_mutation").len().alias("frequency"),
        ]).pivot(
            index="gen",
            on="my_mutation",
            values=["mean_fitness_difference", "max_fitness_difference", "frequency"]
        ).fill_null(0).pipe(lambda df: df.select(sorted(df.columns)))


# -------------------- COMPONENTS DICT

def count_unique_components(component_dict, max_gen):
    """Count the number of unique components per generation from a component dictionary.
    """
    gen_counts = {gen: 0 for gen in range(1, max_gen + 1)}
    # Count components for each generation
    for component_data in component_dict.values():
        for gen in component_data["fitnesses"].keys():
            gen_counts[gen] += 1
    
    return  pl.DataFrame({
        "gen": list(gen_counts.keys()),
        "num_unique_components": list(gen_counts.values())
    }).sort("gen")

# -------------------- RESULTS DF

def get_mapped_setupname_df(df, setup_map):
    """
    Accepts a dictionary that maps setup_{i} name to a name specified in setup_map.

    Args:
        setup_map (dict): A dictionary that maps an integer key to a new setup name.
                          The setup_map looks like this: {i: "new_setup_name"}

    Returns:
        dict: A dictionary with the updated setup names.
    """
    return df.with_columns([
        pl.col("setupname").map_elements(
            lambda x: setup_map.get(
                int(x.split('_')[1]), 
                x
            )
        ).alias("setupname")
    ])

def get_best_setups(final_report_df: pl.DataFrame, setup_map: dict = None) -> pl.DataFrame:
    """Simply returns the aggregated setups sorted by mean max fitness"""
    if setup_map:
        final_report_df = get_mapped_setupname_df(final_report_df, setup_map)
    return final_report_df.group_by("setupname").agg(
        pl.col("max_fitness").mean().alias("mean_fitness"),
        pl.col("max_fitness").max().alias("max_fitness"),
        pl.col("max_fitness").median().alias("median_fitness"),
        pl.col("max_fitness").std().alias("std_fitness"),
    ).sort("mean_fitness", descending=True)

################################################################################
#################### CRAWLING THE RESULTS ######################################
################################################################################

def exec_results_crawler(
        root_path: str,
        save_dfs = True,
        force_recalculation = False,
    ) -> dict:
    """
    Process execution data from a directory structure containing genetic algorithm runs.
    
    Args:
        root_path (str): Path to the root directory containing execution_data
        save_dfs (bool): Whether to save aggregated results to disk
        force_recalculation (bool): If True, recalculate all aggregations even if cached results exist
        
    Returns:
        dict: Dictionary containing processed execution data with structure:
            {
                'final_report': polars.DataFrame,
                'setups': {
                    1: {
                        'params': dict,
                        'gen_info_agg': pl.DataFrame,
                        'mutation_stats_agg': pl.DataFrame,
                        'spawn_rank_agg': dict
                        }
                    2: {...},
                    ...
                }
            }
    """
    # Initialize results dictionary
    execution_results = {
        'final_report': None,
        'setups': {}
    }
    
    # Convert string path to Path object
    root_path = Path(root_path)

    # load max gen, assumes that all runs have same num of gens & stop_cond == "gen"
    with open(root_path / f"{root_path.name}.json") as f:
        maxgen = json.load(f)["setups"][0]["stop_cond"]["val"]
    
    # Load final report
    final_report_path = root_path / "execution_data" / "final_report_df.feather"
    if final_report_path.exists():
        execution_results['final_report'] = pl.read_ipc(final_report_path)
    else:
        raise FileNotFoundError(f"Final report not found at {final_report_path}")
    
    # Process each setup directory
    execution_data_path = root_path / "execution_data"
    for setup_dir in tqdm(execution_data_path.iterdir(), desc="Processing setup directories"):
        # Check if directory name matches setup pattern
        if setup_dir.is_dir() and setup_dir.name.startswith("setup_"):
            try:
                # Extract setup number
                setup_num = int(setup_dir.name.split("_")[1])
                
                # Initialize setup in results
                setup_aggregation = {}
                
                # Load setup parameters
                params_path = setup_dir / f"{setup_dir.name}_params.json"
                with open(params_path) as f:
                    params = json.load(f)
                setup_aggregation['params'] = params

                # Check for cached results
                aggregated_runs_dir = setup_dir / "aggregated_runs"
                cache_exists = aggregated_runs_dir.exists()
                
                if cache_exists and not force_recalculation:
                    try:
                        print(f"\nFound cached results for {setup_dir.name}")
                        # Load cached results
                        setup_aggregation['gen_info_agg'] = pl.read_ipc(
                            aggregated_runs_dir / "gen_info_agg.feather"
                        )
                        setup_aggregation['mutation_stats_agg'] = pl.read_ipc(
                            aggregated_runs_dir / "mutation_stats_agg.feather"
                        )
                        with open(aggregated_runs_dir / "spawn_rank_agg.json", 'r') as f:
                            setup_aggregation['spawn_rank_agg'] = json.load(f)
                        
                        print(f"Successfully loaded cached results for {setup_dir.name}")
                        
                    except Exception as e:
                        print(f"Error loading cached results for {setup_dir.name}: {e}")
                        print("Falling back to recalculation...")
                        force_recalculation = True  # Force recalculation for this setup
                
                if not cache_exists or force_recalculation:
                    # Check for speciation strategy and popsize
                    is_speciation = params.get('selection_strategy') == 'speciation'
                    popsize = params.get('popsize')

                    agg_gen_info = []
                    agg_mutation_stats = []
                    agg_spawn_ranks = []
                    
                    # Process each run directory
                    for run_dir in setup_dir.iterdir():
                        if run_dir.is_dir():
                            data_dir = run_dir / "data"
                            # load the gen_info and mutation_stats dfs
                            gen_info_df = pl.read_ipc(data_dir / "gen_info.feather")
                            mutation_stats_df = pl.read_ipc(data_dir / "mutation_stats_df.feather")
                            # load and add the component data of that run to the df
                            cdict = load_compressed_pickle(data_dir / "component_dict.pkl.gz")
                            gen_info_df = gen_info_df.join(count_unique_components(cdict, maxgen), "gen")
                            # fetch data from pop df, calculate spawn ranks
                            pop_df = pl.read_ipc(data_dir / "population.feather")
                            agg_spawn_ranks.append(analyze_spawns_by_fitness_rank(pop_df, popsize, maxgen))
                            # calculate mean metrics and join to gen info df
                            mean_metrics = pop_df.group_by('gen').agg([pl.col('^metric_.*$').mean()])
                            gen_info_df = gen_info_df.join(mean_metrics, "gen")
                            # calculate mutation stats and join them
                            mutation_stats = get_mutation_stats_expanded(pop_df)
                            gen_info_df = gen_info_df.join(mutation_stats, "gen")
                            # append to aggregation list
                            agg_gen_info.append(gen_info_df)
                            agg_mutation_stats.append(mutation_stats_df)

                    setup_aggregation['gen_info_agg'] = aggregate_geninfo_dataframes(agg_gen_info)
                    setup_aggregation['mutation_stats_agg'] = aggregate_mutation_dataframes(agg_mutation_stats)
                    setup_aggregation['spawn_rank_agg'] = aggregate_spawn_ranks(agg_spawn_ranks)

                    # Save aggregated results if requested
                    if save_dfs:
                        try:
                            # Create aggregated_runs directory if it doesn't exist
                            aggregated_runs_dir.mkdir(exist_ok=True)
                            
                            # Save the aggregated results
                            setup_aggregation['gen_info_agg'].write_ipc(
                                aggregated_runs_dir / "gen_info_agg.feather"
                            )
                            setup_aggregation['mutation_stats_agg'].write_ipc(
                                aggregated_runs_dir / "mutation_stats_agg.feather"
                            )
                            with open(aggregated_runs_dir / "spawn_rank_agg.json", 'w') as f:
                                json.dump(setup_aggregation['spawn_rank_agg'], f)
                                
                            print(f"\nSaved aggregated results for {setup_dir.name}")
                            
                        except Exception as e:
                            print(f"Error saving aggregated results for {setup_dir.name}: {e}")

                execution_results['setups'][setup_num] = setup_aggregation
                
            except (ValueError, IndexError) as e:
                print(f"Error processing directory {setup_dir}: {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"Error reading params file for {setup_dir}: {e}")
                continue
    
    return execution_results


def search_setups(res_dict: dict, search_dict: dict):
    contains_params = lambda d1, d2: all(k in d2 and d2[k] == v for k, v in d1.items())
    search_res = {}
    # key: params to search, value: group to put it in
    for setup_name, setup in res_dict["setups"].items():
        for search_name, search_params in search_dict.items():
            if contains_params(search_params, setup["params"]):
                search_res.setdefault(search_name, []).append(setup_name)
    return search_res


def search_and_aggregate_param_results(res_dict: dict, search_dict: dict):
    """
    Filters a results dict returned by exec_results_crawler() for specific params.

    Args:
        results (dict): The results dictionary returned by exec_results_crawler().
        search (dict): A dictionary specifying the parameters to filter by. 
            The schema for the search dict accepts dicts like this one:
            search = {
                "label for setup": {"search_param": value}, ...
            }

    Returns:
        searches the results dicts for runs with matching params, and aggregates
        all matching dicts further
        agg_dict: {
            "label for setup": pl.DataFrame
        }
            
    """
    search_res = search_setups(res_dict, search_dict)

    # aggregation results
    agg_dict = {}
    for search_name, setups in search_res.items():
        dfs = []
        for setup in setups:
            dfs.append(res_dict["setups"][setup]["gen_info_agg"])
        agg_df = aggregate_geninfo_dataframes(dfs)
        agg_dict[search_name] = agg_df
                
    return agg_dict

################################################################################
#################### AGGREGATED PLOTTING FUNCTIONS #############################
################################################################################

def components_fitness_scatter(df: pl.DataFrame, setup_map: dict = None) -> None:
    """
    Accepts a results DataFrame and visualizes all runs on a scatterplot: 
    fitness ~ num_components, with the mean of each setup marked by a cross.
    
    Args:
        df (pl.DataFrame): Input Polars DataFrame containing results
        setup_map (dict, optional): Dictionary mapping setup numbers to new names
    """
    # If a setup_map is provided, rename the setupname column values
    if setup_map:
        df = get_mapped_setupname_df(df, setup_map)
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique setupnames for colors
    setupnames = df.get_column("setupname").unique().to_list()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(setupnames)))
    
    # Plot each setupname with a different color
    for setupname, color in zip(setupnames, colors):
        # Filter data for current setup
        setup_data = df.filter(pl.col("setupname") == setupname)
        
        # Convert to numpy arrays for plotting
        components = setup_data.get_column("num_components").to_numpy()
        fitness = setup_data.get_column("max_fitness").to_numpy()
        
        # Scatter plot for individual points
        ax.scatter(components, 
                  fitness,
                  c=[color], 
                  label=setupname,
                  alpha=0.7)
        
        # Calculate and plot mean without adding to legend
        mean_components = float(setup_data.select(pl.col("num_components").mean()).item())
        mean_fitness = float(setup_data.select(pl.col("max_fitness").mean()).item())
        ax.scatter(mean_components, 
                  mean_fitness, 
                  c=[color], 
                  marker='X', 
                  s=200, 
                  edgecolors='black', 
                  linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Max Fitness')
    ax.set_title('Max Fitness vs Number of Components (with Setup Means)')
    
    # Add legend
    ax.legend()
    
    # Add a text annotation explaining the 'X' markers
    ax.text(0.95, 0.05, 
            "'X' markers represent setup means", 
            transform=ax.transAxes, 
            ha='right', 
            va='bottom', 
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def create_subplot_grid(num_plots: int) -> tuple[int, int]:
    """Helper function to determine the grid layout based on number of plots.
    """
    if num_plots == 1:
        return 1, 1
    elif num_plots == 2:
        return 2, 1  # Changed to stack vertically
    elif num_plots == 3:
        return 2, 2
    else:
        return 2, 2


def generalized_lineplot(
    plt_layout: List[List[str]],
    data_sources: Dict[str, pl.DataFrame],
    y_ax: str,
    x_ax: str = "gen",
    title: Optional[str] = None,
    subplt_titles: Optional[List[str]] = None,
    figsize: tuple[int, int] = (12, 8),
    legend_loc = "lower right"
    ) -> None:
    """
    Create a dynamic multi-subplot figure with line plots.
    
    Parameters:
    -----------
    plt_layout : List[List[str]]
        Nested list defining the plot structure. Outer list determines subplots,
        inner lists determine lines within each subplot.
    data_sources : Dict[str, pl.DataFrame]
        Dictionary mapping data source names to polars DataFrames.
    y_ax : str
        Column name to plot on y-axis for all plots.
    x_ax : str, optional
        Column name to sort by and plot on x-axis. Defaults to "gen".
    title : str, optional
        Main figure title. If None, uses f"{y_ax} by {x_ax}".
    subplt_titles : List[str], optional
        List of titles for each subplot. Must match length of plt_layout if provided.
    figsize : tuple[int, int], optional
        Figure size in inches. Defaults to (12, 8).
    """
    # Input validation
    if not plt_layout:
        raise ValueError("plt_layout cannot be empty")
    
    if subplt_titles and len(subplt_titles) != len(plt_layout):
        raise ValueError("subplt_titles must match the number of subplots in plt_layout")
    
    # Determine grid layout
    rows, cols = create_subplot_grid(len(plt_layout))
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Convert axes to array for consistent indexing
    if len(plt_layout) == 1:
        axes = np.array([axes])  # Single plot
    elif len(plt_layout) == 2:
        axes = np.array([axes[0], axes[1]])  # Two plots stacked vertically
    else:
        axes = axes.flatten()  # Multiple plots in a grid

    
    # Set main title
    if title is None:
        title = f"{y_ax.replace("_", " ")} by {x_ax}"
    fig.suptitle(title, fontsize=14, y=1.02)
    
    # Create plots
    for idx, (subplot_data, ax) in enumerate(zip(plt_layout, axes)):
        # Plot each line in the subplot
        for line_data in subplot_data:
            if line_data not in data_sources:
                raise ValueError(f"Data source '{line_data}' not found in data_sources")
            
            df = data_sources[line_data]
            
            # Sort data by x_ax column
            df = df.sort(x_ax)
            
            # Plot the line
            ax.plot(
                df[x_ax].to_numpy(),
                df[y_ax].to_numpy(),
                label=line_data,
            )
        
        # Set subplot title if provided
        if subplt_titles:
            ax.set_title(subplt_titles[idx])
        
        # Add labels and legend
        ax.set_xlabel(x_ax)
        ax.set_ylabel(y_ax)
        ax.legend(loc=legend_loc)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide empty subplots if any
    for idx in range(len(plt_layout), len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig


def generalized_barplot(
    plt_layout: List[List[str]],
    data_sources: Dict[str, pl.DataFrame],
    y_ax: str,
    gen: int = -1,
    title: Optional[str] = None,
    group_titles: Optional[List[str]] = None,
    figsize: tuple[int, int] = (12, 6),
    label_lambda = lambda l: l,
    label_rot = 90
    ) -> None:
    """
    Create a grouped bar plot where each group contains multiple bars.
    
    Parameters:
    -----------
    plt_layout : List[List[str]]
        List of lists where each inner list represents a group of bars to be plotted together.
    data_sources : Dict[str, pl.DataFrame]
        Dictionary mapping data source names to polars DataFrames.
    y_ax : str
        Column name to plot as bar heights.
    gen : int, optional
        Generation number to plot. If -1 (default), uses the last generation.
    title : str, optional
        Plot title. If None, uses f"{y_ax} at generation {gen}"
    group_titles : List[str], optional
        Labels for each group of bars. Must match length of plt_layout.
    figsize : tuple[int, int], optional
        Figure size in inches. Defaults to (12, 6).
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    # Set main title
    if title is None:
        generation = gen if gen != -1 else "final generation"
        title = f"{y_ax.replace('_', ' ')} at {generation}"
    ax.set_title(title)
    
    # Calculate number of groups and bars
    num_groups = len(plt_layout)
    max_bars = max(len(group) for group in plt_layout)
    
    # Set up bar width and positions
    bar_width = 0.8 / max_bars
    x = np.arange(num_groups)
    
    # Plot each set of bars
    for bar_idx in range(max_bars):
        values = []
        labels = []
        
        # Collect values for this set of bars
        for group_idx, group in enumerate(plt_layout):
            if bar_idx < len(group):
                data_key = group[bar_idx]
                if data_key not in data_sources:
                    raise ValueError(f"Data source '{data_key}' not found in data_sources")
                
                df = data_sources[data_key]
                
                # Get the correct generation
                if gen == -1:
                    target_gen = df['gen'].max()
                else:
                    target_gen = gen
                
                # Get the value for the specified generation
                value = df.filter(pl.col('gen') == target_gen)[y_ax].item()
                values.append(value)
                labels.append(data_key)
            else:
                values.append(0)  # Add placeholder for missing bars
                labels.append("")
        
        # Plot bars for this set
        offset = bar_width * bar_idx
        rects = ax.bar(x + offset, values, bar_width, 
                      label=f"Bar Set {bar_idx + 1}")
        
        # Add value labels on bars
        for rect, label, value in zip(rects, labels, values):
            if value > 0:  # Only label non-zero bars
                ax.text(rect.get_x() + rect.get_width() / 2., rect.get_height() - (rect.get_height() * 0.05),
                        label_lambda(f'{label}\n{value:.0f}'),
                        ha='center', va='top', color='black', rotation=label_rot)
    
    # Customize the plot
    ax.set_ylabel(y_ax.replace('_', ' '))
    if group_titles:
        ax.set_xticks(x + bar_width * (max_bars - 1) / 2)
        ax.set_xticklabels(group_titles)
    else:
        ax.set_xticks(x + bar_width * (max_bars - 1) / 2)
        ax.set_xticklabels([f"Group {i+1}" for i in range(num_groups)])
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    return fig


import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy

def run_regression(results_dict: dict, predictors: list):
    """
    Perform linear regression analysis to study parameter impacts on num_components and max_fitness.
    
    Args:
        results_dict (dict): Dictionary containing execution results from exec_crawler
        predictors (list): List of parameter names to use as predictors in regression
        
    Returns:
        dict: Dictionary containing regression results for each target variable
    """
    # Extract the results dataframe
    results_df = results_dict['final_report']
    
    # Create new dataframe with setup numbers
    df = results_df.with_columns(
        setup_num=pl.col('setupname').str.extract(r'setup_(\d+)').cast(pl.Int32)
    )
    
    # Add predictor columns from params
    for predictor in predictors:
        # Create a mapping of setup numbers to parameter values
        param_values = {
            setup_num: setup_data['params'].get(predictor)
            for setup_num, setup_data in results_dict['setups'].items()
        }
        
        # Convert param_values to a list maintaining order
        param_series = [param_values.get(i) for i in df['setup_num']]
        
        # Add the column to the dataframe
        df = df.with_columns(pl.Series(predictor, param_series))
    
    # Prepare data for regression
    X = df.select(predictors).to_numpy()
    targets = ["num_components", "max_fitness"]
    
    # Initialize scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store regression results
    regression_results = {}
    
    # Perform regression for each target
    for target in targets:
        y = df.select(target).to_numpy().ravel()
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate R-squared and adjusted R-squared
        r2 = model.score(X_scaled, y)
        n = X.shape[0]  # number of observations
        p = X.shape[1]  # number of predictors
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Calculate p-values
        n = X.shape[0]
        p = X.shape[1]
        dof = n - p - 1  # degrees of freedom
        mse = np.sum((y - model.predict(X_scaled)) ** 2) / dof  # mean squared error
        var_b = mse * np.linalg.inv(np.dot(X_scaled.T, X_scaled)).diagonal()
        sd_b = np.sqrt(var_b)
        t_stat = model.coef_ / sd_b
        p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), dof))
        
        # Store results
        regression_results[target] = {
            'model': model,
            'coefficients': dict(zip(predictors, model.coef_)),
            'intercept': model.intercept_,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'p_values': dict(zip(predictors, p_values)),
            'feature_importance': dict(zip(
                predictors,
                np.abs(model.coef_ * np.std(X, axis=0))  # standardized coefficients
            ))
        }
        
        # Print summary
        print(f"\nRegression Results for {target}")
        print("-" * 50)
        print(f"R-squared: {r2:.4f}")
        print(f"Adjusted R-squared: {adjusted_r2:.4f}")
        print("\nStandardized Coefficients:")
        for pred, coef, p_val in zip(predictors, model.coef_, p_values):
            print(f"{pred:20} {coef:10.4f} (p={p_val:.4f})")
            
    # Add the processed dataframe to the results
    regression_results['processed_df'] = df
    
    return regression_results

################################################################################
#################### ONE-OFF PLOTTING FUNCTIONS ################################
################################################################################

def plot_offspring_distribution(rank_spawns):
    """Creates a histogram of spawn counts by fitness rank.
    """
    ranks = list(rank_spawns.keys())
    spawn_counts = list(rank_spawns.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(ranks, spawn_counts, width=1)
    plt.xlabel('Fitness Rank of Parent')
    plt.ylabel('Number of Offspring')
    plt.title('Distribution of Offspring by Parent Fitness Rank')
    plt.xlim(0, 500)
    # plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()