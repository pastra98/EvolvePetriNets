# %%
from collections import defaultdict
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from statistics import mean
import gzip
import pickle
from tqdm import tqdm
import re
import seaborn as sns
import pandas as pd
import numpy as np
import pickle, gzip
import matplotlib.patheffects as path_effects


################################################################################
#################### PROCESSING AND COMBINING DATAFRAMES #######################
################################################################################

#################### CONSTANTS USED FOR PLOTS ##################################
TICKFONT = 16
AXLABELFONT = 18
TITLEFONT = 24
SUBPLOTITLEFONT = 20
LEGENDFONT = 16
plt.rcParams['font.size'] = TICKFONT  # Base font size
plt.rcParams['axes.titlesize'] = TITLEFONT
plt.rcParams['axes.labelsize'] = AXLABELFONT
plt.rcParams['xtick.labelsize'] = TICKFONT
plt.rcParams['ytick.labelsize'] = TICKFONT
plt.rcParams['legend.fontsize'] = LEGENDFONT
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
    return aggregate_dataframes(dataframes, "gen", ["best_genome", "best_species"], True)

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
        use_setup_num = True,
        load_best_genomes = True
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
        if setup_dir.is_dir():
            try:
                # Extract setup number, use dir name if not use_setup_num
                print(setup_dir.name)
                setupname = int(setup_dir.name.split("_")[1]) if use_setup_num else setup_dir.name
                
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
                        
                        # still load the genomes - duplicate code but this whole function is AI slop anyways at this point
                        if load_best_genomes:
                            best_genomes = []
                            for run_dir in setup_dir.iterdir():
                                if run_dir.is_dir() and run_dir.name != "aggregated_runs":
                                    # print(run_dir.name)
                                    best_g = load_compressed_pickle(run_dir / "best_genome.pkl.gz")
                                    del best_g.pop_component_tracker
                                    best_genomes.append(best_g)
                            setup_aggregation['best_genomes'] = best_genomes

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
                    fitness_variances = []
                    best_genomes = []
                    
                    # Process each run directory
                    for run_dir in setup_dir.iterdir():
                        if run_dir.is_dir() and run_dir.name != "aggregated_runs":
                            data_dir = run_dir / "data"
                            # load the gen_info and mutation_stats dfs
                            gen_info_df = pl.read_ipc(data_dir / "gen_info.feather")
                            mutation_stats_df = pl.read_ipc(data_dir / "mutation_stats_df.feather")
                            maxgen = len(gen_info_df)
                            # load best genome and append to list (if there is still component tracker, delete it ya moron)
                            if load_best_genomes:
                                best_g = load_compressed_pickle(run_dir / "best_genome.pkl.gz")
                                del best_g.pop_component_tracker
                                best_genomes.append(best_g)
                            # load and add the component data of that run to the df
                            cdict = load_compressed_pickle(data_dir / "component_dict.pkl.gz")
                            gen_info_df = gen_info_df.join(count_unique_components(cdict, maxgen), "gen")
                            # fetch data from pop df, calculate spawn ranks
                            pop_df = pl.read_ipc(data_dir / "population.feather")
                            agg_spawn_ranks.append(analyze_spawns_by_fitness_rank(pop_df, popsize, maxgen))
                            # get the fitness variances from the pop df
                            fitness_stats = pop_df.group_by("gen").agg([
                                pl.col("fitness").var().alias("fitness_variance"),
                                (pl.col("fitness").std() / pl.col("fitness").count().sqrt()).alias("fitness_stderr")
                            ]).sort("gen")
                            fitness_variances.append(fitness_stats)
                            # calculate mean metrics and join to gen info df
                            mean_metrics = pop_df.group_by('gen').agg([pl.col('^metric_.*$').mean()])
                            gen_info_df = gen_info_df.join(mean_metrics, "gen")
                            # calculate mutation stats and join them
                            mutation_stats = get_mutation_stats_expanded(pop_df)
                            gen_info_df = gen_info_df.join(mutation_stats, "gen")
                            # append to aggregation list
                            agg_gen_info.append(gen_info_df)
                            agg_mutation_stats.append(mutation_stats_df)

                    # first aggregate the gen info, then the other dataframes
                    agg_gen_info = aggregate_geninfo_dataframes(agg_gen_info)
                    aggregated_fitness_stats = aggregate_dataframes(fitness_variances, "gen", [], True)
                    setup_aggregation['gen_info_agg'] = agg_gen_info.join(aggregated_fitness_stats, "gen")
                    # the other dataframes
                    setup_aggregation['mutation_stats_agg'] = aggregate_mutation_dataframes(agg_mutation_stats)
                    setup_aggregation['spawn_rank_agg'] = aggregate_spawn_ranks(agg_spawn_ranks)
                    # add best genomes
                    setup_aggregation['best_genomes'] = best_genomes

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

                execution_results['setups'][setupname] = setup_aggregation
                
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


def search_and_aggregate_param_results(res_dict: dict, search_dict: dict, search_res=None):
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
    if not search_res:
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

### ---- fitness boxplot ---- ####
def create_fitness_boxplots(df):
    """
    Creates boxplots for fitness data grouped by size and selection method.
    
    Args:
        df: Polars or Pandas DataFrame with columns: setupname, max_fitness
        
    Returns:
        tuple: (matplotlib figure, pandas DataFrame with 5-point summary)
    """
    # Convert to pandas if it's a polars dataframe
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()
    
    # Split setupnames and create new columns
    df[['selection_method', 'size', 'log']] = df['setupname'].str.split('_', expand=True)
    
    # Define colors for each method
    colors = {
        "truncation": "blue",
        "roulette": "orangered",
        "speciation": "green"
    }
    
    # Initialize storage for summary statistics
    summary_stats = []
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('', fontsize=TITLEFONT)
    
    # Define sizes and their order
    sizes = ['small', 'medium', 'big']
    
    # Process each size
    for idx, size in enumerate(sizes):
        # Get data for this size
        size_data = df[df['size'] == size]
        
        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        
        for method in ['roulette', 'truncation', 'speciation']:
            method_data = size_data[size_data['selection_method'] == method]['max_fitness']
            data_to_plot.append(method_data)
            
            # Calculate 5-point summary
            summary = method_data.describe()
            summary_stats.append({
                'size': size,
                'method': method,
                'min': summary['min'],
                'q1': summary['25%'],
                'median': summary['50%'],
                'q3': summary['75%'],
                'max': summary['max']
            })
            label = method + "\n(" + str(round(method_data.mean(), 3)) + ")"
            labels.append(label)
        
        # Create boxplot with colors and meanlines
        bp = axes[idx].boxplot(data_to_plot, 
                             labels=labels,
                             patch_artist=True,  # Required for filling boxes with color
                             meanline=True,      # Show mean lines
                             showmeans=True)     # Show mean markers
        
        # Customize box colors
        for box, method in zip(bp['boxes'], ['roulette', 'truncation', 'speciation']):
            box.set(facecolor=colors[method], alpha=0.3)
        
        # Set meanline and mean marker properties
        for mean in bp['means']:
            mean.set(color='black', linestyle='--', linewidth=1)
        
        # Add horizontal grid lines
        axes[idx].yaxis.grid(True, linestyle=':', alpha=0.7)
        axes[idx].set_title(f'{size.capitalize()} Log', fontsize=SUBPLOTITLEFONT)
        axes[idx].set_ylabel('Max Fitness' if idx == 0 else '', fontsize=AXLABELFONT)
        axes[idx].tick_params(axis='both', labelsize=TICKFONT)
        axes[idx].set_xticklabels(labels, rotation=45)
        
        # Set background grid below boxes
        axes[idx].set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_stats)
    
    return fig, summary_df

### ---- fitness boxplot ---- ####


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


from matplotlib.colors import LinearSegmentedColormap


def generalized_lineplot(
    plt_layout: List[List[str]],
    data_sources: Dict[str, pl.DataFrame],
    y_ax: str,
    x_ax: str = "gen",
    title: Optional[str] = None,
    subplt_titles: Optional[List[str]] = None,
    figsize: tuple[int, int] = (8, 8),
    legend_loc = "lower right",
    show_all_legends: bool = True,
    legend_lambda=lambda l: l,
    gen_limit=None,
    use_gradient_colscheme=False,
    no_legend=False,
    show_legend_outside=False
    ) -> None:
    """
    Create a dynamic multi-subplot figure with line plots.
    Handles 1, 2, or 3 subplots arranged horizontally.
    
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
        Figure size for each subplot in inches. Defaults to (8, 8).
    legend_loc : str, optional
        Location of the legend. Defaults to "lower right".
    show_all_legends : bool, optional
        If True, shows legends for all subplots. If False, only shows legend for the first subplot.
        Defaults to True.
    """
    # Input validation
    if not plt_layout:
        raise ValueError("plt_layout cannot be empty")
    
    if subplt_titles and len(subplt_titles) != len(plt_layout):
        raise ValueError("subplt_titles must match the number of subplots in plt_layout")
    
    num_plots = len(plt_layout)
    if num_plots > 3:
        raise ValueError("This function supports a maximum of 3 subplots")
    
    # Create figure and subplots
    # For 3 subplots, create them in one row with the specified figsize for each
    if num_plots == 3:
        total_width = figsize[0] * 3
        total_height = figsize[1]
        fig, axes = plt.subplots(1, 3, figsize=(total_width, total_height))
    elif num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    else:  # num_plots == 1
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = np.array([axes])  # Wrap in array for consistent indexing
    
    # Ensure axes is always a numpy array
    if num_plots > 1:
        axes = np.array(axes)
    
    if use_gradient_colscheme:
        colors = plt.cm.coolwarm(np.linspace(1, 0, len(data_sources)))
    else:
        colors = [None] * len(data_sources)  # Use default matplotlib colors

    # Set main title
    if title is None:
        title = f"{y_ax.replace('_', ' ')} by {x_ax}"
    fig.suptitle(title, fontsize=TITLEFONT, y=1.02)
    
    # Create plots
    for idx, (subplot_data, ax) in enumerate(zip(plt_layout, axes)):
        # Plot each line in the subplot
        for i, line_data in enumerate(subplot_data):
            if line_data not in data_sources:
                raise ValueError(f"Data source '{line_data}' not found in data_sources")
            
            df = data_sources[line_data]
            
            # Sort data by x_ax column
            df = df.sort(x_ax)
            if gen_limit:
                df = df.filter(pl.col("gen") < gen_limit)
            
            # Plot the line
            ax.plot(
                df[x_ax].to_numpy(),
                df[y_ax].to_numpy(),
                label=legend_lambda(line_data),
                color=colors[i] if use_gradient_colscheme else None
            )
        
        # Set subplot title if provided
        if subplt_titles:
            ax.set_title(subplt_titles[idx].replace("_", " "), fontsize=SUBPLOTITLEFONT)
        
        # Add labels and legend
        ax.tick_params(labelsize=TICKFONT)
        if show_all_legends or idx == 0:
            ax.legend(loc=legend_loc, fontsize=LEGENDFONT)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(x_ax.replace("_", " "), fontsize=AXLABELFONT)
        ax.set_ylabel(y_ax.replace("_", " "), fontsize=AXLABELFONT)
    
    # Adjust layout to prevent overlap
    if no_legend:
        ax.legend().remove()
    else:
        if not show_legend_outside:
            plt.tight_layout()
        else:
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=LEGENDFONT)
    return fig

def generalized_barplot(
        plt_layout,
        data_sources,
        y_ax,
        gen=-1,
        title=None,
        group_titles=None, 
        figsize=(12, 6),
        label_lambda=lambda l: l,
        label_rot=90,
        fp_precision=2,
        show_errors=True,
        x_label_rot=0,
        ytick_lambda=lambda y: y,
        ylim=None):
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    if title is None:
        generation = gen if gen != -1 else "final generation"
        title = f"{y_ax.replace('_', ' ')} at {generation}"
    ax.set_title(title, fontsize=TITLEFONT)
    
    num_groups = len(plt_layout)
    max_bars = max(len(group) for group in plt_layout)
    bar_width = 0.8 / max_bars
    x = np.arange(num_groups)
    
    for bar_idx in range(max_bars):
        values = []
        errors = []
        labels = []
        
        for group_idx, group in enumerate(plt_layout):
            if bar_idx < len(group):
                data_key = group[bar_idx]
                if data_key not in data_sources:
                    raise ValueError(f"Data source '{data_key}' not found in data_sources")
                
                df = data_sources[data_key]
                target_gen = df['gen'].max() if gen == -1 else gen
                
                value = df.filter(pl.col('gen') == target_gen)[y_ax].item()
                values.append(value)
                labels.append(data_key)
                
                if y_ax == "best_genome_fitness" and show_errors:
                    error = df.filter(pl.col('gen') == target_gen)["fitness_stderr"].item()
                    errors.append(error)
            else:
                values.append(0)
                labels.append("")
                if y_ax == "best_genome_fitness":
                    errors.append(0)
        
        offset = bar_width * bar_idx
        if y_ax == "best_genome_fitness" and show_errors:
            rects = ax.bar(x + offset, values, bar_width, 
                          yerr=errors, capsize=5,
                          label=f"Bar Set {bar_idx + 1}")
        else:
            rects = ax.bar(x + offset, values, bar_width, 
                          label=f"Bar Set {bar_idx + 1}")
        
        for rect, label, value in zip(rects, labels, values):
            if value > 0:
                ax.text(rect.get_x() + rect.get_width() / 2., rect.get_height() - (rect.get_height() * 0.05),
                        label_lambda(f'{label}\n{round(value,fp_precision)}'),
                        ha='center', va='top', color='yellow', rotation=label_rot,
                        fontsize=AXLABELFONT,
                        path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]) # White outline
    
    ax.set_ylabel(y_ax.replace('_', ' '), fontsize=AXLABELFONT)
    ax.set_xticks(x + bar_width * (max_bars - 1) / 2)
    ax.set_xticklabels(group_titles if group_titles else [f"Group {i+1}" for i in range(num_groups)], 
                       fontsize=TICKFONT, rotation=x_label_rot)
    
    ax.tick_params(labelsize=TICKFONT)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    if ylim:
        ax.set_ylim(ylim)
    
    # Apply lambda function to all y-ticks
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([ytick_lambda(y) for y in yticks])

    return fig


def generalized_boxplot(
   plt_layout: List[List[str]],
   data_sources: Dict[str, pl.DataFrame],
   y_ax: str,
   gen: int = -1,
   title: Optional[str] = None,
   group_titles: Optional[List[str]] = None,
   figsize: tuple[int, int] = (12, 6),
   ) -> None:

   # Create figure and axis
   fig, ax = plt.subplots(figsize=figsize, layout='constrained')
   
   # Set title
   if title is None:
       generation = gen if gen != -1 else "final generation"
       title = f"{y_ax.replace('_', ' ')} at {generation}"
   ax.set_title(title)
   
   # Calculate layout
   num_groups = len(plt_layout)
   max_boxes = max(len(group) for group in plt_layout)
   box_width = 0.8 / max_boxes
   x = np.arange(num_groups)
   
   # Plot each set of boxes
   for box_idx in range(max_boxes):
       values = []
       labels = []
       
       # Collect values for this set of boxes
       for group_idx, group in enumerate(plt_layout):
           if box_idx < len(group):
               data_key = group[box_idx]
               if data_key not in data_sources:
                   raise ValueError(f"Data source '{data_key}' not found")
               
               df = data_sources[data_key]
               
               # Get the generation
               target_gen = df['gen'].max() if gen == -1 else gen
               
               value = df.filter(pl.col('gen') == target_gen)[y_ax].to_list()
               values.append(value)
               labels.append(data_key)
           else:
               values.append([])
               labels.append("")

       # Plot boxes
       print(values)
       offset = box_width * box_idx
       bp = ax.boxplot(values, positions=x + offset, widths=box_width*0.8, 
                      patch_artist=True, labels=labels)
               
   # Customize plot
   ax.set_ylabel(y_ax.replace('_', ' '))
   if group_titles:
       ax.set_xticks(x + box_width * (max_boxes - 1) / 2)
       ax.set_xticklabels(group_titles)
   
   ax.grid(True, axis='y', linestyle='--', alpha=0.7)
   
   return fig

import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy

def run_regression(results_dict: dict, predictors: list, include_species_target=False):
    """
    Perform linear regression analysis and return results in a pandas DataFrame.
    
    Args:
        results_dict (dict): Dictionary containing execution results from exec_crawler
        predictors (list): List of parameter names to use as predictors in regression
        
    Returns:
        pd.DataFrame: DataFrame containing regression results with predictors as rows
                     and effects on targets as columns
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    import scipy.stats
    
    # Extract the results dataframe
    df = results_dict['final_report'].to_pandas()
    
    # Create new dataframe with setup numbers
    df['setup_num'] = df['setupname'].str.extract(r'setup_(\d+)').astype(int)
    
    # Add predictor columns from params
    for predictor in predictors:
        param_values = {
            setup_num: setup_data['params'].get(predictor)
            for setup_num, setup_data in results_dict['setups'].items()
        }
        df[predictor] = df['setup_num'].map(param_values)
    
    # Prepare data for regression
    X = df[predictors].values
    targets = ["num_components", "max_fitness"]
    if include_species_target:
        targets.append("num_species_last_gen")
    
    # Initialize scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize results storage
    index = pd.Index([p.replace("_", "\_") for p in predictors])
    columns = []
    data = []

    
    # Perform regression for each target
    for target in targets:
        y = df[target].values
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate R-squared and adjusted R-squared
        r2 = model.score(X_scaled, y)
        n = X.shape[0]
        p = X.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Calculate p-values
        dof = n - p - 1
        mse = np.sum((y - model.predict(X_scaled)) ** 2) / dof
        var_b = mse * np.linalg.inv(np.dot(X_scaled.T, X_scaled)).diagonal()
        sd_b = np.sqrt(var_b)
        t_stat = model.coef_ / sd_b
        p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), dof))
        
                # Format results for this target
        col_name = "\makecell{\\textbf{"+f"{target.replace('_', '\_')}"+"}\\\ (Adj R² = "+f"{adjusted_r2:.4f})" + "}".replace("_", "\_")
        columns.append(col_name)
        
        # Format coefficients and p-values
        target_results = ["\makecell{"+f"{coef:.4f} \\\ (p$\\approx${p_val:.4f})"+"}"
                         for coef, p_val in zip(model.coef_, p_values)]
        data.append(target_results)
    
    # Create final DataFrame
    final_results = pd.DataFrame(np.array(data).T, index=index, columns=columns)
    print("\\begin{table}[h]\n\\centering\n",
          final_results.to_latex().replace("\\\n", "\\[12pt]\n").replace("\\toprule","\\toprule \\rowcolor[gray]{0.9}"),
          "\\caption{XXX}\n\\label{tab:XXX}\n\\end{table}")
    return final_results
    


################################################################################
#################### ONE-OFF PLOTTING FUNCTIONS ################################
################################################################################

def plot_offspring_distribution(rank_spawns, title='Distribution of Offspring by Parent Fitness Rank', cap_limit=None, fsize=(12,6)):
    """Creates a histogram of spawn counts by fitness rank.
    Returns the figure object for saving or further modification.
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    ranks = list(rank_spawns.keys())
    spawn_counts = list(rank_spawns.values())
    
    fig = plt.figure(figsize=fsize)
    # Set align='edge' and width=1.0 to make bars touch each other
    plt.bar(ranks, spawn_counts, width=1.1, align='edge')
    plt.xlabel('Fitness Rank of Parent', fontsize=AXLABELFONT)
    plt.ylabel('Number of Offspring', fontsize=AXLABELFONT)
    plt.title(title, fontsize=TITLEFONT)
    plt.xlim(1, cap_limit if cap_limit else len(ranks))
    # plt.yscale('log')
    plt.tick_params(labelsize=TICKFONT)
    plt.xticks(range(-1, cap_limit if cap_limit else len(ranks), 50))
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    return fig

def get_species_similarities_setup(setup_path):
    similarities = []
    for run_dir in Path(setup_path).iterdir():
        if run_dir.is_dir():
            data_path = run_dir / "data"
            similarities.append(species_similarity_over_time(data_path.as_posix() + "/"))
    # Sum the values for each key
    sum_dict = defaultdict(float)
    count = len(similarities)
    for d in similarities:
        for key, value in d.items():
            sum_dict[key] += value
    # Calculate the average for each key
    return {key: value / count for key, value in sum_dict.items()}

def species_similarity_over_time(data_path):
    # First part remains the same
    pop_df = pl.read_ipc(data_path + "population.feather")
    species_components = {}
    for species_id in pop_df['species_id'].unique():
        species_components[species_id] = {}
        species_data = pop_df.filter(pl.col('species_id') == species_id)
        # Process each generation for this species
        for gen in species_data['gen'].unique():
            gen_data = species_data.filter(pl.col('gen') == gen)
            # Combine all component lists into a single set
            all_components = set()
            for comp_list in gen_data['my_components']:
                all_components.update(comp_list)
            # Store the set in our nested dictionary
            species_components[species_id][gen] = all_components
    
    # Calculate Jaccard similarities for each generation
    species_similarities = {}
    all_generations = sorted(set(gen for species_dict in species_components.values() 
                               for gen in species_dict.keys()))
    
    for gen in all_generations:
        # Get all species that exist in this generation
        species_in_gen = [species_id for species_id, gens in species_components.items() 
                         if gen in gens]
        
        similarities = []
        # Compare each pair of species
        for i in range(len(species_in_gen)):
            for j in range(i + 1, len(species_in_gen)):
                species1 = species_in_gen[i]
                species2 = species_in_gen[j]
                
                set1 = species_components[species1][gen]
                set2 = species_components[species2][gen]
                
                # Calculate Jaccard similarity
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                similarity = intersection / union if union > 0 else 0
                
                similarities.append(similarity)
        
        # Store average similarity for this generation
        species_similarities[gen] = sum(similarities) / len(similarities) if similarities else 0
    
    return species_similarities

def plot_similarities(similarity_dict_collection, title):
    """
    Plot multiple similarity dictionaries on the same graph.
    
    Args:
        similarity_dict_collection (dict): Dictionary where keys are names/labels and 
            values are dictionaries of generation -> similarity mappings
        title (str): Title for the plot
    """
    
    plt.figure(figsize=(8, 5))
    
    # Plot each similarity dictionary
    for name, similarities in similarity_dict_collection.items():
        generations = sorted(similarities.keys())
        avg_similarities = [similarities[gen] for gen in generations]
        
        plt.plot(generations, avg_similarities, '-', label=name, markersize=4)
    
    plt.xlabel('Generation')
    plt.ylabel('Average Jaccard Similarity')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf()  # Return the figure object



#-------------------------------------------------------------------------------
# temporary
import shutil

def cleanup_runs(root_path: str, keep_first_n=1):
    """
    Deletes all files in run directories except for data/gen_info.feather
    
    Args:
        root_path (str): Path to the root directory containing execution_data
    """
    root_path = Path(root_path)
    execution_data_path = root_path / "execution_data"
    
    # Process each setup directory
    for setup_dir in tqdm(execution_data_path.iterdir(), desc="Processing setup directories"):
        if setup_dir.is_dir():
            print(f"\nCleaning {setup_dir.name}")
            
            # Process each run directory
            for run_dir in setup_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        data_dir = run_dir / "data"
                        run_nr = int(run_dir.name.split("_")[0])
                        if run_nr > keep_first_n:
                            if data_dir.exists() and data_dir.is_dir():
                                shutil.rmtree(data_dir)
                        
                    except Exception as e:
                        print(f"Error processing {run_dir}: {e}")
                        continue
    
    print("\nCleanup completed!")


def extract_run_metrics(data_path: str | Path) -> pl.DataFrame:
    """
    Extract metrics from run directories and create a summary dataframe.
    
    Args:
        data_path (str | Path): Path to the data directory containing setup folders
        
    Returns:
        pl.DataFrame: DataFrame containing run metrics
    """
    data_path = Path(data_path)
    
    # Lists to store extracted data
    extracted_data = []
    
    # Process each setup directory
    for setup_dir in tqdm(data_path.iterdir(), desc="Processing setups"):
        if setup_dir.is_dir():
            setupname = setup_dir.name
            
            # Process each run directory in this setup
            for run_dir in setup_dir.iterdir():
                if run_dir.is_dir() and run_dir.name != "aggregated_runs":
                    # Extract run number from directory name
                    run_nr = int(run_dir.name.split("_")[0])
                    
                    # Read and parse report.txt
                    report_path = run_dir / "report.txt"
                    if not report_path.exists():
                        continue
                        
                    with open(report_path, 'r') as f:
                        report_content = f.read()
                    
                    # Extract max fitness using regex
                    fitness_match = re.search(r'Best fitness:\n([\d.]+)', report_content)
                    max_fitness = float(fitness_match.group(1)) if fitness_match else None
                    
                    # Extract num components using regex
                    components_match = re.search(r'Total components discovered:\n(\d+)', report_content)
                    num_components = int(components_match.group(1)) if components_match else None

                    runtime_match = re.search(r'Duration of run:\n(\d+):(\d+):(\d+\.\d+)', report_content)
                    if runtime_match:
                        hours, minutes, seconds_str = runtime_match.groups()
                        seconds_part, microseconds = seconds_str.split('.')
                        
                        # Calculate total seconds directly
                        runtime_seconds = (int(hours) * 3600 + 
                                        int(minutes) * 60 + 
                                        float(f"{seconds_part}.{microseconds}"))
                    else:
                        runtime_seconds = None

                    # Add data to list
                    extracted_data.append({
                        'setupname': setupname,
                        'run_nr': run_nr,
                        'max_fitness': max_fitness,
                        'num_components': num_components,
                        'runtime_seconds': runtime_seconds,
                        'Exceptions': False
                    })



    
    # Create DataFrame from extracted data
    if not extracted_data:
        return pl.DataFrame(schema={
            'setupname': pl.Utf8,
            'run_nr': pl.Int64,
            'max_fitness': pl.Float64,
            'num_components': pl.Int64,
            'runtime_seconds': pl.Float64,
            'Exceptions': pl.Boolean
        })
    
    return pl.DataFrame(extracted_data)

#-------------------------------------------------------------------------------
def plot_fitness_components(df: pl.DataFrame):
    """
    Generates a scatter plot of max_fitness vs. num_components for different setup sizes and selection methods.

    Args:
        df: Polars DataFrame with columns 'setupname', 'max_fitness', and 'num_components'.
    """

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))  # Adjust figure size as needed

    sizes = ["small", "medium", "big"]
    selection_methods = ["truncation", "roulette", "speciation"]
    colors = ["blue", "orangered", "green"]  # Define colors for selection methods

    for i, size in enumerate(sizes):
        size_df = df.filter(pl.col("setupname").str.contains(size))

        for j, method in enumerate(selection_methods):
            method_df = size_df.filter(pl.col("setupname").str.starts_with(method))
            axes[i].scatter(
                method_df["num_components"],
                method_df["max_fitness"],
                color=colors[j],
                label=method if i == 0 else None,  # Only label once per method
                alpha=0.7
            )
        axes[i].set_title(size.capitalize(), fontsize=SUBPLOTITLEFONT)

    # Set common labels outside the loop
    fig.text(0.5, 0.0005, "\nNumber of Components", ha='center', va='center', fontsize=AXLABELFONT) # x label
    fig.text(0.06, 0.5, "Max Fitness", ha='center', va='center', rotation='vertical', fontsize=AXLABELFONT) # y label
    fig.legend(bbox_to_anchor=(0.9, 0.5), loc='center left', fontsize=AXLABELFONT)
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=TICKFONT)
    # Adjust space between plots
    fig.subplots_adjust(wspace=0.4)  # Adjust wspace to add more space between plots
    return fig


def compare_mutation_boxplots(df1, df2, mutation: str, names: tuple, title: str):
    """
    Generate boxplots comparing a specific mutation between two dataframes.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    def get_boxplot_data(df, mutation):
        row = df.filter(pl.col("my_mutation") == mutation).to_pandas()
        if len(row) == 0:
            raise ValueError(f"Mutation {mutation} not found in dataframe")
        data = {
            'med': row['median'].iloc[0],
            'q1': row['25%'].iloc[0],
            'q3': row['75%'].iloc[0],
            'whislo': row['min'].iloc[0],
            'whishi': row['max'].iloc[0],
            'mean': row['mean'].iloc[0]
        }
        print(f"\n5-point summary for {mutation} ({names[0] if df is df1 else names[1]}):")
        print(f"Min: {data['whislo']:.4f}")
        print(f"Q1:  {data['q1']:.4f}")
        print(f"Med: {data['med']:.4f}")
        print(f"Q3:  {data['q3']:.4f}")
        print(f"Max: {data['whishi']:.4f}")
        return [data]

    bp1 = ax.bxp(get_boxplot_data(df1, mutation), positions=[1], 
                 patch_artist=True, meanline=True, showmeans=True, showfliers=False,
                 widths=0.5)
    bp2 = ax.bxp(get_boxplot_data(df2, mutation), positions=[2], 
                 patch_artist=True, meanline=True, showmeans=True, showfliers=False,
                 widths=0.5)
    
    for bp in [bp1, bp2]:
        plt.setp(bp['boxes'], color='tab:blue', linewidth=2)
        plt.setp(bp['means'], color='tab:blue', linewidth=2)
        plt.setp(bp['medians'], color='tab:blue', linewidth=2)
        plt.setp(bp['whiskers'], color='tab:blue', linewidth=2)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
    
    ax.set_ylabel('Fitness Impact Distribution', fontsize=AXLABELFONT)
    ax.set_title(title, fontsize=TITLEFONT, pad=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(names, fontsize=TICKFONT)
    ax.tick_params(axis='y', labelsize=TICKFONT)
    ax.axhline(y=0, color='lightgray', linestyle='--')
    
    plt.tight_layout()
    return fig, ax
################################################################################
#################### Fitness scatterplot distribution ##########################
################################################################################
import matplotlib.pyplot as plt
import numpy as np

def fitness_distribution_by_generation(df, title="Fitness Distribution by Generation", 
                                     color_species=False):
    """
    Create a scatter plot of fitness values across generations for a genetic algorithm.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing 'gen', 'fitness', and 'species_id' columns
    title : str, optional
        Plot title (default: "Fitness Distribution by Generation")
    color_species : bool, optional
        If True, colors points by species_id. If False, uses default grey (default: False)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color_species:
        # Get unique species for coloring
        unique_species = df['species_id'].unique().to_list()
        
        # Create color map
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_species)))
        
        # Plot each species with different color
        for species, color in zip(unique_species, colors):
            species_data = df.filter(pl.col('species_id') == species)
            ax.scatter(species_data['gen'], 
                      species_data['fitness'],
                      alpha=0.5,
                      color=color,
                      s=10,
                      label=f'Species {species}')
        
        # Add legend
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
        #          fontsize=TICKFONT)
        
    else:
        # Single color plot
        ax.scatter(df['gen'], df['fitness'],
                  alpha=0.5,
                  color='grey',
                  s=10)
    
    # Set labels and title with specified font sizes
    ax.set_xlabel('Generation', fontsize=AXLABELFONT)
    ax.set_ylabel('Fitness', fontsize=AXLABELFONT)
    ax.set_title(title, fontsize=TITLEFONT, pad=20)
    
    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=TICKFONT)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    if color_species:
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    else:
        plt.tight_layout()
    
    return fig

################################################################################
#################### Plotting component t distribution #########################
################################################################################
def plot_t_value_distributions(filepath_dict, generation, generation_end=None):
    """
    Plot T-value distributions from multiple pickle files containing component dictionaries.
    Creates a composite plot with density curves and boxplots sharing the same x-axis.
    
    Parameters
    ----------
    filepath_dict : dict
        Dictionary where keys are labels for the distributions (e.g., 'random', 'guided')
        and values are paths to gzipped pickle files containing component dictionaries
    generation : int
        Starting generation number to analyze. If generation_end is None, only this generation
        is analyzed
    generation_end : int, optional
        If provided, analyzes T-values from generation through generation_end inclusive
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object which can be further modified or saved
        
    Example
    -------
    >>> dfs = {
        'random': 'path/to/random/components.pkl.gz',
        'guided': 'path/to/guided/components.pkl.gz'
    }
    >>> fig = plot_t_value_distributions(dfs, 300)
    >>> fig.savefig('t_value_distribution.png', dpi=300, bbox_inches='tight')
    """
    
    # Load and process data
    processed_dfs = {}
    for label, filepath in filepath_dict.items():
        # Load pickle file
        with gzip.open(filepath, 'rb') as file:
            component_dict = pickle.load(file)
        
        # Convert to DataFrame
        df_data = []
        for outer_dict in component_dict.values():
            row = {int(k): v for k, v in outer_dict['t_val'].items()}
            df_data.append(row)
        processed_dfs[label] = pd.DataFrame(df_data)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                  gridspec_kw={'height_ratios': [3, 1]}, 
                                  sharex=True)
    
    colors = {}
    all_values = []
    
    # Top subplot: density plots
    for label, df in processed_dfs.items():
        if generation_end is None:
            values = df[generation].dropna()
        else:
            columns = range(generation, generation_end + 1)
            values = df[columns].values.flatten()
            values = values[~np.isnan(values)]
        
        all_values.extend(values)
        line = sns.kdeplot(data=values, label=label, ax=ax1)
        colors[label] = line.get_lines()[-1].get_color()
    
    ax1.grid(axis='x', color='lightgrey', linestyle='-', alpha=0.3)
    
    # Bottom subplot: boxplots
    positions = np.linspace(0, 1, len(processed_dfs) + 2)[1:-1]
    
    for (label, df), pos in zip(processed_dfs.items(), positions):
        if generation_end is None:
            values = df[generation].dropna()
        else:
            columns = range(generation, generation_end + 1)
            values = df[columns].values.flatten()
            values = values[~np.isnan(values)]
        
        bp = ax2.boxplot(values, 
                        positions=[pos], 
                        vert=False,
                        widths=[0.15],
                        sym='',
                        whis=[5, 95],
                        patch_artist=False,
                        meanline=True,
                        showmeans=True)
        
        plt.setp(bp['boxes'], color=colors[label])
        plt.setp(bp['whiskers'], color=colors[label])
        plt.setp(bp['caps'], color=colors[label])
        plt.setp(bp['medians'], color=colors[label])
        plt.setp(bp['means'], color=colors[label], linestyle='--')
    
    ax2.grid(axis='x', color='lightgrey', linestyle='-', alpha=0.3)
    
    # Set x-ticks and styling
    min_val, max_val = min(all_values), max(all_values)
    xticks = np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1.0)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    
    title = f'T-Value Distribution for Generation {generation}' if generation_end is None else \
            f'T-Value Distribution for Generations {generation}-{generation_end}'
    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel('', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.legend()
    
    ax2.set_xlabel('T-Value', fontsize=14)
    ax2.set_ylabel('Boxplot', fontsize=14)
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    # Print statistics
    for label, df in processed_dfs.items():
        if generation_end is None:
            values = df[generation].dropna()
        else:
            columns = range(generation, generation_end + 1)
            values = df[columns].values.flatten()
            values = values[~np.isnan(values)]
            
        print(f"\nStatistics for {label}:")
        print(f"Count of non-NA values: {len(values)}")
        print(f"Mean: {values.mean():.3f}")
        print(f"Std: {values.std():.3f}")
        print(f"Min: {values.min():.3f}")
        print(f"Max: {values.max():.3f}")
    
    return fig

def merge_two_results_df(df1_path, df2_path, savepath):
    """Just a dirty little helper to merge two results dfs
    """
    df1 = pl.read_ipc(df1_path)
    df1 = df1.with_columns([pl.col("num_components").cast(pl.Float64)])
    df2 = pl.read_ipc(df2_path)
    df2 = df2.with_columns([pl.col("num_components").cast(pl.Float64)])
    df3 = pl.concat([df1, df2])
    df3.write_ipc(savepath + "/final_report_df_merged.feather")
    print(f"saved merged df at {savepath}\nlength: {len(df3)}\n", df3)


def analyze_best_genome_comps(data: dict, compare_re=False):
    """This function takes in the result from the setup crawler (with best genomes loaded)
    It then returns:
        * A component comparison df
        * A dict mapping all the unique components in all the setups present in the res,
        mapped to unique ids

    Explaining the df:
    The df includes for each setup:
        * how many components are there in total among all genomes
        * what is the avg. number of components in the best g from every run
        * the delta between total unique compos and avg. The lower the number, the
        more similar the final genomes of that setup. A delta of 0 is perfect convergence
        * The set of all unique components in that setup. comparing those sets indicates
        if two setups found the same components
    """
    all_unique_comps = {}
    df_data = []
    next_component_id = 0
    target_components_simple = {((('pay compensation', 'p'), 1), (('reject request', 'p'), 1)),
        ((('check ticket', 'p'), 1), (('p', 'decide'), 1)),
        ((('decide', 'p'), 1), (('p', 'pay compensation'), 1), (('p', 'reinitiate request'), 1), (('p', 'reject request'), 1)),
        ((('p', 'examine casually'), 1), (('p', 'examine thoroughly'), 1), (('register request', 'p'), 1), (('reinitiate request', 'p'), 1)),
        ((('p', 'check ticket'), 1), (('register request', 'p'), 1), (('reinitiate request', 'p'), 1)),
        ((('p', 'register request'), 1),),
        ((('examine casually', 'p'), 1), (('examine thoroughly', 'p'), 1), (('p', 'decide'), 1))
        }

    for sname, sdata in data["setups"].items():
        setup_all_comps = []
        setup_fitnesses = []
        rediscovery = []
        setup_unique_comps = set()
        # iterate all components
        for g in sdata["best_genomes"]:
            cset = g.get_unique_component_set()
            setup_fitnesses.append(g.fitness)
            g_comp_ids = []
            if compare_re:
                if cset == target_components_simple: rediscovery.append(1)
                else: rediscovery.append(0)
            # if new component, add it to dict and save
            for comp in cset:
                if comp not in all_unique_comps:
                    all_unique_comps[comp] = next_component_id
                    curr_comp_id = next_component_id
                    setup_unique_comps.add(curr_comp_id)
                    next_component_id += 1
                # else fetch id of existing comp
                else:
                    curr_comp_id = all_unique_comps[comp]
                    setup_unique_comps.add(curr_comp_id)
                # add to comp ids
                g_comp_ids.append(curr_comp_id)
            # add component ids of that genome to all setup components
            setup_all_comps.append(g_comp_ids)

        mean_compcount = mean([len(c) for c in setup_all_comps])
        total_unique_comps = len(setup_unique_comps)
        comp_delta = total_unique_comps - mean_compcount
        df_data.append({
            "name": sname,
            "total num components": total_unique_comps,
            "mean compcount": mean_compcount,
            "comp delta": comp_delta,
            "setup unique comps": setup_unique_comps,
            "mean fitness": mean(setup_fitnesses),
            "mean discovery score": mean(rediscovery) if compare_re else 0,
            "eval score": mean(rediscovery) / total_unique_comps * 10
            })

    return pl.DataFrame(df_data), all_unique_comps


def get_jaccard_dist_between_comps(df, setup1_name, setup2_name):
    """Expects a df returned by analyze_best_genome_comps
    """
    c1 = df.filter(pl.col("name")==setup1_name).select("setup_unique_comps").item()
    c2 = df.filter(pl.col("name")==setup2_name).select("setup_unique_comps").item()
    print(f"Jaccard similarity {setup1_name} - {setup2_name}\n",
          len(c1.intersection(c2)) / len(c1.union(c2)))


################################################################################
#################### PROCESSING LOGS OF TWO RUNS AND SPIT OUT DIFF #############
################################################################################

import re

def compare_ga_logs(log_path1, log_path2):
    """
    Analyzes and compares evaluation times from two GA log files.
    Prints statistics and returns the speed difference.
    """
    # Read and process both logs
    with open(log_path1, 'r') as f:
        # Extract evaluation times using regex
        times1 = [float(x) for x in re.findall(r"'evaluate_curr_generation':\s*([\d.]+)", f.read())]
    
    with open(log_path2, 'r') as f:
        times2 = [float(x) for x in re.findall(r"'evaluate_curr_generation':\s*([\d.]+)", f.read())]
    
    # Calculate statistics
    avg1 = sum(times1) / len(times1)
    total1 = sum(times1)
    
    avg2 = sum(times2) / len(times2)
    total2 = sum(times2)
    
    # Calculate speed difference
    speed_difference = total2 / total1
    
    # Print results
    print(f"\nLog 1:")
    print(f"Average evaluation time: {avg1:.4f} seconds")
    print(f"Total evaluation time: {total1:.4f} seconds")
    print(f"Number of generations: {len(times1)}")
    
    print(f"\nLog 2:")
    print(f"Average evaluation time: {avg2:.4f} seconds")
    print(f"Total evaluation time: {total2:.4f} seconds")
    print(f"Number of generations: {len(times2)}")
    
    print(f"\nLog 1 is {speed_difference:.2f}x faster than Log 2")
    
    return speed_difference

################################################################################
#################### PLOTTING GENOMES IN A MORE READABLE WAY ###################
################################################################################

running_example_remap = {
    'check ticket'      : "check t.",
    'decide'            : "decide",
    'examine casually'  : "casual ex.",
    'examine thoroughly': "thorough ex.",
    'pay compensation'  : "pay comp.",
    'register request'  : "register r.",
    'reinitiate request': "reinitiate",
    'reject request'    : "reject"
}

def plot_digraph(g, fontsize=36, node_labels=running_example_remap):
    """Plot digraph with custom font size and optional node relabeling"""
    gviz = g.get_gviz()
    gviz.graph_attr['fontsize'] = str(fontsize)
    gviz.node_attr['fontsize'] = str(fontsize)
    gviz.edge_attr['fontsize'] = str(fontsize)
    
    new_body = []
    for e in gviz.body:
        new_e = e
        if node_labels:
            for old_name, new_name in node_labels.items():
                if old_name in e:
                    new_e = e.replace(old_name, new_name).replace("fontsize=12", f"fontsize={fontsize}")
                    break
            new_body.append(new_e)
        else:
            new_e = e.replace("fontsize=12", f"fontsize={fontsize}")
            new_body.append(new_e)

    gviz.body = new_body

                
    return gviz

################################################################################
#################### GET THE F SCORE FOR PDC LOGS ##############################
################################################################################
from neatutils import log
from importlib import reload

from scripts.analysis_scripts.useful_functions import load_genome
from pprint import pprint
reload(log)


def get_pdc_f_score(genome, pdc_model_code: str, pdc_year: str):
    total_n_errors = 0
    USE_PM4PY_REPLAY = True

    def get_replay_scores(log_to_replay):
        nonlocal total_n_errors
        if USE_PM4PY_REPLAY:
            # horribly ineffecient but idgaf - different order for pm4py thus need to make new map. idk, im tired
            map_trace_to_case = {}
            for i, v in enumerate(log_to_replay["variants"]):
                map_trace_to_case[v] = i
            #
            replay_res = {}
            traces = get_pm4py_quality(genome, log_to_replay)["traces"]
            for t in traces:
                transitions = tuple(t["activated_transitions"])
                try:
                    i = map_trace_to_case[transitions]
                except:
                    total_n_errors += 1
                    continue
                case_id = log_to_replay["case_variant_map"][i]
                replay_res[case_id] = t["trace_fitness"]
            return replay_res
        else:
            model_eval = genome.build_fc_petri(log_to_replay).evaluate()
            replay_res = {}
            for v_id, r in enumerate(model_eval["replay"]):
                case_id = log_to_replay["case_variant_map"][v_id]
                replay_res[case_id] = r["fitness"]
            return replay_res

    # get the replay for the test log
    test_log = log.get_log_from_xes(f"I:/EvolvePetriNets/pm_data/pdc_logs/{pdc_year}/Test Logs/pdc{pdc_year}_{pdc_model_code}.xes", is_pdc_log=True)
    test_rep = get_replay_scores(test_log)
    # get the replay for the base log
    base_log = log.get_log_from_xes(f"I:/EvolvePetriNets/pm_data/pdc_logs/{pdc_year}/Base Logs/pdc{pdc_year}_{pdc_model_code}.xes", is_pdc_log=True)
    base_rep = get_replay_scores(base_log)
    # get the dict for the Ground Truth log
    gt_log_df = log.get_log_from_xes(f"I:/EvolvePetriNets/pm_data/pdc_logs/{pdc_year}/Ground Truth Logs/pdc{pdc_year}_{pdc_model_code}.xes", is_pdc_log=True)["dataframe"]
    gt_log_df = gt_log_df[["case:concept:name", "case:pdc:isPos"]].drop_duplicates()
    gt_map = dict(zip(gt_log_df["case:concept:name"], gt_log_df["case:pdc:isPos"]))
    # compare the two
    total_missing = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for case_id, correct_is_pos in gt_map.items():
        test_class = test_rep.get(case_id)
        base_class = base_rep.get(case_id)
        # for some reason some cases are not in both logs... grrrr
        if not (test_class and base_class):
            # print("missing:", case_id, test_class, base_class)
            total_missing += 1
            continue
        my_is_pos = test_class > base_class
        if my_is_pos and correct_is_pos:
            tp += 1
        elif not my_is_pos and not correct_is_pos:
            tn += 1
        elif my_is_pos and not correct_is_pos:
            fp += 1
        elif not my_is_pos and correct_is_pos:
            fn += 1
    # print("missing perc", total_missing / len(gt_map))
    
    # Calculate precision, recall, and F-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("total errors", total_n_errors, pdc_model_code, pdc_year)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "f-precision": precision,
        "f-recall": recall,
        "f-score": f_score
    }


################################################################################
#################### LOAD A PNML FILE INTO A GENOME ############################
################################################################################
from neat import initial_population
from pm4py.objects.petri_net.importer.variants.pnml import import_net as import_pnml

def load_genome_from_pnml(pn_fp: str):
    net, im, fm = import_pnml(pn_fp)
    tl = [t.label for t in net.transitions]
    return initial_population.construct_genome_from_mined_net(net, im, fm, tl, infer_start_end=True)


################################################################################
#################### PLOT FITNESS METRICS ############################
################################################################################
def plot_metrics(df, title="Fitness Metrics", figsize=(5,5)):
    metrics_to_plot = [
        "metric_aggregated_replay_fitness",
        "metric_over_enabled_trans",
        "metric_num_arcs",
        "metric_remaining_score",
        "metric_sink_score"
    ]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in metrics_to_plot:
        label = col.replace('metric_', '').replace('_', ' ')
        ax.plot(df.to_pandas().index, df[col].to_numpy(), label=label, linewidth=2)
    
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.tick_params(axis='both', labelsize=TICKFONT)
    
    ax.set_xlabel('gen', fontsize=AXLABELFONT)
    ax.set_ylabel('metric value', fontsize=AXLABELFONT)
    ax.set_title(title, fontsize=TITLEFONT, pad=20, loc="left")
    
    ax.legend(bbox_to_anchor=(1.05, 1), 
             loc='upper left', 
             fontsize=LEGENDFONT,
             borderaxespad=0.)
    
    plt.tight_layout()
    
    return fig

################################################################################
#################### PLOT FITNESS IMPACTS ############################
################################################################################
def calculate_metric_deltas(df):
    filtered_df = df.filter(pl.col("gen") != 1)
    metric_cols = [col for col in df.columns if col.startswith("metric_")]
    
    result = []
    for row in filtered_df.iter_rows(named=True):
        parent = df.filter((pl.col("gen") == row["gen"] - 1) & (pl.col("id") == row["parent_id"]))
        parent_values = {col: parent.select(col).item() for col in metric_cols}
        deltas = {f"{col}_delta": row[col] - parent_values[col] for col in metric_cols}
        result.append({**row, **deltas})
    
    return pl.DataFrame(result)


def plot_fitness_impacts(df, cols, gens, title="", figsize=(4,4)):
    filtered_df = df.filter((df["gen"] >= gens[0]) & (df["gen"] <= gens[1]))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(filtered_df[cols[0]], filtered_df[cols[1]], alpha=0.1)
    
    x_label = cols[0].replace("metric_", "").replace("_", " ")
    y_label = cols[1].replace("metric_", "").replace("_", " ")
    
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.8)
    ax.axvline(x=0, color='grey', linestyle='--', alpha=0.8)

    
    ax.set_xlabel(x_label, fontsize=AXLABELFONT)
    ax.set_ylabel(y_label, fontsize=AXLABELFONT)
    ax.set_title(title, fontsize=TITLEFONT, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=TICKFONT)
    
    plt.tight_layout()
    return fig


def compare_impacts(df, cols, gens, title="", figsize=(4, 4)):
    gen_start, gen_end = gens
    filtered_df = df.filter(
        (df['gen'] >= gen_start) & 
        (df['gen'] <= gen_end)
    )
    
    positive_mask = filtered_df[cols[0]] > 0.05
    filtered_df = filtered_df.filter(positive_mask)
    
    data = [
        filtered_df[cols[0]].to_numpy(),
        filtered_df[cols[1]].to_numpy()
    ]
    
    labels = []
    for col in cols:
        label = col.replace('metric_', '').replace('_', '\n').rstrip("_delta")
                    #   f"{'\n' if len(col.split("_")) < 4 else ""}({round(filtered_df[col].mean(), 3)})")
        label = label + "\n" if label.count("\n") == 2 else label
        labels.append(label + f"\n({round(filtered_df[col].mean(), 3)})")

    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.8)
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, meanline=True, showmeans=True, showfliers=False, widths=0.4)
    
    for box in bp['boxes']:
        box.set(facecolor='lightblue')
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=TICKFONT)
    
    ax.tick_params(axis='both', labelsize=TICKFONT)
    # ax.set_ylabel('Value', fontsize=AXLABELFONT)
    
    if title:
        ax.set_title(title, fontsize=TITLEFONT, pad=20)
    
    plt.tight_layout()
    
    return fig


################################################################################
#################### COMPONENT STACKPLOT ############################
################################################################################
from collections import Counter


def get_component_frequencies(pop_df):
    """
    Process raw component frequency data from population DataFrame.
    
    Args:
        pop_df (polars.DataFrame): Population dataframe with 'gen' and 'my_components' columns
    
    Returns:
        tuple: (generations, component_names, frequency_matrix)
        - generations: list of generation numbers
        - component_names: list of component names
        - frequency_matrix: numpy array of shape (n_generations, n_components)
    """
    # Convert to pandas for easier manipulation
    df = pop_df.to_pandas()
    
    # Get unique generations for x-axis
    generations = sorted(df['gen'].unique())
    
    # Process component frequencies per generation
    gen_component_counts = {}
    all_components = set()
    
    # Count components for each generation
    for gen in generations:
        gen_data = df[df['gen'] == gen]
        # Flatten list of lists and count components
        all_gen_components = [comp for row in gen_data['my_components'] for comp in row]
        counts = Counter(all_gen_components)
        gen_component_counts[gen] = counts
        all_components.update(counts.keys())
    
    # Convert to frequency matrix
    all_components = sorted(all_components)
    frequency_matrix = []
    
    for gen in generations:
        counts = gen_component_counts[gen]
        row = [counts.get(comp, 0) for comp in all_components]
        frequency_matrix.append(row)
    
    return generations, all_components, np.array(frequency_matrix)


def plot_component_stackplot(generations, component_names, frequency_matrix, 
                           stop_gen=None, min_frequency=None,
                           figsize=(12, 6), title="Component Frequencies Across Generations"):
    """
    Create a stackplot visualization from component frequency data.
    
    Args:
        generations (list): List of generation numbers for x-axis
        component_names (list): List of component names
        frequency_matrix (np.ndarray): Matrix of component frequencies (shape: n_generations x n_components)
        stop_gen (int, optional): Last generation to include
        min_frequency (int, optional): Minimum frequency threshold for including components
        figsize (tuple, optional): Figure dimensions (width, height)
        title (str, optional): Plot title
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Filter by generation if specified
    if stop_gen is not None:
        gen_mask = np.array(generations) <= stop_gen
        generations = np.array(generations)[gen_mask]
        frequency_matrix = frequency_matrix[gen_mask]
    
    # Filter components by minimum frequency if specified
    if min_frequency is not None:
        max_frequencies = frequency_matrix.max(axis=0)
        keep_components = max_frequencies >= min_frequency
        frequency_matrix = frequency_matrix[:, keep_components]
        component_names = [comp for comp, keep in zip(component_names, keep_components) if keep]
    
    plt.figure(figsize=figsize)
    plt.stackplot(generations, frequency_matrix.T)
    
    plt.xlabel('Generation')
    plt.ylabel('Component Count')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def plot_eval_times(df, title="Generation Evaluation Times", figsize=(10, 6)):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = df['gen'].to_numpy()
    y = df['time_evaluate_curr_generation'].to_numpy()
    
    # Plot original data with low alpha
    ax.plot(x, y, '-', color='#2E86C1', linewidth=1, alpha=0.3)
    
    # Calculate and plot smoothed line
    window = 10  # Adjust window size as needed
    y_smooth = pd.Series(y).rolling(window=window, center=True).mean()
    ax.plot(x, y_smooth, '-', color='#2E86C1', linewidth=2)
    
    ax.set_xlabel('Generation', fontsize=AXLABELFONT)
    ax.set_ylabel('Population Eval Time (s)', fontsize=AXLABELFONT)
    ax.set_title(title, fontsize=TITLEFONT)
    
    ax.tick_params(axis='both', which='major', labelsize=TICKFONT)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig

################################################################################
#################### GET MINED MODELS ############################
################################################################################
# literally copied from the initial pop module and modified slightly
from pm4py.objects.petri_net.obj import PetriNet as pn
from pm4py.discovery import (
    discover_petri_net_alpha as alpha,
    discover_petri_net_alpha_plus as alpha_plus,
    discover_petri_net_inductive as inductive,
    discover_petri_net_heuristics as heuristics,
    discover_petri_net_ilp as ilp
)


def get_pm4py_mined_genomes(log):
    """Returned genomes are already evaluated
    """
    bootstrap_setup = [
        alpha,
        inductive,
        heuristics,
        ilp
    ]
    mined_nets = {}
    for miner in bootstrap_setup:
        name = miner.__name__.removeprefix("discover_petri_net_")
        net, im, fm = miner(log["dataframe"])
        g = initial_population.construct_genome_from_mined_net(net, im, fm, log["task_list"], None)
        # evaluate fitness right here
        g.evaluate_fitness(log)
        mined_nets[name] = g
    return mined_nets

################################################################################
#################### PM4PY FITNESS METRICS #####################################
################################################################################
from pm4py.algo.conformance.tokenreplay.variants.token_replay import apply as get_replayed_traces
from pm4py.algo.evaluation.replay_fitness.variants.token_replay import evaluate as get_fitness_dict
from pm4py.algo.evaluation.precision.variants.etconformance_token import apply as get_precision
from pm4py.algo.evaluation.generalization.variants.token_based import get_generalization
from pm4py.algo.evaluation.simplicity.variants.arc_degree import apply as get_simplicity

def get_pm4py_quality(g, eventlog):
    eventlog = eventlog["dataframe"]
    net, im, fm = g.build_petri()
    # fitness eval
    # default_params = {"show_progress_bar": False}
    default_params = {"show_progress_bar": False, "consider_remaining_in_fitness": False, "case_id_key": "concept:name", "return_names": True}
    aligned_traces = get_replayed_traces(eventlog, net, im, fm, default_params)
    trace_fitness = get_fitness_dict(aligned_traces)
    return {
        "perc_fit_traces" : trace_fitness["perc_fit_traces"] / 100,
        "average_trace_fitness" : trace_fitness["average_trace_fitness"],
        "log_fitness" : trace_fitness["log_fitness"],
        "precision" : get_precision(eventlog, net, im, fm, default_params),
        "generalization" : get_generalization(net, aligned_traces),
        "simplicity" : get_simplicity(net),
        "traces": aligned_traces
    }


def get_log_stats(log):
    """prints some stats about a log
    """
    df = log["dataframe"]
    return pd.DataFrame([{
        "tasks": len(df["concept:name"].unique()),
        "events": len(df),
        "variants": len(log["variants"])
    }], index=None).transpose().rename(columns={0: 'count'})


################################################################################
#################### PM4PY FITNESS METRICS #####################################
################################################################################

def get_barplot_final_results(results, name, showlegend=False):

    def get_capped_error(value, variance):
        error = np.sqrt(variance)
        # Cap the upper error bar at 1.0
        upper_error = min(1.0 - value, error)
        lower_error = min(value, error)
        return np.array([[lower_error], [upper_error]])

    # Existing code to prepare data
    thedict = results[name]
    renamedict = {"name": "name", "perc fit traces": "replayable traces", "pm4py avg trace fit": "avg. trace fitness", "f-score": "f-score"}
    comp_df = thedict["comparison df"][list(renamedict.keys())].rename(columns=renamedict)
    plt_title = name.replace("_", " ").title()
    
    # Get variance values for error bars
    replayable_trace_variance = thedict["gwfm pm4py percfit distribution"]["var"]
    avg_fitness_variance = thedict["gwfm pm4py fit distribution"]["var"]
    fscore_variance = thedict["gwfm fscore distribution"]["var"]
    
    # Get bootstrap values and variances
    bsdict = results[name+"_bootstrap"]
    bs_replayable_trace_variance = bsdict["gwfm pm4py percfit distribution"]["var"]
    bs_replayble_traces_mean = thedict["gwfm pm4py percfit distribution"]["mean"]
    bs_avg_fitness_variance = bsdict["gwfm pm4py fit distribution"]["var"]
    bs_avg_fitness_mean = thedict["gwfm pm4py fit distribution"]["mean"]
    bs_fscore_variance = bsdict["gwfm fscore distribution"]["var"]
    bs_fscore_mean = thedict["gwfm fscore distribution"]["mean"]
    
    # Reorder dataframe to put GWFM-B after GWFM
    gwfm_row = comp_df[comp_df['name'] == 'GWFM']
    other_rows = comp_df[comp_df['name'] != 'GWFM']
    
    # Create bootstrap row
    bootstrap_row = pd.DataFrame({
        'name': ['GWFM-B'],
        'replayable traces': [bs_replayble_traces_mean],
        'avg. trace fitness': [bs_avg_fitness_mean],
        'f-score': [bs_fscore_mean]
    })
    
    # Concatenate in the desired order
    comp_df = pd.concat([gwfm_row, bootstrap_row, other_rows], ignore_index=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Add light gray horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', linewidth=2)
    ax.set_axisbelow(True)  # This puts the grid lines behind the bars
    
    # Set width of bars and positions of the bars
    barWidth = 0.25
    r1 = np.arange(len(comp_df))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    bars1 = ax.bar(r1, comp_df['replayable traces'], width=barWidth, label='Replayable Traces')
    bars2 = ax.bar(r2, comp_df['avg. trace fitness'], width=barWidth, label='Avg. Trace Fitness')
    bars3 = ax.bar(r3, comp_df['f-score'], width=barWidth, label='F-score')
    
    # Add error bars for GWFM and GWFM-B
    for idx, method in enumerate(comp_df['name']):
        if method == 'GWFM':
            ax.errorbar(r1[idx], comp_df['replayable traces'].iloc[idx], 
                       yerr=get_capped_error(comp_df['replayable traces'].iloc[idx], replayable_trace_variance),
                       fmt='k', capsize=5)
            ax.errorbar(r2[idx], comp_df['avg. trace fitness'].iloc[idx], 
                       yerr=get_capped_error(comp_df['avg. trace fitness'].iloc[idx], avg_fitness_variance),
                       fmt='k', capsize=5)
            ax.errorbar(r3[idx], comp_df['f-score'].iloc[idx], 
                       yerr=get_capped_error(comp_df['f-score'].iloc[idx], fscore_variance),
                       fmt='k', capsize=5)
        elif method == 'GWFM-B':
            ax.errorbar(r1[idx], comp_df['replayable traces'].iloc[idx], 
                       yerr=get_capped_error(comp_df['replayable traces'].iloc[idx], bs_replayable_trace_variance),
                       fmt='k', capsize=5)
            ax.errorbar(r2[idx], comp_df['avg. trace fitness'].iloc[idx], 
                       yerr=get_capped_error(comp_df['avg. trace fitness'].iloc[idx], bs_avg_fitness_variance),
                       fmt='k', capsize=5)
            ax.errorbar(r3[idx], comp_df['f-score'].iloc[idx], 
                       yerr=get_capped_error(comp_df['f-score'].iloc[idx], bs_fscore_variance),
                       fmt='k', capsize=5)

    
    # Add labels and title
    ax.set_ylabel('Score')
    ax.set_title(plt_title)
    ax.set_xticks([r + barWidth for r in range(len(comp_df))])
    ax.set_xticklabels(comp_df['name'], rotation=35)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Add legend below the plot
    if showlegend:
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

def get_legend_only():
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    fig = plt.figure(figsize=(8, 0.3))
    ax = fig.add_subplot(111)
    plots = []
    plots.append(plt.Rectangle((0,0), 1, 1, fc='#1f77b4'))
    plots.append(plt.Rectangle((0,0), 1, 1, fc='#ff7f0e'))
    plots.append(plt.Rectangle((0,0), 1, 1, fc='#2ca02c'))
    ax.legend(plots, ['Replayable Traces', 'Avg. Trace Fitness', 'F-score'], 
             loc='center', ncol=3, frameon=False)
    ax.axis('off')
    return fig