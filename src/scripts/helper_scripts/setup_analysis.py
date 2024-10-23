import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional

################################################################################
#################### PROCESSING AND COMBINING DATAFRAMES #######################
################################################################################

# -------------------- GEN INFO DF

def combine_and_aggregate_geninfo_dataframes(dataframes, use_species=False):
    """
    Combines and aggregates the avg of two gen_info dataframes, with an optional
    arg if species columns should be included in the aggregation
    """
    # Columns to aggregate
    agg_columns = [
        "num_total_components",
        "best_genome_fitness",
        "avg_pop_fitness",
        "time_evaluate_curr_generation",
        "time_pop_update"
    ] 
    if use_species:
        agg_columns += ["num_total_species", "best_species_avg_fitness"]
    
    # Concatenate all dataframes vertically
    combined_df = pl.concat(dataframes)
    
    # Group by 'gen' and calculate mean for specified columns
    result_df = combined_df.group_by("gen").agg([
        pl.col(col).mean().alias(f"{col}") for col in agg_columns
    ]).sort("gen")
    
    return result_df

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

################################################################################
#################### CRAWLING THE RESULTS ######################################
################################################################################

def exec_results_crawler(root_path: str) -> dict:
    """
    Process execution data from a directory structure containing genetic algorithm runs.
    
    Args:
        root_path (str): Path to the root directory containing execution_data
        
    Returns:
        dict: Dictionary containing processed execution data with structure:
            {
                'final_report': polars.DataFrame,
                'setups': {
                    1: {'params': dict, ...},
                    2: {'params': dict, ...},
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
    for setup_dir in execution_data_path.iterdir():
        # Check if directory name matches setup pattern
        if setup_dir.is_dir() and setup_dir.name.startswith("setup_"):
            try:
                # Extract setup number
                setup_num = int(setup_dir.name.split("_")[1])
                
                # Initialize setup in results
                execution_results['setups'][setup_num] = {}
                
                # Load setup parameters
                params_path = setup_dir / f"{setup_dir.name}_params.json"
                with open(params_path) as f:
                    params = json.load(f)
                execution_results['setups'][setup_num]['params'] = params
                
                # Check for speciation strategy
                is_speciation = params.get('selection_strategy') == 'speciation'

                execution_results['setups'][setup_num]['is_speciation'] = is_speciation

                agg_gen_info = []
                
                # Process each run directory
                for run_dir in setup_dir.iterdir():
                    if run_dir.is_dir():
                        gen_info_path = run_dir / "data" / "gen_info.feather"
                        if gen_info_path.exists():
                            agg_gen_info.append(pl.read_ipc(gen_info_path))

                execution_results['setups'][setup_num]['gen_info_agg'] = combine_and_aggregate_geninfo_dataframes(agg_gen_info)
                
            except (ValueError, IndexError) as e:
                print(f"Error processing directory {setup_dir}: {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"Error reading params file for {setup_dir}: {e}")
                continue
    
    return execution_results


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
    contains_params = lambda d1, d2: all(k in d2 and d2[k] == v for k, v in d1.items())
    # aggregation results
    agg_dict = {}
    # key: params to search, value: group to put it in
    for setup_name, setup in res_dict["setups"].items():
        for search_name, search_params in search_dict.items():
            if contains_params(search_params, setup["params"]):
                agg_dict.setdefault(search_name, []).append(setup["gen_info_agg"])
    
    for search_name, dfs in agg_dict.items():
        agg_df = combine_and_aggregate_geninfo_dataframes(dfs)
        agg_dict[search_name] = agg_df
                
    return agg_dict

################################################################################
#################### PLOTTING FUNCTIONS ########################################
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
    figsize: tuple[int, int] = (12, 8)
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
        title = f"{y_ax} by {x_ax}"
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
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide empty subplots if any
    for idx in range(len(plt_layout), len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

