"""this takes the standard directory structure of a results folder, and just
extracts the Total components discovered and aggregates it
"""
# %%
"""Scraping the data
"""
import os
import re

def extract_folder_info(root_path):
    n_components_dict = {}
    
    for top_level_folder in os.listdir(root_path):
        top_level_path = os.path.join(root_path, top_level_folder)
        
        if os.path.isdir(top_level_path):
            n_components_dict[top_level_folder] = []
            
            for subfolder in os.listdir(top_level_path):
                subfolder_path = os.path.join(top_level_path, subfolder)
                
                if os.path.isdir(subfolder_path):
                    report_path = os.path.join(subfolder_path, 'report.txt')
                    
                    if os.path.exists(report_path):
                        with open(report_path, 'r') as f:
                            content = f.read()
                            match = re.search(r'Total components discovered:\s*(\d+)', content)
                            if match:
                                n_components_dict[top_level_folder].append(int(match.group(1)))
    
    return n_components_dict

# Specify the root path where the directory structure is contained
# Extract the information and save it to n_components_dict
root_path = r"C:\path\to\your\directory\structure"
n_components_dict = extract_folder_info("E:/migrate_o/github_repos/EvolvePetriNets/results/results_used_in_analysis/compare_search_width")

# %%
"""Visualizing it
"""
from matplotlib import pyplot as plt
import numpy as np

def create_stacked_scatter_plot(n_components_dict):
    methods = list(n_components_dict.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        y = n_components_dict[method]
        x = [i] * len(y)  # Create x-coordinates for stacking
        # Calculate and plot the mean
        mean_y = np.mean(y)
        ax.scatter(i, mean_y, color='black', marker='D', s=100, label=f'{method} mean' if i == 0 else "")
    
        ax.scatter(x, y, alpha=0.6, s=50)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Number of Components Discovered')
    ax.set_title('Components Discovered per Method')
    
    plt.tight_layout()
    plt.show()

create_stacked_scatter_plot(n_components_dict)