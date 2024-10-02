# One-off analysis of results data

## scrape & plot the num components from results folder
path:
    src/tool_scripts/analysis_scripts/summarize_component_data.py
purpose:
    this takes the standard directory structure of a results folder, and just extracts the Total components discovered and aggregates it


## Testing endreports before they are productionized
path:
    src/tool_scripts/analysis_scripts/look_at_dfs.py
purpose:
    Purpose of this script is to have a space where I can work with run data, and test implementing new plots before they are integrated into the endreports module Nothing that is in here should not be implemented in endreports at some point, except for the tests at the bottom of the file