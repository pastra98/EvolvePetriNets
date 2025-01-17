# Scripts glossary
Here I plan to keep track about all the ^^temporary scripts and notebooks I made. Currently those are all in the `src/scripts/` folder.

## One-off analysis (of results data)

### scrape & plot the num components from results folder
**path:**
src/scripts/analysis_scripts/summarize_component_data.py

**purpose:**
this takes the standard directory structure of a results folder, and just extracts the Total components discovered and aggregates it

### Testing endreports before they are productionized & other visualizations
**path:**
src/scripts/analysis_scripts/look_at_dfs.py

**purpose:**
Purpose of this script is to have a space where I can work with run data, and test implementing new plots before they are integrated into the endreports module Nothing that is in here should not be implemented in endreports at some point, except for the tests at the bottom of the file There may be some visualizations such as fitness scatterplot, spawn-rank histogram that will be included in some analysis, but not part of endreports.
Also contains comparison of saving pandas vs polars population dataframe

### Plotting average compatibility scores per generation
**path:**
src/scripts/script_dict.md

**purpose:**
The entire script can probably be scrapped, but the idea is still interesting for trying to explain num species growth. Plots the average compatibility score of all genomes in a generation. Results can be seen in my onenote, "Reducing the number of species"

### Testing different metrics to identify "good components"
**path:**
src/scripts/analysis_scripts/component_fitness_analysis.py

**purpose:**
This analyzes which metric is suitable for identifying good components. The idea was to use this info for guided mutations. Lots of time spent here trying to understand and optimize t-value computation to guide mutations in the right direction. Was ultimately abandoned, because if the fitness function does not point in the right direction, there is no point in guiding the mutations.

### Test startconfigs
**path:**
src/scripts/analysis_scripts/test_startconfigs.py

**purpose:**
Tests log splicing techniques (i.e. feeding the miners only a subset of the variants in the log), incl. funcs to quickly iterate pm4py implemented miners on a given log & convert them into genomes. Also hosts the Measuring genomic drift plots (onenote: 1st.  progress report, fig. 4), as well as code for printing fitness metrics of various mined models. 

**todo:**
move measuring genomic drift visualization into notebook

### Implement token-based-replay
**path:**
src/scripts/analysis_scripts/implement_tbr.py

**purpose:**
Used this when implementing token-replay from scratch, to compare with pm4py implementation. Could also be used later for comparing numpy-based implementation to current OOP implementation.

### PM4PY simplicity standalone implementation
**path:**
src/scripts/analysis_scripts/simplicity.py

**purpose:**
Copied pm4py simplicity code out and make it work standalone. Turns out pm4py simplicity has nothing to do with ProDiGen implementation. 

**todo:**
check if the pm4py simplicity is implemented as a metric in fitnesscalc

### Useful helper functions
**path:**
src/scripts/analysis_scripts/useful_functions.py

**purpose:**
Small helper functions that can be re-used in other analysis scripts for stuff like visualizing genomes, printing traces & variants, printing metrics, resetting the ga.

### Analyzing process discovery contest (PDC) logs
**path:**
src/scripts/analysis_scripts/pdc_log_analysis.py

**purpose:**
Select pdc logs based on filters and print some info about them and their models

## Helper scripts

### Setup Analysis
**path:**
src/scripts/helper_scripts/setup_analysis.py

**purpose:**
Module to be imorted by the analysis notebook. Reads, aggregates and plots data from a config run

### Config maker
**path:**
src/scripts/helper_scripts/config_maker.py

**purpose:**
GUI to select parameters and generate a config with possible permutations of a number of parameters

## Interacting with pm4py

### Conformance checking, print traces, vis petri nets (from file)
**path:**
src/scripts/pm4py_stuff/conformance_nb.py

**purpose:**
Most of this file will probably need to be scrapped, it was last touched in 2021 while doing early testing for my bachelor thesis. Maybe there is still some useful boilerplate for using pm4py, which is why I haven't deleted it yet.

**todo:**
move the useful functions in there out into testing notebook/wherever they are useful

### Check how many places in pm4py.analysis.maximal_decomposition
**path:**
src/scripts/pm4py_stuff/component_size_analysis.py

**purpose:**
This analyzes the size and order of components, need to know if components contain more than one unique place.  The dataframe needs to be generated with minimal_serialization = False

### Problems with pm4py token-replay
**path:**
src/scripts/pm4py_scripts/tbr_outputs.ipynb

**purpose:**
The purpose of this notebook is to identify problems I have with the token-replay implementation of pm4py - focusing specificly on the outputs of the `tokenreplay.variants.token_replay.apply_trace` method.  Pickles of the genomes are also saved in this folder.

### pkl analysis tool
**path:**
src/scripts/analysis_scripts/pkl_analysis.ipynb

**purpose:**
Serialized genomes are big for some reason. Claude helped me write an analysis tool