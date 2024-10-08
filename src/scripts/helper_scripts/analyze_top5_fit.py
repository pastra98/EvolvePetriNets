"""
This piece of crap one-off helper script was used in the Bachelor thesis to help me
quickly rate how I percieved the quality of process models. I don't think I've ever used
those ratings and strongly doubt I'll have any use for them in the Master thesis.
Even If I were to rate the all the models in the Master thesis, I should write a better
tool for this. Keeping it for sentimentality and ideas (yeah right) for now.
"""
# %%
import sys # syspath fuckery to run from same location as main
import os # set cwd to same as project cwd from vscode
from pathlib import Path

from pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star import PARAM_SYNC_COST_FUNCTION

cwd = Path.cwd()

# RUN ONLY ONCE
if not os.getcwd().endswith("EvolvePetriNets"): # rename dir on laptop to repo name as well
    sys.path.append(str(cwd.parent.parent / "src")) # src from where all the relative imports work
    os.chdir(cwd.parent.parent) # workspace level from where I execute scripts

# notebook specific - autoreload modules
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

# import various other shit
import pickle as pkl
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from src import neat
from copy import copy
import pprint as pp
import json

import PIL
from IPython.display import Image
from matplotlib.pyplot import imshow
import numpy as np


# %%
ipython.magic("matplotlib inline")


def rate_results(dir_to_analyze: str = None):
    def save_and_quit(rating_d):
        with open(dir_to_analyze + "/rating.txt", "w") as f:
            json.dump(rating_d, f, indent=4)

    with open(dir_to_analyze + "/rating.txt", "r") as f:
        ratings = json.load(f)

    try: last_visit = list(ratings.keys())[-1]; reached_lv = False
    except: last_visit = None; reached_lv = True

    i = 0
    print(last_visit)
    for r_nr in [1,2]:
        for root, dirs, files in os.walk(dir_to_analyze):
            for file in files:
                root = root.replace("\\", "/")
                imgpath = f"{root}/{file}"
                if last_visit:
                    reached_lv = last_visit == imgpath
                if (root.endswith("reports") and
                    root.split("/")[-2].startswith(f"{r_nr}_") and
                    file.endswith("best_genome.png") and
                    reached_lv):
                    print(imgpath)
                    i += 1
                    last_visit = None
                    im = Image(imgpath)
                    display(im)
                    while (usr_in := input("Quality of model: ")) not in ["1","2","3","4","exit"]:
                        print("rating must be 1, 2, 3 or 4")
                    if usr_in in ["1","2","3","4"]:
                        ratings[imgpath] = int(usr_in)
                        print(usr_in)
                        print(f"evaluated: {i}")
                    elif usr_in == "exit":
                        save_and_quit(ratings)
                        return
    print("Reached the end, saving results and exiting")
    save_and_quit(ratings)


rate_results("results/data/speciation_harder_01-20-2022_21-07-32")

# %%
ratings_df = pd.read_json("results/data/speciation_harder_01-20-2022_21-07-32/rating.txt", typ='series').to_frame()
ratings_df["full_path"] = ratings_df.index
ratings_df["rating"] = ratings_df[0]
ratings_df.reset_index(inplace=True)
ratings_df.drop(["index", 0], axis=1, inplace=True)
ratings_df[["params", "run"]] = ratings_df["full_path"].str.split("/", expand=True)[[3, 4]]
ratings_df["run"] = ratings_df["run"].str[0]
ratings_df.sort_values(by="rating", ascending=False, inplace=True)
ratings_df
