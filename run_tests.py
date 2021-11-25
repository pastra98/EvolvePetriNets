# %%
from src.tests import read_and_save_new_params as rsp
from src.tests import create_population as cp
from src.tests import test_startconfigs as ts
import importlib

# rsp.run()
# ts.run()
# cp.run()

tg = ts.get_test_genomes(save=False, view=True)

# %%
importlib.reload(ts)

for g in tg:
    print(g.id)
    print(ts.get_fitness(g))
