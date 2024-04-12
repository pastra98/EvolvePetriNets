# %%
from neat import params
import importlib

def save_new_params(old_name, new_name, save_current=True):
    importlib.reload(params)
    params.load(old_name)
    print(params.popsize)
    params.new_param_json(new_name, save_current=True)

# %%
# overwrite the existing test params

save_new_params(
    old_name="../params/testing/speciation_test.json",
    new_name="../params/testing/speciation_test.json",
)
