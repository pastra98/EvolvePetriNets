from src.neat import params
import importlib
importlib.reload(params)

def save_new_params(old_name, new_name, save_current=True):
    params.load(old_name)
    print(params.popsize)
    params.new_param_json(new_name, save_current=True)