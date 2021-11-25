from src.neat import params

def run():
    param_path = "param_files/test.json"
    params.read_file(param_path)
    print(params.popsize)