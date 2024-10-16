import sys, os, json, datetime
from multiprocessing import Pool, Manager
from tqdm import tqdm

from neatutils import neatlogger as nl
from neatutils.setuprunner import run_setup
import pandas as pd

def main(conf: dict) -> None:
    conf_name = conf["name"]
    exec_start_time = datetime.datetime.now()
    results_path = f"results/data/{conf_name}_{nl.fs_compatible_time(exec_start_time)}"
    os.makedirs(results_path)
    main_logger = nl.get_logger(results_path, "main", True)
    main_logger.info(f"Execution started at: {exec_start_time}")

    manager = Manager()
    progress_dict = manager.dict()

    argslist = []
    for setup in conf["setups"]:
        for run_nr in range(1, setup["n_runs"] + 1):
            argslist.append((run_nr, main_logger, setup, results_path, progress_dict))

    with Pool() as p:
        setup_fitnesses = p.starmap(run_setup, argslist)

    show_progress_bar = conf["setups"][0].get("show_progress_bar", False)
    if show_progress_bar:
        display_progress(progress_dict, conf["setups"])

    exec_end_time = datetime.datetime.now()
    dur = exec_end_time - exec_start_time
    main_logger.info(f"Execution finished at: {exec_end_time}\nTime: {dur}")
    main_logger.info("saving final fitness report")

    df = pd.DataFrame(setup_fitnesses)
    df.to_feather(f"{results_path}/final_report_df.feather")
    with open(f"{results_path}/execution_report.txt", "w") as f:
        f.write(f"times:\nstart: {exec_start_time}\nend: {exec_end_time}\nduration: {dur}\n")
        f.write("\nsorted by fitness:\n" + str(df.sort_values(by="max_fitness", ascending=False)))
        f.write("\nreport grouped by setup:\n" + str(df[["setupname", "max_fitness"]].groupby("setupname").describe()))
    main_logger.info("Data successfully saved, quitting all processes")

def display_progress(progress_dict, setups):
    total_runs = sum(setup["n_runs"] for setup in setups)
    with tqdm(total=total_runs, desc="Overall Progress") as overall_pbar:
        setup_pbars = {setup["setupname"]: tqdm(total=setup["n_runs"], desc=f"Setup {setup['setupname']}") for setup in setups}
        
        completed_runs = set()
        while len(completed_runs) < total_runs:
            for key, value in progress_dict.items():
                setup_name, run_id = key.rsplit('_', 1)
                if value == 1 and key not in completed_runs:
                    overall_pbar.update(1)
                    setup_pbars[setup_name].update(1)
                    completed_runs.add(key)
                elif key not in completed_runs:
                    setup_pbars[setup_name].n = value
                    setup_pbars[setup_name].refresh()

        for pbar in setup_pbars.values():
            pbar.close()

if __name__ == "__main__":
    try:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    except:
        raise Exception("passed invalid config filepath!")
    main(config)