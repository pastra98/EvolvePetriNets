import sys, os, json, datetime
from multiprocessing import Pool

from pm4py.objects.log.importer.xes import importer as xes_importer

from neatutils import neatlogger as nl
from neatutils.setuprunner import run_setup
import pandas as pd


def main(conf: dict) -> None:
    conf_name = conf["name"]
    exec_start_time = datetime.datetime.now()
    # create new dir for the current conf execution
    results_path = f"results/data/{conf_name}_{nl.fs_compatible_time(exec_start_time)}"
    os.makedirs(results_path)
    # setup main logger
    main_logger = nl.get_logger(results_path, "main", True)
    main_logger.info(f"Execution started at: {exec_start_time}")

    argslist = []
    for setup in conf["setups"]:
        for run_nr, args in enumerate([[setup, results_path]] * setup["n_runs"], start=1):
            argslist.append(tuple([run_nr, main_logger, *args]))

    with Pool() as p:
        setup_fitnesses = p.starmap(run_setup, argslist)

    # info about overall execution (may put log in there) TODO: dump output log here
    exec_end_time = datetime.datetime.now()
    dur = exec_end_time - exec_start_time
    main_logger.info(f"Execution finished at: {exec_end_time}\nTime: {dur}")
    main_logger.info("saving final fitness report")

    # make the final reports
    df = pd.DataFrame(setup_fitnesses)
    df.to_feather(f"{results_path}/final_report_df.feather")
    with open(f"{results_path}/execution_report.txt", "w") as f:
        f.write(f"times:\nstart: {exec_start_time}\nend: {exec_end_time}\nduration: {dur}\n")
        f.write("\nsorted by fitness:\n" + str(df.sort_values(by="max_fitness", ascending=False)))
        f.write("\nreport grouped by setup:\n" + str(df[["setupname", "max_fitness"]].groupby("setupname").describe()))
    main_logger.info("Data successfully saved, quitting all processes")


if __name__ == "__main__":
    # TODO: SET UP MODE "W" to start a new file every time, make sure info also gets printed to console
    try:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    except:
        raise Exception("passed invalid config filepath!")
    main(config)
    