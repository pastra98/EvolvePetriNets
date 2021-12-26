import sys, os, json, datetime
from multiprocessing import Pool

from pm4py.objects.log.importer.xes import importer as xes_importer

from neatutils import neatlogger as nl
from neatutils.setuprunner import run_setup


def main(conf: dict) -> None:
    conf_name = conf["name"]
    exec_start_time = datetime.datetime.now()
    # create new dir for the current conf execution
    results_path = f"results/data/{conf_name}_{nl.fs_compatible_time(exec_start_time)}"
    os.makedirs(results_path)
    # setup main logger
    main_logger = nl.get_logger(results_path, "main", True)
    main_logger.info(f"Execution started at: {exec_start_time}")

    args = [(setup, results_path) for setup in conf["setups"]]
    with Pool() as p:
        setup_fitnesses = p.starmap(run_setup, args)

    # info about overall execution (may put log in there) TODO: dump output log here
    exec_end_time = datetime.datetime.now()
    dur = exec_end_time - exec_start_time
    main_logger.info(f"Execution finished at: {exec_end_time}\nTime: {dur}")

    fit_list = list(zip([s["setupname"] for s in conf["setups"]], setup_fitnesses))
    fit_list.sort(key=lambda t: list(t[1].values())[0])
    with open(f"{results_path}/execution_report.txt", "w") as f:
        f.write(f"{exec_start_time}\n{exec_end_time}\n{dur}\n")
        json.dump(fit_list, fp=f, indent=4)



if __name__ == "__main__":
    # TODO: SET UP MODE "W" to start a new file every time, make sure info also gets printed to console
    try:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    except:
        raise Exception("passed invalid config filepath!")
    main(config)
else:
    raise Exception("this file should not be imported!")
    