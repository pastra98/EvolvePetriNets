import sys, os, json, datetime

from neatutils import neatlogger as nl
from neatutils.setuprunner import run_setup

import multiprocessing as mp
import pandas as pd

import threading
from queue import Empty
from tqdm import tqdm


class SetupTracker(threading.Thread):
    """A listener Thread that fetches from the queue shared by all setuprunner
    threads. Setuprunner only puts into the queue after a setup has finished running.
    Shows a progress bar for the number of finished setups.
    """
    def __init__(self, queue, n_runs):
        super().__init__()
        self._queue = queue
        self._stop_event = threading.Event()
        self.pbar = tqdm(total=n_runs, desc=f"Running {n_runs} setups")


    def run(self):
        while not self._stop_event.is_set():
            try:
                value = self._queue.get(timeout=5)
                self.pbar.update(1)
            except Empty:
                continue
    
    def stop(self):
        self.pbar.close()
        self._stop_event.set()


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
        for run_nr in range(1, setup["n_runs"] + 1):
            # Create tuple matching the function signature exactly
            args = (run_nr, setup, results_path)
            argslist.append(args)

    with mp.Manager() as manager:
        shared_queue = manager.Queue()

        # n_runs = sum([s["n_runs"] for s in conf["setups"]])
        pl = SetupTracker(shared_queue, len(argslist))
        pl.start()

        with mp.Pool() as p:
            # Prepend queue to each argument tuple
            setup_fitnesses = p.starmap(
                run_setup,
                [(run_nr, shared_queue, setup, results_path) for (run_nr, setup, results_path) in argslist]
            )
        
        pl.stop()
        pl.join()



    # info about overall execution (may put log in there)
    exec_end_time = datetime.datetime.now()
    dur = exec_end_time - exec_start_time
    main_logger.info(f"Execution finished at: {exec_end_time}\nTime: {dur}")
    main_logger.info("saving final fitness report")

    # make the final reports
    df = pd.DataFrame(setup_fitnesses)
    df.to_feather(f"{results_path}/final_report_df.feather")
    with open(f"{results_path}/execution_report.txt", "w") as f:
        f.write(f"times:\nstart: {exec_start_time}\nend: {exec_end_time}\nduration: {dur}\n")
        f.write("\nsorted by fitness:\n" + str(df.sort_values(by=["setupname", "max_fitness"], ascending=False).to_markdown()))
        f.write("\n\nreport grouped by setup:\n" + str(df[["setupname", "max_fitness"]].groupby("setupname").describe()))
    main_logger.info("Data successfully saved, quitting all processes")


if __name__ == "__main__":
    try:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    except:
        raise Exception("passed invalid config filepath!")
    main(config)
    