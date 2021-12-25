import cProfile, sys, os, json, traceback, datetime, pickle, pprint, gc
from pm4py.objects.log.importer.xes import importer as xes_importer
from neat import ga
from neatutils import endreports
import logging

def main(conf: dict) -> None:
    conf_name = conf["name"]
    exec_start_time = datetime.datetime.now()
    # create new dir for the current conf execution
    results_path = f"results/data/{conf_name}_{fs_compatible_time(exec_start_time)}"
    os.makedirs(results_path)
    # setup main logger
    main_logger = setup_logger(results_path, "main", True)
    run_fitness_dict = {}

    for setup in conf["setups"]:

        for run in range(setup["n_runs"]):
            run_start = datetime.datetime.now()
            # create a dir for the current run, along with subdir for reports
            run_name = f"{setup['setupname']}_{run}___{fs_compatible_time(run_start)}"
            run_dir = f"{results_path}/{run_name}"
            os.makedirs(f"{run_dir}/reports")
            # setup run_logger, use setup config to determine if send to console
            run_logger = setup_logger(run_dir, run_name, setup["send_gen_info_to_console"])

            # run the current setup once, profile if enabled in setup, save result
            main_logger.info(f"\n{80*'-'}\n{run_start}: loading new ga with params {setup['parampath']}\n")
            try:
                if setup["is_profiled"]:
                    with cProfile.Profile() as pr:
                        run_result = run_setup(setup, run_logger)
                    profpath = f"{run_dir}/{run_name}_profile.prof"
                    pr.dump_stats(profpath)
                    main_logger.info(f"profile dumped! Location\n{profpath}\n")
                else:
                    run_result = run_setup(setup, run_logger)
            except: # woops, maybe config or log path or something messed up
                run_result = {"EXCEPTION": traceback.format_exc()}
                exc_str = f"Error while running {setup['setupname']}, check log at: {run_logger.handlers[1].baseFilename}"
                main_logger.exception(exc_str)
                
            # update results of this run with times
            run_end = datetime.datetime.now()
            run_result |= {"start": run_start, "end": run_end, "time": run_end - run_start}
            main_logger.info(f"{80*'/'}\nRun finished at {run_end}")
            # write results of run to pkl file
            if "EXCEPTION" in run_result:
                run_name += "___EXCEPTION"
            # if no exception occured, save some plots, update run_fitness_dict
            elif conf["save_reports"]:
                endreports.save_report(
                    run_result,
                    f"{run_dir}/reports",
                    conf["show_report_plots"],
                    conf["save_reduced_history_df"]
                )
                run_fitness_dict[run_name] = run_result["max_fitness"]
            results_name = f"{run_dir}/{run_name}_results.pkl"
            if setup["dump_pickle"]:
                main_logger.info("dumping results in pickle")
                with open(results_name, "wb") as f:
                    pickle.dump(run_result, f)
                main_logger.info(f"File saved as:\n{results_name}")

            del run_result
            gc.collect()
    
    # info about overall execution (may put log in there) TODO: dump output log here
    exec_end_time = datetime.datetime.now()
    dur = exec_end_time - exec_start_time
    main_logger.info(f"Execution finished at: {exec_end_time}\nTime: {dur}")
    with open(f"{results_path}/execution_report.txt", "w") as f:
        f.write(f"{exec_start_time}\n{exec_end_time}\n{dur}\n")
        d = {k: v for k, v in sorted(run_fitness_dict.items(), key=lambda item: item[1])}
        json.dump(d, fp=f, indent=4)


def run_setup(setup: logging.Logger, logger):
    log = xes_importer.apply(setup["logpath"])
    stopvar, stopval = setup["stop_cond"]["var"], setup["stop_cond"]["val"]
    # initialize GeneticAlgorithm with setup info
    curr_ga = ga.GeneticAlgorithm(
        setup["parampath"],
        log,
        is_minimal_serialization=setup["ga_kwargs"]["is_minimal_serialization"],
        is_pop_serialized=setup["ga_kwargs"]["is_pop_serialized"],
        is_timed=setup["ga_kwargs"]["is_timed"]
    )
    # run current ga
    stop_ga = False
    while not stop_ga:
        # try to go to the next generation
        try:
            gen_info = curr_ga.next_generation()
            # info always printed, debug depends on logging level (send_gen_info_to_console)
            logger.info(f"GA {setup['setupname']} GEN: {gen_info['gen']}")
            logger.debug(f"{pprint.pformat(gen_info)}\n{8*'-'}")
        # on exception save the ga, return to main
        except Exception:
            logger.exception(f"GA_{setup['parampath']}\nEXCEPTION in generation {curr_ga.curr_gen}")
            logger.info(f"Saving curr ga state after exception")
            result = curr_ga.get_ga_final_info()
            result["EXCEPTION"] = traceback.format_exc()
            return result
        # check if reach stopping codition, could be anything
        if gen_info[stopvar] == stopval: # TODO probably will have to think about other operators
            logger.info(f"{setup['setupname']} reached {stopvar} of {stopval}")
            result = curr_ga.get_ga_final_info()
            del curr_ga
            return result


def fs_compatible_time(dt) -> str:
    return dt.strftime('%m-%d-%Y_%H-%M-%S')


def setup_logger(log_path: str, logname: str, send_to_console: bool) -> logging.Logger:
    # set up main logger for the entire execution
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG) # root level of logger, handlers cannot go deeper, default for all handlers
    # set level of console handler, only relevant for run loggers when printing gen info
    c_main_handler = logging.StreamHandler()
    if send_to_console:
        c_main_handler.setLevel(logging.DEBUG) # means that gen info will be printed
    else:
        c_main_handler.setLevel(logging.INFO) # means that only a new gen will be printed
    # file handler always writes in debug mode, meaning gen info will be included
    f_main_handler = logging.FileHandler(f"{log_path}/{logname}.log", mode="w")
    # Create formatters and add it to handlers
    lformat = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n")
    c_main_handler.setFormatter(lformat)
    f_main_handler.setFormatter(lformat)
    # Add handlers to the logger
    logger.addHandler(c_main_handler)
    logger.addHandler(f_main_handler)
    return logger


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
    