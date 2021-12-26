import cProfile, os, json, traceback, datetime, pickle, pprint, gc, logging
from pm4py.objects.log.importer.xes import importer as xes_importer

from . import endreports as er
from . import neatlogger as nl
from neat import ga

def run_setup(setup, results_path) -> dict:
    setup_fitness_dict = {}
    setup_path = f"{results_path}/{setup['setupname']}"
    os.makedirs(setup_path)
    setup_logger = nl.get_logger(setup_path, setup["setupname"], True)
    for run in range(1, setup["n_runs"]):

        # create a dir for the current run, along with subdir for reports
        run_start = datetime.datetime.now()
        run_name = f"{run}_{nl.fs_compatible_time(run_start)}"
        run_dir = f"{setup_path}/{run_name}"
        os.makedirs(f"{run_dir}/reports")

        # setup run_logger, use setup config to determine if send to console
        run_logger = nl.get_logger(run_dir, run_name, setup["send_gen_info_to_console"])

        # run the current setup once, profile if enabled in setup, save result
        setup_logger.info(f"\n{80*'-'}\n{run_start}: loading new ga with params {setup['parampath']}\n")
        try:
            if setup["is_profiled"]:
                with cProfile.Profile() as pr:
                    run_result = run_ga(setup, run_logger)
                profpath = f"{run_dir}/{run_name}_profile.prof"
                pr.dump_stats(profpath)
                setup_logger.info(f"profile dumped! Location\n{profpath}\n")
            else:
                run_result = run_ga(setup, run_logger)
        except: # woops, maybe config or log path or something messed up
            run_result = {"EXCEPTION": traceback.format_exc()}
            exc_str = f"Error while running {setup['setupname']}, check log at: {run_logger.handlers[1].baseFilename}"
            setup_logger.exception(exc_str)
            
        # update results of this run with times
        run_end = datetime.datetime.now()
        run_result |= {"start": run_start, "end": run_end, "time": run_end - run_start}
        setup_logger.info(f"{80*'/'}\nRun finished at {run_end}")
        # write results of run to pkl file
        if "EXCEPTION" in run_result:
            run_name += "___EXCEPTION"
        # if no exception occured, save some plots, update setup_fitness_dict
        elif setup["save_reports"]:
            er.save_report(
                run_result,
                f"{run_dir}/reports",
                setup["show_report_plots"],
                setup["save_reduced_history_df"]
            )
            setup_fitness_dict[run_name] = run_result["max_fitness"]

        if setup["save_params"]:
            params_name = f"{run_dir}/{run_name}_params.json"
            with open(params_name, "w") as f:
                json.dump(run_result["param_values"], f, indent=4)

        if setup["dump_pickle"]:
            results_name = f"{run_dir}/{run_name}_results.pkl"
            setup_logger.info(f"Dumping results in pickle file!")
            with open(results_name, "wb") as f:
                pickle.dump(run_result, f)
            setup_logger.info(f"File saved as:\n{results_name}")

        del run_result
        gc.collect()
    
    # write setup results to file
    setup_fitness_dict = {k: v for k, v in sorted(setup_fitness_dict.items(), key=lambda item: item[1])}
    with open(f"{setup_path}/execution_report.txt", "w") as f:
        json.dump(setup_fitness_dict, fp=f, indent=4)

    return setup_fitness_dict


def run_ga(setup: logging.Logger, logger):
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
            return result

