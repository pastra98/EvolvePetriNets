import cProfile, os, json, traceback, datetime, pickle, pprint, gc, logging, shutil
from pm4py.objects.log.importer.xes import importer as xes_importer

from neatutils import endreports as er
from neatutils import log as lg
from neatutils import neatlogger as nl
from neat import ga

def run_setup(run_nr, main_logger, setup, results_path, progress_dict) -> dict:
    setup_path = f"{results_path}/{setup['setupname']}"
    try:
        os.makedirs(setup_path)
    except:
        pass

    run_start = datetime.datetime.now()
    run_name = f"{run_nr}_{nl.fs_compatible_time(run_start)}"
    run_dir = f"{setup_path}/{run_name}"
    os.makedirs(run_dir)

    shutil.copy(setup["parampath"], f"{run_dir}/{run_name}_params.json")

    run_logger = nl.get_logger(
        run_dir, 
        run_name, 
        setup["send_gen_info_to_console"],
        progress_dict,
        setup['setupname'],
        run_nr,
        setup['stop_cond']['val']
    )

    run_logger.info(f"\n{80*'-'}\n{run_start}: loading new ga with params {setup['parampath']}\n")
    main_logger.info(f"\n{80*'-'}\n{run_start}: loading new ga with params {setup['parampath']}\nthis is {setup['setupname']} run {run_nr}\n")
    
    try:
        if setup["is_profiled"]:
            with cProfile.Profile() as pr:
                run_result = run_ga(setup, run_logger)
            profpath = f"{run_dir}/{run_name}_profile.prof"
            pr.dump_stats(profpath)
            run_logger.info(f"profile dumped! Location\n{profpath}\n")
        else:
            run_result = run_ga(setup, run_logger)
    except:
        run_result = {"EXCEPTION": traceback.format_exc()}
        run_name += "___EXCEPTION"
        exc_str = f"Error while running {setup['setupname']}, check log at: {run_logger.handlers[1].baseFilename}"
        run_logger.exception(exc_str)
        main_logger.exception(exc_str)
        
    run_end = datetime.datetime.now()
    run_result |= {"start": run_start, "end": run_end, "time": str(run_end - run_start)}
    run_logger.info(f"{80*'/'}\nRun finished at {run_end}")
    main_logger.info(f"{80*'/'}\nRun {run_nr} of setup {setup['setupname']} finished at {run_end}")

    if not "EXCEPTION" in run_result:
        er.save_report(run_result, f"{run_dir}")
        main_logger.info(f"reports saved at:\n{run_dir}")

    gc.collect()
    return {
        "setupname": setup['setupname'],
        "run_nr": run_nr,
        "max_fitness": run_result["best_genome"].fitness if not "EXCEPTION" in run_result else None,
        "Exceptions": "EXCEPTION" in run_result
    }

def run_ga(setup: dict, logger: logging.Logger):
    log = lg.get_log_from_xes(setup["logpath"])
    stopvar, stopval = setup["stop_cond"]["var"], setup["stop_cond"]["val"]
    curr_ga = ga.GeneticAlgorithm(setup["parampath"], log)
    stop_ga = False
    
    while not stop_ga:
        try:
            gen_info = curr_ga.next_generation()
            logger.info(f"GA {setup['setupname']} GEN: {gen_info['gen']}")
            logger.debug(f"{pprint.pformat(gen_info)}\n{8*'-'}")
            
        except Exception:
            logger.exception(f"GA_{setup['parampath']}\nEXCEPTION in generation {curr_ga.curr_gen}")
            logger.info(f"Saving curr ga state after exception")
            result = curr_ga.get_ga_final_info()
            result["EXCEPTION"] = traceback.format_exc()
            return result
        
        if gen_info[stopvar] == stopval:
            logger.info(f"{setup['setupname']} reached {stopvar} of {stopval}")
            result = curr_ga.get_ga_final_info()
            return result