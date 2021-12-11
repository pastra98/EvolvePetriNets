import cProfile, sys, os, json, traceback, datetime, pickle, pprint
from pm4py.objects.log.importer.xes import importer as xes_importer
from neat import ga

def main(conf: dict) -> None:
    conf_name = conf["name"]
    exec_start_time = datetime.datetime.now()
    # create new dir for the current conf execution
    results_path = f"results/data/{conf_name}_{fs_compatible_time(exec_start_time)}"
    os.mkdir(results_path)

    for setup in conf["setups"]:

        for run in range(setup["n_runs"]):
            run_start = datetime.datetime.now()
            # create a dir for the current run, along with subdir for reports
            run_name = f"{setup['setupname']}_{run}_{fs_compatible_time(run_start)}"
            run_dir = f"{results_path}/{run_name}"
            os.makedirs(f"{run_dir}/reports")

            # run the current setup once, profile if enabled in setup, save result
            print(f"\n{80*'-'}\n{run_start}: loading new ga with params {setup['parampath']}\n")
            try:
                if setup["is_profiled"]:
                    with cProfile.Profile() as pr:
                        run_result = run_setup(setup)
                    pr.dump_stats(f"{run_dir}/{run_name}_profile.prof")
                    print("profile dumped!")
                else:
                    run_result = run_setup(setup)
            except: # woops, maybe config or log path or something messed up
                run_result = {"EXCEPTION": traceback.format_exc()}
                print(traceback.format_exc())
                
            # update results of this run with times
            run_end = datetime.datetime.now()
            run_result |= {"start": run_start, "end": run_end, "time": run_end - run_start}
            print(f"{80*'/'}\nRun finished at {run_end}, dumping results in pickle file!")
            # write results of run to pkl file
            if "EXCEPTION" in run_result:
                run_name += "___EXCEPTION"
            results_name = f"{run_dir}/{run_name}_results.pkl"
            with open(results_name, "wb") as f:
                pickle.dump(run_result, f)
            print(f"File saved as:\n{results_name}")
    
    # info about overall execution (may put log in there) TODO: dump output log here
    exec_end_time = datetime.datetime.now()
    dur = exec_end_time - exec_start_time
    print(f"Execution finished at: {exec_end_time}\nTime: {dur}")
    with open(f"{results_path}/times.txt", "w") as f:
        f.write(f"""{fs_compatible_time(exec_start_time)}
            {fs_compatible_time(exec_end_time)}\n{dur}""")


def run_setup(setup):
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
            if setup["print_gen_info"]:
                print(f"GA {setup['setupname']} GEN: {curr_ga.curr_gen}")
                print(f"{pprint.pformat(gen_info)}\n{8*'-'}")
        # on exception save the ga, return to main
        except Exception:
            print(f"GA_{setup['parampath']}\nEXCEPTION in generation {curr_ga.curr_gen}")
            print(f" -> {traceback.format_exc()}\nSaving curr ga state!")
            result = curr_ga.get_ga_final_info()
            result["EXCEPTION"] = traceback.format_exc()
            return result
        # check if reach stopping codition, could be anything
        if gen_info[stopvar] == stopval: # TODO probably will have to think about other operators
            print(f"{setup['setupname']} reached {stopvar} of {stopval}")
            result = curr_ga.get_ga_final_info()
            return result


def fs_compatible_time(dt) -> str:
    return dt.strftime('%m-%d-%Y_%H-%M-%S')


if __name__ == "__main__":
    try:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    except:
        raise Exception("passed invalid config filepath!")
    main(config)
else:
    raise Exception("this file should not be imported!")