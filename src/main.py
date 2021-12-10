import cProfile

import datetime
import pickle
import pprint as pp

from neat import ga
from pm4py.objects.log.importer.xes import importer as xes_importer


def main():
    log_path = "pm_data/running_example.xes" # "pm_data/m1_log.xes"
    log = xes_importer.apply(log_path)

    param_files = ["speciation_params"] # list of param file(names)

    results = {}
    stop_cond = "xyz"
    stop_gen = 10

    for p in param_files:
        # measure time, initialize new ga
        ga_start_time = datetime.datetime.now()
        print(f"\n{80*'-'}\n{ga_start_time}: loading new ga with params {p}\n")
        new_ga = ga.GeneticAlgorithm(p, log, True, True)

        # run current ga
        stop_ga = False
        while not stop_ga:

            # try to go to the next generation
            try:
                gen_info = new_ga.next_generation()
                print(f"GA {p} GEN: {new_ga.curr_gen}\n{pp.pformat(gen_info)}\n{8*'-'}")
            # on exception save the ga
            except Exception as exception:
                print(f"GA_{p} encountered an exception in generation {new_ga.curr_gen}")
                print(f" -> {exception}\nThe current state of the ga will be saved!")
                results[f"{p}_ga_params_EXCEPTION"] = new_ga.history | {
                    "time": datetime.datetime.now(),
                    "exception" : exception
                    }
                break

            # check if reach stopping codition, could be anything
            if gen_info["best genome fitness"] == stop_cond or new_ga.curr_gen == stop_gen:
                # print info about current ga
                ga_end_time = datetime.datetime.now()
                ga_total_time = ga_end_time - ga_start_time 
                print(f"\n{40*'-'}\nGA_{p} stop time: {ga_end_time}, time: {ga_total_time}")
                print(f"GA_{p} reached {stop_cond}: {gen_info['best genome fitness']}\n\n")

                # save results, incl. time
                results[f"{p}_ga_params"] = new_ga.history 
                results | {"time took": ga_total_time}
                stop_ga = True

    # finished with all configurations, stop the process
    finish_time = datetime.datetime.now()
    print(f"Process finished at {finish_time}, dumping results in pickle file!")

    # write results to pkl file
    results_fname = f"results/data/{finish_time.strftime('%m-%d-%Y_%H-%M-%S')}_results.pkl"
    with open(results_fname, "wb") as f:
        pickle.dump(results, f)
    print(f"File saved as:\n{results_fname}")

    return


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()  
    t = datetime.datetime.now()
    pr.dump_stats(f"{t.strftime('%m-%d-%Y_%H-%M-%S')}_profile.prof")
    print("profile dumped\nStopping program!")
else:
    raise Exception("this file should not be imported!")