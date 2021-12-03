import datetime
import pickle
import pprint as pp

from neat import ga
from pm4py.objects.log.importer.xes import importer as xes_importer


def main():
    log_path = "pm_data/m1_log.xes"
    log = xes_importer.apply(log_path)

    param_files = ["speciation_params"] # list of param file(names)

    results = {}
    stop_cond = 3

    for p in param_files:
        # measure time, initialize new ga
        ga_start_time = datetime.datetime.now()
        print(f"\n{80*'-'}\n{ga_start_time}: loading new ga with params {p}\n")
        new_ga = ga.GeneticAlgorithm(p, log)

        # run current ga
        stop_ga = False
        while not stop_ga:
            gen_info = new_ga.next_generation()
            print(f"GA_{p} {pp.pprint(gen_info)}:\n{4*' '}")

            # check if reach stopping codition, could be anything
            if gen_info["gen"] == stop_cond:
                # print info about current ga
                ga_end_time = datetime.datetime.now()
                ga_total_time = ga_end_time - ga_start_time 
                print(f"\n{40*'-'}\nGA_{p} stop time: {ga_end_time}, time: {ga_total_time}")
                print(f"GA_{p} reached {stop_cond}: {gen_info['gen']}\n\n")

                # save results, incl. time
                results[f"{p}_ga_params"] = new_ga.get_ga_info() | {"time_took": ga_total_time}
                stop_ga = True

    # finished with all configurations, stop the process
    finish_time = datetime.datetime.now()
    print(f"Process finished at {finish_time}, dumping results in pickle file!")

    # write results to pkl file
    results_fname = f"results/data/{finish_time.strftime('%m-%d-%Y %H-%M-%S')}_results.pkl"
    with open(results_fname, "wb") as f:
        pickle.dump(results, f)
    print(f"File saved as:\n{results_fname}\nStopping program!")

    return


if __name__ == "__main__":
    main()
else:
    raise Exception("this file should not be imported!")