import ga
import datetime
import pickle

from pm4py.objects.log.importer.xes import importer as xes_importer


log_path = "./pm_data/m1_log.xes"
log = xes_importer.apply(log_path)

param_path = "./neat/param_files/"
param_files = [param_path + "test_params.json"] # list of param file(names)

results = {}
stop_cond = ""

def main():
    for p in param_files:
        # measure time, initialize new ga
        ga_start_time = datetime.datetime.now()
        print(f"\n{80*'-'}\n{ga_start_time}: loading new ga with params {p}\n")
        new_ga = ga.GeneticAlgorithm(p, log)

        # run current ga
        stop_ga = False
        while not stop_ga:
            gen_info = new_ga.next_generation()
            print(f"GA_{p} {gen_info['gen']}:\n{4*' '}{gen_info['other_stuff']}")

            # check if reach stopping codition
            if gen_info["some_condition"] == stop_cond:
                # print info about current ga
                ga_end_time = datetime.datetime.now()
                ga_total_time = ga_end_time - ga_start_time 
                print(f"\n{40*'-'}\nGA_{p} stop time: {ga_end_time}, time: {ga_total_time}")
                print(f"GA_{p} reached {stop_cond}: {gen_info['some_condition']}\n\n")

                # save results, incl. time
                results[f"{p}_ga_params"] = ga.get_evaluation() | {"time_took": ga_total_time}
                stop_ga = True

    # finished with all configurations, stop the process
    finish_time = datetime.datetime.now()
    print(f"Process finished at {finish_time}, dumping results in pickle file!")

    # write results to pkl file
    results_fname = f"./results/{finish_time}_results.pkl"
    with open(results_fname, "wb") as f:
        pickle.dump(results, f)
    print(f"File saved as:\n{results_fname}\nStopping program!")

    return


if __name__ == "__main__":
    main()
else:
    raise Exception("this file should not be imported!")