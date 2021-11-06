import pandas as pd
import pm4py

# log_path = "./pm_data/m1_log.xes"
log_path = "./pm_data/BPI_Challenge_2012.xes"

log = pm4py.read_xes(log_path)
log_converter = pm4py.objects.conversion.log.converter
log_df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

tasks = log_df["concept:name"].unique()
print(tasks)