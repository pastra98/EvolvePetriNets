# %%
import pandas as pd
import pm4py

class LogInterface:
    def __init__(self, logpath:str)-> None:
        self.log = pm4py.read_xes(logpath)
        # convert log to dataframe
        self.df = self.get_dataframe()
        self.tasks = self.log_df["concept:name"].unique()
    
    def get_dataframe(self) -> pd.DataFrame:
        # convert to dataframe
        log_converter = pm4py.objects.conversion.log.converter
        df = log_converter.apply(
                self.log,
                variant=log_converter.Variants.TO_DATA_FRAME
            )
        # rename cols and keep only task, time and case cols
        df = df.rename(columns={
                    "concept:name": "task",
                    "time:timestamp": "time",
                    "case:concept:name": "case"
                })[["task", "time", "case"]]
        return df

    # def get_dataframe(log) -> pd.DataFrame:
    #     log_converter = pm4py.objects.conversion.log.converter
    #     return log_converter.apply(
    #         log,
    #         variant=log_converter.Variants.TO_DATA_FRAME
    #     )

# %%
# lf = LogInterface(logpath)
# lf.log_df

# l1, l2, l3, l4 = pm4py.read_xes(logpath)
# print(f"l1 \n{l1}")
# print(f"l2 \n{l2}")
# print(f"l3 \n{l3}")
# print(f"l4 \n{l4}")

# %%
logpath = "../pm_data/bpi2016/GroundTruthLogs/pdc_2016_5.xes"
# logpath = "../pm_data/m1_log.xes"
# logpath = "../pm_data/BPI_Challenge_2012.xes"
# logpath = "../pm_data/pdc_2016_6.xes"

log = pm4py.read_xes(logpath)
log_converter = pm4py.objects.conversion.log.converter

df = log_converter.apply(
    log,
    variant=log_converter.Variants.TO_DATA_FRAME
)

# df = df.rename(columns={
#             "concept:name": "task",
#             "time:timestamp": "time",
#             "case:concept:name": "case"
#         })[["task", "time", "case"]]

# task_counts = df.groupby("task")["task"].count().sort_values(ascending=False)
df
# task_counts = df.value_counts("task").reset_index(name='count')
# case_counts = df.value_counts("case").reset_index(name='count')

# from pm4py.algo.filtering.pandas.start_activities import start_activities_filter
# sa_filter = pm4py.algo.filtering.pandas.start_activities.start_activities_filter
# sa_filter = pm4py.algo.filtering.pandas.start_activities.start_activities_filter

# sa_filter = pm4py.algo.filtering.pandas.start_activities.start_activities_filter

# from pm4py.algo.filtering.pandas.start_activities import start_activities_filter
# df_auto_sa = start_activities_filter.apply_auto_filter(
#     df,
#     parameters={start_activities_filter.Parameters.DECREASING_FACTOR: 0.6}
#     )
# df_auto_sa

# %%
for trace in log:
    print(f"\nTrace {trace._get_attributes()['concept:name']}:")
    tr_events = []
    for event in trace:
        tr_events.append(event["concept:name"])
    print(" -> ".join(tr_events))

# sa_filter = pm4py.algo.filtering.log.start_activities.start_activities_filter
# sa_filter = pm4py.algo.filtering
