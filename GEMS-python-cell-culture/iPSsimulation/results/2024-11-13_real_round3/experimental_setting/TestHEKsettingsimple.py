from dataclasses import dataclass
from datetime import datetime
import uuid
import numpy as np
import polars as pl
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import LinearPenalty, LinearWithRangePenalty, CyclicalRestPenaltyWithLinear
from gems_python.one_machine_problem_interval_task.task_info import TaskGroup, Task
from gems_python.one_machine_problem_interval_task.transition_manager import Experiment, State
from dataclasses import dataclass
import polars as pl

from curve import calculate_optimal_time_from_df

# 1731423600 2024_11_13_00_00_00 UNIX

UNIX_2024_11_13_00_00_00_IN_JP = 1731423600
def get_now_min_unix() -> int:
    now = datetime.now()
    now = now.astimezone()
    now_unix = now.timestamp()
    return int(now_unix//60)

"""HEK~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


HEK_PROCESSING_TIME = {
    "HEKExpire": 0,
    "HEKPassage": 120,
    "HEKGetImage": 10,
    "HEKSampling": 120,
    "HEKWaiting": 0
}

def get_current_time()->int:
    current_time: float = datetime.now().timestamp()
    # Convert to minutes
    current_time = int(current_time / 60)
    return current_time


HEK_OPERATION_INTERVAL = (12 * 60)
HEK_OPERATION_INTERVAL_FIRST_GET_IMAGE = (24 * 60)
HEK_PASSAGE_DENSITY = 0.8
HEK_SAMPLING_DENSITY = 0.4




class HEKExpireState(State):

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        current_time = df["time"].max()
        optimal_time = int(current_time+HEK_OPERATION_INTERVAL)
        return TaskGroup(
            optimal_start_time = get_now_min_unix(),#optimal_time,
            penalty_type=LinearPenalty(penalty_coefficient=1),
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKExpire"],
                    interval=0, 
                    experiment_operation="HEKExpire"
                    )
            ]
        )
        

    def transition_function(self, df: pl.DataFrame) -> str:
        return "HEKExpireState"




class HEKGetImageState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return TaskGroup(
            optimal_start_time = get_now_min_unix(),#int(optimal_time-HEK_PROCESSING_TIME["HEKPassage"]-HEK_PROCESSING_TIME["HEKGetImage"]),
            penalty_type=LinearPenalty(penalty_coefficient=1000),
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    experiment_operation="HEKGetImage"
                ),
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKPassage"],
                    experiment_operation="HEKPassage"
                ),
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKSampling"],
                    experiment_operation="HEKSampling"
                ),
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKWaiting"],
                    interval = 10000,
                    experiment_operation="HEKWaiting"
                ),
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "HEKGetImageState"
        return "ExpireState"



class HEKExperiment(Experiment):
    def __init__(self, current_state_name, shared_variable_history=None, experiment_uuid=uuid.uuid4()):
        # Define the experiment using the states

        # Create a shared variable history DataFrame (empty for this example)
        if shared_variable_history is None:
            # │ 34568 ┆ null     ┆ Passage   │
            # │ 36008 ┆ 0.411028 ┆ GetImage  │
            # │ 37448 ┆ 0.534656 ┆ GetImage  │
            # │ 38888 ┆ 0.632666 ┆ GetImage
            shared_variable_history = pl.DataFrame(
                {
                    "time": [60*60*24*365*10],
                    "operation": ["HEKPassage"]
                }
            )
            
        # Define the initial state and experiment
        super().__init__(
            experiment_name="HEKExperiment",
            states=[
                HEKGetImageState(),
            ],
            current_state_name=current_state_name,
            shared_variable_history=shared_variable_history,
            experiment_uuid=experiment_uuid
        )

TIME = UNIX_2024_11_13_00_00_00_IN_JP

def HEKAsetting():
    i = 0
    shared_variable_history = pl.DataFrame(
        {
            "time": [int(TIME/60 - 120*(i+5))],
            "operation": ["HEKPassage"]
        }
    )
    new_experiment = HEKExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="HEKGetImageState",
        experiment_uuid = f"HEK_Experiment_{i}"
    )
    return new_experiment
