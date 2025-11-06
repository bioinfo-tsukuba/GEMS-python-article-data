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

"""HEK~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


HEK_PROCESSING_TIME = {
    "HEKExpire": 0,
    "HEKPassage": 60, # < 60 min
    "HEKGetImage": 10, # < 10 min
    "HEKSampling": 40, # < 40 min
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
            optimal_start_time=optimal_time,
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




class HEKPassageState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        optimal_time: float = calculate_optimal_time_from_df(df, target_density=HEK_PASSAGE_DENSITY, density_operation_name="HEKGetImage", passage_operation_name="HEKPassage")
        print(f"passage:{optimal_time=}")
        if np.isinf(optimal_time):
            raise ValueError("Optimal time is inf")
        return TaskGroup(
            optimal_start_time=int(optimal_time-HEK_PROCESSING_TIME["HEKPassage"]-HEK_PROCESSING_TIME["HEKGetImage"]),
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
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    interval=HEK_OPERATION_INTERVAL_FIRST_GET_IMAGE,
                    experiment_operation="HEKGetImage"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "HEKGetImageState"
        return "ExpireState"
    


class HEKFirstGetImageAfterPassageState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        previous_operation_time = df["time"].max()
        optimal_time = int(previous_operation_time) + HEK_OPERATION_INTERVAL_FIRST_GET_IMAGE
        return TaskGroup(
            optimal_start_time=optimal_time,
            penalty_type=LinearPenalty(penalty_coefficient=1),
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    experiment_operation="HEKGetImage"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "HEKGetImageState"
        return "ExpireState"




class HEKRestart20241123GetImageState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        previous_operation_time = df["time"].max()
        optimal_time = int(previous_operation_time) + HEK_OPERATION_INTERVAL_FIRST_GET_IMAGE

        normal_task_group = TaskGroup(
            optimal_start_time=optimal_time,
            penalty_type=LinearPenalty(penalty_coefficient=100),
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    experiment_operation="HEKGetImage"
                )
            ]
        )

        return normal_task_group
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "HEKGetImageState"
        return "ExpireState"
    


class HEKGetImageState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        previous_operation_time = df["time"].max()
        optimal_time = int(previous_operation_time) + HEK_OPERATION_INTERVAL

        normal_task_group = TaskGroup(
            optimal_start_time=optimal_time,
            penalty_type=LinearPenalty(penalty_coefficient=1),
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    experiment_operation="HEKGetImage"
                )
            ]
        )

        dummy_task_group = TaskGroup(
            optimal_start_time = previous_operation_time,
            penalty_type=LinearPenalty(penalty_coefficient=1),
            tasks=[
                Task(
                    processing_time=0,
                    experiment_operation="HEKWaiting"
                ),
            ]
        )



        getimage_count = len(df.filter(pl.col("operation") == "HEKGetImage"))
        passage_count = len(df.filter(pl.col("operation") == "HEKPassage"))
        if getimage_count < 6:
            return normal_task_group
        elif passage_count >= 3:
            optimal_time = calculate_optimal_time_from_df(df, target_density=HEK_SAMPLING_DENSITY, density_operation_name="HEKGetImage", passage_operation_name="HEKPassage")
            if np.isinf(optimal_time):
                return normal_task_group
            time_to_optimal_time = optimal_time - int(df["time"].max())
            if time_to_optimal_time <= HEK_OPERATION_INTERVAL:
                return dummy_task_group
            else:
                return normal_task_group
        else:
            optimal_time = calculate_optimal_time_from_df(df, target_density=HEK_PASSAGE_DENSITY, density_operation_name="HEKGetImage", passage_operation_name="HEKPassage")
            if np.isinf(optimal_time):
                return normal_task_group
            time_to_optimal_time = optimal_time - int(df["time"].max())
            if time_to_optimal_time <= HEK_OPERATION_INTERVAL:
                return dummy_task_group
            else:
                return normal_task_group
    
    def transition_function(self, df: pl.DataFrame) -> str:
        getimage_count = len(df.filter(pl.col("operation") == "HEKGetImage"))
        passage_count = len(df.filter(pl.col("operation") == "HEKPassage"))
        if getimage_count < 2:
            return "HEKGetImageState"
        elif passage_count >= 3:
            optimal_time = calculate_optimal_time_from_df(df, target_density=HEK_SAMPLING_DENSITY, density_operation_name="HEKGetImage", passage_operation_name="HEKPassage")
            if np.isinf(optimal_time):
                return "HEKGetImageState"
            time_to_optimal_time = optimal_time - int(df["time"].max())
            if time_to_optimal_time <= HEK_OPERATION_INTERVAL:
                return "HEKSamplingState"
            else:
                return "HEKGetImageState"
        else:
            optimal_time = calculate_optimal_time_from_df(df, target_density=HEK_PASSAGE_DENSITY, density_operation_name="HEKGetImage", passage_operation_name="HEKPassage")
            if np.isinf(optimal_time):
                return "HEKGetImageState"
            time_to_optimal_time = optimal_time - int(df["time"].max())
            if time_to_optimal_time <= HEK_OPERATION_INTERVAL:
                return "HEKPassageState"
            else:
                return "HEKGetImageState"
            
        return "HEKExpireState"
    

class HEKSamplingState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        optimal_time: float = calculate_optimal_time_from_df(df, target_density=HEK_SAMPLING_DENSITY, density_operation_name="HEKGetImage", passage_operation_name="HEKPassage")
        print(f"passage:{optimal_time=}")
        if np.isinf(optimal_time):
            raise ValueError("Optimal time is inf")
        return TaskGroup(
            optimal_start_time=int(optimal_time-HEK_PROCESSING_TIME["HEKSampling"]-HEK_PROCESSING_TIME["HEKGetImage"]),
            penalty_type=CyclicalRestPenaltyWithLinear(
                cycle_start_time = UNIX_2024_11_13_00_00_00_IN_JP//60,
                cycle_duration = 60*24,
                rest_time_ranges = [(0, 60*10), (60*16, 60*24)],
                penalty_coefficient = 100
            ),
            # cycle_start_time: int
            # cycle_duration: int
            # rest_time_ranges: List[Tuple[int, int]]
            # penalty_coefficient: int
            tasks=[
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKGetImage"],
                    experiment_operation="HEKGetImage"
                ),
                Task(
                    processing_time=HEK_PROCESSING_TIME["HEKSampling"],
                    experiment_operation="HEKSampling"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="HEKWaiting"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "HEKExpireState"




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
                HEKExpireState(),
                HEKGetImageState(),
                HEKPassageState(),
                HEKSamplingState(),
                HEKFirstGetImageAfterPassageState(),
                HEKRestart20241123GetImageState()
            ],
            current_state_name=current_state_name,
            shared_variable_history=shared_variable_history,
            experiment_uuid=experiment_uuid
        )

TIME = UNIX_2024_11_13_00_00_00_IN_JP

def HEKAsetting():
    i = 0
    shared_variable_history = pl.read_csv("iPSsimulation/results/2024-11-13_real_round3/step_00000231/experiments/HEKExperiment_HEK_Experiment_0_shared_variable_history.csv")
    new_experiment = HEKExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="HEKRestart20241123GetImageState",
        experiment_uuid = f"HEK_Experiment_{i}"
    )
    return new_experiment

def HEKBsetting():
    i = 1
    shared_variable_history = pl.read_csv("iPSsimulation/results/2024-11-13_real_round3/step_00000231/experiments/HEKExperiment_HEK_Experiment_1_shared_variable_history.csv")
    new_experiment = HEKExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="HEKRestart20241123GetImageState",
        experiment_uuid = f"HEK_Experiment_{i}"
    )
    return new_experiment

def HEKCsetting():
    i = 2
    shared_variable_history = pl.read_csv("iPSsimulation/results/2024-11-13_real_round3/step_00000231/experiments/HEKExperiment_HEK_Experiment_2_shared_variable_history.csv")
    new_experiment = HEKExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="HEKRestart20241123GetImageState",
        experiment_uuid = f"HEK_Experiment_{i}"
    )
    return new_experiment

def HEKDsetting():
    i = 3
    shared_variable_history = pl.read_csv("iPSsimulation/results/2024-11-13_real_round3/step_00000231/experiments/HEKExperiment_HEK_Experiment_3_shared_variable_history.csv")
    new_experiment = HEKExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="HEKRestart20241123GetImageState",
        experiment_uuid = f"HEK_Experiment_{i}"
    )
    return new_experiment

def HEKEsetting():
    i = 4
    shared_variable_history = pl.read_csv("iPSsimulation/results/2024-11-13_real_round3/step_00000231/experiments/HEKExperiment_HEK_Experiment_4_shared_variable_history.csv")
    new_experiment = HEKExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="HEKRestart20241123GetImageState",
        experiment_uuid = f"HEK_Experiment_{i}"
    )
    return new_experiment




"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

if __name__ == "__main__":
    # Create an instance of the experiment
    experiment = HEKExperiment(current_state_name="HEKExpireState")
    print(experiment.current_state_name)
    experiment.show_experiment_with_tooltips()