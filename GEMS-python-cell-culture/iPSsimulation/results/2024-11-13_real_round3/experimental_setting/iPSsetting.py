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


IPS_PROCESSING_TIME = {
    "iPSExpire": 0,
    "iPSPassage": 80,
    "iPSGetImage": 10,
    "iPSMediumChange": 20,
    "iPSPlateCoating": 20,
    "iPSSampling": 60,
    "iPSWaiting": 0
}


OPERATION_INTERVAL = (24 * 60)
PASSAGE_DENSITY = 0.3
SAMPLING_DENSITY = 0.3
MEDIUM_CHANGE_1_DENSITY = 0.05




class ExpireState(State):

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        current_time = df["time"].max()
        optimal_time = int(current_time+OPERATION_INTERVAL)
        return TaskGroup(
            optimal_start_time=optimal_time,
            penalty_type=LinearPenalty(penalty_coefficient=1),
            tasks=[
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSExpire"], 
                    interval=0, 
                    experiment_operation="iPSExpire"
                    )
            ]
        )
        

    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"



class PassageInitialState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        optimal_time: float = get_now_min_unix()
        print(f"passage:{optimal_time=}")
        return TaskGroup(
            optimal_start_time=int(optimal_time-1*60-IPS_PROCESSING_TIME["iPSPlateCoating"]-IPS_PROCESSING_TIME["iPSGetImage"]),
            penalty_type=LinearPenalty(penalty_coefficient=1000),
            tasks=[
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSPlateCoating"],
                    experiment_operation="iPSPlateCoating"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSGetImage"],
                    interval=1*60,
                    experiment_operation="iPSGetImage"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSPassage"],
                    experiment_operation="iPSPassage"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "MediumChange1State"
        return "ExpireState"

class PassageState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        optimal_time: float = calculate_optimal_time_from_df(df, target_density=PASSAGE_DENSITY, density_operation_name="iPSGetImage", passage_operation_name="iPSPassage")
        print(f"passage:{optimal_time=}")
        if np.isinf(optimal_time):
            raise ValueError("Optimal time is inf")
        return TaskGroup(
            optimal_start_time=int(optimal_time-1*60-IPS_PROCESSING_TIME["iPSPlateCoating"]-IPS_PROCESSING_TIME["iPSGetImage"]),
            penalty_type=LinearPenalty(penalty_coefficient=1000),
            tasks=[
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSPlateCoating"],
                    experiment_operation="iPSPlateCoating"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSGetImage"],
                    interval=1*60,
                    experiment_operation="iPSGetImage"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSPassage"],
                    experiment_operation="iPSPassage"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "MediumChange1State"
        return "ExpireState"


class MediumChange1State(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return TaskGroup(
            optimal_start_time=int(df["time"].max()) + OPERATION_INTERVAL,
            penalty_type=LinearPenalty(penalty_coefficient=100),
            tasks=[
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSGetImage"],
                    experiment_operation="iPSGetImage"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSMediumChange"],
                    experiment_operation="iPSMediumChange"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSGetImage"],
                    interval=OPERATION_INTERVAL,
                    experiment_operation="iPSGetImage"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "MediumChange2State"
        return "ExpireState"

class MediumChange2State(State):

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return TaskGroup(
            optimal_start_time=int(df["time"].max()) + OPERATION_INTERVAL,
            penalty_type=LinearPenalty(penalty_coefficient=100),
            tasks=[
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSGetImage"],
                    experiment_operation="iPSGetImage"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSMediumChange"],
                    experiment_operation="iPSMediumChange"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        getimage_count = len(df.filter(pl.col("operation") == "iPSGetImage"))
        passage_count = len(df.filter(pl.col("operation") == "iPSPassage"))
        if getimage_count < 2:
            return "MediumChange2State"
        elif passage_count >= 3:
            optimal_time = calculate_optimal_time_from_df(df, target_density=SAMPLING_DENSITY, density_operation_name="iPSGetImage", passage_operation_name="iPSPassage")
            if np.isinf(optimal_time):
                return "MediumChange2State"
            time_to_optimal_time = optimal_time - int(df["time"].max())
            if time_to_optimal_time <= OPERATION_INTERVAL:
                return "SamplingState"
            else:
                return "MediumChange2State"
        else:
            optimal_time = calculate_optimal_time_from_df(df, target_density=PASSAGE_DENSITY, density_operation_name="iPSGetImage", passage_operation_name="iPSPassage")
            if np.isinf(optimal_time):
                return "MediumChange2State"
            time_to_optimal_time = optimal_time - int(df["time"].max())
            if time_to_optimal_time <= OPERATION_INTERVAL:
                return "PassageState"
            else:
                return "MediumChange2State"

        return "ExpireState"
    
class SamplingState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        optimal_time: float = calculate_optimal_time_from_df(df, target_density=SAMPLING_DENSITY, density_operation_name="iPSGetImage", passage_operation_name="iPSPassage")
        print(f"passage:{optimal_time=}")
        if np.isinf(optimal_time):
            raise ValueError("Optimal time is inf")
        return TaskGroup(
            optimal_start_time=int(optimal_time-IPS_PROCESSING_TIME["iPSSampling"]-IPS_PROCESSING_TIME["iPSGetImage"]),
            penalty_type=CyclicalRestPenaltyWithLinear(
                cycle_start_time = 9*60,
                cycle_duration = 60*24,
                rest_time_ranges = [(0, 60*10), (60*16, 60*24)],
                penalty_coefficient = 1000
            ),
            tasks=[
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSGetImage"],
                    interval=1*60,
                    experiment_operation="iPSGetImage"
                ),
                Task(
                    processing_time=IPS_PROCESSING_TIME["iPSSampling"],
                    experiment_operation="iPSSampling"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="iPSWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="iPSWaiting"
                ),
                Task(
                    processing_time=0,
                    interval=24*60*7,
                    experiment_operation="iPSWaiting"
                )
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"


class IPSExperiment(Experiment):
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
                    "operation": ["iPSPassage"]
                }
            )
            
        # Define the initial state and experiment
        super().__init__(
            experiment_name="IPSExperiment",
            states=[
                ExpireState(),
                PassageState(),
                PassageInitialState(),
                MediumChange1State(),
                MediumChange2State(),
                SamplingState()                
            ],
            current_state_name=current_state_name,
            shared_variable_history=shared_variable_history,
            experiment_uuid=str(experiment_uuid)
        )

TIME = UNIX_2024_11_13_00_00_00_IN_JP

def iPSAsetting():
    # Create an instance of the experiment
    i = 0
    shared_variable_history = pl.DataFrame(
        {
            "time": [int(TIME/60 - 120*i)],
            "operation": ["Dummy"]
        }
    )
    new_experiment = IPSExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="PassageInitialState",
        experiment_uuid = f"iPS_Experiment_{i}"
    )
    print(new_experiment)
    return new_experiment

def iPSBsetting():
    # Create an instance of the experiment
    i = 1
    shared_variable_history = pl.DataFrame(
        {
            "time": [int(TIME/60 - 120*i)],
            "operation": ["Dummy"]
        }
    )
    new_experiment = IPSExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="PassageInitialState",
        experiment_uuid = f"iPS_Experiment_{i}"
    )
    return new_experiment

def iPSCsetting():
    # Create an instance of the experiment
    i = 2
    shared_variable_history = pl.DataFrame(
        {
            "time": [int(TIME/60 - 120*i)],
            "operation": ["Dummy"]
        }
    )
    new_experiment = IPSExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="PassageInitialState",
        experiment_uuid = f"iPS_Experiment_{i}"
    )
    return new_experiment

def iPSDsetting():
    # Create an instance of the experiment
    i = 3
    shared_variable_history = pl.DataFrame(
        {
            "time": [int(TIME/60 - 120*i)],
            "operation": ["Dummy"]
        }
    )
    new_experiment = IPSExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="PassageInitialState",
        experiment_uuid = f"iPS_Experiment_{i}"
    )
    return new_experiment

def iPSEsetting():
    # Create an instance of the experiment
    i = 4
    shared_variable_history = pl.DataFrame(
        {
            "time": [int(TIME/60 - 120*i)],
            "operation": ["Dummy"]
        }
    )
    new_experiment = IPSExperiment(
        shared_variable_history=shared_variable_history,
        current_state_name="PassageInitialState",
        experiment_uuid = f"iPS_Experiment_{i}"
    )
    return new_experiment

if __name__ == "__main__":
    # Create an instance of the experiment
    # experiment = IPSExperiment(current_state_name="ExpireState")
    # print(experiment.current_state_name)
    # experiment.show_experiment_with_tooltips()

    print(iPSAsetting())

    
    