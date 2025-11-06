from dataclasses import dataclass
from datetime import datetime
import uuid
import numpy as np
import polars as pl
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import LinearPenalty, LinearWithRangePenalty
from gems_python.one_machine_problem_interval_task.task_info import TaskGroup, Task
from gems_python.one_machine_problem_interval_task.transition_manager import Experiment, State
from dataclasses import dataclass
import polars as pl

from curve import calculate_optimal_time_from_df

# 1731423600 2024_11_13_00_00_00 UNIX

UNIX_2024_11_17_16_00_00_IN_JP = 1731826800

PROCESSING_TIME = {
    "LabwareRefill": 120,
}

class LabwareRefillState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        optimal_time = int(UNIX_2024_11_17_16_00_00_IN_JP//60)
        print(f"passage:{optimal_time=}")
        return TaskGroup(
            optimal_start_time=int(optimal_time),
            penalty_type=LinearPenalty(penalty_coefficient=100000),
            tasks=[
                Task(
                    processing_time=0,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=24*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                # 5
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
                Task(
                    processing_time=PROCESSING_TIME["LabwareRefill"],
                    interval=22*60,
                    experiment_operation="LabwareRefill"
                ),
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        return "LabwareRefillState"
        return "ExpireState"


class LabwareRefillExperiment(Experiment):
    def __init__(self, current_state_name, shared_variable_history=None, experiment_uuid=uuid.uuid4()):
        # Define the experiment using the states

        # Create a shared variable history DataFrame (empty for this example)
        if shared_variable_history is None:
            shared_variable_history = pl.DataFrame(
            )
            
        # Define the initial state and experiment
        super().__init__(
            experiment_name="LabwareRefillExperiment",
            states=[
                LabwareRefillState()              
            ],
            current_state_name=current_state_name,
            shared_variable_history=shared_variable_history,
            experiment_uuid=str(experiment_uuid)
        )

TIME = 60*60*24*365*10

def LabwareRefillAsetting():
    # Create an instance of the experiment
    new_experiment = LabwareRefillExperiment(
        current_state_name="LabwareRefillState",
    )
    return new_experiment
