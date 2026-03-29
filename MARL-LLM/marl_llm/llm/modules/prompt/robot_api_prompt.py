"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

import os
import traceback
import yaml
from modules.framework.parser import CodeParser

robot_api_prompt = """
def get_neighbor_id_list(id):
    '''
    Description: Get the IDs of the neigboring robots of the robot.
    Returns:
    - list: The IDs of the neigboring robots. ID is a unique identifier for each robot in the environment.
    '''

def get_robot_position_and_velocity(id):
    '''
    Description: Get the robot position and velocity for the robot.
    Input:
    - int: The ID of the robot
    Returns:
    - numpy.ndarray, numpy.ndarray: The current position and velocity of the robot.
    '''

def get_prey_position_and_velocity():
    '''
    Description: Get the prey position and velocity.
    Returns:
    - numpy.ndarray, numpy.ndarray: The current position and velocity of the prey.
    ''' 

def get_obstacle_positions(id):
    '''
    Description: Get obstacle positions around the robot.
    Returns:
    - numpy.ndarray: The obstacle positions around the robot.
    ''' 

def get_unoccupied_cells_position(id):
    '''
    Description: Get the positions of the sensed unoccupied area within the robot sensing radius.
    Input:
    - int: The ID of the robot
    Returns:
    - numpy.ndarray: The current positions of the sensed unoccupied area.
    '''

def get_target_cell_position(id):
    '''
    Description: Get the position of the target cell for the robot.
    Input:
    - int: The ID of the robot
    Returns:
    - numpy.ndarray, numpy.ndarray: The current position of the target cell of the robot.
    '''

def is_within_target_region(id):
    '''
    Description: Determine whether the robot with the specified ID is within the target area.
    Input:
    - int: The ID of the robot
    Returns:
    - bool: "true" indicates the robot is within the shape, while "false" indicates it is not.
    '''
""".strip()

class RobotApi:
    def __init__(self, content):
        self.content = content
        code_obj = CodeParser()
        code_obj.parse_code(self.content)
        self.apis = code_obj.function_defs

        self.base_apis = [
            "get_neighbor_id_list",
            "get_robot_position_and_velocity",
        ]

        # Updated API scope mapping to support multiple scopes per API
        self.api_scope = {
            "get_robot_position_and_velocity": ["local"],
            "get_prey_position_and_velocity": ["local"],
            "get_neighbor_id_list": ["local"],
            "get_obstacle_positions": ["local"],
            "get_unoccupied_cells_position": ["local"],
            "get_target_cell_position": ["local"],
            "is_within_target_region": ["local"],
        }

        self.base_prompt = [self.apis[api] for api in self.base_apis]
        self.task_apis = {
            "bridging": ["get_obstacle_positions"],
            "encircling": ["get_prey_position_and_velocity", "get_obstacle_positions"],
            "flocking": ["get_obstacle_positions"],
            "assembly": ["get_unoccupied_cells_position", "get_target_cell_position", "is_within_target_region"],
        }

    def get_api_prompt(
        self, task_name: str = None, scope: str = None, only_names: bool = False
    ) -> str | list:
        """
        Get the prompt of the robot API.
        Parameters:
            task_name: str, the name of the task to get the prompt.
            scope: str, optional, 'global' or 'local' to filter APIs by scope.
                   Default is None, which means to get all the prompts.
            only_names: bool, optional, if True, only return the API names instead of full text.
                        Default is False, which returns the full text.
        Returns:
            str or list: The prompt of the robot API.
        """
        if task_name is None:
            all_apis = self.apis.keys()
            if scope:
                filtered_apis = [
                    api for api in all_apis if scope in self.api_scope.get(api, [])
                ]
            else:
                filtered_apis = all_apis

            if only_names:
                return "\n".join(filtered_apis)
            return "\n".join([self.apis[api] for api in filtered_apis])

        try:
            task_prompt = self.base_prompt.copy()
            specific_apis = [
                self.apis[api] for api in self.task_apis.get(task_name, [])
            ]
            task_prompt.extend(specific_apis)

            if scope:
                task_prompt = [
                    api
                    for api in task_prompt
                    if scope in self.api_scope.get(self.get_api_name(api), [])
                ]

            if only_names:
                return [self.get_api_name(api) for api in task_prompt]
            return "\n".join(task_prompt)
        except KeyError:
            traceback.print_exc()
            raise SystemExit(
                f"Error in get_api_prompt: Task name '{task_name}' not found. Current existing tasks: {list(self.task_apis.keys())}."
            )
        except Exception as e:
            raise SystemExit(
                f"Error in get_api_prompt: {e}, current existing apis: {self.apis.keys()},"
                f"input task name: {task_name}, expected name: {self.task_apis.get(task_name, [])}"
            )

    def get_api_name(self, api: str) -> str:
        """
        Helper method to retrieve API name from the function definition.
        """
        return next(
            (name for name, content in self.apis.items() if content == api), None
        )


robot_api = RobotApi(content=robot_api_prompt)

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(script_dir, "../../config/", "experiment_config.yaml")
with open(yaml_file_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)
task_name = data["arguments"]["--run_experiment_name"]["default"][0]

# GLOBAL_ROBOT_API = robot_api.get_api_prompt(task_name, scope="global")
ROBOT_APIS = robot_api.get_api_prompt(task_name, scope="local")

global_import_list = robot_api.get_api_prompt(
    task_name, scope="global", only_names=True
)
local_import_list = robot_api.get_api_prompt(task_name, scope="local", only_names=True)
local_import_list = (
    local_import_list.split("\n")
    if isinstance(local_import_list, str)
    else local_import_list
)
local_import_list.append("get_assigned_task")
import_list = global_import_list + local_import_list