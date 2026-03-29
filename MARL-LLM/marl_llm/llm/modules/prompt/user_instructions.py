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
import yaml

tasks = {
    "assembly": """The robot swarm need to assemble into a specific shape that aligns with the shape of the region. In such a shape, robots should avoid
                    collisions, synchronize movements with neighbors to reduce oscillations, and attempt to explore unoccupied cells.""".replace("\n                    ", " ")
}

def get_user_commands(task_name: str | list = None) -> list[str]:
    """
    Description: Get the user commands to be implemented.
    Args:
        task_name: str|list, the name of the task to be implemented (default is None).
        When task_name is None, return all the user commands.
    Returns:
        list[str]: The list of user commands to be implemented.
    """

    if task_name is None:
        return list(tasks.values())
    elif isinstance(task_name, str):
        return [tasks[task_name]]
    elif isinstance(task_name, list):
        return [tasks[task] for task in task_name]

def get_commands_name() -> list[str]:
    """
    Description: Get the names of the user commands to be implemented.
    Returns:
        list[str]: The list of names of user commands to be implemented.
    """
    return list(tasks.keys())

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(script_dir, "../../config/", "experiment_config.yaml")
with open(yaml_file_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)
task_name = data["arguments"]["--run_experiment_name"]["default"][0]

TASK_DES = get_user_commands(task_name)[0]
