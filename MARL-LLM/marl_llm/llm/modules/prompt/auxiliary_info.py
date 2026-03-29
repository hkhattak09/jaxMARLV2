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

auxiliary_info = {
    "assembly": {
        "COT": [
            "5. Consider how to determinate the inner position of an unoccupied area."
        ],
        "Notes": [
            "- The robot swarm assembling into a specific shape is not a constraint, but a goal.",
        ]
    }
}

def get_auxiliary_info(task_name: str | list = None) -> list[str]:
    auxi_info = auxiliary_info[task_name]
    cot = connect_info(auxi_info['COT'])
    notes = connect_info(auxi_info['Notes'])
    return cot, notes


def connect_info(infos: str | list = None) -> str:
    if len(infos) > 1:
        connected_info = ''
        for info in infos:
            connected_info += info
            connected_info += '\n'
        return connected_info
    elif len(infos) == 1:
        return infos[0]
    else:
        return ''
    
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(script_dir, "../../config/", "experiment_config.yaml")
with open(yaml_file_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)
task_name = data["arguments"]["--run_experiment_name"]["default"][0]

COT, NOTES = get_auxiliary_info(task_name)
