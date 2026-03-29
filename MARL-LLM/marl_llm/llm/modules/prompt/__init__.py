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

from .rl_generation_prompt import GENERATION_PROMPT_TEMPLATE
from .rl_code_review_prompt import CODEREVIEW_PROMPT_TEMPLATE
from .user_instructions import TASK_DES
from .env_descriptions import ENV_DES
from .robot_api_prompt import ROBOT_APIS
from .auxiliary_info import COT, NOTES

__all__ = [
    "GENERATION_PROMPT_TEMPLATE",
    "CODEREVIEW_PROMPT_TEMPLATE",
    "TASK_DES",
    "ENV_DES",
    "ROBOT_APIS",
    "COT",
    "NOTES"
]
