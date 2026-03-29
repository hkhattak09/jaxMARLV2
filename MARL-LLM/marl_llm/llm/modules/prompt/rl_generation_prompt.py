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

GENERATION_PROMPT_TEMPLATE: str = """
## These are the environment description:
{env_des}

## This is the task description:
{task_des}

## Existing robot APIs:
The following APIs are already implemented and can be called directly. You may use some of them.
```python
{api_des}
```

## Role setting:
You are a task analysis assistant. Please analyze the above task and consider how a robot can accomplish it. 
You need to provide two functions: one is a reward function that returns a reward of 1 if the task is completed and 0 otherwise. 
The other is a prior policy function, which may not guarantee task completion but can ensure that the robots consistently possess some basic capabilities.

## The output TEXT format is as follows:
### Reasoning: (reason step by step about how to design reward and policy function)
1. Consider what constraints need to be satisfied to complete the task.
2. Consider what constitutes basic constraints and what constitutes complex constraints.
3. Consider which constraints must be included in the reward function.
4. Consider what basic capabilities the robots should possess.
{auxiliary_cot}
### Code:
```python
def compute_reward(input1, input2, ...):
    '''
    Description: Refine this description in detail.
    Input:
        input1: type, description
        input2: type, description
        ...
    Return:
        type, description
    '''
    function content

def robot_policy(input1, input2, ...):
    '''
    Description: Refine this description in detail.
    Input:
        input1: type, description
        input2: type, description
        ...
    Return:
        type, description
    '''
    function content
```
```json
{{
    "key_task_sub_goal": [sub-goal1, sub-goal2, sub-goal3, ...],
    "basic_capabilities": [capability1, capability2, capability3, ...]
}}
```

## Notes:
- You should reasoning first, and then output the reward and policy function.
- The output must strictly follow the given format.
- The output of the reward function should have a shape of 1 x n, where n represents the number of robots.
{auxiliary_notes}
""".strip()