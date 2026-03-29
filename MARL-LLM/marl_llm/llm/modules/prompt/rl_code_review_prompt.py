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

CODEREVIEW_PROMPT_TEMPLATE: str = """
## Expected basic skills:
The following are the expected basic skills that the policy function should achieve:
{basic_skills}

## Key task sub-goals:
The following are the key sub-goals that the reward function should evaluate:
{key_sub_goals}

## Exsiting policy function and reward function:
```python
{generated_code}
```

## Role setting:
You are a code review assistant tasked with two objectives: 
first, to verify whether the policy function implements the expected basic skills; 
second, to ensure the reward function evaluates all key task sub-goals.

## The output TEXT format is as follows:
### Result: (the output of the code review)
```json
{{
  "review_output": [
    {{
      "is_policy_function_ok": "true/false",
      "description": "Describe why this result is given.“
    }},
    {{
      "is_reward_function_ok": "true/false",
      "description": "Describe why this result is given."
    }}
  ]
}}
```

""".strip()

