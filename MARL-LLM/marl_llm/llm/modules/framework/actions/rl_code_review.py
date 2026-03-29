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

from modules.file import logger
from modules.framework.action import ActionNode
from modules.prompt import (
    CODEREVIEW_PROMPT_TEMPLATE,
)
from modules.framework.parser import *
from modules.utils import root_manager
from datetime import datetime
from tqdm.asyncio import tqdm

class RLCodeReview(ActionNode):
    def __init__(self, next_text, node_name=""):
        super().__init__(next_text, node_name)

    def _build_prompt(self):
        self.prompt = None
        self.context._generated_codes[0] = '\n'.join([f"{i+1}. {goal}" for i, goal in enumerate(self.context._generated_codes[0])])
        self.context._generated_codes[1] = '\n'.join([f"{i+1}. {goal}" for i, goal in enumerate(self.context._generated_codes[1])])
        self.context._generated_codes[2] = '\n'.join(self.context._generated_codes[2])
        self.prompt = CODEREVIEW_PROMPT_TEMPLATE.format(
            basic_skills=self.context._generated_codes[0],
            key_sub_goals=self.context._generated_codes[1],
            generated_code=self.context._generated_codes[2]
        )

    async def _process_response(self, response: str) -> str:
        
        logger.log(f"Output Review Success", "success")

async def process_single_query(i, semaphore):
    """
    Process a single mini_batch
    """
    async with semaphore:  # Limit concurrency
        rl_critic = RLCodeReview("")
        await rl_critic.run()

async def ask_llm_concurrent(args):
    """
    Main function to execute concurrent tasks
    """
    # Create semaphore
    semaphore = asyncio.Semaphore(30)

    # Create progress bar
    task_num = 1
    progress_bar = tqdm(total=task_num, desc="Processing batches", unit="batch")

    # Create all tasks
    tasks = [
        process_single_query(i, semaphore)
        for i in range(task_num)
    ]

    # Use asyncio.as_completed to handle tasks
    for idx, task_future in enumerate(asyncio.as_completed(tasks)):
        result = await task_future
        progress_bar.update(1)

    # Complete progress bar display
    progress_bar.close()


if __name__ == "__main__":
    import asyncio

    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    root_manager.update_root(f"./llm/test/{curr_run}")

    asyncio.run(ask_llm_concurrent(None))

