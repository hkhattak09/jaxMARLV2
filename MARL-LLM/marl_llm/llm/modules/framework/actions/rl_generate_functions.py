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

import time

from modules.file import logger
from modules.framework.action import ActionNode, ActionLinkedList
from modules.utils import root_manager
from datetime import datetime
from tqdm.asyncio import tqdm

class RLGenerateFunctions(ActionNode):
    def __init__(self, next_text: str = "", node_name: str = "", run_mode="layer"):
        from modules.framework.actions import (
            RLGeneration,
            RLCodeReview,
        )

        super().__init__(next_text, node_name)
        self._cost_time = 0
        logger.log(f"Function generation begin ...", "warning")
        design_functions = RLGeneration("")
        code_review = RLCodeReview("")

        # link actions
        self._actions = ActionLinkedList("Generate Functions", design_functions)
        self._actions.add(code_review)

    def _build_prompt(self):
        pass

    def _process_response(self, response: str) -> str:
        pass

    async def _run(self) -> str:
        start_time = time.time()
        finish = False
        while not finish:
            time.sleep(1)
            await self._actions.run_internal_actions()
            finish = True

        end_time = time.time()
        self._cost_time = end_time - start_time
        logger.log(
            f"Generate functions and review cost time: {self._cost_time}",
            "warning",
        )

async def process_single_query(i, semaphore):
    """
    Process a single mini_batch
    """
    async with semaphore:  # Limit concurrency
        rl_critic = RLGenerateFunctions("")
        await rl_critic.run()

async def ask_llm_concurrent(args):
    """
    Main function to execute concurrent tasks
    """
    # Create semaphore
    semaphore = asyncio.Semaphore(30)

    # Create progress bar
    task_num = 20
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

    # generate_function = RLGenerateFunctions("")
    # asyncio.run(generate_function.run())

    asyncio.run(ask_llm_concurrent(None))
