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

import argparse
import pickle
from modules.file import File
from modules.framework.code import FunctionTree

from .context import Context


class WorkflowContext(Context):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()

        return cls._instance

    def _initialize(self):
        self.user_command = File(name="command.md")
        self.feedbacks = []
        self.run_code = File(name="run.py")
        self.args = argparse.Namespace()
        self._generated_codes = []

        self.scoop = "global"
        self.vlm = False

    def save_to_file(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self._instance, file)

    def load_from_file(self, file_path):
        with open(file_path, "rb") as file:
            self._instance = pickle.load(file)

    def set_root_for_files(self, root_value):
        for file_attr in vars(self).values():
            if isinstance(file_attr, File):
                file_attr.root = root_value
            if isinstance(file_attr, FunctionTree):
                file_attr.file.root = root_value

    @property
    def command(self):
        return self._instance.user_command.message

    @command.setter
    def command(self, value):
        self._instance.user_command.message = value

