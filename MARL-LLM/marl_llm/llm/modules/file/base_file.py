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

from abc import abstractmethod


class BaseFile:
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self, content):
        pass

    @property
    def message(self):
        pass

    @message.setter
    def message(self, value):
        pass
