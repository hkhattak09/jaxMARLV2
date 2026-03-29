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

from .function_node import FunctionNode


class FunctionLayer:
    def __init__(self, function_set: set = None):
        if function_set is None:
            function_set = set()
        self._layer: set[FunctionNode] = function_set
        self._next: FunctionLayer = None
        self._index = 0

    @property
    def next(self):
        return self._next

    @property
    def functions(self):
        return list(self._layer)

    @property
    def set_callers(self):
        result = set()
        for function_node in self._layer:
            result |= function_node.callers
        return result

    @next.setter
    def next(self, value: "FunctionLayer"):
        self._next = value

    def __len__(self):
        return len(self.functions)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._layer):
            self._index = 0
            raise StopIteration
        value = self.functions[self._index]
        self._index += 1
        return value

    def add_function(self, function: FunctionNode):
        self._layer.add(function)
