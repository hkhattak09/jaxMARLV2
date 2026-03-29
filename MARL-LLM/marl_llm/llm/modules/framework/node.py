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

from abc import ABC


class Node(ABC):
    def __init__(self, name, description):
        self._name = name
        self._description = description
        self._connections: set[Node] = set()

    @property
    def brief(self):
        return f"**{self._name}**: {self._description}"

    @property
    def connections(self):
        return self._connections

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    def connect_to(self, node: "Node"):
        if node not in self._connections:
            self._connections.add(node)
            node.connect_to(self)

    def has_no_connections(self):
        return len(self._connections) == 0
