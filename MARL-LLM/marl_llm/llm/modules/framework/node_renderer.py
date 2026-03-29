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

from abc import ABC, abstractmethod


class NodeRenderer(ABC):
    def set_node(self, node):
        self._node = node

    @abstractmethod
    def flow_content(self, visited: set) -> str:
        # Abstract method for generating flow content
        pass

    @abstractmethod
    def graph_struct(self, level: int) -> str:
        # Abstract method for generating graph structure
        pass


class ActionNodeRenderer(NodeRenderer):
    def flow_content(self, visited: set) -> str:
        # generating flow content in mermaid style
        if not self._node._next or self._node in visited:
            return ""
        visited.add(self._node)
        content = f"\t\t{str(self._node)} -->|{self._node._next_text}| {str(self._node._next)}\n"
        if self._node.error_handler:
            content += (
                f"\t\t{str(self._node)} -->|failed| {str(self._node.error_handler)}\n"
            )
            content += self._node.error_handler.display(visited)

        content += self._node._next.flow_content(visited)
        return content

    def graph_struct(self, level: int) -> str:
        # Method for generating graph structure
        return str(self._node)


class ActionLinkedListRenderer(NodeRenderer):
    def graph_struct(self, level: int) -> str:
        # Method for generating graph structure in mermaid style
        level += 1
        tables = "\t"
        content = f"subgraph {self._node._name}\n"
        node = self._node._head
        while node and node != self._node._tail:
            content += tables * (level) + f"{node.graph_struct(level)}\n"
            node = node._next
        if node == self._node._tail:
            content += tables * (level) + f"{node.graph_struct(level)}\n"

        content += tables * (level - 1) + "end"
        return content

    def flow_content(self, visited: set) -> str:
        return self._node._head.flow_content(visited)


def display_all(node, error_handler):
    graph = node.graph_struct(level=1)
    visited = set()
    res = node.flow_content(visited)
    text = f"""
```mermaid
graph TD;
    Start((Start)) --> {str(node)}
{res}

{graph}

subgraph chain of handlers
{error_handler.struct()}
end
```
    """
    return _clean_graph(text)


def _clean_graph(graph: str):
    lines = set()
    unique_lines = []
    for line in graph.split("\n"):
        content = line.strip()
        if content not in lines or content == "end":
            unique_lines.append(line)
            lines.add(line.strip())

    return "\n".join(unique_lines)
