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
from modules.framework.code_error import *
from modules.framework.action import BaseNode


class Handler(ABC):
    def __init__(self):
        self._logger = logger
        self._successor = None
        self._next_action = None

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def successor(self):
        return self._successor

    @successor.setter
    def successor(self, value):
        self._successor = value

    @property
    def next_action(self):
        return self._next_action

    @next_action.setter
    def next_action(self, value: BaseNode):
        if isinstance(value, BaseNode):
            self._next_action = value
        else:
            raise TypeError("type of action must be BaseNode")

    @abstractmethod
    def handle(self, request: CodeError) -> BaseNode:
        pass

    def display(self, visited):
        content = f"\t\t{str(self)} -->|skip| {str(self._successor)}\n"
        if self._next_action:
            content += f"\t\t{str(self)} -->|error message| {str(self._next_action)}\n"
            content += self._next_action.flow_content(visited)
        return content + (self._successor.display(visited) if self._successor else "")

    def struct(self):
        return f"\t{str(self)}\n" + (
            self._successor.struct() if self._successor else ""
        )


class BugLevelHandler(Handler):
    def handle(self, request: CodeError) -> BaseNode:
        if isinstance(request, Bug | Bugs):
            self._logger.log("Handled by BugLevelHandler")
            self._next_action.setup(request)
            return self._next_action
        elif self._successor:
            return self._successor.handle(request)


class CriticLevelHandler(Handler):
    def handle(self, request: CodeError) -> BaseNode:
        if isinstance(request, CriticNotSatisfied):
            self._logger.log("Handled by CriticLevelHandler")
            return self._next_action
        elif self._successor:
            return self._successor.handle(request)


class FeedbackHandler(Handler):
    def handle(self, request: CodeError) -> BaseNode:
        if isinstance(request, Feedback):
            self._logger.log("Handled by HumanFeedbackHandler")
            self._next_action.feedback = request.feedback
            return self._next_action
        elif self._successor:
            return self._successor.handle(request)


if __name__ == "__main__":
    # h1 = BugLevelHandler()
    # h2 = CriticLevelHandler()
    # h3 = HumanFeedbackHandler()
    #
    # h2.successor = h1
    # h1.successor = h3
    #
    # handle_pipeline = h2
    #
    # e1 = Bug()
    # e2 = CriticNotSatisfied()
    # e3 = Feedback()
    #
    # handle_pipeline.display()
    #
    # errors = [e1, e2, e3]
    # for error in errors:
    #     handle_pipeline.handle(error)
    pass
