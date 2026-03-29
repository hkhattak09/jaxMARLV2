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


class TextParseError(Exception):
    pass


class CodeParseError(Exception):
    pass


class GrammarError(Exception):
    def __init__(self, message, grammar_error) -> None:
        super().__init__(message)
        self._grammar_error = grammar_error
