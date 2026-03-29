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

import re
import subprocess

from modules.file import logger


class GrammarParser:
    def check_code_errors(self, file_path: str):
        errors = self._run_pylint_check(file_path)
        if errors:
            for e in errors:
                error_function_name, _ = self._find_function_name_from_error(
                    file_path=file_path, error_line=e["line"]
                )
                logger.log(
                    f"{error_function_name}: {e['error_message']}", level="error"
                )
                e["function_name"] = error_function_name
        return errors

    def _run_pylint_check(self, file_path: str):
        command = [
            "pylint",
            # '--disable=W,C,I,R --enable=E,W0612',
            "--disable=W,C,I,R ",
            file_path,
        ]

        try:
            process = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            result = process.stdout + process.stderr

            pattern = re.compile(r"(.*?):(\d+):(\d+): (\w+): (.*) \((.*)\)")
            matches = pattern.findall(result)

            errors = []
            for match in matches:
                file_path, line, column, error_code, error_message, _ = match
                errors.append(
                    {
                        "file_path": file_path,
                        "line": int(line),
                        "column": int(column),
                        "error_code": error_code,
                        "error_message": error_message,
                    }
                )

            return errors
        except Exception as e:
            logger.log(f"Error occurred when check grammar: {e}", level="error")
            raise Exception(f"Error occurred when check grammar: {e}")

    def _find_function_name_from_error(self, file_path, error_line):
        with open(file_path, "r") as file:
            lines = file.readlines()
            error_code_line = lines[error_line - 1].strip()
            for i in range(error_line - 2, -1, -1):
                if lines[i].strip().startswith("def "):
                    function_name = lines[i].strip().split("(")[0].replace("def ", "")
                    return function_name, error_code_line
        return None, error_code_line
