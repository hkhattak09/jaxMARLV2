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

import ast

from modules.framework.error import CodeParseError


class CodeParser(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self._code_str = None
        self._imports = set()
        self._function_dict: dict[str, str] = {}
        self._function_defs: dict[str, str] = {}
        self._function_lines: dict[str, int] = {}
        self._comment_lines: dict[str, int] = {}

    @property
    def imports(self):
        return self._imports

    @property
    def function_dict(self):
        return self._function_dict

    @property
    def function_contents(self):
        return self._function_dict.values()

    @property
    def function_names(self):
        return self._function_dict.keys()

    @property
    def function_defs(self):
        return self._function_defs

    @property
    def function_lines(self):
        return self._function_lines

    @property
    def comment_lines(self):
        return self._comment_lines

    def parse_code(self, code_str):
        self._code_str = code_str
        tree = ast.parse(code_str)
        self.visit(tree)

    # visit_xxx functions are automatically executed in visit()
    # see details in ast.NodeVisitor
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            import_str = f"import {alias.name}"
            if alias.asname:
                import_str += f" as {alias.asname}"
            self._imports.add(import_str)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            import_str = f"from {module} import {alias.name}"
            if alias.asname:
                import_str += f" as {alias.asname}"
            self._imports.add(import_str)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        def reconstruct_function_definition(function_node: ast.FunctionDef):
            defaults_start_index = len(function_node.args.args) - len(
                function_node.args.defaults
            )

            parameters = [
                ast.unparse(arg)
                + (
                    f"={ast.unparse(function_node.args.defaults[i - defaults_start_index])}"
                    if i >= defaults_start_index
                    else ""
                )
                for i, arg in enumerate(function_node.args.args)
            ]

            func_header = f"def {function_node.name}({', '.join(parameters)}):"
            docstring = ast.get_docstring(function_node)
            docstring_part = ""
            if docstring:
                indented_docstring = "\n".join(
                    "    " + line for line in docstring.split("\n")
                )
                docstring_part = f'    """\n{indented_docstring}\n    """\n'
            body_part = ""
            return f"{func_header}\n{docstring_part}{body_part}"

        function_body_with_comments = ast.get_source_segment(self._code_str, node)

        self._function_dict[node.name] = function_body_with_comments.strip()
        self._function_defs[node.name] = reconstruct_function_definition(node)
        start_line = node.lineno
        end_line = node.end_lineno
        self._function_lines[node.name] = end_line - start_line + 1
        code_without_comment = ast.unparse(node).strip()
        line_count = len(code_without_comment.splitlines()) - 1

        self.comment_lines[node.name] = self._function_lines[node.name] - line_count


class SingleFunctionParser(CodeParser):
    def __init__(self):
        super().__init__()

    @property
    def function_name(self):
        return list(self.function_names)[0]

    @property
    def function_definition(self):
        return list(self._function_defs.values())[0]

    def parse_code(self, code_str):
        super().parse_code(code_str)
        self._check_error()

    def _check_error(self):
        if not self._function_dict:
            raise CodeParseError(
                "Failed: No function detected in the response", "error"
            )
            # return ''
        if len(self._function_dict) > 1:
            raise CodeParseError(
                "Failed: More than one function detected in the response"
            )

    def check_function_name(self, desired_function_name):
        function_name = list(self._function_dict.keys())[0]
        if function_name != desired_function_name:
            raise CodeParseError(
                f"Function name mismatch: {function_name} != {desired_function_name}"
            )
        if not function_name:
            raise CodeParseError(f"Failed: No function detected in the response")


if __name__ == "__main__":
    # Example usage
    code = """

    def example_function(param1, param2='default'):
        '''
        aaa
        '''
        # This is a comment
        result = param1 + param2
        return result


    """

    # parser = CodeParser()
    # parser.parse_code(code)
    # print(f"Function Lines: {parser.function_lines}")
    # print(f"Comment Lines: {parser.comment_lines}")
