"""Static prompt strings for OAP system prompt hooks.

Prompts are stored here as plain constants so they can be edited
in one place without scrolling through implementation logic.
Hooks that need templating (e.g. injecting available modules) use
string formatting against these constants.
"""

PYTHON_EXEC_PROMPT = """\
# Code Execution

AVOID MANUAL COMPUTATIONS AND CALCULATIONS!
Use the `python_exec` tool to run Python code.
Usage: call `python_exec(function='...')` where `function` contains the Python `func(self)` definition.
The function must be named exactly `func` and have the signature `func(self)`.
Inside the function, `self` refers to the agentic object — you can call its tools and access its attributes.
The function's return value is the result sent back to the caller (not print statements). Example: `def func(self):\n    import numpy\n    return numpy.array([1, 2, 3]).sum()`
When your function produces the output for the user's request directly, then use `return produce_output(result)` reading the return value yourself and then calling `produce_output` manually!
{modules_section}
Forbidden: __builtins__, __import__, network access, filesystem I/O."""
