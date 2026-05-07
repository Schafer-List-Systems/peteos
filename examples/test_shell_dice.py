#!/usr/bin/env python3
"""Test shell example with dice roll - runs for fixed time."""

import subprocess
import sys
import time

def main():
    cmd = [
        sys.executable,
        "examples/shell_example.py"
    ]

    # Run with a timeout, sending commands
    input_lines = [
        "/new test\n",
        "roll a dice\n",
        "/quit\n"
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Send all input at once
    full_input = "".join(input_lines)
    stdout, _ = proc.communicate(input=full_input, timeout=35)

    # Filter debug logs and show relevant output
    for line in stdout.split("\n"):
        if "httpcore" in line or "DEBUG" in line:
            continue
        print(line)

if __name__ == "__main__":
    main()