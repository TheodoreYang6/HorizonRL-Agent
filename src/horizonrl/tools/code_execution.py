"""Safe Python code execution tool for agent use."""

from __future__ import annotations

import asyncio
import io
import sys
import traceback


class CodeExecutionTool:
    """Execute Python code in a sandboxed environment.

    For production use, this should be replaced with a proper sandbox
    (Docker, gVisor, or RestrictedPython).
    """

    name = "code_execution"
    description = "Execute Python code and return stdout/stderr."

    def __init__(self, timeout: float = 30.0, max_output_chars: int = 10000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    async def execute(self, code: str) -> dict[str, str]:
        """Execute code asynchronously with timeout.

        Args:
            code: Python source code to execute.

        Returns:
            Dict with 'stdout', 'stderr', 'success', 'error' keys.
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_sync, code),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return {"stdout": "", "stderr": "", "success": False, "error": "Execution timed out"}

    def _execute_sync(self, code: str) -> dict[str, str]:
        """Synchronous code execution with output capture."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Restricted globals for safety
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "any": any,
                    "all": all,
                    "isinstance": isinstance,
                    "round": round,
                },
            }

            exec(code, safe_globals, {})

            stdout = stdout_capture.getvalue()[: self.max_output_chars]
            stderr = stderr_capture.getvalue()[: self.max_output_chars]

            return {"stdout": stdout, "stderr": stderr, "success": True, "error": None}

        except Exception:
            return {
                "stdout": stdout_capture.getvalue()[: self.max_output_chars],
                "stderr": stderr_capture.getvalue()[: self.max_output_chars],
                "success": False,
                "error": traceback.format_exc()[: self.max_output_chars],
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def __call__(self, code: str) -> dict[str, str]:
        return asyncio.run(self.execute(code))
