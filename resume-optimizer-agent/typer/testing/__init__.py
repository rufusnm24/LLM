"""Testing helper for the lightweight Typer shim."""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import List


@dataclass
class Result:
    exit_code: int
    stdout: str


class CliRunner:
    def invoke(self, app, args: List[str]):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                app.invoke(args)
                exit_code = 0
            except SystemExit as exc:  # Typer raises SystemExit to exit
                try:
                    exit_code = int(exc.code or 0)
                except (TypeError, ValueError):
                    buffer.write(str(exc))
                    exit_code = 1
            except Exception as exc:  # pragma: no cover - debugging aid
                buffer.write(str(exc))
                exit_code = 1
        return Result(exit_code=exit_code, stdout=buffer.getvalue())
