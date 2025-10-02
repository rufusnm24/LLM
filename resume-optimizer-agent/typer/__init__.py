"""Minimal Typer-compatible interface for constrained environments."""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Option:
    default: Any = ...
    help: str | None = None
    exists: bool = False


class Typer:
    def __init__(self, *, help: str | None = None) -> None:
        self.help = help or ""
        self._commands: Dict[str, Callable[..., Any]] = {}

    def command(self, name: Optional[str] = None, *, help: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            command_name = (name or func.__name__).replace("-", "_")
            self._commands[command_name] = func
            return func

        return decorator

    def _dispatch(self, argv: List[str]) -> Any:
        if not argv:
            raise SystemExit(0)
        command_name = argv[0].replace("-", "_")
        func = self._commands.get(command_name)
        if not func:
            raise SystemExit(f"Unknown command {argv[0]}")
        kwargs = _parse_arguments(func, argv[1:])
        return func(**kwargs)

    def invoke(self, argv: List[str]) -> Any:
        return self._dispatch(argv)

    def __call__(self, argv: Optional[List[str]] = None) -> Any:
        argv = list(sys.argv[1:] if argv is None else argv)
        return self._dispatch(argv)


def _parse_arguments(func: Callable[..., Any], argv: List[str]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    parameters = signature.parameters
    result: Dict[str, Any] = {}

    for name, param in parameters.items():
        if isinstance(param.default, Option):
            default_value = param.default.default
            result[name] = default_value if default_value is not ... else None
        elif param.default is not inspect._empty:
            result[name] = param.default
        else:
            result[name] = None

    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if not token.startswith("--"):
            raise SystemExit(f"Unexpected argument {token}")
        key = token[2:].replace("-", "_")
        if key not in parameters:
            raise SystemExit(f"Unknown option {token}")
        param = parameters[key]
        annotation = _resolve_annotation(param.annotation)
        if annotation in (bool, Optional[bool]) or (
            isinstance(param.default, Option) and isinstance(param.default.default, bool)
        ):
            value = True
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                next_token = argv[idx + 1]
                value = next_token.lower() not in {"false", "0", "no"}
                idx += 1
            result[key] = value
        elif annotation in (List[str], list):
            values: List[str] = [] if result[key] is None else list(result[key])
            while idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                values.append(argv[idx + 1])
                idx += 1
            result[key] = values
        else:
            if idx + 1 >= len(argv):
                raise SystemExit(f"Option {token} requires a value")
            value_token = argv[idx + 1]
            idx += 1
            result[key] = _convert_value(annotation, value_token)
        idx += 1

    for name, param in parameters.items():
        if result[name] is None:
            if isinstance(param.default, Option) and param.default.default is not ...:
                result[name] = param.default.default
            elif param.default is inspect._empty:
                raise SystemExit(f"Missing option --{name.replace('_', '-')}")
    return result


def _convert_value(annotation: Any, token: str) -> Any:
    target = _resolve_annotation(annotation)
    if getattr(annotation, "__origin__", None) is Optional:
        target = annotation.__args__[0]
    if target in (str, inspect._empty):
        return token
    if isinstance(target, type) and issubclass(target, Path):
        return target(token)
    if target is float:
        return float(token)
    if target is int:
        return int(token)
    return token


def _resolve_annotation(annotation: Any) -> Any:
    if isinstance(annotation, str):
        lowered = annotation.lower()
        if lowered == "path":
            return Path
        if lowered == "bool":
            return bool
        if lowered == "float":
            return float
        if lowered == "int":
            return int
        if lowered == "list[str]":
            return list
    return annotation


__all__ = ["Typer", "Option"]
