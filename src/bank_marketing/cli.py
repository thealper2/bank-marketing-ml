"""Single entry point: ``bank-ml <command> [options]``.
Each command lives in its own module and exposes ``main(argv) -> int``.
Modules are imported lazily, so ``bank-ml --help`` stays fast and a command
whose module does not exist yet fails with a clear message.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable


from bank_marketing import __version__

COMMANDS: dict[str, tuple[str, str]] = {
    "data": ("bank_marketing.data", "download / prepare the dataset"),
    "eda": ("bank_marketing.eda", "data quality report and EDA figures"),
    "train": ("bank_marketing.train", "compare, tune and save a model"),
    "evaluate": ("bank_marketing.evaluate", "final evaluation on the test split"),
    "predict": ("bank_marketing.predict", "batch or single-record predictions"),
}

def _usage() -> str:
    lines = ["usage: bank-ml <command> [options]", "", "commands:"]
    lines += [f" {name:<10} {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += ["", "Run `bank-ml <command> --help` for command options."]
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if args[0] == "--version":
        print(f"bank-ml {__version__}")
        return 0
    
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"Unknown command '{command}'.\n\n{_usage()}", file=sys.stderr)
        return 2
    module_name = COMMANDS[command][0]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise # a real missing dependency, not an unimplemented command
        print(
            f"Command '{command}' is not implemented yet ({module_name}).",
            file=sys.stderr,
        )
        return 2
    entry: Callable[[list[str]], int] = module.main
    return entry(rest)

if __name__ == "__main__":
    raise SystemExit(main())
