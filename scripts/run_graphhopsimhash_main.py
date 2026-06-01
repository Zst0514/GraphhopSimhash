#!/usr/bin/env python3
"""Launch GraphhopSimhash-main as the GraphhopSimhash package."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def main() -> None:
    repo_dir = pathlib.Path(__file__).resolve().parents[1]
    init_path = repo_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "GraphhopSimhash",
        init_path,
        submodule_search_locations=[str(repo_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create package spec from {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["GraphhopSimhash"] = module
    spec.loader.exec_module(module)

    from GraphhopSimhash import cli

    cli.main()


if __name__ == "__main__":
    main()
