#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _load_repo_package():
    repo_dir = Path(__file__).resolve().parents[1]
    repo_parent = repo_dir.parent
    cleaned_sys_path = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            cleaned_sys_path.append(entry)
            continue
        if resolved == repo_dir:
            continue
        cleaned_sys_path.append(entry)
    sys.path[:] = cleaned_sys_path
    if str(repo_parent) not in sys.path:
        sys.path.insert(0, str(repo_parent))
    for top_level_name in ("models", "data"):
        if top_level_name not in sys.modules:
            sys.modules[top_level_name] = importlib.import_module(top_level_name)
    init_path = repo_dir / "__init__.py"
    module_name = "graphhopsimhash_main_pkg"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(repo_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load package spec from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module_name


def main():
    module_name = _load_repo_package()
    cli_module = importlib.import_module(f"{module_name}.cli")
    cli_module.main()


if __name__ == "__main__":
    main()
