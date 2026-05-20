import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
ONEFORALL_DIR = os.path.join(REPO_ROOT, "OneForAll")

def ensure_repo_paths():
    for import_path in (REPO_ROOT, ONEFORALL_DIR, os.getcwd()):
        if import_path and import_path not in sys.path:
            sys.path.insert(0, import_path)
