import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
ONEFORALL_DIR = os.path.join(REPO_ROOT, "OneForAll")
MODEL_ROOT = os.environ.get("GRAPHHOP_MODEL_ROOT", os.path.join(REPO_ROOT, "models"))

def ensure_repo_paths():
    for import_path in (REPO_ROOT, ONEFORALL_DIR, os.getcwd()):
        if import_path and import_path not in sys.path:
            sys.path.insert(0, import_path)


def resolve_repo_path(path):
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def resolve_model_path(path, env_var=None):
    if env_var and os.environ.get(env_var):
        return os.path.expandvars(os.path.expanduser(os.environ[env_var]))

    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return path

    if path.startswith("models/"):
        return os.path.join(MODEL_ROOT, path[len("models/") :])
    if path.startswith("./") or path.startswith("../"):
        return os.path.abspath(os.path.join(REPO_ROOT, path))

    # HuggingFace/modelscope repo IDs also contain slashes, so keep non-local
    # paths unchanged unless the caller explicitly makes them relative.
    return path
