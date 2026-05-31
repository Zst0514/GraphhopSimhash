#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./push.sh "commit message"
  ./push.sh -m "commit message"
  bash scripts/push_to_github.sh "commit message"
  bash scripts/push_to_github.sh -m "commit message"

Optional environment variables:
  REMOTE=origin          Git remote to push to.
  BRANCH=<branch-name>   Branch to push. Defaults to current branch.

Examples:
  ./push.sh "docs: update graphbit notes"
  bash scripts/push_to_github.sh "docs: update graphbit notes"
  REMOTE=origin BRANCH=main bash scripts/push_to_github.sh -m "feat: add script"
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-}"
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      shift
      if [[ $# -eq 0 ]]; then
        echo "[Error] Missing commit message after -m/--message." >&2
        exit 1
      fi
      COMMIT_MSG="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${COMMIT_MSG}" ]]; then
        COMMIT_MSG="$1"
      else
        COMMIT_MSG="${COMMIT_MSG} $1"
      fi
      ;;
  esac
  shift
done

if [[ -z "${COMMIT_MSG}" ]]; then
  COMMIT_MSG="update: sync local changes"
fi

cd "${REPO_DIR}"

if [[ ! -d .git ]]; then
  echo "[Error] ${REPO_DIR} is not a git repository." >&2
  exit 1
fi

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
fi

if [[ "${BRANCH}" == "HEAD" ]]; then
  echo "[Error] Detached HEAD. Please checkout a branch before pushing." >&2
  exit 1
fi

if ! git remote get-url "${REMOTE}" >/dev/null 2>&1; then
  echo "[Error] Remote '${REMOTE}' does not exist." >&2
  echo "Available remotes:"
  git remote -v
  exit 1
fi

echo "[Git] repo:   ${REPO_DIR}"
echo "[Git] remote: ${REMOTE} ($(git remote get-url "${REMOTE}"))"
echo "[Git] branch: ${BRANCH}"

git status --short

git add -A

if git diff --cached --quiet; then
  echo "[Git] No staged changes. Pushing existing local commits, if any."
else
  echo "[Git] commit: ${COMMIT_MSG}"
  git commit -m "${COMMIT_MSG}"
fi

echo "[Git] push ${REMOTE} ${BRANCH}"
git push "${REMOTE}" "${BRANCH}"

echo "[Done] Pushed to ${REMOTE}/${BRANCH}."
