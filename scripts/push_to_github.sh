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
  SKIP_REBASE=1          Skip fetch + pull --rebase. Use only when intentional.

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

if git diff --cached --name-only | grep -Ei '^CAM_sim/.*\.pdf$' >/dev/null; then
  echo "[Error] Refusing to commit PDF files under CAM_sim/." >&2
  echo "Remove them first, or keep PDFs outside the git-tracked CAM_sim tree." >&2
  git diff --cached --name-only | grep -Ei '^CAM_sim/.*\.pdf$' >&2
  exit 1
fi

if git diff --cached --quiet; then
  echo "[Git] No staged changes. Pushing existing local commits, if any."
else
  echo "[Git] diff --check"
  git diff --check --cached
  echo "[Git] commit: ${COMMIT_MSG}"
  git commit -m "${COMMIT_MSG}"
fi

if [[ "${SKIP_REBASE:-0}" != "1" ]]; then
  echo "[Git] fetch ${REMOTE} ${BRANCH}"
  git fetch "${REMOTE}" "${BRANCH}"

  if git show-ref --verify --quiet "refs/remotes/${REMOTE}/${BRANCH}"; then
    echo "[Git] pull --rebase --autostash ${REMOTE} ${BRANCH}"
    git pull --rebase --autostash "${REMOTE}" "${BRANCH}"
  else
    echo "[Git] No remote tracking ref ${REMOTE}/${BRANCH}; first push may create it."
  fi
fi

echo "[Git] push ${REMOTE} ${BRANCH}"
git push "${REMOTE}" "${BRANCH}"

echo "[Done] Pushed to ${REMOTE}/${BRANCH}."
