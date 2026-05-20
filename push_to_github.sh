#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:Zst0514/GraphhopSimhash.git}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${1:-Update GraphhopSimhash}"

cd "$(dirname "$0")"

echo "[GraphhopSimhash] Working directory: $(pwd)"
echo "[GraphhopSimhash] Remote: ${REPO_URL}"
echo "[GraphhopSimhash] Branch: ${BRANCH}"

if ! command -v git >/dev/null 2>&1; then
  echo "[Error] git is not installed or not in PATH." >&2
  exit 1
fi

if [ ! -d .git ]; then
  echo "[Git] Initializing repository..."
  git init
fi

git branch -M "${BRANCH}"

if git remote get-url origin >/dev/null 2>&1; then
  current_remote="$(git remote get-url origin)"
  if [ "${current_remote}" != "${REPO_URL}" ]; then
    echo "[Git] Updating origin remote:"
    echo "      old: ${current_remote}"
    echo "      new: ${REPO_URL}"
    git remote set-url origin "${REPO_URL}"
  fi
else
  echo "[Git] Adding origin remote..."
  git remote add origin "${REPO_URL}"
fi

if ! git config user.name >/dev/null; then
  echo "[Warn] git user.name is not set. Set it with:"
  echo "       git config --global user.name \"Zst0514\""
fi

if ! git config user.email >/dev/null; then
  echo "[Warn] git user.email is not set. Set it with:"
  echo "       git config --global user.email \"your_email@example.com\""
fi

echo "[Git] Staging files..."
git add -A

if git diff --cached --quiet; then
  echo "[Git] No local changes to commit."
else
  echo "[Git] Committing: ${COMMIT_MSG}"
  git commit -m "${COMMIT_MSG}"
fi

echo "[Git] Checking remote branch..."
if git ls-remote --exit-code --heads origin "${BRANCH}" >/dev/null 2>&1; then
  echo "[Git] Remote ${BRANCH} exists. Pulling with rebase before push..."
  git pull --rebase origin "${BRANCH}" --allow-unrelated-histories
else
  echo "[Git] Remote ${BRANCH} does not exist yet. Skipping pull."
fi

echo "[Git] Pushing to origin/${BRANCH}..."
git push -u origin "${BRANCH}"

echo "[Done] GraphhopSimhash has been pushed to ${REPO_URL} (${BRANCH})."
