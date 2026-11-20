#!/usr/bin/env bash
# scripts/dev/cleanup.sh [--apply] [--idle H] [--build-idle H] [--archive-before YYYY-MM-DD]: keep the repo, its worktrees and the build
# dirs from piling up (team rule 2026-11-20: merge the same day, delete merged branches / worktrees; devspeed runs this daily).
# Dry run by default: prints what it would do. --apply does it. Run it from any worktree of the repo: `bash scripts/dev/cleanup.sh`.
#   1. remote branches merged into origin/master (tip an ancestor), last commit older than --idle hours (default 6): deleted with
#      --force-with-lease on the tip they were checked at; the local branch of the same name too when no worktree has it checked out
#   2. worktrees (not the main one) whose HEAD is merged into origin/master, with no tracked change, no untracked file outside build*,
#      no process holding a cwd / exe / open file in them, no symlink into them from /vesuvius/ufsm or ~ (depth 4), and no file but build
#      output newer than --idle hours: `git worktree remove --force`, then their local branch
#   3. build dirs (build*/ with a CMakeCache.txt or build.ninja) of the remaining worktrees whose HEAD is merged (all worktrees with
#      --build-unmerged), untouched (.ninja_log) for --build-idle hours (default 48) and not in use: deleted (a rebuild takes ~4 min)
#   4. unmerged remote branches: listed with age, commit count, subject and files; with --archive-before DATE, those whose last
#      commit is older than DATE and whose worktree (if any) is idle and unused are tagged archive/<branch> (pushed) and deleted with a
#      lease (lead 2026-11-20: approved for parked code branches from before 11-19)
# Every deletion is one command on a literal path or ref, printed before it runs. Exit 0.
set -u
APPLY=0; IDLE=6; BIDLE=48; ARCH=; BUNM=0
while [ $# -gt 0 ]; do case $1 in
  --apply) APPLY=1;; --build-unmerged) BUNM=1;; --idle) IDLE=$2; shift;; --build-idle) BIDLE=$2; shift;; --archive-before) ARCH=$2; shift;;
  *) sed -n '2,17p' "$0"; exit 2;; esac; shift; done
R=$(git rev-parse --path-format=absolute --git-common-dir 2> /dev/null) || { echo "cleanup: not in a git repo" >&2; exit 2; }
cd "${R%/.git}" || exit 2
MAIN=$(pwd)
git fetch -q --prune origin || { echo "cleanup: fetch failed" >&2; exit 2; }
NOW=$(date +%s); ACT="dry run"; [ $APPLY = 1 ] && ACT=APPLY
do_() { echo "  + $*"; [ $APPLY = 1 ] && "$@"; return 0; }
merged() { git merge-base --is-ancestor "$1" origin/master 2> /dev/null; }

# what running processes use: cwd, exe and open files of every process (one find over /proc)
USED=$(find /proc/[0-9]*/cwd /proc/[0-9]*/exe /proc/[0-9]*/fd -maxdepth 1 -printf '%l\n' 2> /dev/null | grep '^/' | sort -u)
inuse() { grep -qF "$1/" <<< "$USED"; }
# symlinks pointing into a dir (targets resolved), from the data tree and ~
LINKS=$(find /vesuvius/ufsm "$HOME" -maxdepth 4 -type l -printf '%p\t%l\n' 2> /dev/null)
linkedinto() { awk -F'\t' -v d="$1/" 'index($2, d) == 1 {print $1; exit}' <<< "$LINKS"; }
# a file other than build output newer than H hours under worktree $1
recent() { [ -n "$(find "$1" -path "$1/build*" -prune -o -path "$1/.git" -prune -o -type f -newermt "-$2 hours" -print -quit 2> /dev/null)" ]; }

n_rb0=$(git branch -r | grep -vc 'HEAD'); n_wt0=$(git worktree list | wc -l)
echo "cleanup ($ACT, $(date '+%F %T')): $n_rb0 remote branches, $n_wt0 worktrees, $(df -h / | awk 'NR == 2 {print $4}') free on /"

# worktrees: path, head, branch
declare -A WTB   # branch -> worktree
mapfile -t WT < <(git worktree list --porcelain | awk '/^worktree / {w = $2} /^HEAD / {h = $2} /^branch / {b = $2; sub("refs/heads/", "", b)} /^detached/ {b = "-"} /^$/ {print w "\t" h "\t" b; b = ""}')
for l in "${WT[@]}"; do IFS=$'\t' read -r w h b <<< "$l"; [ "$b" != "-" ] && WTB[$b]=$w; done

echo "== 1. merged remote branches (idle > ${IDLE} h)"
for rb in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin); do
  b=${rb#origin/}; case $b in HEAD|master|origin) continue;; esac
  sha=$(git rev-parse "$rb"); merged "$sha" || continue
  age=$(( (NOW - $(git log -1 --format=%ct "$sha")) / 3600 )); [ $age -lt $IDLE ] && { echo "  keep $b (merged, last commit ${age} h ago)"; continue; }
  do_ git push -q origin --force-with-lease="$b:$sha" --delete "$b"
  if [ -z "${WTB[$b]:-}" ] && git show-ref -q --verify "refs/heads/$b"; then merged "refs/heads/$b" && do_ git branch -q -D "$b"; fi
done

echo "== 2. merged, idle worktrees"
KEEPWT=()
for l in "${WT[@]}"; do
  IFS=$'\t' read -r w h b <<< "$l"; [ "$w" = "$MAIN" ] && continue
  why=
  merged "$h" || why="not merged"
  [ -z "$why" ] && [ -n "$(git -C "$w" status --porcelain --untracked-files=no 2> /dev/null)" ] && why="uncommitted changes"
  [ -z "$why" ] && [ -n "$(git -C "$w" status --porcelain 2> /dev/null | grep '^??' | grep -v ' build')" ] && why="untracked files"
  [ -z "$why" ] && inuse "$w" && why="in use by a process"
  [ -z "$why" ] && { s=$(linkedinto "$w"); [ -n "$s" ] && why="symlink into it: $s"; }
  [ -z "$why" ] && recent "$w" "$IDLE" && why="files changed < ${IDLE} h"
  if [ -n "$why" ]; then KEEPWT+=("$w"); printf '  keep %-30s %-26s %s\n' "${w##*/}" "${b}" "$why"; continue; fi
  echo "  remove ${w} (${b}, merged, idle)"
  do_ git worktree remove --force "$w"
  [ "$b" != "-" ] && git show-ref -q --verify "refs/heads/$b" && do_ git branch -q -D "$b"
done
[ $APPLY = 1 ] && git worktree prune

echo "== 3. idle build dirs of the remaining worktrees (> ${BIDLE} h)"
for w in "${KEEPWT[@]}"; do
  [ -d "$w" ] || continue
  [ $BUNM = 0 ] && ! merged "$(git -C "$w" rev-parse HEAD)" && continue
  for d in "$w"/build*/; do
    d=${d%/}; [ -f "$d/CMakeCache.txt" ] || [ -f "$d/build.ninja" ] || continue
    t=$(stat -c %Y "$d/.ninja_log" 2> /dev/null || stat -c %Y "$d"); age=$(( (NOW - t) / 3600 ))
    [ $age -lt $BIDLE ] && continue
    inuse "$d" && { echo "  keep $d (in use)"; continue; }
    s=$(linkedinto "$d"); [ -n "$s" ] && { echo "  keep $d (symlink into it: $s)"; continue; }
    echo "  $(du -sh "$d" 2> /dev/null | cut -f1) $d (idle ${age} h)"
    do_ rm -rf -- "$d"
  done
done

echo "== 4. unmerged remote branches"
for rb in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin); do
  b=${rb#origin/}; case $b in HEAD|master|origin) continue;; esac
  sha=$(git rev-parse "$rb"); merged "$sha" && continue
  d=$(git log -1 --format=%cd --date=format:%F "$sha")
  printf '  %-30s %s %3s commits  %-60.60s  wt %s\n' "$b" "$d" "$(git rev-list --count "origin/master..$sha")" "$(git log -1 --format=%s "$sha")" "${WTB[$b]:--}"
  if [ -n "$ARCH" ] && [[ "$d" < "$ARCH" ]]; then
    w=${WTB[$b]:-}
    if [ -n "$w" ] && { inuse "$w" || recent "$w" 24; }; then echo "    (not archived: its worktree is in use / recent)"; continue; fi
    do_ git tag -f "archive/$b" "$sha"
    do_ git push -q origin "refs/tags/archive/$b"
    do_ git push -q origin --force-with-lease="$b:$sha" --delete "$b"
    [ -n "$w" ] && [ -z "$(git -C "$w" status --porcelain --untracked-files=no 2> /dev/null)" ] && do_ git worktree remove --force "$w"
    git show-ref -q --verify "refs/heads/$b" && do_ git branch -q -D "$b"
  fi
done

[ $APPLY = 1 ] && git fetch -q --prune origin
echo "cleanup ($ACT) done: $(git branch -r | grep -vc 'HEAD') remote branches, $(git worktree list | wc -l) worktrees, $(df -h / | awk 'NR == 2 {print $4}') free on /"
exit 0
