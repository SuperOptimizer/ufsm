#!/usr/bin/env bash
# up.sh "<lftp commands>": runs lftp against the dl.ash2txt.org sftp upload account; credentials read from /root/.ash2txt into a 600
# command file that is deleted afterwards (never printed, never on a command line)
set -u
val() { sed -n "s/^$1:[[:space:]]*//p" /root/.ash2txt | head -1; }
esc() { printf "%s" "$1" | sed -e "s/\\\\/\\\\\\\\/g" -e "s/\"/\\\\\"/g"; }
F=$(mktemp /root/.lftp.XXXXXX); chmod 600 "$F"
{ printf "set sftp:auto-confirm yes\nset net:max-retries 8\nset net:reconnect-interval-base 10\nset net:timeout 60\nset cmd:fail-exit yes\n"
  printf "open -p %s sftp://%s\n" "$(val port)" "$(val host)"
  printf "user \"%s\" \"%s\"\n" "$(esc "$(val user)")" "$(esc "$(val pass)")"
  printf "%s\n" "$1"; } > "$F"
lftp -f "$F"; rc=$?; rm -f "$F"; exit $rc
