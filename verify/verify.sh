#!/usr/bin/env bash
# Recompute the README's numbers in languages other than the one that produced
# them, and fail if any of them disagrees.
#
# check_numbers.py already re-derives every quoted figure from the search log,
# but it is Python re-checking Python: the same reading of the same CSV by the
# same arithmetic. These do it again from the raw logs in SQL and in R, and a
# mistake would have to be made identically in all three to survive.
#
# Each is skipped with a message if its toolchain is missing, so a partial
# install still runs the rest. CI has both.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# The SQL has no assertion of its own. It prints one string per figure it
# recomputed, and each of those must appear verbatim in README.md.
# sqlite3 reads stdin, which inside a script is the script itself, so stdin is
# closed here; its CSV output is CRLF, so the carriage return is stripped.
check_sql () {
    local out missing=0 n=0
    out=$(sqlite3 -init verify/search_log.sql :memory: "" < /dev/null | tr -d '\r') || return 1
    [ -n "$out" ] || { echo "sqlite produced no output"; return 1; }
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        n=$((n + 1))
        if grep -qF -- "$line" README.md; then
            printf '  ok   %s\n' "$line"
        else
            printf '  FAIL %s\n' "$line"
            missing=$((missing + 1))
        fi
    done <<< "$out"
    if [ "$missing" -gt 0 ]; then
        printf '%d of %d figures recomputed in SQL are not in README.md as written\n' \
               "$missing" "$n"
        return 1
    fi
    printf 'SQL reproduces all %d figures in README sections 4, 5 and 8\n' "$n"
}

run "SQL, search log aggregation" sqlite3 check_sql
run "R, statistical inference"    Rscript Rscript verify/stats.R "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
