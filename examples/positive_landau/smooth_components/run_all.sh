#!/usr/bin/env bash
# The smoothly connected components of Y ∖ Sing(Y) for every diagram: smooth.jl, with the
# singular locus in ../results/<name>.txt. For a diagram that has none yet (env and pentb),
# ../<name>.m2 computes it first, with its own time limit; those diagrams run in the
# background, Macaulay2 then Julia, while the others go through Julia one after another.
#
#   examples/positive_landau/smooth_components/run_all.sh              # every diagram
#   examples/positive_landau/smooth_components/run_all.sh A3 env       # these
#
# To leave it running overnight:
#
#   nohup examples/positive_landau/smooth_components/run_all.sh > run_all.out 2>&1 &
#
# A diagram that already has results/<name>.txt here (and finished its connectivity step)
# is skipped; FORCE=1 runs it again. Only one run_all.sh runs at a time.
#
#   M2_LIMIT=4h              a Macaulay2 script is stopped after this long; the diagram is
#                            then reported as timed out and not routed
#   M2_MEMORY_GB=48          and cannot use more memory than this
#   MONODROMY_SECONDS=3600   monodromy stops after this long; the routing points found by
#                            then are used, and some may be missing
#   JULIA_LIMIT=3h           a Julia script is stopped after this long; the report then has
#                            the routing points only, if they were found
#
# 0 lifts a limit. DRY_RUN=1 only lists what would be run.
#
# logs/summary.txt is written as it goes; logs/<name>.m2.log and logs/<name>.jl.log have
# the output of each step, and results/<name>.txt the report of each diagram.

set -u

DIAGRAMS=(A3 A4 B2 B3 B4 par acn env npltrb tdetri debox tdebox pltrb dbox pentb)

M2_LIMIT=${M2_LIMIT:-4h}
M2_MEMORY_GB=${M2_MEMORY_GB:-48}
MONODROMY_SECONDS=${MONODROMY_SECONDS:-3600}
JULIA_LIMIT=${JULIA_LIMIT:-3h}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LANDAU=$(cd "$DIR/.." && pwd)
ROOT=$(cd "$DIR/../../.." && pwd)
LOGS=$DIR/logs
SUMMARY=$LOGS/summary.txt
if [[ $DRY_RUN == 1 ]]; then
    SUMMARY=/dev/null
else
    mkdir -p "$LOGS" "$DIR/results"
    exec 9> "$LOGS/run_all.lock"
    if ! flock -n 9; then
        echo "run_all.sh is already running (see $SUMMARY). Not starting a second one." >&2
        exit 1
    fi
fi

(( $# > 0 )) && DIAGRAMS=("$@")
for name in "${DIAGRAMS[@]}"; do
    [[ -f $LANDAU/$name.m2 ]] || { echo "no diagram $name in $LANDAU" >&2; exit 2; }
done

# lines are written whole, so the background diagrams do not garble the summary
say() {
    echo "$*"
    echo "$*" >> "$SUMMARY"
}
clock() { printf '%d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

outcome() {
    case $1 in
        0) echo "ok" ;;
        124 | 137) echo "STOPPED at its limit of $2" ;;
        *) echo "FAILED (exit $1)" ;;
    esac
}

has_singular_locus() { [[ -e $LANDAU/results/$1.txt && $(tail -n 1 "$LANDAU/results/$1.txt") == done ]]; }
finished() { [[ -e $DIR/results/$1.txt ]] && grep -q "smoothly connected component" "$DIR/results/$1.txt"; }

# run_step LIMIT LOG COMMAND...: the command with a time limit and its output in LOG
child=
run_step() {
    local limit=$1 log=$2
    shift 2
    timeout -k 60 "$limit" "$@" > "$log" 2>&1 &
    child=$!
    wait "$child"
    local status=$?
    child=
    return $status
}

run_m2() {
    local name=$1 start=$SECONDS status
    local m2=(M2 --script "$LANDAU/$name.m2")
    (( M2_MEMORY_GB > 0 )) && m2=(bash -c 'ulimit -v "$0" && exec "$@"' $((M2_MEMORY_GB * 1024 * 1024)) "${m2[@]}")
    run_step "$M2_LIMIT" "$LOGS/$name.m2.log" "${m2[@]}"
    status=$?
    say "$(printf '%-8s' "$name")Macaulay2 $(outcome $status "$M2_LIMIT")  ($(clock $((SECONDS - start))))"
    return $status
}

run_julia() {
    local name=$1 start=$SECONDS status label
    label=$(printf '%-8s' "$name")
    local options=()
    (( MONODROMY_SECONDS > 0 )) && options+=("--monodromy-timeout=$MONODROMY_SECONDS")
    # unset: with them HomotopyContinuation can segfault
    run_step "$JULIA_LIMIT" "$LOGS/$name.jl.log" env -u LD_LIBRARY_PATH -u LD_PRELOAD \
        julia -t auto --project="$ROOT" "$DIR/smooth.jl" "$name" "${options[@]}"
    status=$?
    {
        echo "${label}Julia $(outcome $status "$JULIA_LIMIT")  ($(clock $((SECONDS - start))))"
        (( status == 0 )) || echo "        see logs/$name.jl.log"
        grep -q "monodromy stopped at its timeout" "$LOGS/$name.jl.log" &&
            echo "        monodromy stopped at its limit of $MONODROMY_SECONDS s: routing points may be missing"
        grep -q "paths reached no index 0" "$LOGS/$name.jl.log" &&
            echo "        some paths reached no index 0 point: components may be split"
        grep -E '^[0-9]+ (routing points|components),' "$LOGS/$name.jl.log" | sed 's/^/        /'
        if [[ -e $DIR/results/$name.txt ]]; then
            grep -E '^[0-9]+ smoothly connected component|^component [0-9]+:|smooth components not computed' \
                "$DIR/results/$name.txt" | sed 's/^/        /'
        fi
    } > "$LOGS/$name.summary"
    cat "$LOGS/$name.summary"
    cat "$LOGS/$name.summary" >> "$SUMMARY"
    rm -f "$LOGS/$name.summary"
}

# a diagram without a singular locus: Macaulay2, then Julia only if it finished
run_with_m2() {
    local name=$1
    trap 'trap "" TERM; [[ -n $child ]] && kill "$child" 2> /dev/null; wait; exit 130' TERM INT
    if run_m2 "$name" && has_singular_locus "$name"; then
        run_julia "$name"
    else
        say "        $name: Macaulay2 timed out or failed, so there is no singular locus and no components (see logs/$name.m2.log)"
        { echo "$name: Macaulay2 did not finish within $M2_LIMIT (or failed); no singular locus, so the"
          echo "smooth components were not computed. See logs/$name.m2.log."; } > "$DIR/results/$name.txt"
    fi
}

background=()
stop() {
    trap '' INT TERM
    say "interrupted $(date '+%F %T')"
    [[ -n $child ]] && kill "$child" 2> /dev/null
    for pid in "${background[@]}"; do kill "$pid" 2> /dev/null; done
    wait 2> /dev/null
    exit 130
}
trap stop INT TERM

total=$SECONDS
say ""
say "run_all.sh started $(date '+%F %T')"
say "limits: Macaulay2 $M2_LIMIT and $M2_MEMORY_GB GB, Julia $JULIA_LIMIT, monodromy $MONODROMY_SECONDS s (0 = none)"
say "diagrams: ${DIAGRAMS[*]}"
say ""

# first the diagrams that need Macaulay2, in the background
foreground=()
for name in "${DIAGRAMS[@]}"; do
    label=$(printf '%-8s' "$name")
    if finished "$name" && [[ $FORCE != 1 ]]; then
        say "${label}already done: smooth_components/results/$name.txt is there (FORCE=1 runs it again)"
    elif has_singular_locus "$name"; then
        foreground+=("$name")
        [[ $DRY_RUN == 1 ]] && say "${label}would run smooth.jl"
    elif [[ $DRY_RUN == 1 ]]; then
        say "${label}would run $name.m2 (limit $M2_LIMIT) in the background, then smooth.jl"
    else
        say "${label}no singular locus yet: $name.m2 starts in the background"
        run_with_m2 "$name" &
        background+=($!)
    fi
done
[[ $DRY_RUN == 1 ]] && exit 0

for name in "${foreground[@]}"; do
    run_julia "$name"
done

for pid in "${background[@]}"; do wait "$pid"; done
say ""
say "finished $(date '+%F %T'), after $(clock $((SECONDS - total)))"
