#!/usr/bin/env bash
# The whole computation, one diagram after another: <name>.m2 (the components of Y and its
# singular locus), then <name>.jl (the routing points and the positive ones).
#
#   examples/positive_landau/run_all.sh                 # every diagram, in the order below
#   examples/positive_landau/run_all.sh debox dbox      # these two, in this order
#
# To leave it running overnight:
#
#   nohup examples/positive_landau/run_all.sh > run_all.out 2>&1 &
#
# A diagram that already has results/<name>_positive.txt is skipped, so A3, par and acn are
# not repeated, and a run that was interrupted can simply be started again. FORCE=1 runs
# them all the same. Only one run_all.sh runs at a time: a second one refuses to start.
#
# Each step has a limit, so that one diagram cannot take the night:
#
#   M2_LIMIT=2h              a Macaulay2 script is stopped after this long
#   M2_MEMORY_GB=48          and cannot use more memory than this
#   MONODROMY_SECONDS=3600   monodromy stops after this long on each stratum; the routing
#                            points found by then are reported, and some may be missing
#   JULIA_LIMIT=3h           a Julia script is stopped after this long, and then reports nothing
#
# 0 lifts a limit, e.g.  M2_LIMIT=8h MONODROMY_SECONDS=0 JULIA_LIMIT=0 run_all.sh env
# DRY_RUN=1 only lists what would be run.
#
# A Macaulay2 script writes results/<name>.txt only when it is done. If there is none
# after it stopped, the Julia script runs with --no-singular-locus: f = α₁⋯α_{E-1}·M, and the
# report says so. An older complete results/<name>.txt is used if there is one.
#
# logs/summary.txt is written as it goes; logs/<name>.m2.log and logs/<name>.jl.log have
# the output of each step, and results/<name>_positive.txt the report of each diagram.

set -u

DIAGRAMS=(A3 A4 B2 B3 B4 par acn env npltrb tdetri debox tdebox pltrb dbox pentb)

M2_LIMIT=${M2_LIMIT:-2h}
M2_MEMORY_GB=${M2_MEMORY_GB:-48}
MONODROMY_SECONDS=${MONODROMY_SECONDS:-3600}
JULIA_LIMIT=${JULIA_LIMIT:-3h}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$DIR/../.." && pwd)
LOGS=$DIR/logs
SUMMARY=$LOGS/summary.txt
if [[ $DRY_RUN == 1 ]]; then
    SUMMARY=/dev/null
else
    mkdir -p "$LOGS" "$DIR/results"
    # one run at a time: two would do the same diagrams and write over each other's logs
    exec 9> "$LOGS/run_all.lock"
    if ! flock -n 9; then
        echo "run_all.sh is already running (see $SUMMARY). Not starting a second one." >&2
        exit 1
    fi
fi

(( $# > 0 )) && DIAGRAMS=("$@")
for name in "${DIAGRAMS[@]}"; do
    [[ -f $DIR/$name.m2 && -f $DIR/$name.jl ]] || { echo "no diagram $name in $DIR" >&2; exit 2; }
done

say() {                       # to the screen and to the summary
    echo "$*"
    echo "$*" >> "$SUMMARY"
}
clock() { printf '%d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

# run_step LIMIT LOG COMMAND...: the command with a time limit and its output in LOG. It
# runs in the background and is waited for, so that the trap below can stop it at once.
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
stop() {
    trap '' INT TERM
    say "interrupted $(date '+%F %T')"
    if [[ -n $child ]]; then
        kill "$child" 2> /dev/null
        wait "$child" 2> /dev/null
    fi
    exit 130
}
trap stop INT TERM

# what a step's exit status means; timeout gives 124, and 137 if it had to kill
outcome() {
    case $1 in
        0) echo "ok" ;;
        124 | 137) echo "STOPPED at its limit of $2" ;;
        *) echo "FAILED (exit $1)" ;;
    esac
}

run_diagram() {
    local name=$1
    local label
    label=$(printf '%-8s' "$name")
    local singular=$DIR/results/$name.txt positive=$DIR/results/${name}_positive.txt

    if [[ -e $positive && $FORCE != 1 ]]; then
        say "${label}already done: results/${name}_positive.txt is there (FORCE=1 runs it again)"
        return
    fi
    if [[ $DRY_RUN == 1 ]]; then
        say "${label}would run $name.m2, then $name.jl"
        return
    fi

    # Macaulay2, with a cap on its memory
    local start=$SECONDS status
    local m2=(M2 --script "$DIR/$name.m2")
    (( M2_MEMORY_GB > 0 )) && m2=(bash -c 'ulimit -v "$0" && exec "$@"' $((M2_MEMORY_GB * 1024 * 1024)) "${m2[@]}")
    run_step "$M2_LIMIT" "$LOGS/$name.m2.log" "${m2[@]}"
    status=$?
    say "${label}Macaulay2 $(outcome $status "$M2_LIMIT")  ($(clock $((SECONDS - start))))"

    # Julia, without the singular locus if Macaulay2 left no complete results
    local options=()
    if [[ -e $singular && $(tail -n 1 "$singular") == done ]]; then
        (( status == 0 )) || say "        using results/$name.txt from an earlier run"
    else
        options+=(--no-singular-locus)
        say "        no results/$name.txt: Julia runs without the singular locus"
    fi
    (( MONODROMY_SECONDS > 0 )) && options+=("--monodromy-timeout=$MONODROMY_SECONDS")

    start=$SECONDS
    # Julia is started with these two variables unset: with them HomotopyContinuation can segfault
    run_step "$JULIA_LIMIT" "$LOGS/$name.jl.log" env -u LD_LIBRARY_PATH -u LD_PRELOAD \
        julia -t auto --project="$ROOT" "$DIR/$name.jl" "${options[@]}"
    status=$?
    say "        Julia $(outcome $status "$JULIA_LIMIT")  ($(clock $((SECONDS - start))))"
    (( status == 0 )) || say "        see logs/$name.jl.log"
    grep -q "monodromy stopped at its timeout" "$LOGS/$name.jl.log" &&
        say "        monodromy stopped at its limit of $MONODROMY_SECONDS s: routing points may be missing"

    # the counts of each stratum, and the verdict on each component of Y
    grep -E '^stratum [0-9]+ \(.*\): [0-9]+ points' "$LOGS/$name.jl.log" | sed 's/^/        /' | tee -a "$SUMMARY"
    (( status == 0 )) && grep -E '^   [0-9]+\. dim ' "$positive" | sed 's/^/        /' | tee -a "$SUMMARY"
    return 0
}

total=$SECONDS
say ""
say "run_all.sh started $(date '+%F %T')"
say "limits: Macaulay2 $M2_LIMIT and $M2_MEMORY_GB GB, Julia $JULIA_LIMIT, monodromy $MONODROMY_SECONDS s on each stratum (0 = none)"
say "diagrams: ${DIAGRAMS[*]}"
say ""
for name in "${DIAGRAMS[@]}"; do
    run_diagram "$name"
done
say ""
say "finished $(date '+%F %T'), after $(clock $((SECONDS - total)))"
