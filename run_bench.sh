#!/usr/bin/env bash

set -eu
set -o pipefail

BENCH_NAMES=(rsg-nhbr-allgather moore-nhbr-allgather)

: ${ITR:=500}

: ${FRNDSHIP_THR:=4}

NUM_PROCS=16

EXEC_ROOT=.

OUT_ROOT=.

moore-nhbr-allgather()
{
    moore-nhbr-coll allgather
}

moore-nhbr-coll()
{
    local coll=$1
    local bench_exec=$EXEC_ROOT/nhbr-$coll

    dims=( 2 )
    rads=( 2 )

    for d in ${dims[@]}; do
        for r in ${rads[@]}; do
            echo "=== #procs = $NUM_PROCS, dimension = $d, radius = $r," \
                 "friendship_thr = $FRNDSHIP_THR, alg = $alg ==="
            local options="-t moore -d $d -r $r -n $ITR"
            mpiexec -n "$NUM_PROCS" "$bench_exec" $options \
            | tee -a "$out_dir/${alg}.d${d}.r${r}.${NUM_PROCS}"
        done
    done
}

rsg-nhbr-allgather()
{
    rsg-nhbr-coll allgather
}

rsg-nhbr-coll()
{
    local coll=$1
    local bench_exec=$EXEC_ROOT/nhbr-$coll

    local prob_list=( 0.05 )
    local n_runs=1

    for p in ${prob_list[@]}; do
        for i in $(seq "$n_runs"); do
            echo "=== Run $i/$n_runs, #procs = $NUM_PROCS, sparsity p = $p,"\
                 "friendship_thr = $FRNDSHIP_THR, alg = $alg ==="
            local options="-t rsg -p $p -n $ITR"
            mpiexec -n "$NUM_PROCS" "$bench_exec" $options\
            | tee -a "$out_dir/${alg}.p${p}.${NUM_PROCS}"
        done
    done
}

is_valid_bench()
{
    local bench_name=$1
    for name in ${BENCH_NAMES[@]}; do
        if [ "$bench_name" == $name ]; then
            return 0
        fi
    done
    return 1
}

print_usage()
{
    cat << EOF
$(basename "$0")[options] <benchmark>
options:
    -h:           print this usage message
    -l:           list benchmark names
    -n <number>:  number of processes. Default = $NUM_PROCS
    -o <dir>:     output directory. Default = $OUT_ROOT
    -e <dir>:     executables directory. Default = $EXEC_ROOT
EOF
}

while getopts ":hln:o:e:" opt; do
    case $opt in
        h)
            print_usage
            exit 0
            ;;
        l)
            echo ${BENCH_NAMES[@]}
            exit 0
            ;;
        n)
            NUM_PROCS=$OPTARG
            ;;
        o)
            OUT_ROOT=$OPTARG
            ;;
        e)
            EXEC_ROOT=$OPTARG
            ;;
        \?)
            echo "Invalid option"
            print_usage
            exit 1
            ;;
    esac
done

shift $(( OPTIND - 1 ))

if [ $# -ne 1 ]; then
    print_usage
    exit 1
fi

bench_name=$1
out_dir=$OUT_ROOT/${bench_name}.out/$NUM_PROCS

if [ ! -d "$out_dir" ]; then
    mkdir -p "$out_dir"
fi

for alg in {auto,comb}; do
    export MPICH_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM=$alg
    export MPICH_NEIGHBOR_COLL_MSG_COMB_FRNDSHP_THRSHLD=$FRNDSHIP_THR
    if ! is_valid_bench $bench_name; then
        echo "Invalid benchmark name"
        print_usage
        exit 1
    fi
    $bench_name
done
