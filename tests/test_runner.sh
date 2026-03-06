#!/bin/sh
# Copyright (c) 2026 Cornelis Networks. All rights reserved.

# Wrapper to run MPI test programs with mpirun
set -e
MPIRUN="${MPIRUN:-mpirun}"
MPIRUN_FLAGS="${MPIRUN_FLAGS:-}"
TEST_BIN="$1"
shift

# test_allreduce_correctness requires -c count; supply a default for make check
case "$(basename "$TEST_BIN")" in
    test_allreduce_correctness|.libs/test_allreduce_correctness)
        set -- -c 1000 "$@"
        ;;
esac
PASS=0
FAIL=0
run_test() {
    NP="$1"
    shift
    echo "--- Running $TEST_BIN with -np $NP ---"
    # shellcheck disable=SC2086
    if $MPIRUN $MPIRUN_FLAGS -np "$NP" "$TEST_BIN" "$@"; then
        echo "PASS: np=$NP"
        PASS=$((PASS + 1))
    else
        echo "FAIL: np=$NP"
        FAIL=$((FAIL + 1))
    fi
}
run_test 4 "$@"
run_test 3 "$@"
run_test 1 "$@"
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
