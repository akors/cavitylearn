#!/usr/bin/env bash

SOURCEFILES="/home/akorsunsky/envs:/share/apps/software/virtualenvs/cavitylearn/bin/activate"


SCRIPT="$1"
shift

# ARGUMENT_ARRAY=("--batchsize" "200" "--datasets" "test-mini" "$1" "$2")
ARGUMENT_ARRAY=$@

# pack arguments for pbs
IFS=';' eval 'ARGUMENTS="${ARGUMENT_ARRAY[*]}"'


basename=${SCRIPT##*/}
basename=${basename%.py}


export SOURCEFILES ARGUMENTS SCRIPT STDOUTFILE

qsub -N ${basename} -v SOURCEFILES,SCRIPT,ARGUMENTS py3-run.pbs
