#!/usr/bin/env bash

SOURCEFILES="/home/akorsunsky/envs:/share/apps/software/virtualenvs/cavitylearn/bin/activate"


thispath=$(readlink -f "$0")
thispath=$(dirname "$thispath")

if [ -z "${PPN+set}" ]; then
  PPN=64
fi

SCRIPT="$1"
shift

# ARGUMENT_ARRAY=("--batchsize" "200" "--datasets" "test-mini" "$1" "$2")
ARGUMENT_ARRAY=$@

# pack arguments for pbs
IFS=';' eval 'ARGUMENTS="${ARGUMENT_ARRAY[*]}"'


basename=${SCRIPT##*/}
basename=${basename%.py}


export SOURCEFILES ARGUMENTS SCRIPT

qsub -N ${basename} -lnodes=1:ppn=${PPN} -v SOURCEFILES,SCRIPT,ARGUMENTS "${thispath}/py3-run.pbs"
