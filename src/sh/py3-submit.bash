#!/usr/bin/env bash

SOURCEFILES="/home/akorsunsky/envs:/share/apps/software/virtualenvs/tensorflow-dev/bin/activate"


thispath=$(readlink -f "$0")
thispath=$(dirname "$thispath")

if [ -z "${PPN+set}" ]; then
  PPN=64
fi

if [ -z "${NODES+set}" ]; then
  NODESARG="1"
else
  NODESARG="nodes=${NODES}"
fi

SCRIPT="$1"
shift


ARGUMENT_ARRAY=$@

# pack arguments for pbs
IFS=';' eval 'ARGUMENTS="${ARGUMENT_ARRAY[*]}"'


basename=${SCRIPT##*/}
basename=${basename%.py}


export SOURCEFILES ARGUMENTS SCRIPT

qsub -N ${basename} -l${NODESARG}:ppn=${PPN} -v SOURCEFILES,SCRIPT,ARGUMENTS "${thispath}/py3-run.pbs"
