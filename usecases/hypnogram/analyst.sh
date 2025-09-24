#!/bin/bash
HOST=$1
SCAN=$2
QUERY=$3
./analyst ${HOST} ${SCAN} ${QUERY} output/${SCAN}_${QUERY}.txt && python gui/utils/plot_hypnogram.py output/${SCAN}_${QUERY}.txt
