#!/usr/bin/env bash

outdir="logs/"
if [ ! -d "$outdir" ]; then
    echo "Creating outdir $outdir..."
    mkdir $outdir
fi

outfile="$outdir/run_$(date +"%Y-%m-%d_%H-%M-%S").log"
python3 -u runner.py | tee $outfile
