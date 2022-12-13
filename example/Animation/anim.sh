#!/bin/bash

mkdir -p ./frames

frames=360

for (( i=0; i < $frames ; i++ )) do
    x=`echo "-c($i * 4 * a(1) / 180)" | bc -l`
    y=`echo "1 - s($i * 4 * a(1) / 180)" | bc -l`

    echo "frame: $i"

    ../../raytrace --obj sph r: 0.4 pos: 0 1 0 emit \
                   --obj sph r: 0.2 pos: $x $y -0.2 glass: 0.2 opacity: 0 \
                   --obj pln pos: 0 0 -0.4 rough: 1 \
                   --cam exp: 0.5 \
                   --sample 256 -o ./frames/out$i.png
                   --ssaa 2
                
done

ffmpeg -framerate 60 -f image2 -i ./frames/out%d.png out.mp4
