#!/bin/bash

# to do - pass the path as argument 
for value in {1..20}
do
  python prepare_gaze.py -p /scratch/user/ravikt/airsim/data/stationary_truck/truck_mountains$value/
done
