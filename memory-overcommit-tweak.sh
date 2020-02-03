#!/bin/bash

# Taken from: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type

cat /proc/sys/vm/overcommit_memory
sudo echo 1 > /proc/sys/vm/overcommit_memory
cat /proc/sys/vm/overcommit_memory
