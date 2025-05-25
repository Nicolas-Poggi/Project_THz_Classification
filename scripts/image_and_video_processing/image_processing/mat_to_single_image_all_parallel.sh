#!/bin/bash

# This script runs the imageCreatorParallel.py script with different parameters 
# It loops through 4 mat files and generates 3 different Scaling factors for each mat file.)
# The script is designed to be run in a bash shell. 
chmod +x mat_to_single_image_all_parallel.sh
echo "Start of Bash Script"

# Start the timer
start=$(date +%s)

for i in {0..3}; do
    #for j in {1..3}; do
    j=3
    echo "Running: python 1_single_image_creator_parallel.py $i $j"
    python 1_single_image_creator_parallel.py $i $j
    #done
done

# End the timer
end=$(date +%s)
runtime=$((end - start))

echo "End of Bash Script"
printf "\n\nTotal execution time: %02d:%02d (mm:ss)\n" $((runtime / 60)) $((runtime % 60))
