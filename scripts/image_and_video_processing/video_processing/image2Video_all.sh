#!/bin/bash

# This script runs the videoCreator.py script with different parameters. 
# It loops through 4 mat files and generates 3 different Scaling factors for each mat file.)
# The script is designed to be run in a bash shell. 
chmod +x image2Video_all.sh
echo "Start of Bash Script"

# Start the timer
start=$(date +%s)

for i in {0..3}; do
    for j in {1..3}; do
        echo "Running: python video_creator.py $i $j"
        python video_creator.py $i $j
    done
done


# End the timer
end=$(date +%s)
runtime=$((end - start))

echo "End of Bash Script"
printf "\n\nTotal execution time: %02d:%02d (mm:ss)\n" $((runtime / 60)) $((runtime % 60))