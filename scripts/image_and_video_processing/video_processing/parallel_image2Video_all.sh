#!/bin/bash

# This script runs the videoCreator.py script with different parameters. 
# It loops through 4 mat files and generates 3 different Scaling factors for each mat file.)
# The script is designed to be run in a bash shell. 

echo "Start of Bash Script"
start=$(date +%s)

# Export the function or script call

run_script() {
    i=$1
    j=$2
    echo "Running: python video_creator.py $i $j"
    python video_creator.py "$i" "$j"
}
export -f run_script
# Generate combinations and run in parallel

parallel --jobs 16 run_script ::: {0..3} ::: {1..3}

end=$(date +%s)
runtime=$((end - start))

echo "End of Bash Script"
printf "Total execution time: %02d:%02d (mm:ss)\n" $((runtime / 60)) $((runtime % 60))
