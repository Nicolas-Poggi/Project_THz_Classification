#!/bin/bash

chmod +x combined_image_to_combined_crops_all_parallel.sh
echo "Start of Bash Script"

# Start the timer
start=$(date +%s)

for i in {0..3}; do
    #for j in {1..3}; do
    j=3
    echo "Running: python 3_create_crops_combined.py $i $j"
    python 3_create_crops_combined.py $i $j
    #done
done

# End the timer
end=$(date +%s)
runtime=$((end - start))

echo "End of Bash Script"
printf "\n\nTotal execution time: %02d:%02d (mm:ss)\n" $((runtime / 60)) $((runtime % 60))
