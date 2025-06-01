#!/usr/bin/bash
#SBATCH --job-name=THz-One-Shot-ICL
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_mi300 #  or  gpu_mi300 or  gpu_a100_il or gpu_h100_il 
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/2_experiment/one_shot_ICL/Calculate-THz_%A_%a.out 
#SBATCH --error=slurm/2_experiment/one_shot_ICL/Calculate-THz_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.poggi.gomes.da.silva@students.uni-mannheim.de

echo "Started job on $(date)"
starttime=$(date +%s)

#Comment SBATCH --exclusive

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amd

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

DESCRIPTION="Experiment 1 - For Models Qwen and Mistral (One-Shot-ICL)"

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    #python nico_get_thz_result.py --model qwen --should_add_context True --test_description "Experiment 1 - For Models Qwen and Mistral" --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_first_experiment/1_one_shot_in_context_learning
    python nico_get_thz_result.py --model qwen --should_add_context True --test_description "$DESCRIPTION" --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/2_experiment/1_one_shot_in_context_learning
else
    #python nico_get_thz_result.py --model mistral --should_add_context True --test_description "Experiment 1 - For Models Qwen and Mistral" --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_first_experiment/1_one_shot_in_context_learning 
    python nico_get_thz_result.py --model mistral --should_add_context True --test_description "$DESCRIPTION" --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/2_experiment/1_one_shot_in_context_learning
fi

echo "Ended job on $(date)"
endtime=$(date +%s)

runtime=$((endtime - starttime))
echo "Total runtime: $runtime seconds"

