#!/usr/bin/bash
#SBATCH --job-name=THz-Zero-Shot
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_mi300 #  or  gpu_mi300 or  gpu_a100_il or gpu_h100_il 
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/whole-introOnlyPrompt/Calculate-THz_%A_%a.out 
#SBATCH --error=slurm/whole-introOnlyPrompt/Calculate-THz_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.poggi.gomes.da.silva@students.uni-mannheim.de

echo "Started job on $(date)"
starttime=$(date +%s)

#Comment SBATCH --exclusive

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amd

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

MESSAGE="Experiment 1 - For Models Qwen and Mistral"

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    python nico_get_thz_results.py --intro_prompt_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/THz_Classification/prompts/1_nico_intro_prompt.txt --testnumber 1 --message "$MESSAGE"
else
    python nico4_one_frame_per_prompt_only_intro_prompt.py --intro_prompt_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/THz_Classification/prompts/1_shashank_intro_prompt.txt --testnumber 2 --message "$MESSAGE"
fi

echo "Ended job on $(date)"
endtime=$(date +%s)

runtime=$((endtime - starttime))
echo "Total runtime: $runtime seconds"

