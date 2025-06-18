#!/usr/bin/bash
#SBATCH --job-name=THz-Experiment-Zero-and-One-Shot
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_mi300 #  or  gpu_mi300 or  gpu_a100_il or gpu_h100_il 
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/1_experiment/combined/Calculate-THz_%A_%a.out 
#SBATCH --error=slurm/1_experiment/combined/Calculate-THz_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.poggi.gomes.da.silva@students.uni-mannheim.de

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amd

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

MODEL="llava-hf/llava-1.5-13b-hf"
DESCRIPTION="Experiment 1 - Model"


#-----ZERO SHOT-----
#Zero Shot Inference
python nico_get_thz_result.py --model "$MODEL" --should_add_context False --test_description "$DESCRIPTION" --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/0_zero_shot

#Zero Shot Read Inference
python ../evaluation_processing/parse_model_classification.py --model "$MODEL" --folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/0_zero_shot

#Zero Shot Evaluate Inference
python ../evaluation_processing/evaluate_model_classification.py --model "$MODEL" --folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/0_zero_shot

#-----ONE SHOT-----
python nico_get_thz_result.py --model "$MODEL" --should_add_context True --test_description "$DESCRIPTION" --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/1_one_shot_in_context_learning

#Zero Shot Read Inference
python ../evaluation_processing/parse_model_classification.py --model "$MODEL" --folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/1_one_shot_in_context_learning

#Zero Shot Evaluate Inference
python ../evaluation_processing/evaluate_model_classification.py --model "$MODEL" --folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/1_one_shot_in_context_learning
