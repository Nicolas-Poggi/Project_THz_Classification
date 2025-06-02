# Classification and Reasoning on THz data using VLMs

## Project Overview / Description of Folders
* [experiments](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/experiments) - Shows all Experiment Results regarding the classification of THz Images. The main experiment results that should be looked at (for both Zero-Shot and In-Context Learning) can be found in the [[1_experiment](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/experiments/1_experiment)] folder. The other folders are "past experiments" that were conducted for testing reasons (not important). A experiment result has 3 main parts:
  * **1-[model_name].txt** - It contains the experiment settings and classification output of the VLM model [model_name] for the 1400 THz Images of one THz Dataset.

  * **2-eval-[model_name].csv** - It contains the "True_Label" of each THz Frame (in total 1400) and the "Predicted_Label" that the used VLM model classified for that frame.

  * **3-eval-[model_name].txt** - It contains the evaluation results of the THz classification for the given model.  

* [prompts](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/prompts) - Contains the prompts utilized for the experiment (including some test prompts).
  
* [scripts](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts) - Contains the utilized scripts for python scripts for:
  * Running a classification - [main_scripts](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/main_scripts)
  * Creating and Processing THz Images and Videos - [image_and video_processing](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/image_and_video_processing)
  * Preparing Labeled Files and Creating Evaluations from Model Classifications  - [evaluation_processing](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/evaluation_processing)


## Setup

1. create conda environtment and install essential libraries

```
conda create -n THz python=3.11 -y
conda activate THz
pip install -U transformers
pip install accelerate
pip install tensorflow
```

Please note, Qwen requires slightly different (conflicting) libraries, so for Qwen models create a diffirent environment and install the required libraries

```
conda create -n THz_qwen python=3.11 -y
conda activate THz_qwen
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
```

2. Install other required libraries

```
pip install einops torchvision ipdb tqdm numpy pandas spaces opencv-python
pip install --upgrade --no-deps --force-reinstall torch
```
## Workflow
This guide provides a complete step-by-step workflow for running THz image classification experiments using Vision-Language Models (VLMs), exporting predictions and evaluating results.
<br>
<br>
<br>
### 0. Create Folder structure 📁
Before starting the experiment, add a folder, with a name of your choosing, to the "experiments" folder. In this folder please create the following sub_folders: "0_zero_shot" and "1_one_shot_in_context_learning". These folders are where your Experiment results for "Zero-Shot" and "One-Shot In-Context-Learning" will be saved. Here is what Your folder structure should look like (See Below):
<br>
<br>
Project_THz_Classification/<br>
│<br>
├──experiments/<br>
│ ├──YOUR_EXPERIMENT_FOLDER/<br>
│ │ ├──0_zero_shot/<br>
│ │ └──1_one_shot_in_context_learning/
<br>
<br>
<br>
### 1. Generate a Model Classification 🚀
To generate predictions using a vision language model (VLM) on the processed image frames please use the [nico_get_thz_result.py](https://github.com/Nicolas-Poggi/Project_THz_Classification/blob/main/scripts/main_scripts/nico_get_thz_result.py) script. Please set the following parameters at the beginning part of the "main" method in the script, to your projects needs.
<br><br>
For more information read the README.md description in the [main_scripts](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/main_scripts) folder.
<br>
<br>
* model_name - "qwen" or "mistral" - sets the vision language model (VLM) to be used
* should_add_context - boolean flag for turning on in-context learning with a one frame context. TRUE for running script with "In-context learning", FALSE for running script without "In-context learning". 
* image_folder_filepath - path to dataset root, such that images of different frames are inside this path. (!!! IMPORTANT WITHOUT BACKSLASH AT END!!!)
* output_folder_filepath - path to save the output txt file of the generated output. Set this to the previously created "0_zero_shot" or "1_one_shot_in_context" folder depending on your experiment. (!!! IMPORTANT WITHOUT BACKSLASH AT END!!!)

After setting the parameters in the python script, you can run the following command, when in the [main_scripts](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/main_scripts) folder, in order to run the python script:
```
python -W ignore nico_get_thz_result.py
```
<br>
<br>

### 2. Export Classification to CSV 📤
After generating predictions, you have to convert the ".txt" output into a structured CSV file. To do this we use the [parse_model_classification.py](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/evaluation_processing) script. Please set the following parameters at the beginning part of the "main" method in the script, to your projects needs.
<br><br>
For more information read the README.md description in the [evaluation_processing](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/evaluation_processing) folder.
<br>
<br>
* model_name - "qwen" or "mistral" - sets the vision language model (VLM) to be used
* input_csv_filepath - path to csv file containing the true labels of the used Dataset.
* folder_filepath - path to save the output csv file of the generated output. Set this to the previously created "0_zero_shot" or "1_one_shot_in_context" folder depending on your experiment. (!!! IMPORTANT WITHOUT BACKSLASH AT END!!!)

After setting the parameters in the python script, you can run the following command, when in the [evaluation_processing](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/evaluation_processing) folder, in order to run the python script:
```
python parse_model_classification.py
```

#### WARNING CHECK CSV FILE FOR ROWS WHERE REGEX STATEMENT DIDN\'T WORK / HAS WEIRD OUTPUT") - CHECK BY FILTERING FOR "None"
<br>
<br>

### 3. Run Evaluation Script 📊
To evaluate the classification results (e.g. against a ground-truth label set) please use the [evaluate_model_classification.py](https://github.com/Nicolas-Poggi/Project_THz_Classification/blob/main/scripts/evaluation_processing/evaluate_model_classification.py) script. This generates standard classification metrics, such as: 
- Accuracy
- Precision
- Recall
- F1-Score

Please set the following parameters at the beginning part of the "main" method in the script, to your projects needs. 

* model_name - "qwen" or "mistral" - sets the vision language model (VLM) to be used
* folder_filepath - path to save the output csv file of the generated output. Set this to the previously created "0_zero_shot" or "1_one_shot_in_context" folder depending on your experiment. (!!! IMPORTANT WITHOUT BACKSLASH AT END!!!)

After setting the parameters in the python script, you can run the following command, when in the [evaluation_processing](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/scripts/evaluation_processing) folder, in order to run the python script:
```
python evaluate_model_classification.py
```
<br>
<br>

### Summary ✅ 
You should now have tho following:
 * **1-[model_name].txt** - It contains the experiment settings and classification output of the VLM model [model_name] for the 1400 THz Images of one THz Dataset.
 * **2-eval-[model_name].csv** - It contains the "True_Label" of each THz Frame (in total 1400) and the "Predicted_Label" that the used VLM model classified for that frame.
 * **3-eval-[model_name].txt** - It contains the evaluation results of the THz classification for the given model.  
