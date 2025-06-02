# Classification and Reasoning on THz data using VLMs

## Project Overview / Description of Folders
* [experiments](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/experiments) - Shows all Experiment Results regarding the classification of THz Images. Main experiment results (for both Zero-Shot and In-Context Learning) can be found in the [[1_experiment](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/experiments/1_experiment)] folder. The other folders are "past experiments" that were conducted for testing reasons. The experiment result has 3 main parts:
  * **1-[model_name].txt** - It contains the classification output of the VLM model [model_name] for the 1400 THz Images of one THz Dataset.

  * **2-eval-[model_name].csv** - It  contains the "True_Label" of each THz Frame (in total 1400) and the "Predicted_Label" that the used VLM model classified for that frame.

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


## Usage / Running a model classification
The Project contains two methods for running a model classification (MAIN METHOD is preferred):

### Using Nico's Script (MAIN Method)


### Using Shashank's Script (OLD Method) 

Currently this script is only set up for testing purposes (testing history capabilities of models). In order to run the classification "normally" (i.e. without testing the history) you have to change the "should_test_history" Parameter to "False". 

Important parameters can be set in 2 different ways:
1. In the main Method of the script [[shashank_get_thz_result.py](https://github.com/Nicolas-Poggi/Project_THz_Classification/blob/main/scripts/main_scripts/shashank_get_thz_result.py)]
2. As "Arguments" when running the model from the Command Line [EXCEPTION is the "should_test_history" Parameter].
<br><br>

> model_name : "qwen" or "mistral"
<br>

> should_test_history - boolean flag for testing if the model remebers the history (past prompts), without passing them in the "chat_template". 
<br>

> image_folder_filepath : path to dataset root, such that images of different frames are inside this path.
```
for example, one images path is: "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/depth_image_layer_0001.png"
then,
data_path : "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/"
```
<br>

> output_folder_filepath : path to save the output txt file of the generated output.
```
Example output_folder_filepath: "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/history_test/shashank_results"
```
<br>

Examples of running the script could look like this:
* Setting parameters within the script:
```
python -W ignore shashank_get_thz_result.py
```

* Setting parameters as arguments:
```
python -W ignore shashank_get_thz_result.py --model qwen --image_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/ --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/history_test/shashank_results
```
