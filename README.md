# Classification and Reasoning on THz data using VLMs

## Project Overview / Description of Folders
* [experiments](https://github.com/Nicolas-Poggi/Project_THz_Classification/tree/main/experiments) - Shows all Experiment Results regarding the classification of THz Images. An experiment result has 3 parts:
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


## Usage

> model_name : "qwen", "llama4" or "mistral"

> category_name : luxury", "culture" or "maintenance"
The prompts for each policy category are in ./policies

> data_path : path to dataset root, such that images of different frames are inside this path.
```
for example, one images path is: "/ceph/sagnihot/projects/THz_imaging/lens_softmax_frames/frame_0000.png"
then,
data_path : "/ceph/sagnihot/projects/THz_imaging/lens_softmax_frames"
```

> output_path: path to save the output txt file of the generated output, default: "./experiments"


```
python -W ignore get_answer_one_frame.py --model mistral --model model_name --data_path data_path --output_path output_path
```
