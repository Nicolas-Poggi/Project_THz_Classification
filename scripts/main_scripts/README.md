
## Running a model classification

The Project contains two methods for running a model classification (MAIN METHOD is preferred). 
* MAIN Method can do Zero-Shot Classification AND In-Context Learning with a One-Frame Context.
* OLD Method can do Zero-Shot Classification, but cannot do In-Context Learning with a One-Frame Context.

### Using Nico's Script (MAIN Method)

Important parameters (here only the most important are mentioned) can be set in 2 different ways:
1. As "variables" in the "main" method of the script [[shashank_get_thz_result.py](https://github.com/Nicolas-Poggi/Project_THz_Classification/blob/main/scripts/main_scripts/shashank_get_thz_result.py)]
2. As "arguments" when running the script from the command line [EXCEPTION is the "should_test_history" and "image_folder_filepath" Parameter].
<br><br>

> model_name : "qwen" or "mistral"  (if the parameter is used as a argument it's called "model")
<br>

> should_test_history - boolean flag for testing if the model remebers the history (past prompts), without passing them in the "chat_template". TRUE for running script with "History Test", FALSE for running script wihtout "History Test".
<br>

> should_add_context  - boolean flag for turning on in-context learning with a one frame context. TRUE for running script with "In-context learning", FALSE for running script without "In-context learning". 
<br>

> image_folder_filepath : path to dataset root, such that images of different frames are inside this path.
```
for example, one images path is: "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/combined_images/depth_image_layer_0001.png"
then,
data_path : "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/combined_images"
!!! NOTICE WITHOUT BACKSLASH AT END!!!
```
<br>

> output_folder_filepath : path to save the output txt file of the generated output.
```
Example output_folder_filepath: "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/0_zero_shot"
!!! NOTICE WITHOUT BACKSLASH AT END!!!
```
<br>

Examples of running the script could look like this:
* Setting parameters as "variables" in the "main" method of the script:
```
python -W ignore nico_get_thz_result.py
```

* Setting parameters as "arguments" when running the script from the command line:
```
python -W ignore shashank_get_thz_result.py --model qwen --image_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/ --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/history_test/shashank_results
```


### Using Shashank's Script (OLD Method) 

Currently this script is only set up for testing purposes (testing history capabilities of models). In order to run the classification "normally" (i.e. without testing the history) you have to change the "should_test_history" Parameter to "False". 

Important parameters (here only the most important are mentioned) can be set in 2 different ways:
1. As "variables" in the "main" method of the script [[shashank_get_thz_result.py](https://github.com/Nicolas-Poggi/Project_THz_Classification/blob/main/scripts/main_scripts/shashank_get_thz_result.py)]
2. As "arguments" when running the script from the command line [EXCEPTION is the "should_test_history" Parameter].
<br><br>

> model_name : "qwen" or "mistral"
<br>

> should_test_history - boolean flag for testing if the model remebers the history (past prompts), without passing them in the "chat_template". TRUE for running script with "History Test", FALSE for running script wihtout "History Test".
<br>

> image_folder_filepath : path to dataset root, such that images of different frames are inside this path.
```
for example, one images path is: "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/depth_image_layer_0001.png"
then,
data_path : "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/"
!!! NOTICE WITH BACKSLASH AT END!!!
```
<br>

> output_folder_filepath : path to save the output txt file of the generated output.
```
Example output_folder_filepath: "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/history_test/shashank_results"
!!! NOTICE WITHOUT BACKSLASH AT END!!!
```
<br>

Examples of running the script could look like this:
* Setting parameters as "variables" in the "main" method of the script:
```
python -W ignore shashank_get_thz_result.py
```

* Setting parameters as "arguments" when running the script from the command line:
```
python -W ignore shashank_get_thz_result.py --model qwen --image_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/ --output_folder_filepath /pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/history_test/shashank_results
```
