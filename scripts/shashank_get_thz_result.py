import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer
from transformers import GenerationConfig
#import spaces
import cv2
import os
import argparse
from PIL import Image
import glob
import json
from pathlib import Path

testnr = '5'

prompt_explain_path = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/THz_Classification/nico_prompt_one_frame.txt"  
prompt_per_frame_path = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/THz_Classification/test.txt"
#prompt_per_frame_path = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/THz_Classification/nico_prompt_per_frame.txt"


output_filename = f"new_prompt_one_frame_test_{testnr}.txt"
output_filename_full = f"new_prompt_one_frame_test_full_{testnr}.txt"

def get_model_processor(args):
    if args.model.lower() == "qwen":
        from models.qwen import QwenScores
        model_info = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_info, torch_dtype="bfloat16", device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(model_info, trust_remote_code=True)
        pipeline = QwenScores(model=model, processor=processor)
    
    elif args.model.lower() == "llama4":
        from models.llama4 import Llama4Scores
        from transformers import Llama4ForConditionalGeneration
        model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        pipeline = Llama4Scores(model=model, processor=processor)
        
    elif args.model.lower() == "mistral":
        from models.mistral import MistralScores
        from transformers import AutoModelForImageTextToText
        
        model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, 
                                                            device_map="cuda", 
                                                            torch_dtype=torch.bfloat16)
        pipeline = MistralScores(model=model, processor=processor)

        
        
    
    elif args.model.lower() == "molmo":
        from models.molmo import MolmoScores
        model_info = 'allenai/Molmo-7B-D-0924'
        processor = AutoProcessor.from_pretrained(
            model_info,
            trust_remote_code=True,
            torch_dtype='bfloat16',
            device_map='cuda'
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_info,
            trust_remote_code=True,
            torch_dtype='bfloat16',
            device_map='cuda'
        )
        pipeline = MolmoScores(model=model, processor=processor)
    else:
        raise ValueError(
            "Invalid model name. Choose from 'qwen', 'molmo', or 'shield_gemma2'."
            )
    
    return pipeline




def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print("Loading model processor...")
    # Load the model and processor based on the selected model
    pipeline = get_model_processor(args)
    print("Model processor loaded")

    print("Reading Task prompt...") 
    with open(prompt_explain_path, 'r') as file:
        prompt_explain = file.read()
    print("Read Task prompt")

    print("Reading Per Frame prompt...") 
    with open(prompt_per_frame_path, 'r') as file:
        prompt_per_frame = file.read()
    print("Read Per Frame prompt")
    

    print("Reading Frames...")
    #video_path = "/ceph/sagnihot/projects/THz_imaging/0_02625_backside_softmax_video.mp4"
    images_paths = sorted(glob.glob("{}/*.png".format(args.image_path)))
    print("Frames read")

    #Give Mistral the task prompt
    print("Setting up Mistral...")
    if args.model.lower() == "mistral":
      output_explain = pipeline.setup_mistral(prompt=prompt_explain)
    print("Mistral set up")

    with open(os.path.join(args.output_path, output_filename), 'a') as file:
            file.write("------------------------------------------------------------------------------------------------------\n")
            file.write("------INPUT------" + "\n")
            file.write("||||Prompt:\n\n" +prompt_explain + "\n\n\n")
            file.write("------Output------" + "\n")
            file.write("||||Model Answer:\n\n" + output_explain + "\n")


    #Loop through the images and get the answers
    print("Getting answers...")
    all_answers = ""
    count = 0
    for image_path in images_paths:
        print("Processing image: ", image_path)
        count += 1

        prompt_per_frame_with_number = prompt_per_frame.replace("[Image NR]", f"{(count):04d}")

        if count == len(images_paths):
            prompt_per_frame_with_number = prompt_per_frame_with_number + "\n\n<END of Terahertz Images>"


        with open(os.path.join(args.output_path, output_filename), 'a') as file:
            with torch.inference_mode():
                answer = pipeline.get_answer(image_path=image_path, prompt=prompt_per_frame)
            
            file.write("------------------------------------------------------------------------------------------------------\n")
            file.write("------INPUT------" + "\n")
            file.write("||||Image:\n\n" + image_path + "\n\n")
            file.write("||||Prompt:\n\n" +prompt_per_frame_with_number + "\n\n\n")
            file.write("------Output------" + "\n")
            file.write("||||Model Answer:\n\n" + answer + "\n")
            all_answers += answer + "\n"
            print("Answer: ", answer)

    # Get scores from the pipeline

    with open(os.path.join(args.output_path,output_filename_full), 'w') as file:
        file.write(all_answers)
    print("Answer: ", all_answers)
    
    # Save the answer to a file
    

if __name__ == "__main__":
    # Load the model and processor based on the selected model
    parser = argparse.ArgumentParser(description="Give fairness score for generated images")
    parser.add_argument("--model", type=str, required=False, default="mistral", 
                        choices=["qwen", "molmo", "llama4", "mistral", "deepseek-r1"], help="VLM to use")
    parser.add_argument("--image_path", type=str, required=False, 
                        default="/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax", 
                        help="Path to the generated images to be scored.")
    parser.add_argument("--output_path", type=str, required=False, 
                        default="/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/THz_Classification/experiments/debugging_one_frame_nico", 
                        help="Path to the output directory to save the scores.")
    args = parser.parse_args()
    
    args.output_path = os.path.join(args.output_path, args.model)
    
    main(args)
    