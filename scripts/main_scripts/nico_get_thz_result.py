print("importing libraries (torch, transformers, os, argparse, time)")
import torch
from transformers import pipeline
import os 
import argparse
import time

def get_model_path(model_name):
    model_path = ""
    if model_name.lower() == "mistral":
        model_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    elif model_name.lower() == "qwen":
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    elif model_name.lower() == "llama4":
        model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    
    elif model_name.lower() == "molmo":
        model_path = "allenai/Molmo-7B-D-0924"

    return model_path


def get_promt(prompt_filepath):
    with open(prompt_filepath, 'r') as file:
        prompt = file.read()
    return prompt

def get_image_paths(image_path):
    image_paths = sorted([os.path.join(image_path, file) for file in os.listdir(image_path) if file.lower().endswith(('.png'))])  
    # image_paths = image_paths[-10:]  # Limit to the last 10 images
    return image_paths

def get_answer(message_history, pipe, max_tokens):
    output = pipe.__call__(text=message_history , return_full_text=False,  max_new_tokens=max_tokens)
    generated_text = output[0]['generated_text']
    return generated_text

def get_output_filepath(testnumber, output_folder, test_addon):
    return f"{output_folder}/{testnumber}-{test_addon}.txt"

def get_edited_frame_prompt(frame_prompt, counter):
    edit_frame_prompt = frame_prompt.replace("[Image NR]", str(counter))
    if(counter == 1400):
        edit_frame_prompt = edit_frame_prompt + "<END OF Terahertz Images>"
    return edit_frame_prompt

def get_test_information_str(system_prompt_filepath, frame_prompt_filepath,image_folder_path, model_path, max_tokens, max_storage, testnumber, test_description, should_test_history, test_frame_prompt_filepath, test_image_filepath):
    
    test_information_str = "".join([
        "--Test Information--\n",
        f"System Prompt Path:             {system_prompt_filepath}\n",
        f"Frame Prompt Path:              {frame_prompt_filepath}\n",
        f"Image Folder Path:              {image_folder_path}\n",
        f"Model Path:                     {model_path}\n",
        f"Max Tokens:                     {max_tokens}\n",
        f"Max Storage:                    {max_storage}\n",
        f"Test Number:                    {testnumber}\n",
        f"Time Elapsed:                   \n",
        f"Frames Completed:               \n",
    ])

    if(should_test_history):
        test_information_str += "".join([ 
            f"History Test Frame Prompt Path: {test_frame_prompt_filepath}\n",
            f"History Test Image Path:        {test_image_filepath}\n"
    ])  
        
    test_information_str += "".join([
        f"Test Description:                {test_description}",
        "\n--------------------"])
    return test_information_str

def get_message_history_as_string_in_json_format(message_history):
    counter = 1
    message_history_str = ""
    
    message_history_str += "[\n"
    for message in message_history:  
        
        #do rest
        message_history_str += "\t{\n"

        for key, value in message.items():
            message_history_str += "\t"
            if key != 'content':
                message_history_str += f"'{key}': '{value}'"

            else:
                message_history_str += f"'{key}': \n\t\t[\n"
                for content in value:
                    message_history_str += "\t\t\t"

                    if content['type'] != 'text':
                        message_history_str += f"{{'type': '{content['type']}', 'path': '{content['path']}'}}"
                    else:
                        escaped_text = content['text'].replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t').replace("'", "\\'")
                        message_history_str += f"{{'type': '{content['type']}', 'text': '{escaped_text}'}}"

                    if content != value[-1]:
                        message_history_str += ",\n"
                    else:
                        message_history_str += "\n"

                message_history_str += "\t\t]"
            

            if(key != 'content'):
                message_history_str += ",\n"
            else:
                message_history_str += "\n"

        message_history_str += "\t}"
        
        message_history_str += "\n"
        if counter%2 == 1:
            message_history_str += "\n\n"
        counter += 1

    message_history_str += "]"

    return message_history_str

def reduce_history_only_image_of_user_element(message_history, counter, max_storage):
    i = counter - max_storage

    # 1. List [{x1, x2, x3}]     where x = Dictionary and is either "role" System, User, Assistant
    message_history[(1 + 2*(i-1))]['content'].pop(0)
    return message_history

def reduce_history_image_and_text_user_and_text_of_assistant_element(message_history, should_add_context):
    if(should_add_context==False):
        message_history.pop(1) # User Prompt
        message_history.pop(1) # Assistant Prompt
    else:
        message_history.pop(2) # User Prompt
        message_history.pop(2) # Assistant Prompt

    return message_history

def append_chat_element_to_message_history(chat_element_type, message_history, text, image_filepath):
    """
    Appends element to the message history.
    Parameters:
        chat_element_type (str): Type of chat element ("user", "assistant", "system").
        message_history (list): The message history to which the chat element will be appended.
        text (str): The text to append to the message history.
        image_filepath (str): Path to the related image file (only used when chat_element_type is "user").
    Returns:
        None
    """

    if chat_element_type == "user":
       message_history.append({"role": "user", "content":[{"type": "image", "path": image_filepath}, {"type": "text", "text": text}]})

    elif chat_element_type == "assistant":
        message_history.append({"role": "assistant", "content":[{"type": "text", "text": text}]})
    
    elif chat_element_type == "system":
        message_history.append({"role": "system", "content":[{"type": "text", "text": text}]})

def append_chat_element_to_file(chat_element_type, counter, output_filepath, text, image_filepath):
    """
    Appends text to a file in a structured format based on the type of chat element.

    Parameters:
        chat_element_type (str or None): Type of chat element ("user", "assistant", "system", "message_history_json"), 
                                         or None to write plain text without a header.
        counter (int): A counter used to number the sections. (only used if chat_element_type is a valid chat element type)
        output_filepath (str): Path to the file where the text should be appended.
        text (str): The text to append to the file.
        image_filepath (str): Path to the related image file (only used when chat_element_type is "user").

    Returns:
        None
    """
       
    with open(output_filepath, 'a') as file:

        if(chat_element_type != None):
            
            if chat_element_type == "user":
                file.write("\n\n--- " + f"{(counter):04d}" + " ------ Input ---\n\n")
                file.write("Image Path: " + image_filepath +  "\n")
            
            elif chat_element_type == "assistant":
                file.write("\n\n--- " + f"{(counter):04d}" + " ------ Output ---\n\n")
            
            elif chat_element_type == "system":
                file.write("\n\n--- " + f"{(counter):04d}" + " ------ System ---\n\n")
            
            elif chat_element_type == "message_history_json":
                file.write("\n\n--- " + f"{(counter):04d}" + " ------ Message History (JSON) ---\n\n")

            file.write(text + "\n")

        else:
            file.write("\n\n" + text + "\n")

def print_chat_element(chat_element_type, counter, text, image_filepath):
    """
    Prints the chat element in a structured format based on the type of chat element.
    Parameters:
        chat_element_type (str): Type of chat element ("user", "assistant", "system", "message_history_json") or None to write plain text without a header.
        counter (int): A counter used to number the sections.
        text (str): The text to print.
        image_filepath (str): Path to the related image file (only used when type is "user").
    Returns:
        None
    """
    if(chat_element_type != None):

        if chat_element_type == "user":
            print("\n\n--- " + f"{(counter):04d}" + " ------ Input ---\n\n")
            print("Image Path: " + image_filepath +  "\n")
            
        elif chat_element_type == "assistant":
            print("\n\n--- " + f"{(counter):04d}" + " ------ Output ---\n\n")
        
        elif chat_element_type == "system":
            print("\n\n--- " + f"{(counter):04d}" + " ------ System ---\n\n")
        
        elif chat_element_type == "message_history_json":
            print("\n\n--- " + f"{(counter):04d}" + " ------ Message History ---\n\n")
            
        print(text, "\n")

    else:
        print("\n\n",text, "\n")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main(args):

    start_time = time.perf_counter()
    
    #--------------------------------------------------------------------------------------------------
    # Models - When changing the modelpath also change test_addon in parameters for the output file
    #  
    # mistral
    # qwen
    # llama4
    # mistralai/Mistral-Small-3.1-24B-Instruct-2503
    # Qwen/Qwen2.5-VL-7B-Instruct           
    # meta-llama/Llama-4-Scout-17B-16E-Instruct
    #---------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -- SET GLOBAL PARAMETERS --

    # General parameters for the test
    frame_prompt_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/prompts/2_nico_frame_prompt.txt"
    image_folder_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/combined_images"
    model_name = "qwen"
    max_tokens = 200
    max_storage = 1 # max number of prompts/images in memory [including the current prompt being used]
    
    #--------------------------------------------------------------------------------------------------
    # Two different System prompts
    # 1. System prompt from Shashank
    # 2. System prompt from Nico
    #--------------------------------------------------------------------------------------------------
    # system_prompt_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/prompts/1_shashank_system_prompt.txt"    #Shashank
    system_prompt_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/prompts/1_nico_system_prompt.txt"      #Nico
    

    #Parameters for the output file
    testnumber = 1
    test_description = "Testing history capabilities with different models - If they can remember the first image (Mona Lisa) that was shown to them without being in chat_template --> AI remembers history"
    output_folder_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_first_experiment/0_zero_shot"
    output_addon = model_name
    
    #Special Parameters for the test
    should_test_history = False
    test_frame_prompt_one_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/prompts/30_nico_test_history_first.txt"
    test_frame_prompt_two_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/prompts/31_nico_test_history_second.txt"
    test_image_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/history_test/mona_lisa.png"

    #In-Context Learning
    should_add_context = False
    in_context_learning_image_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_Softmax/combined_images_cropped/depth_image_layer_0663/depth_image_layer_0663_crop_0014_row_01_col_02.png"
    in_context_learning_prompt_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/prompts/40_nico_in_context_learning_prompt.txt"


    #Parameters for the script
    counter = 0
    message_history = []
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    if args.model != None: 
        model_name = args.model
        output_addon = args.model
    if args.should_add_context != None: 
        should_add_context = args.should_add_context
    if args.test_description != None: 
        test_description = args.test_description
    if args.output_folder_filepath != None: 
        output_folder_filepath = args.output_folder_filepath

    # Parse Paths & Prompts 
    output_filepath = get_output_filepath(testnumber, output_folder_filepath, output_addon)
    image_paths = get_image_paths(image_folder_filepath)
    system_prompt = get_promt(system_prompt_filepath)
    frame_prompt = get_promt(frame_prompt_filepath)
    in_context_learning_prompt = get_promt(in_context_learning_prompt_filepath)
    model_path = get_model_path(model_name)

    test_frame_prompt_one = ""

    if(should_test_history == True):
        test_frame_prompt_one = get_promt(test_frame_prompt_one_filepath)
        frame_prompt_filepath = test_frame_prompt_two_filepath
        frame_prompt = get_promt(frame_prompt_filepath)


    print(frame_prompt)
    # Create the pipeline where the model is loaded 
    print("creating pipeline")
    pipe = pipeline("image-text-to-text", model=model_path, device="cuda", torch_dtype=torch.bfloat16)
    print("pipeline created")
    


    #--------------------------------------------------------------------------------------------------
    # -- ADD TEST INFO --
    #
    # Parse and Create the Test information for printing and saving
    # Append the Test information to the output file
    # Print the Test information
    #--------------------------------------------------------------------------------------------------
    test_information_str = get_test_information_str(system_prompt_filepath=      system_prompt_filepath, 
                                                    frame_prompt_filepath=      frame_prompt_filepath, 
                                                    image_folder_path=          image_folder_filepath, 
                                                    model_path=                 model_path, 
                                                    max_tokens=                 max_tokens, 
                                                    max_storage=                max_storage,
                                                    testnumber=                 testnumber, 
                                                    test_description=           test_description, 
                                                    should_test_history=        should_test_history, 
                                                    test_frame_prompt_filepath= test_frame_prompt_one_filepath, 
                                                    test_image_filepath=        test_image_filepath)
    append_chat_element_to_file(chat_element_type=None, counter=None, output_filepath=output_filepath, text=test_information_str, image_filepath=None)
    print_chat_element(         chat_element_type=None, counter=None,                                  text=test_information_str, image_filepath=None)



    #--------------------------------------------------------------------------------------------------
    # -- ADD SYSTEM CHAT ELEMENT --
    #
    # Append the new "system" chatElement to the message_history  (input prompt: only text)
    # Append the system chatElement to the output file
    # Print the system chatElement - simple format
    #--------------------------------------------------------------------------------------------------
    append_chat_element_to_message_history(chat_element_type="system", message_history=message_history,                  text=system_prompt, image_filepath=None)
    append_chat_element_to_file(           chat_element_type="system", counter=counter, output_filepath=output_filepath, text=system_prompt, image_filepath=None)
    print_chat_element(                    chat_element_type="system", counter=counter,                                  text=system_prompt, image_filepath=None)
    


    if(should_add_context == True):
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -- ADD ONE CROP AS CONTEXT FOR IN CONTEXT LEARNING --
        
        #--------------------------------------------------------------------------------------------------
        # -- ADD USER CHAT ELEMENT --
        #
        # Append the new "system" chatElement to the message_history  (input prompt: only text)
        # Append the system chatElement to the output file
        # Print the system chatElement - simple format
        #--------------------------------------------------------------------------------------------------
        append_chat_element_to_message_history(chat_element_type="user", message_history=message_history,                  text=in_context_learning_prompt, image_filepath=in_context_learning_image_filepath)
        append_chat_element_to_file(           chat_element_type="user", counter=counter, output_filepath=output_filepath, text=in_context_learning_prompt, image_filepath=in_context_learning_image_filepath)
        print_chat_element(                    chat_element_type="user", counter=counter,                                  text=in_context_learning_prompt, image_filepath=in_context_learning_image_filepath)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    


    counter += 1


  
    if(should_test_history == True):
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -- TESTING HISTORY --
        
        #--------------------------------------------------------------------------------------------------
        # -- ADD USER CHAT ELEMENT  (MONA LISA IMAGE) --
        #
        # 1. Append the new "user" chatElement to the message_history  (input prompt: image of mona lisa + text)
        # 2. Append the user chatElement to the output file
        # 3. Print the user chatElement - simple format
        #--------------------------------------------------------------------------------------------------
        append_chat_element_to_message_history(chat_element_type="user", message_history=message_history,                  text=test_frame_prompt_one, image_filepath=test_image_filepath)
        append_chat_element_to_file(           chat_element_type="user", counter=counter, output_filepath=output_filepath, text=test_frame_prompt_one, image_filepath=test_image_filepath)
        print_chat_element(                    chat_element_type="user", counter=counter,                                  text=test_frame_prompt_one, image_filepath=test_image_filepath)

        #Calculate the model output with the input (i.e. message_history) 
        model_output = get_answer(message_history=message_history, pipe=pipe, max_tokens = max_tokens)

        #--------------------------------------------------------------------------------------------------
        # -- ADD ASSISTANT CHAT ELEMENT --
        #
        # 1. Append the new "assistant" chatElement to the message_history  (output prompt: only text)
        # 2. Append the assistant chatElement to the output file
        # 3. Print the assistant chatElement - simple format
        #--------------------------------------------------------------------------------------------------
        append_chat_element_to_message_history(chat_element_type="assistant", message_history=message_history,                  text=model_output, image_filepath=None)
        append_chat_element_to_file(           chat_element_type="assistant", counter=counter, output_filepath=output_filepath, text=model_output, image_filepath=None)
        print_chat_element(                    chat_element_type="assistant", counter=counter,                                  text=model_output, image_filepath=None)


        #Remove the User and Assistant Element from the message history --> NOW only System Element is in the message history
        reduce_history_image_and_text_user_and_text_of_assistant_element(message_history=message_history)    

        counter += 1
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
   

    for image_filepath in image_paths:
        
        #--------------------------------------------------------------------------------------------------
        #  -- ADD USER CHAT ELEMENT --
        #
        # 1. Edit the InputPrompt to fit the current image Number 
        # 2. Append the new "user" chatElement to the message_history  (input prompt: image + text)
        # 3. Append the user chatElement to the output file
        # 4. Print the user chatElement - simple format
        #--------------------------------------------------------------------------------------------------
        edited_frame_prompt = get_edited_frame_prompt(frame_prompt=frame_prompt, counter=counter)
        append_chat_element_to_message_history(chat_element_type="user", message_history=message_history,                  text=edited_frame_prompt, image_filepath=image_filepath)
        append_chat_element_to_file(           chat_element_type="user", counter=counter, output_filepath=output_filepath, text=edited_frame_prompt, image_filepath=image_filepath)
        print_chat_element(                    chat_element_type="user", counter=counter,                                  text=edited_frame_prompt, image_filepath=image_filepath)


        #Calculate the model output with the input (i.e. message_history) 
        model_output = get_answer(message_history=message_history, pipe=pipe, max_tokens = max_tokens)



        #--------------------------------------------------------------------------------------------------
        # -- ADD ASSISTANT CHAT ELEMENT --
        #
        # 1. Append the new "assistant" chatElement to the message_history  (output prompt: only text)
        # 2. Append the assistant chatElement to the output file
        # 3. Print the assistant chatElement - simple format
        #--------------------------------------------------------------------------------------------------
        append_chat_element_to_message_history(chat_element_type="assistant", message_history=message_history,                  text=model_output, image_filepath=None)
        append_chat_element_to_file(           chat_element_type="assistant", counter=counter, output_filepath=output_filepath, text=model_output, image_filepath=None)
        print_chat_element(                    chat_element_type="assistant", counter=counter,                                  text=model_output, image_filepath=None)



        #--------------------------------------------------------------------------------------------------
        # -- APPEND / PRINT MESSAGE HISTORY (JSON FORMAT) --
        # 
        # Append the message history to the output file
        # Print the message history in the "Json" format
        #--------------------------------------------------------------------------------------------------
        message_history_str_in_json = get_message_history_as_string_in_json_format(message_history)
        append_chat_element_to_file(chat_element_type="message_history_json", counter=counter, output_filepath=output_filepath, text=message_history_str_in_json, image_filepath=None)
        print_chat_element(         chat_element_type="message_history_json", counter=counter,                                  text=message_history_str_in_json, image_filepath=None)



        counter += 1



        #--------------------------------------------------------------------------------------------------
        # -- REDUCE MESSAGE HISTORY --
        # 
        # Multiple options to reduce the message history
        # 1. Reduce the history to only the last x images                                        --> x=Defined by max_storage
        # 2. Reduce the history to only the last x steps (1 step == INPUT & OUTPUT Chat Element) --> x=Defined by max_storage
        # 3. Break the loop if the counter is greater than max_storage
        #--------------------------------------------------------------------------------------------------
        if(counter > max_storage):
            # reduce_history_only_image_of_user_element(message_history=message_history, counter=counter, max_storage=max_storage)
            # break

            if(should_add_context==False):
                reduce_history_image_and_text_user_and_text_of_assistant_element(message_history=message_history, should_add_context=False)    
            else:
                reduce_history_image_and_text_user_and_text_of_assistant_element(message_history=message_history, should_add_context=True)
            

        #If currently testing_History is True - Stop Test after 10 Images (break the loop)
        if(should_test_history & (counter == 11)):
            break



    #--------------------------------------------------------------------------------------------------
    # -- APPEND / PRINT <END OF TEST> & ELAPSED TIME --
    #
    # Calculate the elapsed time
    # Append END OF TEST to the output file
    # Print END OF TEST 
    # Append the elapsed time to the output file
    # Print the elapsed time
    #--------------------------------------------------------------------------------------------------
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    append_chat_element_to_file(chat_element_type=None, counter=None, output_filepath=output_filepath, text="\n\n--End of Test--", image_filepath=None)
    print_chat_element(         chat_element_type=None, counter=None,                                  text="--End of Test--", image_filepath=None)
    append_chat_element_to_file(chat_element_type=None, counter=None, output_filepath=output_filepath, text=f"Elapsed Time: {elapsed_time:.2f} seconds", image_filepath=None)
    print_chat_element(         chat_element_type=None, counter=None,                                  text=f"Elapsed Time: {elapsed_time:.2f} seconds", image_filepath=None)

    

if __name__ == "__main__":

    # Load the model and processor based on the selected model
    parser = argparse.ArgumentParser(description="Give fairness score for generated images")
    
    parser.add_argument("--model", type=str, required=False, 
                        choices=["mistral", "qwen",  "llama4", "molmo"], 
                        help="VLM to use")

    parser.add_argument("--should_add_context", type=str2bool, required=False, 
                        help="Add one Image Context")
    
    parser.add_argument("--test_description", type=str, required=False, 
                        help="description of the Test to be conducted")
    
    parser.add_argument("--output_folder_filepath", type=str, required=False, 
                        help="Output_Folder_Filepath")
    


    args = parser.parse_args()


    main(args)