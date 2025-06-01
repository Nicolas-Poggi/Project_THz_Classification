import pandas as pd
import re

def get_list_of_model_classification_new(classification_filepath):
    """
    Extracts all frame-wise observations (e.g., 'No C4', 'Yes C4') from the file,
    even if the label appears on the line after '### Observation:'.
    
    Args:
        classification_filepath (str): Path to the .txt file containing the model output classification.

    Returns:
        List[str]: A list of observation strings, one per frame (in order).
    """
    observations = ["None"] * 1400
    index = -1
    expect_observation = False

    with open(classification_filepath, 'r') as file:
        for line in file:
            line = line.strip()
            observation_class_match = re.match(r"### Classification:\s*(No C4|Yes C4)", line)
                

            if re.match(r"--- \d{4} ------ Message History (JSON) ---", line):
                expect_observation = False  # reset

            # New frame block
            if re.match(r"--- \d{4} ------ Output ---", line):
                index += 1
                expect_observation = True  # reset

            elif observation_class_match and expect_observation:       
                observations[index] = observation_class_match.group(1)
                expect_observation = False

            # Line following "### Observation:" → should contain "Yes C4" or "No C4"
            elif expect_observation:
                if line in ["Yes C4", "No C4"]:
                    if 0 <= index < len(observations):
                        observations[index] = line
                    expect_observation = False  # reset after reading

    return observations


# def get_list_of_model_classification(classification_filepath):
    # """
    # Extracts all frame-wise observations (e.g., 'No C4', 'Yes C4') from the file.
    
    # Args:
    #     classification_filepath (str): Path to the .txt file containing the model output classification.

    # Returns:
    #     List[str]: A list of observation strings, one per frame (in order).
    # """
    # observations = ["None"] * 1400
    # counter = 0
    # index = counter - 1
    
    # with open(classification_filepath, 'r') as file:
    #     for line in file:
    #         match_output = re.match(r"--- \d{4} ------ Output ---", line.strip())
    #         if match_output:
    #             counter += 1
    #             index += 1 
    #         # Match lines that contain the Observation field
    #         match = re.match(r"### Observation:\s*(.+)", line.strip())
    #         if match:
    #             observations[index] = match.group(1)
    # return observations

def get_csv_as_dataframe(csv_filepath):
    return pd.read_csv(csv_filepath)




def main():
    ########################################################################
    #INPUT VARIABLES
    model_name = "qwen"
    input_csv_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Labeled_Data/0_02625_Backside_Softmax_Labeled.csv"
    folder_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/2_experiment/0_zero_shot"
    input_model_classification_folder_filepath = folder_filepath
    output_csv_folder_filepath = folder_filepath
    ########################################################################


    #Calculate Filepaths

    input_model_classification_filepath = f"{input_model_classification_folder_filepath}/1-{model_name}.txt"
    output_csv_filepath = f"{output_csv_folder_filepath}/2-eval-{model_name}.csv"



    classifications = get_list_of_model_classification_new(classification_filepath=input_model_classification_filepath)

    df = get_csv_as_dataframe(csv_filepath=input_csv_filepath)

    if len(classifications) != len(df):
        print("\n\n",len(classifications),"\n\n")
        raise ValueError("Mismatch between number of classifications and CSV rows.")


    df['Predicted_Label'] = classifications

    df.to_csv(output_csv_filepath, index=False)

    print("\nFile Successfully created!")
    print("IMPORTANT - CHECK CSV FILE FOR ROWS WHERE REGEX STATEMENT DIDN\'T WORK / HAS WEIRD OUTPUT")
    print("CHECK BY FILTERING FOR \"None\"")


if __name__=="__main__":
    main()