import pandas as pd
import re


def get_list_of_model_classification(classification_filepath):
    """
    Extracts all frame-wise observations (e.g., 'No C4', 'Yes C4') from the file.
    
    Args:
        classification_filepath (str): Path to the .txt file containing the model output classification.

    Returns:
        List[str]: A list of observation strings, one per frame (in order).
    """
    observations = ["None"] * 1400
    counter = 0
    index = counter - 1
    
    with open(classification_filepath, 'r') as file:
        for line in file:
            match_output = re.match(r"--- \d{4} ------ Output ---", line.strip())
            if match_output:
                counter += 1
                index += 1 
            # Match lines that contain the Observation field
            match = re.match(r"### Observation:\s*(.+)", line.strip())
            if match:
                observations[index] = match.group(1)

            

    return observations


def add_classification_to_csv(classification_filepath, csv_filepath):
    classifications = get_list_of_model_classification(classification_filepath)
    df = pd.read_csv(csv_filepath)

    if len(classifications) != len(df):
        raise ValueError("Mismatch between number of classifications and CSV rows.")

    df['Predicted_Label'] = classifications
    df.to_csv(csv_filepath, index=False)

def get_csv_as_dataframe(csv_filename):
    return pd.read_csv(csv_filename)





########################################################################
#INPUT VARIABLES
input_csv_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Labeled_Data/0_02625_Backside_Softmax_Labeled.csv"
input_model_classification_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/evaluation_test/4-1-example-no-context-frame-by-frame-full-1400-test-sum.txt"
output_csv_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/evaluation_test/test_labeled.csv"

classifications = get_list_of_model_classification(classification_filepath=input_model_classification_filepath)

df = get_csv_as_dataframe(csv_filename=input_csv_filepath)

if len(classifications) != len(df):
    print("\n\n",len(classifications),"\n\n")
    raise ValueError("Mismatch between number of classifications and CSV rows.")


df['Predicted_Label'] = classifications

df.to_csv(output_csv_filepath, index=False)






