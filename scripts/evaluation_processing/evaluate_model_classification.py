import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

def get_csv_as_dataframe(csv_filepath):
    return pd.read_csv(csv_filepath)

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
    else:
        model_path = model_name
    return model_path



def get_evaluation_output_as_formatted_string(model_name,accuracy,precision,recall,f1):
    return "".join([
        "--EVALUATION--------------------\n",
        f"Model:       {get_model_path(model_name)}\n",
        f"Accuracy:    {accuracy:.4f}\n",
        f"Precision:   {precision:.4f}\n",
        f"Recall:      {recall:.4f}\n",
        f"F1-Score:    {f1:.4f}\n"
        "--------------------------------\n"
    ])

def evaluate_predictions(df):
    y_true = df["True_Label"]
    y_pred = df["Predicted_Label"]
    pos_label = "Yes C4"

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    
    return accuracy, precision, recall, f1

def main(args):
    ########################################################################
    #INPUT VARIABLES
    #modelnames     qwen    mistral
    model_name = "qwen"
    folder_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/1_one_shot_in_context_learning"
    output_addon = model_name
    #########################################################################
    
    if args.model != None: 
        model_name = args.model
        output_addon = args.model.split("/")[-1]
        output_addon = output_addon.replace(".","")
    if args.folder_filepath != None: 
        folder_filepath = args.folder_filepath



    input_evaluation_csv_filepath = f"{folder_filepath}/2-eval-{output_addon}.csv"
    output_model_evaluation_filepath = f"{folder_filepath}/3-eval-{output_addon}.txt"

    df = get_csv_as_dataframe(input_evaluation_csv_filepath)
    accuracy, precision, recall, f1 = evaluate_predictions(df)

    output_str = get_evaluation_output_as_formatted_string(model_name=model_name,accuracy=accuracy, precision=precision, recall=recall, f1=f1)

    # Save results
    with open(output_model_evaluation_filepath, "w") as f:
        f.write(output_str)
    
    print(output_str)

    


if __name__=="__main__":
    
    # Load the model and processor based on the selected model
    parser = argparse.ArgumentParser(description="Give fairness score for generated images")
    
    parser.add_argument("--model", type=str, required=False, 
                        help="VLM to use")
    
    parser.add_argument("--folder_filepath", type=str, required=False, 
                        help="Input_and_Output_Folder_Filepath")
    


    args = parser.parse_args()

    
    main(args)