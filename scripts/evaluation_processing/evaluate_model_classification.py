import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def main():
    ########################################################################
    #INPUT VARIABLES
    #modelnames     qwen    mistral
    model_name = "qwen"
    folder_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Project_THz_Classification/experiments/1_experiment/1_one_shot_in_context_learning"
    #########################################################################

    input_evaluation_csv_filepath = f"{folder_filepath}/2-eval-{model_name}.csv"
    output_model_evaluation_filepath = f"{folder_filepath}/3-eval-{model_name}.txt"

    df = get_csv_as_dataframe(input_evaluation_csv_filepath)
    accuracy, precision, recall, f1 = evaluate_predictions(df)

    output_str = get_evaluation_output_as_formatted_string(model_name=model_name,accuracy=accuracy, precision=precision, recall=recall, f1=f1)

    # Save results
    with open(output_model_evaluation_filepath, "w") as f:
        f.write(output_str)
    
    print(output_str)

    


if __name__=="__main__":
    main()