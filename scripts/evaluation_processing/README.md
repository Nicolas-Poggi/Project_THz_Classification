# 🧪 Evaluation Processing
This folder contains scripts for transforming raw model outputs into a structured format suitable for evaluation. The key utility here is `parse_model_classification.py`, which extracts frame-wise predictions from `.txt` files and merges them with ground-truth labels into a final CSV.

## Script: `parse_model_classification.py`

### Description

This script converts a model-generated `.txt` output file (e.g., from Qwen or Mistral VLMs) into a clean `.csv` file that includes predictions per frame, aligned with the true labels.

It does this by:
- Parsing the `.txt` output using regex patterns
- Extracting one prediction per frame (`"Yes C4"` / `"No C4"`)
- Merging the results into an existing labeled dataset CSV
- Outputting a final `eval` CSV for analysis and metrics computation

### Configuration 🔧

Before running the script, edit the `main()` method and set the following variables to match your project setup:

| Variable | Description |
|----------|-------------|
| `model_name` | The name of the vision-language model used. Options: `"qwen"` or `"mistral"` |
| `input_csv_filepath` | Absolute path to the ground-truth labeled CSV file |
| `folder_filepath` | Directory containing the model’s `.txt` output file and where the output `.csv` should be saved (⚠️ **no trailing slash!**) |

The script expects the classification `.txt` file to be named as follows: 1-{model_name}.txt
<br>
The output will be saved to: 2-eval-{model_name}.csv


### How to Run ▶️ 

Once parameters are configured, execute the script by running:

```
python parse_model_classification.py
```
