import csv

label_data_filepath = "/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Labeled_Data/0_02625_Backside_Softmax_Labeled.csv"

with open(label_data_filepath, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame_Number', 'True_Label'])
    for i in range(1,1401):
        writer.writerow([i,''])
