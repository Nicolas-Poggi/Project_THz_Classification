import time
start_time = time.time()
import sys

if len(sys.argv) != 3:
    print("\nError: Incorrect number of arguments")
    print("Usage: python imageCreatorParallel.py <num1 - Number of File> <num2 - Scaling Type>")
    print("num1: 0) 02625_Backside  1) 02625_Backside_noLens  2) 05250_Backside_noLens  3) USAF")
    print("num2: 1) Normal Scaling  2) Log Scaling   3) Softmax Scaling\n")
    sys.exit(1)

print("\nPython Script - Nicolas Poggi - videoCreator.py")
print("Bachelor Thesis - FSS 2025")
print("Importing Libraries")

import imageio.v2 as imageio
import os


print("Libraries Imported")
print("Defining Functions")

def getScaleName(scaling_type):
    if scaling_type == 1:
        scaling_name = "Normal"
    
    elif scaling_type == 2:
        scaling_name = "Log"
    
    elif scaling_type == 3:
        scaling_name = "Softmax"
    
    else:
        print("Error: Unknown Scaling Type")
        exit(1)
    
    return scaling_name

def getInputFolderPreName(filename_num):
    if filename_num == 0:
        input_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside'
    
    elif filename_num == 1:
        input_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/1_02625_Backside_noLens'
    
    elif filename_num == 2:
        input_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/2_05250_Backside_noLens'
    
    elif filename_num == 3:
        input_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/3_USAF'
    
    else:
        print("Error: Unknown File Number")
        exit(1)
    
    return input_folder_pre

print("Functions Defined")
print("Starting Script")


#---Define Image Folder-Pre Name for importing  & Calculate Output Folder Pre for output video path
#Number of file
#0) 02625_Backside  1) 02625_Backside_noLens  2) 05250_Backside_noLens  3) USAF
filename_num = int(sys.argv[1])
input_folder_pre = getInputFolderPreName(filename_num)

#---Define Scaling Type
#Type of Image Scaling
# 1) Normal Scaling  2) Log Scaling   3) Softmax Scaling
scaling_type = int(sys.argv[2])
scaling_name = getScaleName(scaling_type)

#---Define Input Folder Name
input_folder = f"{input_folder_pre}_{scaling_name}"
folder_name = os.path.basename(os.path.normpath(input_folder))  # Gets '3_USAF_' from the full path

# Define full video path
video_filename = f"{folder_name.lower()}_video.mp4"
video_path = os.path.join("/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/Videos", video_filename)

# Frame rate (frames per second)
fps = 10

print("Downloading Frames.")
# Create a list of image file paths sorted by depth layer
filenames = [
    os.path.join(input_folder, f"depth_image_layer_{i:04d}.png")
    for i in range(1, 1401)  # 1 to 1400
    if os.path.exists(os.path.join(input_folder, f"depth_image_layer_{i:04d}.png"))
]

print("Compiling Video.")
# Read and compile video
with imageio.get_writer(video_path, fps=fps, format="FFMPEG") as writer:
    i =1
    for filename in filenames:
        print(f"Added Frame:  {i:04d}/{len(filenames)}")
        image = imageio.imread(filename)
        writer.append_data(image)
        i = i + 1
print("-------------------------------")

print( "\nVideo Successfully created.")
print(f"Video saved to: {video_path}")
print(f"Video fps:      {fps}")
lengthseconds = len(filenames)/fps
print(f"Video length:   {int(lengthseconds/60)} min {int(lengthseconds%60)} sec")

print()
print(f"File Number:    {filename_num}")
print(f"Scaling Type:   {scaling_type} - {scaling_name}")

end_time = time.time()
print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
print("End of Script")