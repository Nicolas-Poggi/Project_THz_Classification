import time
start_time = time.time()
import sys
if len(sys.argv) != 3:
    print("\nError: Incorrect number of arguments")
    print("Usage: python imageCreatorParallel.py <num1 - Number of File> <num2 - Scaling Type>\n\n")
    print("num1: 0) 02625_Backside  1) 02625_Backside_noLens  2) 05250_Backside_noLens  3) USAF\n")
    print("num2: 1) Normal Scaling  2) Log Scaling   3) Softmax Scaling\n")
    sys.exit(1)

print("\nPython Script - Nicolas Poggi - combined_image_creator_parallel.py")
print("Bachelor Thesis - FSS 2025")
print("Importing Libraries")

sys.path.insert(1, "src_code/")
import process_rdf as prdf
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import skimage
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

matplotlib.use('Agg')
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

def getMatFileName(filename_num):
    if filename_num == 0:
        filename_mat = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/Rohacell_10mm_RDX_300GHz_02625_backside.mat'
    
    elif filename_num == 1:
        filename_mat = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/Rohacell_10mm_RDX_300GHz_02625_backside_no_lens_215mm_rp.mat'
    
    elif filename_num == 2:
        filename_mat = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/Rohacell_10mm_RDX_300GHz_05250_backside_no_lens_215mm_rp.mat'
    
    elif filename_num == 3:
        filename_mat = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/USAF_PCB_01_02625_750_G.mat'
    
    else:
        print("Error: Unknown File Number")
        exit(1)
    
    return filename_mat

def getOutputFolderPreName(filename_num):
    if filename_num == 0:
        output_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside'
    
    elif filename_num == 1:
        output_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/1_02625_Backside_noLens'
    
    elif filename_num == 2:
        output_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/2_05250_Backside_noLens'
    
    elif filename_num == 3:
        output_folder_pre = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/3_USAF'
    
    else:
        print("Error: Unknown File Number")
        exit(1)
    
    return output_folder_pre

def process_and_save(depth_layer):
    print(f"Computing Image: {(depth_layer+1):04d} / {processed_data.shape[0]}")
    
    # Get intensity image at a specific depth layer (without FFT)
    image_depth = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)).transpose(0, 1).detach().cpu().numpy()
    
    # Get phase image at a specific depth layer (without FFT)
    image_phase = torch.flipud(torch.angle(processed_data[depth_layer, ...])).transpose(0, 1).detach().cpu().numpy()

    width = (5.92 * 2)
    height = 4.8

    #------------------------------------------
    # Plotting both images in one figure
    fig, axs = plt.subplots(1, 2, figsize=(width, height), dpi=100)

    #Intensity Image
    im0 = axs[0].imshow(image_depth, cmap='viridis', vmin=global_min, vmax=global_max)
    axs[0].set_title("Intensity")
    axs[0].set_xlabel('X-axis (spatial units)')
    axs[0].set_ylabel('Y-axis (spatial units)')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label= f"{scaling_name} Intensity (arb. units)")

    #Phase Image
    im1 = axs[1].imshow(image_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Phase")
    axs[1].set_xlabel('X-axis (spatial units)')
    axs[1].set_ylabel('Y-axis (spatial units)')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Phase (radians)")
    

    plt.suptitle(f"Frequency {depth_layer+1:04d} / {processed_data.shape[0]}")
    plt.tight_layout(pad=0.5)
    filename = f"depth_image_layer_{(depth_layer+1):04d}.png"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, bbox_inches=None)
    plt.close()
    #------------------------------------------
    
    im = Image.open(save_path)
    target_size = (int(width * 100), int(height * 100))
    if im.size != target_size:
        im = im.resize(target_size, resample=Image.LANCZOS)
        im.save(save_path)
    return save_path


print("Functions Defined")
print("Starting Script")
#---Define File Num for reading (.Mat) & Output Folder Pre for output
#Number of file
#0) 02625_Backside  1) 02625_Backside_noLens  2) 05250_Backside_noLens  3) USAF
filename_num = int(sys.argv[1])
filename_mat  = getMatFileName(filename_num)
output_folder_pre = getOutputFolderPreName(filename_num)

#---Define Scaling Type
#Type of Image Scaling
# 1) Normal Scaling  2) Log Scaling   3) Softmax Scaling
scaling_type = int(sys.argv[2])
scaling_name = getScaleName(scaling_type)

#---Create Output Folder---
output_folder = os.path.join(f"{output_folder_pre}_{scaling_name}", "combined_images")
os.makedirs(output_folder, exist_ok=True)

print("Mat File:                        ",filename_mat)
print("Output Folder:                   ", output_folder)
print()


#---Read and Process Dat
print("Started Reading File.")
device = 'cpu'
complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, int(parameters["NF"]), device=device)
print("File Read Sucessfully.")

# --- Compute global min/max intensity across all frequencies
abs_squared = torch.pow(torch.abs(processed_data), 2)  # shape: [NF, NX, NY]

#Different Methods for Scaling The Images

if scaling_type == 1:
# 1) Normal Scaling
    global_min = abs_squared.min().item()
    global_max = abs_squared.max().item()

if scaling_type == 2:
# 2) Log Scaling
    global_min = torch.log(abs_squared).min().item()
    global_max = torch.log(abs_squared).max().item()

if scaling_type == 3:
    softmax_x = F.softmax(abs_squared, dim=0)
    global_min = softmax_x.min().item()
    global_max = softmax_x.max().item()


print(f"Global min intensity:            {global_min:.4f}")
print(f"Global max intensity:            {global_max:.4f}")


#---Print info about Mat File---
print("NF - Number of Frequencies:      ", processed_data.shape[0])
print("NX - Number of Points on X-Axis: ", processed_data.shape[1])
print("NY - Number of Points on Y-Axis: ", processed_data.shape[2])

# Use a few CPUs for parallel processing
with ProcessPoolExecutor(max_workers=32) as executor:
    executor.map(process_and_save, range((processed_data.shape[0])))


print("\nAll Images Saved Sucessfully.\n")

print(f"Global min intensity:            {global_min:.4f}")
print(f"Global max intensity:            {global_max:.4f}")
print("NF - Number of Frequencies:      ", processed_data.shape[0])
print("NX - Number of Points on X-Axis: ", processed_data.shape[1])
print("NY - Number of Points on Y-Axis: ", processed_data.shape[2])

print("\nFile Number:                     ",filename_num)
print("Scaling Type:                    ",scaling_type, scaling_name)

print("\nMat File:                        ",filename_mat)
print("Output Folder:                   ", output_folder)
end_time = time.time()
print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
print("End of Script")