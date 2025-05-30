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

def get_crop_size(filename_num):
    '''Crop size for width x height
    '''
    if filename_num == 0:

        crop_size = (26,26)
    
    elif filename_num == 1:
        crop_size = (167,115)    
    
    elif filename_num == 2:
        crop_size = (67,77)     
    
    elif filename_num == 3:
        crop_size = (73,73)     #some are 19 x 73

    else:
        print("Error: Unknown File Number")
        exit(1)

    return crop_size

def get_list_of_intensity_layers(processed_data):
    intensity_layers_list = []
    for depth_layer in range(processed_data.shape[0]):
            image_intensity = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)).transpose(0, 1).detach().cpu().numpy()
            intensity_layers_list.append(image_intensity)

    return intensity_layers_list

def get_list_of_phase_layers(processed_data):
    phase_layers_list = []
    for depth_layer in range(processed_data.shape[0]):
            image_phase = torch.flipud(torch.angle(processed_data[depth_layer, ...])).transpose(0, 1).detach().cpu().numpy()
            phase_layers_list.append(image_phase)

    return phase_layers_list


def process_and_save(depth_layer, intensity_layer, phase_layer, global_min, global_max, output_folder, crop_size, scaling_name):
    layer_number = int(depth_layer+1)
    
    print(f"Computing Image: {(layer_number):04d} / {processed_data.shape[0]}")
    
    # # Get intensity image at a specific depth layer (without FFT)
    # image_intensity = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)).transpose(0, 1).detach().cpu().numpy()
    
    # # Get phase image at a specific depth layer (without FFT)
    # image_phase = torch.flipud(torch.angle(processed_data[depth_layer, ...])).transpose(0, 1).detach().cpu().numpy()



    width = (5.92 * 2)
    height = 4.8



    ##CROP DATA 
    output_folder_layer = os.path.join(output_folder, f"depth_image_layer_{layer_number:04d}")
    os.makedirs(output_folder_layer, exist_ok=True)
    output_filename_pre = os.path.join(output_folder_layer, f"depth_image_layer_{layer_number:04d}_")
    
    #Shape of the image in height x width format
    image_height, image_width = intensity_layer.shape
    crop_width, crop_height = crop_size

    counter = 1

    steps_width = int(image_width // crop_width)
    steps_height = int(image_height // crop_height)

    #Loop through the image and create crops
    for row in range (0, steps_height):
        for col in range(0, steps_width):
            
            # print(f"Computing Image: {(depth_layer):04d} / {processed_data.shape[0]} - Crop {counter:02d}")
    
            left = col * crop_width
            top = row * crop_height
            cropped_image_intensity = intensity_layer[top:top + crop_height, left:left + crop_width]
            cropped_image_phase = phase_layer[top:top + crop_height, left:left + crop_width]
            
            # Plotting both images in one figure
            fig, axs = plt.subplots(1, 2, figsize=(width, height), dpi=100)
        
            #Intensity Image
            im0 = axs[0].imshow(cropped_image_intensity, cmap='viridis', vmin=global_min, vmax=global_max)
            axs[0].set_title("Intensity")
            axs[0].set_xlabel('X-axis (spatial units)')
            axs[0].set_ylabel('Y-axis (spatial units)')
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label= f"{scaling_name} Intensity (arb. units)")

            #Phase Image
            im1 = axs[1].imshow(cropped_image_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
            axs[1].set_title("Phase")
            axs[1].set_xlabel('X-axis (spatial units)')
            axs[1].set_ylabel('Y-axis (spatial units)')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Phase (radians)")
        

            plt.suptitle(f"Frequency {layer_number:04d} / {processed_data.shape[0]} - Crop {counter:02d}")
            plt.tight_layout(pad=0.5)

            crop_filepath = f"{output_filename_pre}crop_{(counter):04d}_row_{(row):02d}_col_{(col):02d}.png"
            plt.savefig(crop_filepath, bbox_inches=None)
            plt.close()
            counter += 1


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

#---Define Crop Size
crop_size = get_crop_size(filename_num)

#---Create Output Folder---
output_folder = os.path.join(f"{output_folder_pre}_{scaling_name}", "combined_images_cropped")
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


#########################################################
####OLD CODE - NOT USED
# # Use a few CPUs for parallel processing
# with ProcessPoolExecutor(max_workers=32) as executor:
#     executor.map(process_and_save, )
#########################################################



intensity_layers = get_list_of_intensity_layers(processed_data)
phase_layers = get_list_of_phase_layers(processed_data)



# Use ProcessPoolExecutor to process images in parallel
with ProcessPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(process_and_save, i, intensity_layers[i], phase_layers[i], global_min, global_max, output_folder, crop_size, scaling_name) for i in range((processed_data.shape[0]))]
    for future in futures:
        try:
            future.result()  # Will raise if exception happened in subprocess
        except Exception as e:
            print(f"Error occurred: {e}")



# process_and_save(0)



print("\nAll Images Saved Sucessfully.\n")
print(f"Global min intensity:            {global_min:.4f}")
print(f"Global max intensity:            {global_max:.4f}")
print("NF - Number of Frequencies:      ", processed_data.shape[0])
print("NX - Number of Points on X-Axis: ", processed_data.shape[1])
print("NY - Number of Points on Y-Axis: ", processed_data.shape[2])
print("File Number:                     ",filename_num)
print("Scaling Type:                    ",scaling_type, scaling_name)
print("Mat File:                        ",filename_mat)
print("Output Folder:                   ", output_folder)
end_time = time.time()
print(f"Script finished in                {end_time - start_time:.2f} seconds.")
print("End of Script")