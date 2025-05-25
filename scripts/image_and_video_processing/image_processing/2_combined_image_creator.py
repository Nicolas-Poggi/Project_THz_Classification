import sys
sys.path.insert(1, "src_code/")
import process_rdf as prdf
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image


print()
print("---Script started---")

#---Define Mat File for reading & Output Folder for Images
filename_mat  = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/Rohacell_10mm_RDX_300GHz_02625_backside.mat'

#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/USAF_PCB_01_02625_750_G.mat'
#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/Rohacell_10mm_RDX_300GHz_02625_backside.mat'
#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/Rohacell_10mm_RDX_300GHz_02625_backside_no_lens_215mm_rp.mat'
#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Original_Data/Rohacell_10mm_RDX_300GHz_05250_backside_no_lens_215mm_rp.mat'

output_folder = '/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/Softmax_02625_Backside'

#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside'
#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/1_02625_Backside_noLens'
#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/2_05250_Backside_noLens'
#'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/3_USAF'

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

softmax_x = F.softmax(abs_squared, dim=0)

global_min = softmax_x.min().item()
global_max = softmax_x.max().item()



#global_min = torch.log(abs_squared).min().item()
#global_max = torch.log(abs_squared).max().item()

print(f"Global min intensity:            {global_min:.4f}")
print(f"Global max intensity:            {global_max:.4f}")


#---Print info about Mat File---
print("NF - Number of Frequencies:      ", processed_data.shape[0])
print("NX - Number of Points on X-Axis: ", processed_data.shape[1])
print("NY - Number of Points on Y-Axis: ", processed_data.shape[2])


#---UNCOMMENT TO Loop through all possible Depth Layers.---------------------
for depth_layer in range(processed_data.shape[0]):
    print(f"Computing Image: {(depth_layer+1):04d} / {processed_data.shape[0]}")
    image_depth = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)).transpose(0, 1).detach().cpu().numpy()

    
    # compute image depth as the argmax of the absolut value of the processed data
    #image_depth = np.argmax(np.abs(processed_data.cpu().numpy()), axis=0)

    # Create a figure with exact pixel size 
    #USAF                   Image size: 560x455 (Width x Height) Y N  - 560 464
    #02625-BackSide         Image size: 575x389 (Width x Height) N N  - 576 400
    #02625-Backside-no-Lens Image size: 575x394 (Width x Height) N N    576 400
    #05250-Backside-no-Lens Image size: 562x389 (Width x Height) N N    576 400
    width = 5.92
    height = 4.8

    plt.figure(figsize=(width, height), dpi=100)
    
    #Plotting
    plt.imshow(image_depth, cmap='viridis', vmin=global_min, vmax=global_max)  
    plt.title(f'Frequency: {(depth_layer+1):04d} / {processed_data.shape[0]}')  # Title
    plt.xlabel('X-axis (spatial units)')  # X label
    plt.ylabel('Y-axis (spatial units)')  # Y label
    
    # Colorbar (heatmap)
    cbar = plt.colorbar()  
    cbar.set_label('Intensity (arb. units)')  # Colorbar label
    # Adjust layout to center content
    plt.tight_layout(pad=0.5)

    print(f"Saving Image:    {(depth_layer+1):04d} / {processed_data.shape[0]}")

    #Save Image in Output-Folder
    filename = f"depth_image_layer_{(depth_layer+1):04d}.png"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, bbox_inches=None)
    plt.close()

    print(f"Cropping Image:  {(depth_layer+1):04d} / {processed_data.shape[0]}")

    
    im = Image.open(save_path)

    # If it's off by one, pad it manually
    target_size = ((width * 100),(height*100))
    
    if im.size != target_size:
        print(f"Original size: {im.size}, resizing to {target_size}")
        im = im.resize(target_size, resample=Image.LANCZOS)
        im.save(save_path)
    
    print(f"Saving Crop:     {(depth_layer+1):04d} / {processed_data.shape[0]} - Target Size: {target_size} Final Size: {im.size}")
print("\nAll Images Saved Sucessfully.")

print(f"Global min intensity:            {global_min:.4f}")
print(f"Global max intensity:            {global_max:.4f}")
print("NF - Number of Frequencies:      ", processed_data.shape[0])
print("NX - Number of Points on X-Axis: ", processed_data.shape[1])
print("NY - Number of Points on Y-Axis: ", processed_data.shape[2])
print("Mat File:                        ",filename_mat)
print("Output Folder:                   ", output_folder)

print("End of Script")
#----------------------------------------------------------------------------

# #---UNCOMMENT TO Show one possible frequency---------------------------------
# depth_layer = 0
# print(f"Computing Image: {depth_layer:04d} / {processed_data.shape[0]}")
# image_depth = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)).transpose(0, 1).detach().cpu().numpy()

# # compute image depth as the argmax of the absolut value of the processed data
# #image_depth = np.argmax(np.abs(processed_data.cpu().numpy()), axis=0)

# # Create a figure with exact pixel size   do 592x480 for all picturres
# #USAF                   Image size: 560x455 (Width x Height) Y N  - 560 464   --> try 16+ each value 576x480 
# #02625-BackSide         Image size: 575x389 (Width x Height) N N  - 576 400
# #02625-Backside-no-Lens Image size: 575x394 (Width x Height) N N    576 400
# #05250-Backside-no-Lens Image size: 562x389 (Width x Height) N N    576 400
# width = 5.92
# height = 4.8

# plt.figure(figsize=(width, height), dpi=100)

# #Plotting
# plt.imshow(image_depth)
# plt.title(f'Frequency: {depth_layer:04d} / {processed_data.shape[0]}')  # Title
# plt.xlabel('X-axis (spatial units)')  # X label
# plt.ylabel('Y-axis (spatial units)')  # Y label

# # Colorbar (heatmap)
# cbar = plt.colorbar()  
# cbar.set_label('Intensity (arb. units)')  # Colorbar label
# print(f"Saving Image:    {depth_layer:04d} / {processed_data.shape[0]}")

# # Adjust layout to center content
# plt.tight_layout(pad=0.5)
# #plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

# #Save Image in Output-Folder
# filename = f"depth_image_layer_{depth_layer:04d}.png"
# save_path = os.path.join(output_folder, filename)
# plt.savefig(save_path, bbox_inches=None)
# plt.clf()
# print("Saved at:", save_path)

# from PIL import Image
# im = Image.open(save_path)

# # If it's off by one, pad it manually
# target_size = ((width * 100),(height*100))
# print("Target: ",target_size)

# if im.size != target_size:
#     print(f"Original size: {im.size}, resizing to {target_size}")
#     im = im.resize(target_size, resample=Image.LANCZOS)
#     im.save(save_path)
# print("Final:  ",im.size)  # should print (560, 464)

# #----------------------------------------------------------------------------
