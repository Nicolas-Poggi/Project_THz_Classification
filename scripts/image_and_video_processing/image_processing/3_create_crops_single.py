import time
start_time = time.time()
import sys
import os
if len(sys.argv) != 3:
    print("\nError: Incorrect number of arguments")
    print("Usage: python imageCreatorParallel.py <num1 - Number of File> <num2 - Scaling Type>\n\n")
    print("num1: 0) 02625_Backside  1) 02625_Backside_noLens  2) 05250_Backside_noLens  3) USAF\n")
    print("num2: 1) Normal Scaling  2) Log Scaling   3) Softmax Scaling\n")
    sys.exit(1)

from PIL import Image
from concurrent.futures import ProcessPoolExecutor



print("\nPython Script - Nicolas Poggi - create_crops_creator_parallel.py")
print("Bachelor Thesis - FSS 2025")
print("Importing Libraries")


def create_crops(filenumber):
    """
    Create square crops from an image and save them to the output folder.
    Args:
        filenumber (int): The number of the file to process.
    
    Returns:
        None
    """
    filenumber = filenumber + 1
    
    output_filename_pre = ""

    if(filenumber%2 == 0):
        filenumber = int(filenumber / 2)
        print(f"Computing Crops Intensity Image: {(filenumber+1):04d} / 1400")
        input_image_path = os.path.join(input_folder, f"depth_image_layer_intensity_{filenumber:04d}.png")
        output_folder_layer = os.path.join(output_folder, f"depth_image_layer_{filenumber:04d}")
        os.makedirs(output_folder_layer, exist_ok=True)

        output_filename_pre = os.path.join(output_folder_layer, f"depth_image_layer_intensity_{filenumber:04d}_")
        
    else:
        filenumber = int((filenumber + 1) / 2)
        print(f"Computing Crops Phase Image: {(filenumber):04d} / 1400")
        input_image_path = os.path.join(input_folder, f"depth_image_layer_phase_{filenumber:04d}.png")
        output_folder_layer = os.path.join(output_folder, f"depth_image_layer_{filenumber:04d}")
        os.makedirs(output_folder_layer, exist_ok=True)

        output_filename_pre = os.path.join(output_folder_layer,f"depth_image_layer_phase_{filenumber:04d}_")
        

    img = Image.open(input_image_path)
    width, height = img.size
    crop_width, crop_height = crop_size

    counter = 0
    # Slide over the image with stride 1 (pixel by pixel)
    for row in range (0, int(height/crop_height)):
        for col in range(0, int(width/crop_width)):
            
            left = col * crop_width
            top = row * crop_height

            box = (left, top, left + crop_width, top + crop_height)
            crop = img.crop(box)

            crop_filepath = f"{output_filename_pre}crop_{(counter+1):04d}_row_{(row):02d}_col_{(col):02d}.png"
            crop.save(crop_filepath)
            counter += 1


def get_input_folder(filename_num, scaling_name):
    if filename_num == 0:
        input_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_{scaling_name}/single_images'
    
    elif filename_num == 1:
        input_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/1_02625_Backside_noLens_{scaling_name}/single_images'
    
    elif filename_num == 2:
        input_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/2_05250_Backside_noLens_{scaling_name}/single_images'
    
    elif filename_num == 3:
        input_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/3_USAF_{scaling_name}/single_images'
    
    else:
        print("Error: Unknown File Number")
        exit(1)
    
    return input_folder

def get_output_folder_pre(filename_num, scaling_name):
    if filename_num == 0:
        output_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/0_02625_Backside_{scaling_name}/single_images_cropped'
    
    elif filename_num == 1:
        output_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/1_02625_Backside_noLens_{scaling_name}/single_images_cropped'
    
    elif filename_num == 2:
        output_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/2_05250_Backside_noLens_{scaling_name}/single_images_cropped'
    
    elif filename_num == 3:
        output_folder = f'/pfs/work9/workspace/scratch/ma_npoggigo-bachelor_thesis_fss2025/Thz_Data/Data/Processed_Image_Data/3_USAF_{scaling_name}/single_images_cropped'
    
    else:
        print("Error: Unknown File Number")
        exit(1)
    
    return output_folder

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

def get_crop_size(filename_num):
    '''Crop size for width x height
    '''
    if filename_num == 0:

        crop_size = (26,26)
    
    elif filename_num == 1:
        crop_size = (16.7,20)
    
    elif filename_num == 2:
        crop_size = (25,21)
    
    elif filename_num == 3:
        crop_size = (45.7,43.8) 

    else:
        print("Error: Unknown File Number")
        exit(1)

    return crop_size

def get_input_size(input_image_path):
    img = Image.open(input_image_path)
    return img.size

print("Functions Defined")
print("Starting Script")


#---Define Scaling Type
#Type of Image Scaling
# 1) Normal Scaling  2) Log Scaling   3) Softmax Scaling
scaling_type = int(sys.argv[2])
scaling_name = getScaleName(scaling_type)


filename_num = int(sys.argv[1])
input_folder = get_input_folder(filename_num, scaling_name)
output_folder = get_output_folder_pre(filename_num, scaling_name)
crop_size = get_crop_size(filename_num)

os.makedirs(output_folder, exist_ok=True)
    
create_crops(0)





# with ProcessPoolExecutor(max_workers=32) as executor:
#     executor.map(create_crops, range(0,2800))

#Get the size of the input image - With example image (ALL images have the same size)
input_image_path = os.path.join(input_folder, f"depth_image_layer_intensity_0001.png")
input_size = get_input_size(input_image_path)



print("\nAll Images Saved Sucessfully.\n")
print("File Number:                                       ", filename_num)
print("Scaling Type:                                      ", scaling_type, scaling_name)
print("Input Folder:                                      ", input_folder)
print("Output Folder:                                     ", output_folder)
print("Input Image Size:                                  ", input_size)
print("Crop Size:                                         ", crop_size)
print("Number of Rows Created:                            ", int(input_size[0] / crop_size[0]))
print("Number of Columns Created:                         ", int(input_size[1] / crop_size[1]))
print("Number of Crops Created (oer Image):               ", int((input_size[0] / crop_size[0]) * (input_size[1] / crop_size[1])))
end_time = time.time()
print(f"\nScript finished in                                {end_time - start_time:.2f} seconds.")
print("End of Script")