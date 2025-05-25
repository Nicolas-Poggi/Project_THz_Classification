from PIL import Image
import sys

if len(sys.argv) != 2:
    print("Usage: python get_image_size.py <path_to_png_file>")
    sys.exit(1)

image_path = sys.argv[1]

try:
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"Image size: {width}x{height} (Width x Height)")
except FileNotFoundError:
    print(f"Error: File not found -> {image_path}")
except Exception as e:
    print(f"Error reading image: {e}")



#USAF                   Image size: 560x455 (Width x Height) Y N  - 560 464
#02625-BackSide         Image size: 575x389 (Width x Height) N N  - 576 400
#02625-Backside-no-Lens Image size: 575x394 (Width x Height) N N    576 400
#05250-Backside-no-Lens Image size: 562x389 (Width x Height) N N    576 400
