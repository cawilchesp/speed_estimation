from pathlib import Path

# Folder parameters
ROOT = Path('D:/Data')
SOURCE_FOLDER = ROOT / 'VIDEOS_DAI'
OUTPUT_FOLDER = ROOT / 'VIDEOS_DAI'
MODEL_FOLDER = ROOT / 'models' / 'yolov10'

# Source parameters
INPUT_VIDEO = '00000002761000000_new.mp4'
OUTPUT_NAME = '00000002761000000_new_vTest'
JSON_NAME = '00000002761000000_new_region.json'

# Deep Learning model configuration
MODEL_WEIGHTS = 'yolov10b.pt'

# Inference configuration
CLASS_FILTER = [0,1,2,3,5,7]
IMAGE_SIZE = 640
CONFIDENCE = 0.1
