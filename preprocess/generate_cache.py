"""IMPORT PACKAGES"""
import os
import json
import numpy as np
import pandas as pd
from skimage.measure import label
from scipy import ndimage
from PIL import Image
from numba import jit
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

"""SPECIFY EXTENSIONS AND DATA ROOTS"""
EXT_VID = ['.mp4', '.m4v', '.avi']
EXT_IMG = ['.jpg', '.png', '.tiff', '.tif', '.bmp', '.jpeg']


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""DEFINE SET OF FUNCTIONS FOR SELECTING ROI IN RAW ENDOSCOPE IMAGES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


# Define function for minimum pooling the images
@jit(nopython=True)
def min_pooling(img, g=8):
    print(img.shape)
    # Copy Image
    out = img.copy()

    # Determine image shape and compute step size for pooling
    h, w = img.shape
    nh = int(h / g)
    nw = int(w / g)

    # Perform minimum pooling
    for y in range(nh):
        for x in range(nw):
            out[g * y : g * (y + 1), g * x : g * (x + 1)] = np.min(out[g * y : g * (y + 1), g * x : g * (x + 1)])

    return out


# Define function for finding the largest connected region in images
def getlargestcc(segmentation):
    # Use built-in label method, to label connected regions of an integer array
    labels = label(segmentation)

    # Assume at least 1 CC
    assert labels.max() != 0  # assume at least 1 CC

    # Find the largest of connected regions, return as True and False
    largestcc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largestcc


# Define function for finding bounding box coordinates for ROI in images
def bbox(img):
    # Find rows and columns where a True Bool is encountered
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    # Find the first and last row/column for the bounding box coordinates
    # cmin = left border, cmax = right border, rmin = top border, rmax = bottom border
    # Usage Image.crop((left, top, right, bottom))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


# Define Complete function for finding ROI bounding box coordinates, by combining previous functions
def find_roi(img):
    # Open image as numpy array
    image = np.array(img, dtype=np.uint8)

    # Compute L1 norm of the image
    norm_org = np.linalg.norm(image, axis=-1)

    # Use Gaussian Filter to capture low-frequency information
    img_gauss = ndimage.gaussian_filter(norm_org, sigma=5)

    # Scale pixel values
    img_scaled = ((norm_org - np.min(img_gauss)) / (np.max(img_gauss) - np.min(img_gauss))) * 255
    print('Image scaled shape: ', img_scaled.shape)
    # Use minimum pooling
    img_norm = min_pooling(img_scaled, g=8)

    # Find largest connected region with threshold image as input
    th = 10
    largestcc = getlargestcc(img_norm >= th)

    # Obtain cropping coordinates
    rmin, rmax, cmin, cmax = bbox(largestcc)

    return rmin, rmax, cmin, cmax


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""DEFINE FUNCTION FOR CREATING CACHE WITH METADATA FOR AVAILABLE IMAGES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""


def cache_wle(root_dir, mask_dir, subtlety_dir, quality_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    os.makedirs(storing_folder, exist_ok=True)

    # Create empty dictionary for  image files
    img_files = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):
        # Loop over filenames in files
        for name in files:
            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for image files
    maskdict = dict()

    # Loop over roots (folders in OTMASKDIR), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(mask_dir):
        # if 'Lowerlikelihood' in root:
        # Loop over filenames in files
        for file in files:
            # Extract filename before .ext
            # print('Lowerlikelihood File: ', file)
            maskcase = os.path.splitext(file)[0]
            # print('Maskcase: ', maskcase)

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))
    print('Maskdict: ', maskdict)
    # Read Subtlety database as excel file; Create empty dictionary for subtlety
    subtlety_df = pd.read_excel(subtlety_dir)
    subtlety_dict = dict()

    # Iterate over rows in excel file; initialize img name key in subtlety dict, set value based on subtlety
    for idx, frame in subtlety_df.iterrows():
        if 'Imagename' not in frame.keys():
            print('Error: Imagename not in keys: ', frame.keys())
            continue
        # print('Frame name: ', frame['Imagename'])
        subtlety_dict[os.path.splitext(frame['Imagename'])[0]] = frame['Subtlety (0=easy, 1=medium, 2=hard)']

    # Read Quality database as excel file; Create empty dictionary for quality
    # quality_df = pd.read_excel(quality_dir)
    # quality_dict = dict()

    # # Iterate over rows in excel file; initialize img name key in quality dict, set value based on quality
    # # It can happen that multiple same key occurs multiple times in the dict
    # for idx, frame in quality_df.iterrows():
    #     if 'Imagename' not in frame.keys():
    #         print('Error: Imagename not in keys: ', frame.keys())
    #         continue
    #     if os.path.splitext(frame['Imagename'])[0] not in quality_dict.keys():
    #         quality_dict[os.path.splitext(frame['Imagename'])[0]] = list()
    #         quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])
    #     elif os.path.splitext(frame['Imagename'])[0] in quality_dict.keys():
    #         quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""
    failed_files = []
    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        try:
            frame = np.array(Image.open(img))
        except Exception as e:
            print('Error: ', e)
            failed_files.append(img)
            continue

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General structure filename: hospital_patID_modality_source_pathology_..._..._..._
        # General folder structure: training/validatie > Prospectief/Retrospectief > ndbe/neo > image/frame >
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file
        data['dataset'] = os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[0])[
            1
        ]  # training/validatie
        data['source'] = os.path.split(os.path.split(img)[0])[1]  # image/frames
        data['class'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]  # ndbe/neo
        data['protocol'] = os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]
        data['modality'] = os.path.split(img)[1].split('_')[2]  # modality
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital
        data['subtlety'] = (
            subtlety_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in subtlety_dict.keys()
            else 0
        )
        if 'subtle' in img:
            data['subtlety'] = 2
        # data['quality'] = (
        #     quality_dict[os.path.splitext(os.path.split(img)[1])[0]]
        #     if os.path.splitext(os.path.split(img)[1])[0] in quality_dict.keys()
        #     else []
        # )

        # Open the image as numpy array; extract height and width and place in data dictionary

        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        # print("checking maskdict for: ", os.path.splitext(os.path.split(img)[1])[0])
        # print( os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys())
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]
        print('Number of masks: {}'.format(len(data['masks'])))

        # Print final version of data dictionary with all keys and values in there
        print('Data: ', data)

        # Check whether there is already a json file for this particular image; otherwise create file
        if not os.path.exists(
            os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '.json',
            )
        ):
            jsonfile = os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '.json',
            )
        elif not os.path.exists(
            os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '_2.json',
            )
        ):
            jsonfile = os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '_2.json',
            )
        elif not os.path.exists(
            os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '_3.json',
            )
        ):
            jsonfile = os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '_3.json',
            )
        elif not os.path.exists(
            os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '_4.json',
            )
        ):
            jsonfile = os.path.join(
                os.getcwd(),
                '..',
                'cache folders',
                storing_folder,
                os.path.splitext(imgname)[0] + '_4.json',
            )
        else:
            print('All files already exist')
            continue
            # raise ValueError('All files already exist')

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    print('Failed files: ', failed_files)
"""""" """""" """""" """""" ""
"""EXECUTION OF FUNCTIONS"""
"""""" """""" """""" """""" ""
if __name__ == '__main__':
    # create parser object
    parser = argparse.ArgumentParser(description='Generate cache for WLE images')
    parser.add_argument('--root_dir', type=str, help='Root directory for WLE images')
    parser.add_argument('--mask_dir', type=str, help='Root directory for masks')
    parser.add_argument('--subtlety_dir', type=str, help='Root directory for subtlety')
    parser.add_argument('--quality_dir', type=str, help='Root directory for quality')
    
    
    """DEFINE PATHS AND STORING FOLDERS FOR REGULAR TESTING"""
    parser.add_argument('--cache_dir', type=str, default='cache', help='Folder to store cache files')
    args = parser.parse_args()
    # Execute functions
    cache_wle(root_dir=args.root_dir, mask_dir=args.mask_dir, subtlety_dir=args.subtlety_dir,
              quality_dir=args.quality_dir, storing_folder=args.cache_dir)
