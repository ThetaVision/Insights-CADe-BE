"""IMPORT PACKAGES"""
import os
import json
import numpy as np
import pandas as pd
from skimage.measure import label
from scipy import ndimage
from PIL import Image
from numba import jit

"""SPECIFY EXTENSIONS AND DATA ROOTS"""
EXT_VID = ['.mp4', '.m4v', '.avi']
EXT_IMG = ['.jpg', '.png', '.tiff', '.tif', '.bmp', '.jpeg']


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""DEFINE SET OF FUNCTIONS FOR SELECTING ROI IN RAW ENDOSCOPE IMAGES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


# Define function for minimum pooling the images
@jit(nopython=True)
def min_pooling(img, g=8):
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
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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
        if 1:  # 'Lowerlikelihood' in root:
            # Loop over filenames in files
            for file in files:
                # Extract filename before .ext
                maskcase = os.path.splitext(file)[0]

                # Append filename to mask dictionary if already existing key; otherwise create key and append to list
                if maskcase in maskdict.keys():
                    maskdict[maskcase].append(os.path.join(root, file))
                else:
                    maskdict[maskcase] = list()
                    maskdict[maskcase].append(os.path.join(root, file))

    # Read Subtlety database as excel file; Create empty dictionary for subtlety
    subtlety_df = pd.read_excel(subtlety_dir)
    subtlety_dict = dict()

    # Iterate over rows in excel file; initialize img name key in subtlety dict, set value based on subtlety
    for idx, frame in subtlety_df.iterrows():
        subtlety_dict[os.path.splitext(frame['Imagename'])[0]] = frame['Subtlety (0=easy, 1=medium, 2=hard)']

    # Read Quality database as excel file; Create empty dictionary for quality
    quality_df = pd.read_excel(quality_dir)
    quality_dict = dict()

    # Iterate over rows in excel file; initialize img name key in quality dict, set value based on quality
    # It can happen that multiple same key occurs multiple times in the dict
    for idx, frame in quality_df.iterrows():
        if os.path.splitext(frame['Imagename'])[0] not in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]] = list()
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])
        elif os.path.splitext(frame['Imagename'])[0] in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])

    # Initialize counter for total number of masks and number of mismatching frame and mask shapes
    counter_mismatch = 0
    counter_mask_total = 0

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

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
        data['quality'] = (
            quality_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in quality_dict.keys()
            else []
        )

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]
        print('Number of masks: {}'.format(len(data['masks'])))

        # Build in check for matching shape frame and mask
        if len(data['masks']) > 0:
            counter_mask_total += 1
            mask = np.array(Image.open(data['masks'][0]))
            print('Mask shape: ', mask.shape)
            if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                counter_mismatch += 1

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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    # Print the amount of mismatches in shape
    print('Total number of masks: {}'.format(counter_mask_total))
    print('Number of mismatches in frame and mask shape: {}'.format(counter_mismatch))


def cache_born(root_dir, mask_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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
        if 1:  # 'Lowerlikelihood' in root:
            # Loop over filenames in files
            for file in files:
                # Extract filename before .ext
                maskcase = os.path.splitext(file)[0]

                # Append filename to mask dictionary if already existing key; otherwise create key and append to list
                if maskcase in maskdict.keys():
                    maskdict[maskcase].append(os.path.join(root, file))
                else:
                    maskdict[maskcase] = list()
                    maskdict[maskcase].append(os.path.join(root, file))

    # Initialize counter for total number of masks and number of mismatching frame and mask shapes
    counter_mismatch = 0
    counter_mask_total = 0

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        if 'IWGCO-BORN' in imgname:
            data['patient'] = imgname.split('_')[0].split('-')[2]
        else:
            data['patient'] = imgname.split('_')[0]
        data['modality'] = 'wle'
        data['file'] = img  # path to file
        data['class'] = os.path.split(os.path.split(img)[0])[1]  # ndbe/neo
        data['subtlety'] = 0
        data['quality'] = []

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]
        print('Number of masks: {}'.format(len(data['masks'])))

        # Build in check for matching shape frame and mask
        if len(data['masks']) > 0:
            counter_mask_total += 1
            mask = np.array(Image.open(data['masks'][0]))
            print('Mask shape: ', mask.shape)
            if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                counter_mismatch += 1

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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    # Print the amount of mismatches in shape
    print('Total number of masks: {}'.format(counter_mask_total))
    print('Number of mismatches in frame and mask shape: {}'.format(counter_mismatch))


def cache_argos(root_dir, mask_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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
        if 1:  # 'Lowerlikelihood' in root:
            # Loop over filenames in files
            for file in files:
                # Extract filename before .ext
                maskcase = os.path.splitext(file)[0]

                # Append filename to mask dictionary if already existing key; otherwise create key and append to list
                if maskcase in maskdict.keys():
                    maskdict[maskcase].append(os.path.join(root, file))
                else:
                    maskdict[maskcase] = list()
                    maskdict[maskcase].append(os.path.join(root, file))

    # Initialize counter for total number of masks and number of mismatching frame and mask shapes
    counter_mismatch = 0
    counter_mask_total = 0

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file
        data['dataset'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]  # dataset 3/4/5
        data['class'] = os.path.split(os.path.split(img)[0])[1]  # ndbe/neo
        data['modality'] = 'wle'
        data['subtlety'] = 0
        data['quality'] = []

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]
        print('Number of masks: {}'.format(len(data['masks'])))

        # Build in check for matching shape frame and mask
        if len(data['masks']) > 0:
            counter_mask_total += 1
            mask = np.array(Image.open(data['masks'][0]))
            print('Mask shape: ', mask.shape)
            if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                counter_mismatch += 1

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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    # Print the amount of mismatches in shape
    print('Total number of masks: {}'.format(counter_mask_total))
    print('Number of mismatches in frame and mask shape: {}'.format(counter_mismatch))


def cache_wle_corrupt_test(
    root_dir,
    mask_dir,
    subtlety_dir,
    quality_dir,
    conversion_file,
    storing_folder,
):
    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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
        if 1:  # 'Lowerlikelihood' in root:
            # Loop over filenames in files
            for file in files:
                # Extract filename before .ext
                maskcase = os.path.splitext(file)[0]

                # Append filename to mask dictionary if already existing key; otherwise create key and append to list
                if maskcase in maskdict.keys():
                    maskdict[maskcase].append(os.path.join(root, file))
                else:
                    maskdict[maskcase] = list()
                    maskdict[maskcase].append(os.path.join(root, file))

    # Read Subtlety database as excel file; Create empty dictionary for subtlety
    subtlety_df = pd.read_excel(subtlety_dir)
    subtlety_dict = dict()

    # Iterate over rows in excel file; initialize img name key in subtlety dict, set value based on subtlety
    for idx, frame in subtlety_df.iterrows():
        subtlety_dict[os.path.splitext(frame['Imagename'])[0]] = frame['Subtlety (0=easy, 1=medium, 2=hard)']

    # Read Quality database as excel file; Create empty dictionary for quality
    quality_df = pd.read_excel(quality_dir)
    quality_dict = dict()

    # Iterate over rows in excel file; initialize img name key in quality dict, set value based on quality
    # It can happen that multiple same key occurs multiple times in the dict
    for idx, frame in quality_df.iterrows():
        if os.path.splitext(frame['Imagename'])[0] not in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]] = list()
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])
        elif os.path.splitext(frame['Imagename'])[0] in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])

    # Read conversion file
    df = pd.read_excel(conversion_file)
    original_files = df['original file'].values.tolist()
    corrupted_files = df['corrupted file'].values.tolist()

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Extract original imagename from list
        original_file = original_files[corrupted_files.index(imgname)]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General structure filename: hospital_patID_modality_source_pathology_..._..._..._
        # General folder structure: training/validatie > Prospectief/Retrospectief > ndbe/neo > image/frame >
        data['patient'] = '_'.join(os.path.split(original_file)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file
        data['dataset'] = 'test-corrupt'
        data['original'] = original_file
        data['source'] = ''
        if 'ndbe' in original_file:
            data['class'] = 'ndbe'
        else:
            data['class'] = 'neo'
        data['protocol'] = ''
        data['modality'] = ''
        data['clinic'] = os.path.split(original_file)[1].split('_')[0]  # hospital
        data['subtlety'] = (
            subtlety_dict[os.path.splitext(os.path.split(original_file)[1])[0]]
            if os.path.splitext(os.path.split(original_file)[1])[0] in subtlety_dict.keys()
            else 0
        )
        if 'subtle' in img:
            data['subtlety'] = 2
        data['quality'] = (
            quality_dict[os.path.splitext(os.path.split(original_file)[1])[0]]
            if os.path.splitext(os.path.split(original_file)[1])[0] in quality_dict.keys()
            else []
        )

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]
        print('Number of masks: {}'.format(len(data['masks'])))

        # Build in check for matching shape frame and mask
        if len(data['masks']) > 0:
            mask = np.array(Image.open(data['masks'][0]))
            print('Mask shape: ', mask.shape)

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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_allcomers(root_dir, mask_dir, subtlety_dir, quality_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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
        if 1:  # 'Lowerlikelihood' in root:
            # Loop over filenames in files
            for file in files:
                # Extract filename before .ext
                maskcase = os.path.splitext(file)[0]

                # Append filename to mask dictionary if already existing key; otherwise create key and append to list
                if maskcase in maskdict.keys():
                    maskdict[maskcase].append(os.path.join(root, file))
                else:
                    maskdict[maskcase] = list()
                    maskdict[maskcase].append(os.path.join(root, file))

    # Read Subtlety database as excel file; Create empty dictionary for subtlety
    subtlety_df = pd.read_excel(subtlety_dir)
    subtlety_dict = dict()

    # Iterate over rows in excel file; initialize img name key in subtlety dict, set value based on subtlety
    for idx, frame in subtlety_df.iterrows():
        subtlety_dict[os.path.splitext(frame['Imagename'])[0]] = frame['Subtlety (0=easy, 1=medium, 2=hard)']

    # Read Quality database as excel file; Create empty dictionary for quality
    quality_df = pd.read_excel(quality_dir)
    quality_dict = dict()

    # Iterate over rows in excel file; initialize img name key in quality dict, set value based on quality
    # It can happen that multiple same key occurs multiple times in the dict
    for idx, frame in quality_df.iterrows():
        if os.path.splitext(frame['Imagename'])[0] not in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]] = list()
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])
        elif os.path.splitext(frame['Imagename'])[0] in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General structure filename: hospital_patID_modality_source_pathology_..._..._..._
        # General folder structure: training/validatie > Prospectief/Retrospectief > ndbe/neo > image/frame >
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file
        data['dataset'] = 'all-comers'
        data['class'] = os.path.split(os.path.split(img)[0])[1]
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital
        data['subtlety'] = (
            subtlety_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in subtlety_dict.keys()
            else 0
        )
        if 'subtle' in img:
            data['subtlety'] = 2
        data['quality'] = (
            quality_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in quality_dict.keys()
            else []
        )

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_validation_xxl(root_dir, mask_dir, subtlety_dir, quality_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    if not os.path.exists(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder)):
        os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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
        if 1:  # 'Lowerlikelihood' in root:
            # Loop over filenames in files
            for file in files:
                # Extract filename before .ext
                maskcase = os.path.splitext(file)[0]

                # Append filename to mask dictionary if already existing key; otherwise create key and append to list
                if maskcase in maskdict.keys():
                    maskdict[maskcase].append(os.path.join(root, file))
                else:
                    maskdict[maskcase] = list()
                    maskdict[maskcase].append(os.path.join(root, file))

    # Read Subtlety database as the Excel file; Create empty dictionary for subtlety
    subtlety_df = pd.read_excel(subtlety_dir)
    subtlety_dict = dict()

    # Iterate over rows in the Excel file; initialize img name key in subtlety dict, set value based on subtlety
    for idx, frame in subtlety_df.iterrows():
        subtlety_dict[os.path.splitext(frame['Imagename'])[0]] = frame['Subtlety (0=easy, 1=medium, 2=hard)']

    # Read Quality database as the Excel file; Create empty dictionary for quality
    quality_df = pd.read_excel(quality_dir)
    quality_dict = dict()

    # Iterate over rows in Excel file; initialize img name key in quality dict, set value based on quality
    # It can happen that multiple same key occurs multiple times in the dict
    for idx, frame in quality_df.iterrows():
        if os.path.splitext(frame['Imagename'])[0] not in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]] = list()
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])
        elif os.path.splitext(frame['Imagename'])[0] in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General structure filename: hospital_patID_modality_source_pathology_..._..._..._
        # General folder structure: training/validatie > Prospectief/Retrospectief > ndbe/neo > image/frame >
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file
        data['dataset'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]
        data['class'] = os.path.split(os.path.split(img)[0])[1]
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital
        data['subtlety'] = (
            subtlety_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in subtlety_dict.keys()
            else 0
        )
        if 'subtle' in img:
            data['subtlety'] = 2
        data['quality'] = (
            quality_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in quality_dict.keys()
            else []
        )

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]
        print('Number of masks: {}'.format(len(data['masks'])))

        # # Print final version of data dictionary with all keys and values in there
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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_nbi_ood(root_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General structure filename: hospital_patID_modality_source_pathology_..._..._..._
        # General folder structure: training/validatie > Prospectief/Retrospectief > ndbe/neo > image/frame >
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file
        data['modality'] = 'nbi'
        data['dataset'] = 'nbi-ood'
        data['class'] = 'neo' if 'eac' in imgname or 'hgd' in imgname else 'ndbe'
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital
        data['subtlety'] = 0
        data['quality'] = []

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()

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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_internship_ood(root_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General structure filename: hospital_patID_modality_source_pathology_..._..._..._
        # General folder structure: training/validatie > Prospectief/Retrospectief > ndbe/neo > image/frame >
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file
        data['dataset'] = 'internship-ood'
        data['class'] = 'neo' if 'eac' in imgname or 'hgd' in imgname else 'ndbe'
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital
        data['subtlety'] = 0
        data['quality'] = []
        data['content'] = os.path.split(os.path.split(img)[0])[1]

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        if frame.shape[0] == 1080 and frame.shape[1] == 1920:
            data['roi'] = [0.0, 1079.0, 552.0, 1895.0]
        elif frame.shape[0] == 540 and frame.shape[1] == 960:
            data['roi'] = [0.0, 539.0, 232.0, 895.0]
        else:
            roi = find_roi(frame)
            data['roi'] = [float(x) for x in roi]
            # raise ValueError(f'Incorrect shape: {frame.shape}')

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()

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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_txi_triplets_ood(root_dir, subtlety_dir, quality_dir, storing_folder):
    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder))

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

    # Read Subtlety database as excel file; Create empty dictionary for subtlety
    subtlety_df = pd.read_excel(subtlety_dir)
    subtlety_dict = dict()

    # Iterate over rows in excel file; initialize img name key in subtlety dict, set value based on subtlety
    for idx, frame in subtlety_df.iterrows():
        subtlety_dict[os.path.splitext(frame['Imagename'])[0]] = frame['Subtlety (0=easy, 1=medium, 2=hard)']

    # Read Quality database as excel file; Create empty dictionary for quality
    quality_df = pd.read_excel(quality_dir)
    quality_dict = dict()

    # Iterate over rows in excel file; initialize img name key in quality dict, set value based on quality
    # It can happen that multiple same key occurs multiple times in the dict
    for idx, frame in quality_df.iterrows():
        if os.path.splitext(frame['Imagename'])[0] not in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]] = list()
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])
        elif os.path.splitext(frame['Imagename'])[0] in quality_dict.keys():
            quality_dict[os.path.splitext(frame['Imagename'])[0]].append(frame['Quality'])

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """""" """""" """""" """""" """""" """""" ""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        # print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General structure filename: hospital_patID_modality_source_pathology_..._..._..._
        # General folder structure: training/validatie > Prospectief/Retrospectief > ndbe/neo > image/frame >
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['file'] = img  # path to file

        if 'test' in os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]:
            if 'patienten' in os.path.split(os.path.split(os.path.split(img)[0])[0])[1]:
                data['dataset'] = 'txi-triplets-test-patients'
            else:
                data['dataset'] = 'txi-triplets-test-imgs'
        elif 'validatie' in os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]:
            if 'patienten' in os.path.split(os.path.split(os.path.split(img)[0])[0])[1]:
                data['dataset'] = 'txi-triplets-val-patients'
            else:
                data['dataset'] = 'txi-triplets-val-imgs'
        elif 'training' in os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]:
            data['dataset'] = 'txi-triplets-train-imgs'
        else:
            data['dataset'] = 'txi-triplets-other'

        if 'WLE' in imgname:
            data['modality'] = 'wle'
        elif 'TXI1' in imgname:
            data['modality'] = 'txi1'
        elif 'TXI2' in imgname:
            data['modality'] = 'txi2'
        else:
            raise ValueError('Unrecognized modality')

        data['class'] = os.path.split(os.path.split(img)[0])[1]
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital
        data['subtlety'] = (
            subtlety_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in subtlety_dict.keys()
            else 0
        )
        if 'subtle' in img:
            data['subtlety'] = 2
        data['quality'] = (
            quality_dict[os.path.splitext(os.path.split(img)[1])[0]]
            if os.path.splitext(os.path.split(img)[1])[0] in quality_dict.keys()
            else []
        )

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()

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
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""DEFINE CACHE FUNCTION WITH METADATA FOR AVAILABLE IMAGES IN CAD2.0 DATASET"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""


def cache_wle_development_cad2(root_dir, mask_dir, storing_folder):
    # Create empty dictionary for  image files
    img_files = list()
    img_names = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):
        # Loop over filenames in files
        for name in files:
            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
                img_names.append(os.path.splitext(name)[0])
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for image files
    maskdict = dict()

    # Loop over roots (folders in OTMASKDIR), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            # Extract filename before .ext
            maskcase = os.path.splitext(file)[0]

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))

    # Create a skip list
    skip_list = list()

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR WHOLE DATASET"""
    """""" """""" """""" """""" """""" """""" ""

    # Create directory
    print('\nGenerating cache...')

    # Create dictionary
    data_complete = dict()

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    counter = 0
    for img in img_files:
        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract file and patient information from filename and place in dictionary with corresponding key
        data['file'] = img  # path to file
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital

        # Extract dataset information from filename and place in dictionary with corresponding key
        data['dataset'] = os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]
        data['type'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]
        data['quality'] = ''
        data['class'] = os.path.split(os.path.split(img)[0])[1]
        data['masks'] = (
            maskdict[os.path.splitext(imgname)[0]] if os.path.splitext(imgname)[0] in maskdict.keys() else []
        )

        # Open the image as numpy array; extract height and width and place in data dictionary
        try:
            frame = np.array(Image.open(img))
        except:
            skip_list.append(img)
            continue
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Print final version of data dictionary with all keys and values in there
        counter += 1
        print("\r" + f'Cache progress: {counter}/{len(img_files)} ({round((counter/len(img_files)*100), 4)}%)', end="")

        # Add data dictionary to complete dictionary
        data_complete[os.path.splitext(imgname)[0]] = data

    # Save complete dictionary as json file
    if not os.path.exists(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')):
        jsonfile = os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')
    else:
        raise ValueError

    with open(jsonfile, 'w') as outfile:
        json.dump(data_complete, outfile, indent=2)

    # Print Skipped List
    print('Skipped images: \n')
    print(len(skip_list), skip_list)


def cache_wle_testing_cad2(root_dir, mask_dir, storing_folder):
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
        for file in files:
            # Extract filename before .ext
            maskcase = os.path.splitext(file)[0]

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))

    # Create a skip list
    skip_list = list()

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR WHOLE DATASET"""
    """""" """""" """""" """""" """""" """""" ""

    # Create directory
    print('\nGenerating cache...')

    # Create dictionary
    data_complete = dict()

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    counter = 0
    for img in img_files:
        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract file and patient information from filename and place in dictionary with corresponding key
        data['file'] = img  # path to file
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital

        # data['class'] = os.path.split(os.path.split(img)[0])[1]
        # data['quality'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]
        # data['type'] = os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]
        # data['dataset'] = os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[0])[1]
        # data['masks'] = []

        # Extract dataset information from filename and place in dictionary with corresponding key
        if 'validation-robustness' in img:
            data['dataset'] = 'validation-robustness'
            data['type'] = os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]
            data['quality'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]
            data['class'] = os.path.split(os.path.split(img)[0])[1]
            data['masks'] = []
        else:
            data['dataset'] = os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]
            data['type'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]
            data['quality'] = ''
            data['class'] = os.path.split(os.path.split(img)[0])[1]
            data['masks'] = (
                [maskdict[os.path.splitext(imgname)[0]]] if os.path.splitext(imgname)[0] in maskdict.keys() else []
            )

        # Open the image as numpy array; extract height and width and place in data dictionary
        try:
            frame = np.array(Image.open(img))
        except:
            skip_list.append(img)
            continue
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Print final version of data dictionary with all keys and values in there
        counter += 1
        print("\r" + f'Cache progress: {counter}/{len(img_files)} ({round((counter/len(img_files)*100), 4)}%)', end="")

        # Add data dictionary to complete dictionary
        data_complete[os.path.splitext(imgname)[0]] = data

    # Save complete dictionary as json file
    if not os.path.exists(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')):
        jsonfile = os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')
    else:
        raise ValueError

    with open(jsonfile, 'w') as outfile:
        json.dump(data_complete, outfile, indent=2)

    # Print Skipped List
    print('Skipped images: \n')
    print(len(skip_list), skip_list)


def cache_wle_development_cad2_update(root_dir, mask_dir, storing_folder):
    # Create empty dictionary for  image files
    img_files = list()
    img_names = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):
        # Loop over filenames in files
        for name in files:
            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
                img_names.append(os.path.splitext(name)[0])
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for image files
    maskdict = dict()

    # Loop over roots (folders in OTMASKDIR), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            # Extract filename before .ext
            maskcase = os.path.splitext(file)[0]

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))

    # Create a skip list
    skip_list = list()

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR WHOLE DATASET"""
    """""" """""" """""" """""" """""" """""" ""

    # Create directory
    print('\nGenerating cache...')

    # Create dictionary
    data_complete = dict()

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    counter = 0
    for img in img_files:
        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract file and patient information from filename and place in dictionary with corresponding key
        data['file'] = img  # path to file
        data['patient'] = '_'.join(os.path.split(img)[1].split('_')[:2])  # hospital_patID
        data['clinic'] = os.path.split(img)[1].split('_')[0]  # hospital

        # Extract dataset information from filename and place in dictionary with corresponding key
        data['dataset'] = os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[0])[1]
        data['type'] = os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]
        data['source'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]
        data['class'] = os.path.split(os.path.split(img)[0])[1]
        data['quality'] = ''

        # Extract mask information from filename
        if data['type'] == 'enhanced' and data['class'] == 'neo':
            name_no_ext = os.path.splitext(imgname)[0]
            index = name_no_ext.find('_TXI_off')
            name_no_ext = name_no_ext[:index]
            data['masks'] = maskdict[name_no_ext] if name_no_ext in maskdict.keys() else []
        else:
            data['masks'] = (
                maskdict[os.path.splitext(imgname)[0]] if os.path.splitext(imgname)[0] in maskdict.keys() else []
            )

        # Open the image as numpy array; extract height and width and place in data dictionary
        try:
            frame = np.array(Image.open(img))
        except:
            skip_list.append(img)
            continue
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Print final version of data dictionary with all keys and values in there
        counter += 1
        print("\r" + f'Cache progress: {counter}/{len(img_files)} ({round((counter/len(img_files)*100), 4)}%)', end="")

        # Add data dictionary to complete dictionary
        data_complete[os.path.splitext(imgname)[0]] = data

    # Save complete dictionary as json file
    if not os.path.exists(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')):
        jsonfile = os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')
    else:
        raise ValueError

    with open(jsonfile, 'w') as outfile:
        json.dump(data_complete, outfile, indent=2)

    # Print Skipped List
    print('Skipped images: \n')
    print(len(skip_list), skip_list)


def cache_nbi_development_cad2(root_dir, storing_folder):
    # Create empty dictionary for  image files
    img_files = list()
    img_names = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):
        # Loop over filenames in files
        for name in files:
            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
                img_names.append(os.path.splitext(name)[0])
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create a skip list
    skip_list = list()

    """""" """""" """""" """""" """""" """""" ""
    """CREATE JSON FILE FOR WHOLE DATASET"""
    """""" """""" """""" """""" """""" """""" ""

    # Create directory
    print('\nGenerating cache...')

    # Create dictionary
    data_complete = dict()

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    counter = 0
    for img in img_files:
        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract file and patient information from filename and place in dictionary with corresponding key
        data['file'] = img  # path to file

        # Extract patient ID (CADx2.0 dataset)
        if 'karf' in imgname and '-' in imgname:
            data['patient'] = imgname.split('-')[0]
            data['clinic'] = 'karf'
        elif 'umcu-nbi' in imgname:
            data['patient'] = ('_'.join(imgname.split('_')[:2])).replace('-nbi', '')
            data['clinic'] = 'umcu'
        elif 'umcu_002' in imgname:
            data['patient'] = 'umcu_2'
            data['clinic'] = 'umcu'
        else:
            data['patient'] = '_'.join(imgname.split('_')[:2])
            data['clinic'] = imgname.split('_')[0]

        # Extract dataset information from filename and place in dictionary with corresponding key
        data['dataset'] = os.path.split(
            os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[0])[0]
        )[1]
        data['source'] = os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[0])[1]
        data['class'] = os.path.split(os.path.split(os.path.split(os.path.split(img)[0])[0])[0])[1]
        data['cap'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]
        data['type'] = os.path.split(os.path.split(img)[0])[1]

        # Open the image as numpy array; extract height and width and place in data dictionary
        try:
            frame = np.array(Image.open(img))
        except:
            skip_list.append(img)
            continue
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Print final version of data dictionary with all keys and values in there
        counter += 1
        print(
            "\r" + f'Cache progress: {counter}/{len(img_files)} ({round((counter / len(img_files) * 100), 4)}%)', end=""
        )

        # Add data dictionary to complete dictionary
        data_complete[os.path.splitext(imgname)[0]] = data

    # Save complete dictionary as json file
    if not os.path.exists(os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')):
        jsonfile = os.path.join(os.getcwd(), '..', 'cache folders', storing_folder + '.json')
    else:
        raise ValueError

    with open(jsonfile, 'w') as outfile:
        json.dump(data_complete, outfile, indent=2)

    # Print Skipped List
    print('Skipped images: \n')
    print(len(skip_list), skip_list)


"""""" """""" """""" """""" ""
"""EXECUTION OF FUNCTIONS"""
"""""" """""" """""" """""" ""
if __name__ == '__main__':
    """DEFINE PATHS AND STORING FOLDERS FOR REGULAR TESTING"""
    # # Define paths to the folders for WLE (Images, Masks, Subtlety, Quality)
    # wle_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/OTData11012022'
    # wle_dir_frames_MQ = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/OTData11012022 Frames MQ'
    # wle_dir_frames_LQ = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/OTData11012022 Frames LQ'
    # wle_mask_dir_soft = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/Mask_medical_softspot'
    # wle_mask_dir_plausible = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/Mask_medical_standard'
    # wle_mask_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/Mask_two_experts'
    # wle_subtle_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/Subtiliteit Verdeling/Subtiliteit_verdeling_11012022.xlsx'
    # wle_quality_dir = (
    #     'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/Kwaliteit Scores/Kwaliteit_score_11012022.xlsx'
    # )
    #
    # # Define paths to the folders for Artificial Degraded Test Set
    # wle_corrupt_test_img_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/OT-TestSet-Corrupt/Images'
    # wle_corrupt_test_mask_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/OT-TestSet-Corrupt/Masks'
    # wle_corrupt_test_conv_file = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/OT-TestSet-Corrupt/conversion_file.xlsx'
    #
    # # Define paths to the folders for BORN Module
    # wle_born_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/BORN Module/images'
    # wle_born_mask_medium = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/BORN Module/delineations/mediumspot'
    # wle_born_mask_sweet = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/BORN Module/delineations/sweetspot'
    #
    # Define paths to the folders for ARGOS
    # wle_argos_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/ARGOS Fuji Test Sets/images'
    # wle_argos_mask_soft = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/ARGOS Fuji Test Sets/delineations/consensus'
    # wle_argos_mask_all = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/ARGOS Fuji Test Sets/delineations/all masks'
    # wle_argos_dir = '/gpfs/work5/0/tesr0602/Datasets/ARGOS Fuji/images'
    # wle_argos_mask_soft = '/gpfs/work5/0/tesr0602/Datasets/ARGOS Fuji/delineations/consensus'
    # wle_argos_mask_all = '/gpfs/work5/0/tesr0602/Datasets/ARGOS Fuji/delineations/all masks'

    # # Define paths to the folders for Validation Set XXL
    # wle_validation_xxl_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/CADeV2 - Validation XXL'
    # wle_validation_xxl_mask_dir = (
    #     'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/Masks_plausible_complete'
    # )

    #
    # # # Define paths to the folders for All-Comers Test Set
    # # wle_allcomers_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/WLE Test All-Comers/images'
    # # wle_allcomers_mask_pl_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/WLE Test All-Comers/masks plausible'
    # # wle_allcomers_mask_soft_dir = 'C:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/WLE Test All-Comers/masks soft'
    # #
    # # # Define storing folders for NBI and WL
    # # wle_soft_store = 'cache_wle_train-val-test_soft'
    # # wle_plausible_store = 'cache_wle_train-val-test_plausible'
    # # wle_store = 'cache_wle_train-val-test_all_masks'
    # wle_store_frames_MQ = 'cache_wle_train-val_frames_MQ'
    # wle_store_frames_LQ = 'cache_wle_train-val_frames_LQ'
    # # wle_born_store_medium = 'cache_wle_born_medium'
    # # wle_born_store_sweet = 'cache_wle_born_sweet'
    # # wle_argos_store = 'cache_wle_argos_all_masks'
    # # wle_argos_store_soft = 'cache_wle_argos_soft'
    # # wle_corrupt_test_store = 'cache_wle_corrupt_test'
    # # wle_allcomers_plausible_store = 'cache_wle_allcomers_plausible'
    # # wle_validation_xxl_plausible_store = 'cache_wle_validation_xxl_plausible'
    #
    # # Execute functions
    # # cache_wle(root_dir=wle_dir, mask_dir=wle_mask_dir_soft, subtlety_dir=wle_subtle_dir,
    # #           quality_dir=wle_quality_dir, storing_folder=wle_soft_store)
    # # cache_wle(root_dir=wle_dir, mask_dir=wle_mask_dir_plausible, subtlety_dir=wle_subtle_dir,
    # #           quality_dir=wle_quality_dir, storing_folder=wle_plausible_store)
    # # cache_wle(root_dir=wle_dir, mask_dir=wle_mask_dir, subtlety_dir=wle_subtle_dir,
    # #           quality_dir=wle_quality_dir, storing_folder=wle_store)
    # cache_wle(
    #     root_dir=wle_dir_frames_MQ,
    #     mask_dir=wle_mask_dir,
    #     subtlety_dir=wle_subtle_dir,
    #     quality_dir=wle_quality_dir,
    #     storing_folder=wle_store_frames_MQ,
    # )
    # cache_wle(
    #     root_dir=wle_dir_frames_LQ,
    #     mask_dir=wle_mask_dir,
    #     subtlety_dir=wle_subtle_dir,
    #     quality_dir=wle_quality_dir,
    #     storing_folder=wle_store_frames_LQ,
    # )
    #
    # # BORN Module
    # # cache_born(root_dir=wle_born_dir, mask_dir=wle_born_mask_medium, storing_folder=wle_born_store_medium)
    # # cache_born(root_dir=wle_born_dir, mask_dir=wle_born_mask_sweet, storing_folder=wle_born_store_sweet)
    #
    # # ARGOS Fuji Test Sets
    # # cache_argos(root_dir=wle_argos_dir, mask_dir=wle_argos_mask_soft, storing_folder=wle_argos_store_soft)
    # # cache_argos(root_dir=wle_argos_dir, mask_dir=wle_argos_mask_all, storing_folder=wle_argos_store)
    #
    # # Artificial Degraded Test Set
    # # cache_wle_corrupt_test(root_dir=wle_corrupt_test_img_dir, mask_dir=wle_corrupt_test_mask_dir,
    # #                        subtlety_dir=wle_subtle_dir, quality_dir=wle_quality_dir,
    # #                        conversion_file=wle_corrupt_test_conv_file, storing_folder=wle_corrupt_test_store)
    #
    # # All-Comers Test Set
    # # cache_allcomers(root_dir=wle_allcomers_dir, mask_dir=wle_allcomers_mask_pl_dir, subtlety_dir=wle_subtle_dir,
    # #                 quality_dir=wle_quality_dir, storing_folder=wle_allcomers_plausible_store)
    # #
    # # # Validation Set XXL
    # # cache_validation_xxl(
    # #     root_dir=wle_validation_xxl_dir,
    # #     mask_dir=wle_validation_xxl_mask_dir,
    # #     subtlety_dir=wle_subtle_dir,
    # #     quality_dir=wle_quality_dir,
    # #     storing_folder=wle_validation_xxl_plausible_store,
    # # )
    #
    # """DEFINE PATHS AND STORING FOLDERS FOR OOD DATA"""
    #
    # # # Generate cache for the NBI OoD dataset
    # # wle_nbi_ood_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/WLE Test OoD/Modality/NBI'
    # # wle_nbi_ood_store = 'cache_nbi_ood'
    # # cache_nbi_ood(root_dir=wle_nbi_ood_dir, storing_folder=wle_nbi_ood_store)
    #
    # # Generate cache for the scientific internship dataset
    # # wle_internship_ood_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/WLE Test OoD/Wetenschappelijke stage'
    # # wle_internship_ood_store = 'cache_internship_ood'
    # # cache_internship_ood(root_dir=wle_internship_ood_dir, storing_folder=wle_internship_ood_store)
    #
    # # Generate cache for the TXI Triplets dataset
    # # wle_txi_triplets_ood_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG)/WLE Test OoD/TXI triplets'
    # # wle_txi_triplets_ood_store = 'cache_txi_triplets_ood'
    # # cache_txi_triplets_ood(root_dir=wle_txi_triplets_ood_dir, subtlety_dir=wle_subtle_dir, quality_dir=wle_quality_dir,
    # #                        storing_folder=wle_txi_triplets_ood_store)
    #
    # """DEFINE PATHS AND STORING FOLDERS FOR CAD2.0 DATASET"""
    #
    # # Define local paths to the folders for WLE
    # # wle_dev_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG) V2/WLE-Dev-Data-17122023'
    # # wle_test_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG) V2/WLE-Test-Data-17122023'
    # # mask_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG) V2/WLE-Masks/Plausible Spot'
    #
    # # # Define snellius paths to the folders for WLE
    # # wle_dev_cad2_dir = '/gpfs/work5/0/tesr0602/Datasets/WLEDataV2-CHJKusters/WLE-Dev-Data-17122023'
    # # wle_test_cad2_dir = '/gpfs/work5/0/tesr0602/Datasets/WLEDataV2-CHJKusters/WLE-Test-Data-17122023'
    # # mask_cad2_dir = '/gpfs/work5/0/tesr0602/Datasets/WLEDataV2-CHJKusters/WLE-Masks/Plausible Spot'
    # #
    # # # Define storing folders
    # # wle_dev_cad2_store = 'cache_wle_dev_cad2'
    # # wle_test_cad2_store = 'cache_wle_test_cad2'
    # #
    # # # Execute functions
    # # cache_wle_development_cad2(root_dir=wle_dev_cad2_dir, mask_dir=mask_cad2_dir, storing_folder=wle_dev_cad2_store)
    # # cache_wle_testing_cad2(root_dir=wle_test_cad2_dir, mask_dir=mask_cad2_dir, storing_folder=wle_test_cad2_store)
    #
    # """DEFINE PATHS AND STORING FOLDER FOR TRAINING AND VALIDATION DATA ON CLUSTER/SNELLIUS"""
    # # # Define paths to the folders for WLE
    # # wle_dir = '/share/medical/Sonic/WLEData-CHJKusters/OTData11012022'
    # # wle_mask_dir_plausible = '/share/medical/Sonic/WLEData-CHJKusters/Mask_medical_standard'
    # # wle_subtle_dir = '/share/medical/Sonic/WLEData-CHJKusters/Subtiliteit_verdeling_11012022.xlsx'
    # # wle_quality_dir = '/share/medical/Sonic/WLEData-CHJKusters/Kwaliteit_score_11012022.xlsx'
    #
    # wle_dir = '/gpfs/work5/0/tesr0602/WLEData-CHJKusters/OTData11012022'
    # wle_frames_dir_MQ = '/gpfs/work5/0/tesr0602/WLEData-CHJKusters/OTData11012022 Frames MQ'
    # wle_frames_dir_LQ = '/gpfs/work5/0/tesr0602/WLEData-CHJKusters/OTData11012022 Frames LQ'
    # wle_mask_dir = '/gpfs/work5/0/tesr0602/WLEData-CHJKusters/Mask_two_experts'
    # wle_mask_dir_plausible = '/gpfs/work5/0/tesr0602/WLEData-CHJKusters/Mask_medical_standard'
    # wle_subtle_dir = '/gpfs/work5/0/tesr0602/WLEData-CHJKusters/Subtiliteit_verdeling_11012022.xlsx'
    # wle_quality_dir = '/gpfs/work5/0/tesr0602/WLEData-CHJKusters/Kwaliteit_score_11012022.xlsx'
    #
    # # Define storing folders for NBI and WL
    # wle_plausible_store = 'cache_wle_train-val-test_plausible'
    # wle_all_mask_store = 'cache_wle_train-val-test_all_masks'
    # wle_frames_store_MQ = 'cache_wle_train-val_frames_MQ'
    # wle_frames_store_LQ = 'cache_wle_train-val_frames_LQ'
    #
    # # Execute functions
    # cache_wle(
    #     root_dir=wle_dir,
    #     mask_dir=wle_mask_dir_plausible,
    #     subtlety_dir=wle_subtle_dir,
    #     quality_dir=wle_quality_dir,
    #     storing_folder=wle_plausible_store,
    # )
    #
    # cache_wle(
    #     root_dir=wle_dir,
    #     mask_dir=wle_mask_dir,
    #     subtlety_dir=wle_subtle_dir,
    #     quality_dir=wle_quality_dir,
    #     storing_folder=wle_all_mask_store,
    # )
    #
    # cache_wle(
    #     root_dir=wle_frames_dir_MQ,
    #     mask_dir=wle_mask_dir,
    #     subtlety_dir=wle_subtle_dir,
    #     quality_dir=wle_quality_dir,
    #     storing_folder=wle_frames_store_MQ,
    # )
    # cache_wle(
    #     root_dir=wle_frames_dir_LQ,
    #     mask_dir=wle_mask_dir,
    #     subtlety_dir=wle_subtle_dir,
    #     quality_dir=wle_quality_dir,
    #     storing_folder=wle_frames_store_LQ,
    # )

    """DEFINE PATHS AND STORING FOLDERS FOR CADe2.0 DATASET"""

    # # Define local paths to the folders for WLE
    # wle_dev_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG) V2/WLE-Dev-Data-14052024'
    # cons_mask_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG) V2/WLE-Masks/Plausible Spot'
    # all_mask_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/WLE Algorithm (CL + SEG) V2/WLE-Masks-Experts'
    #
    # # # Define snellius paths to the folders for WLE
    # wle_dev_cad2_dir = '/gpfs/work5/0/tesr0602/Datasets/WLEDataV2-CHJKusters/WLE-Dev-Data-14052024'
    # cons_mask_cad2_dir = '/gpfs/work5/0/tesr0602/Datasets/WLEDataV2-CHJKusters/WLE-Masks/Plausible Spot'
    # all_mask_cad2_dir = '/gpfs/work5/0/tesr0602/Datasets/WLEDataV2-CHJKusters/WLE-Masks-Experts'
    #
    # # Define storing folders
    # wle_dev_cad2_store = 'cache_wle_dev_cad2_new'
    # wle_dev_cad2_store_am = 'cache_wle_dev_cad2_new_am'

    # # Execute functions
    # cache_wle_development_cad2_update(
    #     root_dir=wle_dev_cad2_dir, mask_dir=cons_mask_cad2_dir, storing_folder=wle_dev_cad2_store
    # )
    #
    # cache_wle_development_cad2_update(
    #     root_dir=wle_dev_cad2_dir, mask_dir=all_mask_cad2_dir, storing_folder=wle_dev_cad2_store_am
    # )

    """DEFINE PATHS AND STORING FOLDERS FOR CADx2.0 DATASET"""

    # Define local paths to the folders for NBI
    # nbi_dev_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/NBI Algorithm (CL) V2/NBI-Dev-Data-12082024'
    nbi_test_cad2_dir = 'D:/Data Barretts Esophagus - Olympus/NBI Algorithm (CL) V2/NBI-Test-Data-28082024'

    # # Define snellius paths to the folders for NBI
    # nbi_dev_cad2_dir = '/gpfs/work5/0/tesr0602/Datasets/NBIData-CHJKusters/NBI-Dev-Data-12082024'

    # Define storing folders for NBI
    # nbi_dev_cad2_store = 'cache_nbi_dev_cad2'
    nbi_test_cad2_store = 'cache_nbi_test_cad2'

    # Execute functions
    # cache_nbi_development_cad2(root_dir=nbi_dev_cad2_dir, storing_folder=nbi_dev_cad2_store)
    cache_nbi_development_cad2(root_dir=nbi_test_cad2_dir, storing_folder=nbi_test_cad2_store)
