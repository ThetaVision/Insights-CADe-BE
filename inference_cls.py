"""IMPORT PACKAGES"""
import os
import argparse
import time
import json
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchinfo import summary
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as pr_rec_auc

from data.dataset_cls import read_inclusion, augmentations
from data.dataset_cls import read_inclusion_cad2 as read_inclusion_cad2_nbi
from data.dataset import read_inclusion_cad2
from train_cls import check_cuda, find_best_model
from models.model import Model_CLS as Model

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    criteria['dev'] = {
        'dataset': ['validation'],
        'min_height': None,
        'min_width': None,
    }

    criteria["dev-rob-high-cad2"] = {
        "dataset": ["validation-robustness"],
        "quality": ["high"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev-rob-medium-cad2"] = {
        "dataset": ["validation-robustness"],
        "quality": ["medium"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev-rob-low-cad2"] = {
        "dataset": ["validation-robustness"],
        "quality": ["low"],
        "min_height": None,
        "min_width": None,
    }

    criteria['test'] = {
        'dataset': ['test'],
        'min_height': None,
        'min_width': None,
    }

    criteria['all-comers'] = {
        'dataset': ['all-comers'],
        'min_height': None,
        'min_width': None,
    }

    criteria['test-corrupt'] = {
        'dataset': ['test-corrupt'],
        'min_height': None,
        'min_width': None,
    }

    criteria['born'] = {
        'modality': ['wle'],
        'min_height': None,
        'min_width': None,
    }

    criteria['argos-ds3'] = {
        'modality': ['wle'],
        'dataset': ["Dataset 3"],
        'min_height': None,
        'min_width': None,
    }

    criteria['argos-ds4'] = {
        'modality': ['wle'],
        'dataset': ["Dataset 4"],
        'min_height': None,
        'min_width': None,
    }

    criteria['argos-ds34'] = {
        'modality': ['wle'],
        'dataset': ["Dataset 3", "Dataset 4"],
        'min_height': None,
        'min_width': None,
    }

    criteria['argos-ds5'] = {
        'modality': ['wle'],
        'dataset': ["Dataset 5"],
        'min_height': None,
        'min_width': None,
    }

    criteria["nbi-val"] = {
        "dataset": ["validation"],
        "source": ["images", "frames"],
        "class": ["ndbe", "neo"],
        "cap": ["cap", "no cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-val-images"] = {
        "dataset": ["validation"],
        "source": ["images"],
        "class": ["ndbe", "neo"],
        "cap": ["cap", "no cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-val-frames"] = {
        "dataset": ["validation"],
        "source": ["frames"],
        "class": ["ndbe", "neo"],
        "cap": ["cap", "no cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-val-cap"] = {
        "dataset": ["validation"],
        "source": ["images", "frames"],
        "class": ["ndbe", "neo"],
        "cap": ["cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-val-nocap"] = {
        "dataset": ["validation"],
        "source": ["images", "frames"],
        "class": ["ndbe", "neo"],
        "cap": ["no cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-test"] = {
        "dataset": ["test"],
        "source": ["images"],
        "class": ["ndbe", "neo"],
        "cap": ["no cap"],
        "type": ["focus"],
        "min_height": None,
        "min_width": None,
    }

    return criteria


# Define custom argument type for a list of enhancement settings
def list_of_settings(arg):
    return list(map(str, arg.split(',')))


"""""" """""" """""" """""" """"""
"""" FUNCTIONS FOR INFERENCE """
"""""" """""" """""" """""" """"""


def run(opt, f_txt, exp_name, inf_set):
    # Test Device
    device = check_cuda()

    # Create model output database
    df = pd.DataFrame(columns=['Case', 'CLS', 'CLS Correct'])
    logi = 0

    # Construct data
    criteria = get_data_inclusion_criteria()

    # CADe1.0 datasets
    if inf_set == 'Val':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['dev'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Test':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['test'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'All-Comers':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['all-comers'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Test-Corrupt':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['test-corrupt'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'BORN':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['born'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'ARGOS-DS3':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['argos-ds3'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'ARGOS-DS4':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['argos-ds4'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'ARGOS-DS34':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['argos-ds34'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'ARGOS-DS5':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['argos-ds5'])
        print('Found {} images...'.format(len(val_inclusion)))

    # Journal datasets
    elif inf_set == 'Val-HQ':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['dev'])
        val_inclusion_extra = read_inclusion(path=CACHE_PATH_EXTRA, criteria=criteria['dev'])
        val_inclusion = val_inclusion + val_inclusion_extra
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-MQ':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['dev'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-LQ':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['dev'])
        print('Found {} images...'.format(len(val_inclusion)))

    # Quality Triplets
    elif inf_set == 'Val-Rob-CAD2-High':
        val_inclusion = read_inclusion_cad2(path=CACHE_PATH, criteria=criteria['dev-rob-high-cad2'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-Rob-CAD2-Medium':
        val_inclusion = read_inclusion_cad2(path=CACHE_PATH, criteria=criteria['dev-rob-medium-cad2'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-Rob-CAD2-Low':
        val_inclusion = read_inclusion_cad2(path=CACHE_PATH, criteria=criteria['dev-rob-low-cad2'])
        print('Found {} images...'.format(len(val_inclusion)))

    # CADx2.0 datasets
    elif inf_set == 'Val-NBI':
        val_inclusion = read_inclusion_cad2_nbi(path=CACHE_PATH, criteria=criteria['nbi-val'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-NBI-Images':
        val_inclusion = read_inclusion_cad2_nbi(path=CACHE_PATH, criteria=criteria['nbi-val-images'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-NBI-Frames':
        val_inclusion = read_inclusion_cad2_nbi(path=CACHE_PATH, criteria=criteria['nbi-val-frames'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-NBI-Cap':
        val_inclusion = read_inclusion_cad2_nbi(path=CACHE_PATH, criteria=criteria['nbi-val-cap'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val-NBI-NoCap':
        val_inclusion = read_inclusion_cad2_nbi(path=CACHE_PATH, criteria=criteria['nbi-val-nocap'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Test-NBI':
        val_inclusion = read_inclusion_cad2_nbi(path=CACHE_PATH, criteria=criteria['nbi-test'])
        print('Found {} images...'.format(len(val_inclusion)))

    else:
        raise Exception('Unrecognized DEFINE_SET: {}'.format(inf_set))

    # Construct transforms
    data_transforms = augmentations(opt=opt)

    # Construct Model and load weights
    model = Model(opt=opt)
    best_index = find_best_model(path=os.path.join(SAVE_DIR, exp_name))
    checkpoint = torch.load(os.path.join(SAVE_DIR, exp_name, best_index))['state_dict']

    # Adapt state_dict keys (remove model. from the key and save again)
    if not os.path.exists(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt')):
        checkpoint_keys = list(checkpoint.keys())
        for key in checkpoint_keys:
            checkpoint[key.replace('model.', '')] = checkpoint[key]
            del checkpoint[key]
        model.load_state_dict(checkpoint, strict=True)
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'),
        )

    # Load weights
    weights = torch.load(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'))
    model.load_state_dict(weights, strict=True)

    # Initialize metrics
    tp_cls, tn_cls, fp_cls, fn_cls = 0.0, 0.0, 0.0, 0.0
    y_true, y_pred = list(), list()

    # Push model to GPU and set in evaluation mode
    model.cuda()
    model.eval()
    with torch.no_grad():
        # Loop over the data
        for img in val_inclusion:
            # Extract information from cache
            file = img['file']
            img_name = os.path.splitext(os.path.split(file)[1])[0]
            roi = img['roi']

            # Construct target
            label = img['label']
            if label:
                target = True
                y_true.append(target)
            else:
                target = False
                y_true.append(target)

            # Open Image
            image = Image.open(file).convert('RGB')

            # Crop the image to the ROI
            image = image.crop((roi[2], roi[0], roi[3], roi[1]))

            # Apply transforms to image and mask
            image_t = data_transforms['test'](image)
            image_t = image_t.unsqueeze(0).cuda()

            # Get prediction of model and perform Sigmoid activation
            cls_pred = model(image_t)
            cls_pred = torch.sigmoid(cls_pred).cpu()

            # Process classification prediction; positive prediction if exceed threshold = 0.5
            cls = cls_pred > opt.threshold
            cls = cls.squeeze(axis=0).item()

            # Append values to list
            y_pred.append(cls_pred.item())

            # Update classification metrics
            tp_cls += target * cls
            tn_cls += (1 - target) * (1 - cls)
            fp_cls += (1 - target) * cls
            fn_cls += target * (1 - cls)

            # Add values to the dataframe
            cls_result = cls == target
            df.loc[logi] = [
                img_name,
                round(cls_pred.item(), 5),
                cls_result,
            ]
            logi += 1

            # # Make folders
            # if not os.path.exists(os.path.join(OUTPUT_PATH, 'wrong')):
            #     os.makedirs(os.path.join(OUTPUT_PATH, 'wrong'))
            # if not os.path.exists(os.path.join(OUTPUT_PATH, 'correct')):
            #     os.makedirs(os.path.join(OUTPUT_PATH, 'correct'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'figures')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'figures'))

            # # Create original image with heatmap overlay
            # composite = image
            # draw = ImageDraw.Draw(composite)
            # font = ImageFont.truetype('C:/Users/s157128/Documents/Roboto/Roboto-Regular.ttf', size=48)
            # draw.text(
            #     (0, 0),
            #     "Cls: {:.3f}".format(cls_pred.item()),
            #     (255, 255, 255),
            #     font=font,
            # )
            #
            # # Save the composite images in folders for wrong and correct classifications
            # if 'Val-Rob-CAD2' in inf_set:
            #     if ' ' in img_name:
            #         img_name = img_name.split(' ')[0]
            #     else:
            #         img_name = img_name.split('-')[0]
            #
            # if cls != target:
            #     composite.save(os.path.join(OUTPUT_PATH, 'wrong', img_name + '.jpg'))
            # else:
            #     composite.save(os.path.join(OUTPUT_PATH, 'correct', img_name + '.jpg'))

    # Compute accuracy, sensitivity and specificity for classification
    accuracy_cls = (tp_cls + tn_cls) / (tp_cls + fn_cls + tn_cls + fp_cls)
    sensitivity_cls = tp_cls / (tp_cls + fn_cls)
    specificity_cls = tn_cls / (tn_cls + fp_cls + 1e-16)

    # Print accuracy, sensitivity and specificity
    print('\nClassification Performance')
    print('accuracy_cls: {:.4f}'.format(accuracy_cls))
    print('sensitivity_cls: {:.4f}'.format(sensitivity_cls))
    print('specificity_cls: {:.4f}'.format(specificity_cls))

    # Compute ROC AUC for classification
    auc = roc_auc_score(y_true, y_pred)
    print('auc_cls: {:.4f}'.format(auc))
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Compute precision, recall for classification
    precision_cls = tp_cls / (tp_cls + fp_cls + 1e-16)
    recall_cls = tp_cls / (tp_cls + fn_cls + 1e-16)

    # Print precision and recall
    print('\nprecision_cls: {:.4f}'.format(precision_cls))
    print('recall_cls: {:.4f}'.format(recall_cls))

    # Compute Precision-Recall AUC for classification
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = pr_rec_auc(recall, precision)
    print('pr_auc_cls: {:.4f}\n\n'.format(pr_auc))

    # Write Classification performance to file
    # CADe1.0 datasets
    if inf_set == 'Val':
        f_txt.write(f'### Validation Set (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Test':
        f_txt.write(f'### Test Set (Threshold = {opt.threshold}) ###')
    elif inf_set == 'All-Comers':
        f_txt.write(f'### All-Comers (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Test-Corrupt':
        f_txt.write(f'### Corrupt Test Set (Threshold = {opt.threshold}) ###')
    elif inf_set == 'BORN':
        f_txt.write(f'### BORN Module (Threshold = {opt.threshold}) ###')
    elif inf_set == 'ARGOS-DS3':
        f_txt.write(f'### ARGOS-Dataset 3 (Threshold = {opt.threshold}) ###')
    elif inf_set == 'ARGOS-DS4':
        f_txt.write(f'### ARGOS-Dataset 4 (Threshold = {opt.threshold}) ###')
    elif inf_set == 'ARGOS-DS34':
        f_txt.write(f'### ARGOS-Dataset 3+4 (Threshold = {opt.threshold}) ###')
    elif inf_set == 'ARGOS-DS5':
        f_txt.write(f'### ARGOS-Dataset 5 (Threshold = {opt.threshold}) ###')

    # Journal datasets
    elif inf_set == 'Val-HQ':
        f_txt.write(f'### Validation-HQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-MQ':
        f_txt.write(f'### Validation-MQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-LQ':
        f_txt.write(f'### Validation-LQ (Threshold = {opt.threshold}) ###')

    # CADe2.0 datasets
    elif inf_set == 'Val-Rob-CAD2-High':
        f_txt.write(f'### Validation-Robustness Set CAD2.0 - High (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-Rob-CAD2-Medium':
        f_txt.write(f'### Validation-Robustness Set CAD2.0 - Medium (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-Rob-CAD2-Low':
        f_txt.write(f'### Validation-Robustness Set CAD2.0 - Low (Threshold = {opt.threshold}) ###')

    # CADx2.0 datasets
    elif inf_set == 'Val-NBI':
        f_txt.write(f'### Validation NBI Set (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-NBI-Images':
        f_txt.write(f'### Validation NBI Set - Images (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-NBI-Frames':
        f_txt.write(f'### Validation NBI Set - Frames (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-NBI-Cap':
        f_txt.write(f'### Validation NBI Set - Cap (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-NBI-NoCap':
        f_txt.write(f'### Validation NBI Set - No Cap (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Test-NBI':
        f_txt.write(f'### Test NBI Set (Threshold = {opt.threshold}) ###')

    f_txt.write('\nClassification Performance')
    f_txt.write('\naccuracy_cls: {:.4f}'.format(accuracy_cls))
    f_txt.write('\nsensitivity_cls: {:.4f}'.format(sensitivity_cls))
    f_txt.write('\nspecificity_cls: {:.4f}'.format(specificity_cls))
    f_txt.write('\nauc_cls: {:.4f}'.format(auc))
    f_txt.write('\n\nprecision_cls: {:.4f}'.format(precision_cls))
    f_txt.write('\nrecall_cls: {:.4f}'.format(recall_cls))
    f_txt.write('\npr_auc_cls: {:.4f}\n\n'.format(pr_auc))

    # Plot ROC curve for classification results and save to specified folder
    plt.plot(fpr, tpr, marker='.', label='Classification head')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    major_ticks = np.arange(0.0, 1.01, 0.05)
    plt.xticks(major_ticks, fontsize='x-small')
    plt.yticks(major_ticks)
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.grid(True)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.title('ROC AUC')
    plt.savefig(os.path.join(OUTPUT_PATH, 'figures', 'auc_curve.jpg'))
    plt.close()

    # Save dataframe as csv file
    df.to_excel(os.path.join(OUTPUT_PATH, 'cls_scores.xlsx'))


def fps_check(opt, use_cuda=True):
    # Set parameters
    num_samples = 1000
    reps = 10

    # Test Device
    if use_cuda:
        device = check_cuda()

    # Construct Model and load weights
    model = Model(opt=opt)

    # Create random dummy input
    if use_cuda:
        dummy = torch.rand(1, 3, opt.imagesize, opt.imagesize).cuda()
    else:
        dummy = torch.rand(1, 3, opt.imagesize, opt.imagesize)

    # Push model to GPU
    if use_cuda:
        model.cuda()

    # Do model summary
    summary(model=model, input_size=(1, 3, opt.imagesize, opt.imagesize))

    # Initialize list for inference time
    inference_times = []

    # Set Model in evaluation mode and do fps check
    model.eval()
    for j in range(reps):
        with torch.no_grad():
            starttime = time.time()
            for i in range(num_samples):
                cls_pred = model(dummy)
            stoptime = time.time()

        inference_time = stoptime - starttime
        avg_inference_sample = inference_time / num_samples
        avg_inference_fps = 1 / avg_inference_sample

        # print('Average Inference Time per sample: {} sec.'.format(avg_inference_sample))
        print('Average fps: {}'.format(avg_inference_fps))
        inference_times.append(avg_inference_fps)

    # Print average fps
    print('Average fps over {} repetitions: {}'.format(reps, np.mean(inference_times)))


"""""" """""" """"""
"""" EXECUTION """
"""""" """""" """"""

if __name__ == '__main__':
    """SPECIFY PATH FOR SAVING"""
    SAVE_DIR = os.path.join(os.getcwd(), 'nbi-cadx2')

    """ARGUMENT PARSER"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experimentnames', type=list_of_settings)
    parser.add_argument('--evaluate_sets', type=list_of_settings)
    parser.add_argument('--modality', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--textfile', type=str, default='Results.txt')
    inference_opt = parser.parse_args()

    """LOOP OVER ALL EXPERIMENTS"""
    for exp_name in inference_opt.experimentnames:
        # EXTRACT INFORMATION FROM PARAMETERS USED IN EXPERIMENT
        f = open(os.path.join(SAVE_DIR, exp_name, 'params.json'))
        data = json.load(f)
        opt = {
            'experimentname': exp_name,
            'backbone': data['backbone'],
            'imagesize': data['imagesize'],
            'num_classes': data['num_classes'],
            'label_smoothing': data['label_smoothing'],
            'weights': [data['weights'] if 'weights' in data.keys() else 'GastroNet'][0],
            'modality': inference_opt.modality,
            'threshold': inference_opt.threshold,
            'evaluate_sets': inference_opt.evaluate_sets,
            'textfile': inference_opt.textfile,
        }
        opt = argparse.Namespace(**opt)

        # Create text file for writing results
        f = open(os.path.join(SAVE_DIR, exp_name, opt.textfile), 'x')
        f_txt = open(os.path.join(SAVE_DIR, exp_name, opt.textfile), 'a')

        # Loop over all sets
        for inf_set in opt.evaluate_sets:
            # CADe1.0 datasets
            if inf_set == 'Val':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_train-val-test_plausible')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Validation Set')
            elif inf_set == 'Test':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_train-val-test_plausible')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Test Set')
            elif inf_set == 'All-Comers':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_allcomers_plausible')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'All-Comers Set')
            elif inf_set == 'Test-Corrupt':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_test-corrupt')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Corrupt Test Set')
            elif inf_set == 'BORN':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_born_sweet_subset')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'BORN Module Set')
            elif inf_set == 'ARGOS-DS3':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_argos_soft')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'ARGOS Fuji Set - DS3')
            elif inf_set == 'ARGOS-DS4':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_argos_soft')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'ARGOS Fuji Set - DS4')
            elif inf_set == 'ARGOS-DS34':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_argos_soft')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'ARGOS Fuji Set - DS34')
            elif inf_set == 'ARGOS-DS5':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_argos_soft')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'ARGOS Fuji Set - DS5')

            # Journal datasets
            elif inf_set == 'Val-HQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_train-val-test_plausible')
                CACHE_PATH_EXTRA = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_train-val_frames')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val - HQ')
            elif inf_set == 'Val-MQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_train-val_frames_MQ')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val - MQ')
            elif inf_set == 'Val-LQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_train-val_frames_LQ')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val - LQ')

            # CADe2.0 datasets
            elif inf_set == 'Val-Rob-CAD2-High':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_test_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Validation-Robustness Set - High')
            elif inf_set == 'Val-Rob-CAD2-Medium':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_test_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Validation-Robustness Set - Medium')
            elif inf_set == 'Val-Rob-CAD2-Low':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_test_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Validation-Robustness Set - Low')

            # CADx2.0 datasets
            elif inf_set == 'Val-NBI':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_nbi_dev_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val')
            elif inf_set == 'Val-NBI-Images':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_nbi_dev_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val-Images')
            elif inf_set == 'Val-NBI-Frames':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_nbi_dev_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val-Frames')
            elif inf_set == 'Val-NBI-Cap':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_nbi_dev_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val-Cap')
            elif inf_set == 'Val-NBI-NoCap':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_nbi_dev_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val-NoCap')
            elif inf_set == 'Test-NBI':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_nbi_test_cad2.json')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Test')

            else:
                raise ValueError

            # Run inference
            run(opt=opt, f_txt=f_txt, exp_name=exp_name, inf_set=inf_set)

        # Close text file
        f_txt.close()
