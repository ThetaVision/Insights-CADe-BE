"""IMPORT PACKAGES"""
import os
import argparse
import time
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torchinfo import summary
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as pr_rec_auc

from data.dataset_cls import read_inclusion, augmentations
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

    # Datasets
    if inf_set == 'Val-HQ':
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

            # Make folders
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'wrong')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'wrong'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'correct')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'correct'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'figures')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'figures'))

            # Create original image with heatmap overlay
            composite = image
            draw = ImageDraw.Draw(composite)
            font = ImageFont.truetype('C:/Users/s157128/Documents/Roboto/Roboto-Regular.ttf', size=48)
            draw.text(
                (0, 0),
                "Cls: {:.3f}".format(cls_pred.item()),
                (255, 255, 255),
                font=font,
            )

            # Save the composite images in folders for wrong and correct classifications
            if cls != target:
                composite.save(os.path.join(OUTPUT_PATH, 'wrong', img_name + '.jpg'))
            else:
                composite.save(os.path.join(OUTPUT_PATH, 'correct', img_name + '.jpg'))

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
    if inf_set == 'Val-HQ':
        f_txt.write(f'### Validation-HQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-MQ':
        f_txt.write(f'### Validation-MQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-LQ':
        f_txt.write(f'### Validation-LQ (Threshold = {opt.threshold}) ###')

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


"""""" """""" """"""
"""" EXECUTION """
"""""" """""" """"""

if __name__ == '__main__':
    """SPECIFY PATH FOR SAVING"""
    SAVE_DIR = os.path.join(os.getcwd(), '')

    """ARGUMENT PARSER"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experimentnames', type=list_of_settings)
    parser.add_argument('--evaluate_sets', type=list_of_settings)
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
            if inf_set == 'Val-HQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                CACHE_PATH_EXTRA = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val - HQ')
            elif inf_set == 'Val-MQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val - MQ')
            elif inf_set == 'Val-LQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Val - LQ')
            else:
                raise ValueError

            # Run inference
            run(opt=opt, f_txt=f_txt, exp_name=exp_name, inf_set=inf_set)

        # Close text file
        f_txt.close()
