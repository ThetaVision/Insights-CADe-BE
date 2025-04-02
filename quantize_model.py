"""IMPORT PACKAGES"""
import os
import argparse
import time
import json
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

from torch.ao import quantization

import cv2

from torchinfo import summary

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as pr_rec_auc

from data.dataset import read_inclusion,  augmentations
from train import check_cuda, find_best_model
from models.model import Model
from models.MetaFormer import MetaFormerFPN, MetaFormer
import torch.nn as nn
from utils.metrics import BinaryDiceMetricEval

import matplotlib
import matplotlib.pyplot as plt
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

matplotlib.use('Agg')

from torch.serialization import add_safe_globals 
"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    criteria["train"] = {
        "modality": ["wle"],
        "dataset": ["training"],
        "protocol": ["Retrospectief", "Prospectief"],
        "min_height": None,
        "min_width": None,
    }

    criteria["validation"] = {
        "modality": ["wle"],
        "dataset": ["validation"],
        "protocol": ["Retrospectief", "Prospectief"],
        "min_height": None,
        "min_width": None,
    }

    criteria["extra-validation"] = {
        "modality": ["wle"],
        "dataset": ["extraval"],
        "protocol": ["Prospectief"],
        "min_height": None,
        "min_width": None,
    }

    criteria["test"] = {
        "modality": ["wle"],
        "dataset": ["test"],
        "protocol": ["Prospectief"],
        "min_height": None,
        "min_width": None,
    }

    return criteria


# Define custom argument type for a list of enhancement settings
def list_of_settings(arg):
    return list(map(str, arg.split(',')))


# Define function for extracting masks
def extract_masks(image, masklist):
    # Create dictionary for masks
    mask_dict = {'Soft': 0, 'Plausible': 0, 'Sweet': 0, 'Hard': 0}

    # Extract information on expert
    expert_list = list(set([os.path.split(os.path.split(masklist[i])[0])[1] for i in range(len(masklist))]))
    expert_list.sort()

    # Set Bools for all masks
    lower0, higher0, lower1, higher1 = False, False, False, False
    ll0, hl0, ll1, hl1 = 0, 0, 0, 0

    # Loop over all masks
    for i in range(len(masklist)):
        # Extract information on expert and likelihood
        expert = os.path.split(os.path.split(masklist[i])[0])[1]
        likelihood = os.path.split(os.path.split(os.path.split(masklist[i])[0])[0])[1]

        # If ll0 mask is present
        if expert_list.index(expert) == 0 and 'Lower' in likelihood:
            lower0 = True
            ll0 = Image.open(masklist[i]).convert('1')
            if ll0.size != image.size:
                ll0 = np.array(ll0.resize(image.size, resample=Image.NEAREST))
            else:
                ll0 = np.array(ll0)

        # If hl0 mask is present
        elif expert_list.index(expert) == 0 and 'Higher' in likelihood:
            hl0 = Image.open(masklist[i]).convert('1')
            higher0 = True
            if hl0.size != image.size:
                hl0 = np.array(hl0.resize(image.size, resample=Image.NEAREST))
            else:
                hl0 = np.array(hl0)

        # If ll1 mask is present
        elif expert_list.index(expert) == 1 and 'Lower' in likelihood:
            ll1 = Image.open(masklist[i]).convert('1')
            lower1 = True
            if ll1.size != image.size:
                ll1 = np.array(ll1.resize(image.size, resample=Image.NEAREST))
            else:
                ll1 = np.array(ll1)

        # If hl1 mask is present
        elif expert_list.index(expert) == 1 and 'Higher' in likelihood:
            hl1 = Image.open(masklist[i]).convert('1')
            higher1 = True
            if hl1.size != image.size:
                hl1 = np.array(hl1.resize(image.size, resample=Image.NEAREST))
            else:
                hl1 = np.array(hl1)

        # # If more than 2 experts are available, raise an error
        # else:
        #     raise ValueError('More than 2 experts...')

    # Replace LL with LL U HL if they both exist to enforce the protocol
    if lower0 and higher0:
        ll0 = np.add(ll0, hl0)
    if lower1 and higher1:
        ll1 = np.add(ll1, hl1)

    """Create Consensus masks for each likelihood"""
    # Construct LowerLikelihood building blocks
    if lower0 + lower1 == 2:
        union_ll = np.add(ll0, ll1)
        intersection_ll = np.multiply(ll0, ll1)
    elif lower0 + lower1 == 1:
        if lower0:
            union_ll = ll0
            intersection_ll = ll0
        else:
            union_ll = ll1
            intersection_ll = ll1
    else:
        union_ll = 0
        intersection_ll = 0

    # Construct HigherLikelihood building blocks
    if higher0 + higher1 == 2:
        union_hl = np.add(hl0, hl1)
        intersection_hl = np.multiply(hl0, hl1)
    elif higher0 + higher1 == 1:
        if higher0:
            union_hl = hl0
            intersection_hl = hl0
        else:
            union_hl = hl1
            intersection_hl = hl1
    else:
        union_hl = 0
        intersection_hl = 0

    # Construct consensus masks
    if lower0 + lower1 == 0:
        soft = Image.fromarray(union_hl).convert('1')
        plausible = Image.fromarray(union_hl).convert('1')
        sweet = Image.fromarray(union_hl).convert('1')
        hard = Image.fromarray(intersection_hl).convert('1')
    elif higher0 + higher1 == 0:
        soft = Image.fromarray(union_ll).convert('1')
        plausible = Image.fromarray(intersection_ll).convert('1')
        sweet = Image.fromarray(intersection_ll).convert('1')
        hard = Image.fromarray(intersection_ll).convert('1')
    elif lower0 + lower1 == 1:
        soft = Image.fromarray(np.add(intersection_ll, union_hl)).convert('1')
        plausible = Image.fromarray(np.add(intersection_ll, union_hl)).convert('1')
        sweet = Image.fromarray(union_hl).convert('1')
        hard = Image.fromarray(intersection_hl).convert('1')
    else:
        soft = Image.fromarray(union_ll).convert('1')
        plausible = Image.fromarray(np.add(intersection_ll, union_hl)).convert('1')
        sweet = Image.fromarray(union_hl).convert('1')
        hard = Image.fromarray(intersection_hl).convert('1')

    # Store in dictionary
    mask_dict['Soft'] = soft
    mask_dict['Plausible'] = plausible
    mask_dict['Sweet'] = sweet
    mask_dict['Hard'] = hard

    return mask_dict


"""""" """""" """""" """""" """"""
"""" FUNCTIONS FOR INFERENCE """
"""""" """""" """""" """""" """"""

def run(model):
    # Test Device
    device = check_cuda()
    # Construct transforms
    data_transforms = augmentations(opt=opt)
    # Push model to GPU and set in evaluation mode
    model.cuda()
    model.eval()
                
    # array with all the validation data
    calib_array = []
    with torch.no_grad():
        # Loop over the data
        for img in val_inclusion:
            # Extract information from cache
            file = img['file']
            img_name = os.path.splitext(os.path.split(file)[1])[0]
            roi = img['roi']
            mask = img['mask']

            # Construct target
            label = img['label']


            # Open Image
            image = Image.open(file).convert('RGB')

            # By default set has_mask to zero
            has_mask = 0

            # Set has_mask for NDBE cases
            if label == np.array([0], dtype=np.float32):
                has_mask = 1

            # Open mask for neoplasia cases
            if len(mask) > 0:
                mask_dict = extract_masks(image, mask)
                if opt.ground_truth == 'Soft':
                    mask_gt = (
                        mask_dict['Soft']
                        .crop((roi[2], roi[0], roi[3], roi[1]))
                        .resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST)
                    )
                elif opt.ground_truth == 'Plausible':
                    mask_gt = (
                        mask_dict['Plausible']
                        .crop((roi[2], roi[0], roi[3], roi[1]))
                        .resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST)
                    )
                elif opt.ground_truth == 'Sweet':
                    mask_gt = (
                        mask_dict['Sweet']
                        .crop((roi[2], roi[0], roi[3], roi[1]))
                        .resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST)
                    )
                elif opt.ground_truth == 'Hard':
                    mask_gt = (
                        mask_dict['Hard']
                        .crop((roi[2], roi[0], roi[3], roi[1]))
                        .resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST)
                    )
                has_mask = 1
            # Create mask with all zeros when there are no available ones
            else:
                mask_np = np.zeros(image.size)
                mask_gt = Image.fromarray(mask_np, mode='RGB').convert('1')
                mask_gt = mask_gt.crop((roi[2], roi[0], roi[3], roi[1]))

            # Crop the image to the ROI
            image = image.crop((roi[2], roi[0], roi[3], roi[1]))

            # Apply transforms to image and mask
            image_t, mask_gt = data_transforms['test'](image, mask_gt, has_mask)
            image_t = image_t.unsqueeze(0).cuda()
            mask_dice = mask_gt.unsqueeze(0)

            # extend the calib_array
            calib_array.append(image_t)

            # Get prediction of model and perform Sigmoid activation
            out1, out2 = model(image_t)
            # if opt.precision and inf_set == 'Val':
            #     model_prepared(image_t)
            cls_pred = out1 if out1.dim() == 2 else out2
            seg_pred = out2 if out2.dim() == 4 else out1
            cls_pred = torch.sigmoid(cls_pred).cpu()
            seg_pred = torch.sigmoid(seg_pred).cpu()





"""""" """""" """"""
"""" EXECUTION """
"""""" """""" """"""

if __name__ == '__main__':
    """SPECIFY PATH FOR SAVING"""
    # python inference.py --output_dir "C:\Users\20195435\Documents\theta\projects\cosmo\Insights-CADe-BE\output" --cache_path "C:\Users\20195435\Documents\theta\projects\cosmo\barretts_cache" --experiment_name "C:\Users\20195435\Documents\theta\projects\cosmo\Insights-CADe-BE\experiments\baseline" --evaluate_sets Test Val Extra-Val --ground_truth Plausible --threshold 0.5 --min_sensitivity 0.9 --textfile Results.txt

    """ARGUMENT PARSER"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--cache_path', type=str, default='cache')
    parser.add_argument('--experiment_name', type=str, default='GastroNet', help='path to experiment')
    parser.add_argument('--evaluate_sets', type=list_of_settings)
    parser.add_argument('--ground_truth', type=str, default='Plausible')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min_sensitivity', type=float, default=0.9)
    parser.add_argument('--textfile', type=str, default='Results.txt')
    parser.add_argument("--quantized_precision_model", type=str, default=None, choices=["fp32", "bf16", "int8", "int4", "int4_weight", "int8_weight", "fp8"])
    inference_opt = parser.parse_args()

    SAVE_DIR = inference_opt.output_dir
    CACHE_PATH = inference_opt.cache_path

    exp_name = inference_opt.experiment_name
    # EXTRACT INFORMATION FROM PARAMETERS USED IN EXPERIMENT
    f = open(os.path.join(SAVE_DIR, exp_name, 'local_params.json'))
    data = json.load(f)

    opt = {
        'experimentname': exp_name,
        'backbone': data['backbone'],
        'seg_branch': data['seg_branch'],
        'imagesize': data['imagesize'],
        'num_classes': data['num_classes'],
        'label_smoothing': data['label_smoothing'],
        'threshold': inference_opt.threshold,
        'min_sensitivity': inference_opt.min_sensitivity,
        'ground_truth': inference_opt.ground_truth,
        'evaluate_sets': inference_opt.evaluate_sets,
        'textfile': inference_opt.textfile,
        'weights': data['weights'],
        'precision': inference_opt.quantized_precision_model,
    }
    opt = argparse.Namespace(**opt)

    # Create text file for writing results
    # f = open(os.path.join(exp_name, opt.textfile), 'x')
    f_txt = open(os.path.join(exp_name, opt.textfile), 'w+')
    # Loop over all sets
    for inf_set in opt.evaluate_sets:
        print('Evaluating set: {}'.format(inf_set))
        if inf_set == 'Test':
            OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Test Set (Test)')
        elif inf_set == 'Val':
            OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Validation Set (Val)')
        elif inf_set == 'Extra-Val':
            OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Extra Validation Set (Extra-Val)')
        else:
            raise ValueError('Unrecognized set: {}'.format(inf_set))
        # Run inference

           # Construct Model and load weights
    model = Model(opt=opt)
    best_index = find_best_model(path=os.path.join(exp_name), finetune=False)
    checkpoint = torch.load(os.path.join(exp_name, best_index))['state_dict']

    # Adapt state_dict keys (remove model. from the key and save again)
    if not os.path.exists(os.path.join(exp_name, 'final_pytorch_model.pt')):

        new_state_dict = {}
        for key in checkpoint:
            new_state_dict[key.replace('model.', '')] = checkpoint[key]

        # delete cls_criterion.pos_weight from state_dict
        del new_state_dict['cls_criterion.pos_weight']

        model.load_state_dict(new_state_dict, strict=True)
        torch.save(
            model.state_dict(),
            os.path.join(exp_name, 'final_pytorch_model.pt'),
        )

    # Determine file paths
    base_model_path = os.path.join(exp_name, 'final_pytorch_model.pt')
    quantized_model_path = os.path.join(SAVE_DIR, exp_name, f'final_pytorch_model_{opt.precision}.pt')
    onnx_model_path = os.path.join(SAVE_DIR, exp_name, f'final_pytorch_model_{opt.precision}.onnx')

    # No precision requirement; load the base model weights directly
    weights = torch.load(base_model_path, weights_only=True)
    print('Loading base model weights...')
    model.load_state_dict(weights, strict=True)

    # Construct data
    criteria = get_data_inclusion_criteria()

    # Test Sets
    if inf_set == 'Test':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['test'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'Val':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['validation'])
        print('Found {} images...'.format(len(val_inclusion)))

    elif inf_set == 'Extra-Val':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['extra-validation'])
        print('Found {} images...'.format(len(val_inclusion)))
    else:
        raise Exception('Unrecognized DEFINE_SET: {}'.format(inf_set))

    # Select quantization config
    config = mtq.INT8_SMOOTHQUANT_CFG

    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, config, forward_loop=run)
    mtq.print_quant_summary(model)
    mto.save(model, quantized_model_path)

    # dummy input on gpu
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)

    # Close text file
    f_txt.close()
