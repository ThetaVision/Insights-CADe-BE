"""IMPORT PACKAGES"""
import os
import argparse
import time
import json
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import cv2

import tensorrt as trt
from torch2trt import torch2trt, TRTModule

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as pr_rec_auc

from data.dataset import read_inclusion, augmentations
from train import check_cuda, find_best_model
from models.model import Model
from utils.metrics import BinaryDiceMetricEval

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    # Criteria for CADe1.0 datasets
    criteria['dev'] = {
        'modality': ['wle'],
        'dataset': ['validation'],
        'min_height': None,
        'min_width': None,
    }

    criteria['test'] = {
        'modality': ['wle'],
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
        'mask_only': True,
    }

    criteria['argos-ds34'] = {
        'modality': ['wle'],
        'dataset': ["Dataset 3", "Dataset 4"],
        'min_height': None,
        'min_width': None,
        'mask_only': True,
    }

    criteria['argos-ds5'] = {
        'modality': ['wle'],
        'dataset': ["Dataset 5"],
        'min_height': None,
        'min_width': None,
        'mask_only': True,
    }

    criteria["iq-triplet"] = {
        "dataset": ["validation-robustness"],
        "min_height": None,
        "min_width": None,
    }

    criteria["iq-triplet-high"] = {
        "dataset": ["validation-robustness"],
        "quality": ["high"],
        "min_height": None,
        "min_width": None,
    }

    criteria["iq-triplet-medium"] = {
        "dataset": ["validation-robustness"],
        "quality": ["medium"],
        "min_height": None,
        "min_width": None,
    }

    criteria["iq-triplet-low"] = {
        "dataset": ["validation-robustness"],
        "quality": ["low"],
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


def run(opt, f_txt, exp_name, inf_set):
    # Test Device
    device = check_cuda()

    # Create model output database
    df = pd.DataFrame(columns=['Case', 'CLS', 'SEG', 'AVG', 'CLS Correct', 'SEG Correct', 'AVG Correct', 'DICE'])
    logi = 0

    # Construct data
    criteria = get_data_inclusion_criteria()

    # Test Sets
    if inf_set == 'Test':
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
    elif inf_set == 'ARGOS-DS34':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['argos-ds34'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'ARGOS-DS5':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['argos-ds5'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'IQ-Triplet':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['iq-triplet'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'IQ-Triplet-HQ':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['iq-triplet-high'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'IQ-Triplet-MQ':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['iq-triplet-medium'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif inf_set == 'IQ-Triplet-LQ':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['iq-triplet-low'])
        print('Found {} images...'.format(len(val_inclusion)))

    # Paths for Journal: Validation Set with different quality levels by means of video frames
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
    elif inf_set == 'Val-Img':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['dev-neo'])
        print('Found {} images...'.format(len(val_inclusion)))

    else:
        raise Exception('Unrecognized DEFINE_SET: {}'.format(inf_set))

    # Construct transforms
    data_transforms = augmentations(opt=opt)

    # Construct TensorRT model if it does not exist
    if not os.path.exists(os.path.join(SAVE_DIR, exp_name, 'final_trt_model.pth')):
        # Construct Model and load weights
        model_old = Model(opt=opt)
        best_index = find_best_model(path=os.path.join(SAVE_DIR, exp_name), finetune=False)
        checkpoint = torch.load(os.path.join(SAVE_DIR, exp_name, best_index))['state_dict']

        # Adapt state_dict keys (remove model. from the key and save again)
        if not os.path.exists(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt')):
            checkpoint_keys = list(checkpoint.keys())
            for key in checkpoint_keys:
                checkpoint[key.replace('model.', '')] = checkpoint[key]
                del checkpoint[key]
            model_old.load_state_dict(checkpoint, strict=True)
            torch.save(
                model_old.state_dict(),
                os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'),
            )

        # Load weights
        weights = torch.load(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'))
        model_old.load_state_dict(weights, strict=True)
        model_old.eval().cuda()

        # Convert to ONNX model
        torch.onnx.export(
            model_old,
            torch.randn(1, 3, 256, 256).cuda(),
            os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.onnx'),
            input_names=['input'],
            output_names=['Classification', 'Segmentation'],
        )

        # Convert to TensorRT model
        model_trt = torch2trt(
            model_old,
            [torch.randn(1, 3, 256, 256).cuda()],
            fp16_mode=False,
            max_batch_size=1,
            use_onnx=True,
            onnx_file_path=os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.onnx'),
        )
        torch.save(model_trt.state_dict(), os.path.join(SAVE_DIR, exp_name, 'final_trt_model.pth'))

    # Load TensorRT model
    model = TRTModule()
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, exp_name, 'final_trt_model.pth')), strict=True)

    # Initialize metrics
    dice_score = BinaryDiceMetricEval(threshold=opt.threshold)
    tp_cls, tn_cls, fp_cls, fn_cls = 0.0, 0.0, 0.0, 0.0
    tp_seg, tn_seg, fp_seg, fn_seg = 0.0, 0.0, 0.0, 0.0
    tp_avg, tn_avg, fp_avg, fn_avg = 0.0, 0.0, 0.0, 0.0
    y_true, y_pred_cls, y_pred_seg, y_pred_avg = list(), list(), list(), list()

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
            mask = img['mask']

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

            # Get prediction of model and perform Sigmoid activation
            # cls_pred, seg_pred = model(image_t)
            out1, out2 = model(image_t)
            cls_pred = out1 if out1.dim() == 2 else out2
            seg_pred = out2 if out2.dim() == 4 else out1
            cls_pred = torch.sigmoid(cls_pred).cpu()
            seg_pred = torch.sigmoid(seg_pred).cpu()

            # Process classification prediction; positive prediction if exceed threshold = 0.5
            cls = cls_pred > opt.threshold
            cls = cls.squeeze(axis=0).item()

            # Process segmentation prediction; positive prediction if 1 pixel exceeds threshold = 0.5
            mask = seg_pred.squeeze(axis=0)
            mask_cls_logit = torch.max(mask)
            mask_cls = (torch.max(mask) > opt.threshold).item()

            # Process average prediction from classification and segmentation
            avg_pred = (cls_pred + mask_cls_logit) / 2
            avg_cls = (avg_pred > opt.threshold).item()

            # Process segmentation prediction; Average Dice Score
            dice_score.update(seg_pred, mask_dice, torch.tensor(has_mask))
            dice = dice_score.compute_single(seg_pred, mask_dice, torch.tensor(has_mask))

            # Append values to list
            y_pred_cls.append(cls_pred.item())
            y_pred_seg.append(mask_cls_logit.item())
            y_pred_avg.append(avg_pred.item())

            # Update classification metrics
            tp_cls += target * cls
            tn_cls += (1 - target) * (1 - cls)
            fp_cls += (1 - target) * cls
            fn_cls += target * (1 - cls)

            # Update segmentation metrics
            tp_seg += target * mask_cls
            tn_seg += (1 - target) * (1 - mask_cls)
            fp_seg += (1 - target) * mask_cls
            fn_seg += target * (1 - mask_cls)

            # Update counters (average classification/segmentation)
            tp_avg += target * avg_cls
            tn_avg += (1 - target) * (1 - avg_cls)
            fp_avg += (1 - target) * avg_cls
            fn_avg += target * (1 - avg_cls)

            # Add values to the dataframe
            cls_result = cls == target
            seg_result = mask_cls == target
            avg_result = avg_cls == target
            if target and has_mask == 1:
                df.loc[logi] = [
                    img_name,
                    round(cls_pred.item(), 5),
                    round(mask_cls_logit.item(), 5),
                    round(avg_pred.item(), 5),
                    cls_result,
                    seg_result,
                    avg_result,
                    dice.item(),
                ]
            else:
                df.loc[logi] = [
                    img_name,
                    round(cls_pred.item(), 5),
                    round(mask_cls_logit.item(), 5),
                    round(avg_pred.item(), 5),
                    cls_result,
                    seg_result,
                    avg_result,
                    'No Dice',
                ]
            logi += 1

            # Process predicted mask and save to specified folder
            mask = mask.permute(1, 2, 0)
            maskpred = np.array(mask * 255, dtype=np.uint8)
            maskpred_pil = Image.fromarray(cv2.cvtColor(maskpred, cv2.COLOR_GRAY2RGB), mode='RGB')

            # Make folders
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'wrong')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'wrong'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'cls c-seg w')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'cls c-seg w'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'cls w-seg c')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'cls w-seg c'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'correct')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'correct'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'ROC')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'ROC'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'masks')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'masks'))

            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(mask_gt[0, :, :], cmap='gray')
            ax1.set_title('Ground Truth')
            ax1.axis('off')
            ax2.imshow(maskpred_pil)
            ax2.axis('off')
            ax2.set_title('Generated Mask')
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_PATH, 'masks', img_name + '.png'),
                bbox_inches='tight',
            )
            plt.close()

            # Visualization Journal Masks
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'masks')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'masks'))
            if has_mask == 1 and target:
                plt.imshow(maskpred_pil)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUTPUT_PATH, 'masks', img_name + '.png'),
                    bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=300,
                )
                plt.close()

            #  Transform to heatmap
            heatmap = cv2.cvtColor(
                cv2.applyColorMap(maskpred, cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB,
            )
            heatmap = heatmap / 255.0

            # Define alpha value
            alphavalue = 0.5
            alpha = np.where(maskpred > 0.1, alphavalue, 0.0)

            # Process heatmap to PIL image, resize and convert to RGB
            heatmap = np.array(np.concatenate((heatmap, alpha), axis=-1) * 255, dtype=np.uint8)
            heatmap_pil = Image.fromarray(heatmap, mode='RGBA')
            w = int(image.size[0])
            h = int(image.size[1])
            heatmap_pil = heatmap_pil.resize(size=(w, h), resample=Image.NEAREST)
            heatmap_pil = heatmap_pil.convert('RGB')

            # Create original image with heatmap overlay
            composite = Image.blend(heatmap_pil, image, 0.6)
            draw = ImageDraw.Draw(composite)
            font = ImageFont.truetype('arial.ttf', size=48)
            draw.text(
                (0, 0),
                "Cls: {:.3f}, Seg:{:.3f}".format(cls_pred.item(), mask_cls_logit.item()),
                (255, 255, 255),
                font=font,
            )

            # Save the composite images in folders for wrong and correct classifications
            if mask_cls != target and cls != target:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'wrong', img_name + '.jpg'))
            elif mask_cls != target and cls == target:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'cls c-seg w', img_name + '.jpg'))
            elif mask_cls == target and cls != target:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'cls w-seg c', img_name + '.jpg'))
            else:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'correct', img_name + '.jpg'))

    # Compute accuracy, sensitivity, specificity, AUC, precision, recall, AUPRC and Dice metrics (Classification only)
    accuracy_cls = (tp_cls + tn_cls) / (tp_cls + fn_cls + tn_cls + fp_cls)
    sensitivity_cls = tp_cls / (tp_cls + fn_cls)
    specificity_cls = tn_cls / (tn_cls + fp_cls + 1e-16)
    auc_cls = roc_auc_score(y_true, y_pred_cls)
    fpr_cls, tpr_cls, _ = roc_curve(y_true, y_pred_cls)

    precision_cls = tp_cls / (tp_cls + fp_cls + 1e-16)
    recall_cls = tp_cls / (tp_cls + fn_cls + 1e-16)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_cls)
    pr_auc_cls = pr_rec_auc(recall, precision)

    avg_dice_cls = dice_score.compute()

    # Print accuracy, sensitivity, specificity, AUC, precision, recall and AUPRC metrics (Classification only)
    print('\nClassification Performance')
    print('accuracy_cls: {:.4f}'.format(accuracy_cls))
    print('sensitivity_cls: {:.4f}'.format(sensitivity_cls))
    print('specificity_cls: {:.4f}'.format(specificity_cls))
    print('auc_cls: {:.4f}'.format(auc_cls))
    print('precision_cls: {:.4f}'.format(precision_cls))
    print('recall_cls: {:.4f}'.format(recall_cls))
    print('pr_auc_cls: {:.4f}'.format(pr_auc_cls))
    print('avg_dice_cls: {:.4f}\n'.format(avg_dice_cls.item()))

    # Compute accuracy, sensitivity, specificity, AUC, precision, recall and AUPRC metrics (Segmentation only)
    accuracy_seg = (tp_seg + tn_seg) / (tp_seg + fn_seg + tn_seg + fp_seg)
    sensitivity_seg = tp_seg / (tp_seg + fn_seg)
    specificity_seg = tn_seg / (tn_seg + fp_seg + 1e-16)
    auc_seg = roc_auc_score(y_true, y_pred_seg)
    fpr_seg, tpr_seg, _ = roc_curve(y_true, y_pred_seg)

    precision_seg = tp_seg / (tp_seg + fp_seg + 1e-16)
    recall_seg = tp_seg / (tp_seg + fn_seg + 1e-16)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_seg)
    pr_auc_seg = pr_rec_auc(recall, precision)

    # Print accuracy, sensitivity, specificity, AUC, precision, recall and AUPRC metrics (Segmentation only)
    print('\nSegmentation Performance')
    print('accuracy_seg: {:.4f}'.format(accuracy_seg))
    print('sensitivity_seg: {:.4f}'.format(sensitivity_seg))
    print('specificity_seg: {:.4f}'.format(specificity_seg))
    print('auc_seg: {:.4f}'.format(auc_seg))
    print('precision_seg: {:.4f}'.format(precision_seg))
    print('recall_seg: {:.4f}'.format(recall_seg))
    print('pr_auc_seg: {:.4f}'.format(pr_auc_seg))
    print('avg_dice_seg: {:.4f}\n'.format(avg_dice_cls.item()))

    # Compute accuracy, sensitivity, specificity, AUC, precision, recall and AUPRC metrics (Average)
    accuracy_avg = (tp_avg + tn_avg) / (tp_avg + fn_avg + tn_avg + fp_avg)
    sensitivity_avg = tp_avg / (tp_avg + fn_avg)
    specificity_avg = tn_avg / (tn_avg + fp_avg + 1e-16)
    auc_avg = roc_auc_score(y_true, y_pred_avg)
    fpr_avg, tpr_avg, _ = roc_curve(y_true, y_pred_avg)

    precision_avg = tp_avg / (tp_avg + fp_avg + 1e-16)
    recall_avg = tp_avg / (tp_avg + fn_avg + 1e-16)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_avg)
    pr_auc_avg = pr_rec_auc(recall, precision)

    # Print accuracy, sensitivity, specificity, AUC, precision, recall and AUPRC metrics (Average)
    print('\nAverage Performance')
    print('accuracy_avg: {:.4f}'.format(accuracy_avg))
    print('sensitivity_avg: {:.4f}'.format(sensitivity_avg))
    print('specificity_avg: {:.4f}'.format(specificity_avg))
    print('auc_avg: {:.4f}'.format(auc_avg))
    print('precision_avg: {:.4f}'.format(precision_avg))
    print('recall_avg: {:.4f}'.format(recall_avg))
    print('pr_auc_avg: {:.4f}'.format(pr_auc_avg))
    print('avg_dice_avg: {:.4f}\n'.format(avg_dice_cls.item()))

    # Write Classification performance to file
    if inf_set == 'Test':
        f_txt.write(f'### Test Set (Threshold = {opt.threshold}) ###')
    elif inf_set == 'All-Comers':
        f_txt.write(f'### All-Comers (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Test-Corrupt':
        f_txt.write(f'### Corrupt Test Set (Threshold = {opt.threshold}) ###')
    elif inf_set == 'BORN':
        f_txt.write(f'### BORN Module (Threshold = {opt.threshold}) ###')
    elif inf_set == 'ARGOS-DS34':
        f_txt.write(f'### ARGOS-Dataset 3/4 (Threshold = {opt.threshold}) ###')
    elif inf_set == 'ARGOS-DS5':
        f_txt.write(f'### ARGOS-Dataset 5 (Threshold = {opt.threshold}) ###')
    elif inf_set == 'IQ-Triplet':
        f_txt.write(f'### IQ-Triplet (Threshold = {opt.threshold}) ###')
    elif inf_set == 'IQ-Triplet-HQ':
        f_txt.write(f'### IQ-Triplet-HQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'IQ-Triplet-MQ':
        f_txt.write(f'### IQ-Triplet-MQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'IQ-Triplet-LQ':
        f_txt.write(f'### IQ-Triplet-LQ (Threshold = {opt.threshold}) ###')

    # Datasets for Journal experiments
    elif inf_set == 'Val-HQ':
        f_txt.write(f'### Validation-HQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-MQ':
        f_txt.write(f'### Validation-MQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-LQ':
        f_txt.write(f'### Validation-LQ (Threshold = {opt.threshold}) ###')
    elif inf_set == 'Val-Img':
        f_txt.write(f'### Validation-Images (Threshold = {opt.threshold}) ###')

    # Write performance to textfile (Classification only
    f_txt.write('\nClassification Performance')
    f_txt.write('\n--------------------------')
    f_txt.write('\naccuracy_cls: {:.4f}'.format(accuracy_cls))
    f_txt.write('\nsensitivity_cls: {:.4f}'.format(sensitivity_cls))
    f_txt.write('\nspecificity_cls: {:.4f}'.format(specificity_cls))
    f_txt.write('\nauc_cls: {:.4f}'.format(auc_cls))
    f_txt.write('\n\nprecision_cls: {:.4f}'.format(precision_cls))
    f_txt.write('\nrecall_cls: {:.4f}'.format(recall_cls))
    f_txt.write('\npr_auc_cls: {:.4f}'.format(pr_auc_cls))
    f_txt.write('\n\navg_dice_cls: {:.4f}\n\n'.format(avg_dice_cls.item()))

    # Write performance to textfile (Segmentation only)
    f_txt.write('\nSegmentation Performance')
    f_txt.write('\n------------------------')
    f_txt.write('\naccuracy_seg: {:.4f}'.format(accuracy_seg))
    f_txt.write('\nsensitivity_seg: {:.4f}'.format(sensitivity_seg))
    f_txt.write('\nspecificity_seg: {:.4f}'.format(specificity_seg))
    f_txt.write('\nauc_seg: {:.4f}'.format(auc_seg))
    f_txt.write('\n\nprecision_seg: {:.4f}'.format(precision_seg))
    f_txt.write('\nrecall_seg: {:.4f}'.format(recall_seg))
    f_txt.write('\npr_auc_seg: {:.4f}'.format(pr_auc_seg))
    f_txt.write('\n\navg_dice_seg: {:.4f}\n\n'.format(avg_dice_cls.item()))

    # Write performance to textfile (Average)
    f_txt.write('\nAverage Performance')
    f_txt.write('\n-------------------')
    f_txt.write('\naccuracy_avg: {:.4f}'.format(accuracy_avg))
    f_txt.write('\nsensitivity_avg: {:.4f}'.format(sensitivity_avg))
    f_txt.write('\nspecificity_avg: {:.4f}'.format(specificity_avg))
    f_txt.write('\nauc_avg: {:.4f}'.format(auc_avg))
    f_txt.write('\n\nprecision_avg: {:.4f}'.format(precision_avg))
    f_txt.write('\nrecall_avg: {:.4f}'.format(recall_avg))
    f_txt.write('\npr_auc_avg: {:.4f}'.format(pr_auc_avg))
    f_txt.write('\n\navg_dice_cls: {:.4f}\n\n'.format(avg_dice_cls.item()))

    # Plot ROC curve for classification results and save to specified folder
    plt.plot(fpr_cls, tpr_cls, marker='.', label='Classification head')
    plt.plot(fpr_seg, tpr_seg, marker='.', label='Segmentation head')
    plt.plot(fpr_avg, tpr_avg, marker='.', label='Average')
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    major_ticks = np.arange(0.0, 1.01, 0.05)
    plt.xticks(major_ticks, fontsize='x-small')
    plt.yticks(major_ticks)
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.grid(True)
    plt.grid(alpha=0.5)
    plt.legend(loc='lower right')
    plt.title('ROC AUC')
    plt.savefig(os.path.join(OUTPUT_PATH, 'ROC', 'auc_curve_output_comparison_TRT.jpg'))
    plt.close()

    # Save dataframe as csv file
    df.to_excel(os.path.join(OUTPUT_PATH, 'cls_scores_TRT.xlsx'))


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
    parser.add_argument('--ground_truth', type=str, default='Plausible')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--textfile', type=str, default='Results_TRT.txt')
    inference_opt = parser.parse_args()

    """LOOP OVER ALL EXPERIMENTS"""
    for exp_name in inference_opt.experimentnames:
        # EXTRACT INFORMATION FROM PARAMETERS USED IN EXPERIMENT
        f = open(os.path.join(SAVE_DIR, exp_name, 'params.json'))
        data = json.load(f)
        opt = {
            'experimentname': exp_name,
            'backbone': data['backbone'],
            'seg_branch': data['seg_branch'],
            'imagesize': data['imagesize'],
            'num_classes': data['num_classes'],
            'label_smoothing': data['label_smoothing'],
            'weights': [data['weights'] if 'weights' in data.keys() else 'GastroNet'][0],
            'threshold': inference_opt.threshold,
            'ground_truth': inference_opt.ground_truth,
            'evaluate_sets': inference_opt.evaluate_sets,
            'textfile': inference_opt.textfile,
        }
        opt = argparse.Namespace(**opt)

        # Create text file for writing results
        f = open(os.path.join(SAVE_DIR, exp_name, opt.textfile), 'x')
        f_txt = open(os.path.join(SAVE_DIR, exp_name, opt.textfile), 'a')

        # Loop over all sets
        for inf_set in opt.evaluate_sets:
            if inf_set == 'Test':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Test Set (Test)')
            elif inf_set == 'All-Comers':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'All-Comers Set')
            elif inf_set == 'Test-Corrupt':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'Corrupt Test Set')
            elif inf_set == 'BORN':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'BORN Module Set')
            elif inf_set == 'ARGOS-DS34':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'ARGOS Fuji Set - DS34')
            elif inf_set == 'ARGOS-DS5':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'ARGOS Fuji Set - DS5')
            elif inf_set == 'IQ-Triplet':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'IQ-Triplet Set')
            elif inf_set == 'IQ-Triplet-HQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'IQ-Triplet Set - HQ')
            elif inf_set == 'IQ-Triplet-MQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'IQ-Triplet Set - MQ')
            elif inf_set == 'IQ-Triplet-LQ':
                CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Image Inference', 'IQ-Triplet Set - LQ')

            # Sets for validation
            elif inf_set == 'Val-HQ':
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
