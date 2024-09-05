"""IMPORT PACKAGES"""
import os
import re
import json
import random
import argparse
from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import (
    DATASET_TRAIN_TEST,
    DATASET_VAL,
    read_inclusion,
    read_inclusion_split,
    read_inclusion_cad2,
    sample_weights,
)

from data.dataset import augmentations
from data.dataset import domain_augmentations as d_augmentations
from models.model import Model

from utils.loss import construct_loss_function
from utils.metrics import construct_metric
from utils.optim import construct_optimizer, construct_scheduler

"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


# CADe1.0: Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    criteria["train"] = {
        "modality": ["wle"],
        "dataset": ["training"],
        "protocol": ["Retrospectief", "Prospectief"],
        "min_height": None,
        "min_width": None,
    }

    criteria["finetune"] = {
        "modality": ["wle"],
        "dataset": ["training"],
        "protocol": ["Prospectief"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev"] = {
        "modality": ["wle"],
        "dataset": ["validation"],
        "min_height": None,
        "min_width": None,
    }

    return criteria


# CADe2.0: Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria_cad2():
    criteria = dict()

    criteria["train-images"] = {
        "dataset": ["training"],
        "type": ['original', 'enhanced'],
        "source": ["images"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["train-frames-HQ"] = {
        "dataset": ["training"],
        "type": ['original'],
        "source": ["frames HQ"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["train-frames-MQ"] = {
        "dataset": ["training"],
        "type": ['original'],
        "source": ["frames MQ"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["train-frames-LQ"] = {
        "dataset": ["training"],
        "type": ['original'],
        "source": ["frames LQ"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev-images"] = {
        "dataset": ["validation"],
        "type": ['original', 'enhanced'],
        "source": ["images"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev-frames-HQ"] = {
        "dataset": ["validation"],
        "type": ['original'],
        "source": ["frames HQ"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev-frames-MQ"] = {
        "dataset": ["validation"],
        "type": ['original'],
        "source": ["frames MQ"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev-frames-LQ"] = {
        "dataset": ["validation"],
        "type": ['original'],
        "source": ["frames LQ"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["dev-robustness"] = {
        "dataset": ["validation-robustness"],
        "class": ["ndbe", "neo"],
        "quality": ["high", "medium", "low"],
        "mask_only": False,
        "min_height": None,
        "min_width": None,
    }

    criteria["ARGOS-DS3"] = {
        "dataset": ["Dataset 3"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria["ARGOS"] = {
        "dataset": ["Dataset 3", "Dataset 4", "Dataset 5"],
        "class": ["neo", "ndbe"],
        "min_height": None,
        "min_width": None,
    }

    criteria['train-UEGW'] = {
        "dataset": ["training"],
        "class": ["neo", "ndbe"],
        "mask_only": False,
        "min_height": None,
        "min_width": None,
    }

    criteria['dev-UEGW'] = {
        "dataset": ["validation"],
        "class": ["neo", "ndbe"],
        "mask_only": False,
        "min_height": None,
        "min_width": None,
    }

    return criteria


# Function for checking whether GPU or CPU is being used
def check_cuda():
    print("\nExtract Device...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        device_count = torch.cuda.device_count()
        torch.cuda.empty_cache()
        print("Using device: {}".format(device))
        print("Device name: {}".format(device_name))
        print("Device count: {}\n".format(device_count))
    else:
        device = torch.device("cpu")
        print("Using device: cpu\n")


# Find the best checkpoint model
def find_best_model(path, finetune):
    # Append all files
    files = list()
    values = list()

    # List files with certain extension
    if not finetune:
        for file in os.listdir(path):
            if file.endswith(".ckpt"):
                val = re.findall(r"\d+\.\d+", file)
                auc_seg, auc, hmean = val[0], val[1], val[2]
                value = hmean
                files.append(file)
                values.append(value)
    elif finetune:
        for file in os.listdir(path):
            if file.endswith(".ckpt") and "finetune" in file:
                val = re.findall(r"\d+\.\d+", file)
                auc_seg, auc, hmean = val[0], val[1], val[2]
                value = hmean
                files.append(file)
                values.append(value)

    # Find file with highest value
    max_val = max(values)
    indices = [i for i, x in enumerate(values) if x == max_val]
    max_index = indices[-1]

    return files[max_index]


# Remove keys from checkpoint for finetuning
def remove_keys(opt, ckpt_path):
    # Extract checkpoint name
    filename = os.path.splitext((os.path.split(ckpt_path)[1]))[0]

    # Load checkpoint
    checkpoint = torch.load(ckpt_path)

    # Unpack the keys of the checkpoint
    checkpoint_keys = list(checkpoint["state_dict"].keys())

    # Loop over the keys
    for key in checkpoint_keys:
        # Exclude layers that are to be preserved
        if "ResNet" in opt.backbone or "FCBFormer" in opt.backbone or "ESFPNet" in opt.backbone:
            if "backbone.fc" in key:
                del checkpoint["state_dict"][key]
                print("Deleted key: {}".format(key))
        elif "ConvNeXt" in opt.backbone:
            if "backbone.head" in key:
                del checkpoint["state_dict"][key]
                print("Deleted key: {}".format(key))
        elif "UNet" in opt.backbone:
            if "backbone.classification_head.3" in key:
                del checkpoint["state_dict"][key]
                print("Deleted key: {}".format(key))
        elif "Swin" in opt.backbone and "UperNet" in opt.backbone:
            if "backbone.fc" in key:
                del checkpoint["state_dict"][key]
                print("Deleted key: {}".format(key))

    # Save new checkpoint
    new_filename = filename + "-removehead.ckpt"
    torch.save(checkpoint, os.path.join(SAVE_DIR, opt.experimentname, new_filename))

    return new_filename


"""""" """""" """""" """""" """""" """""" """""" ""
"""" DATA: PYTORCH LIGHTNING DATAMODULES """
"""""" """""" """""" """""" """""" """""" """""" ""
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#why-do-i-need-a-datamodule


class WLEDataModuleTrain(pl.LightningDataModule):
    def __init__(self, data_dir, criteria, transforms, opt):
        super().__init__()
        self.data_dir = data_dir
        self.criteria = criteria
        self.transforms = transforms
        self.train_sampler = None
        self.train_set = None
        self.val_set_train = None
        self.val_set_test = None
        self.opt = opt

    def setup(self, stage: Optional[str] = None):
        # Find data that satisfies the inclusion criteria
        if 'CAD2' in self.opt.experimentname:
            train_inclusion = read_inclusion_cad2(path=self.data_dir, criteria=self.criteria["train-images"])

            if self.opt.training_content == 'Both':
                if self.opt.frame_quality == 'HQ':
                    train_inclusion_frames = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-HQ"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames
                elif self.opt.frame_quality == 'MQ':
                    train_inclusion_frames = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-MQ"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames
                elif self.opt.frame_quality == 'LQ':
                    train_inclusion_frames = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-LQ"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames
                elif self.opt.frame_quality == 'HQ-MQ':
                    train_inclusion_frames_hq = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-HQ"]
                    )
                    train_inclusion_frames_mq = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-MQ"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames_hq = random.sample(
                        train_inclusion_frames_hq, k=int(len(train_inclusion_frames_hq) * self.opt.frame_perc)
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames_mq = random.sample(
                        train_inclusion_frames_mq, k=int(len(train_inclusion_frames_mq) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames_hq + train_inclusion_frames_mq
                elif self.opt.frame_quality == 'HQ-LQ':
                    train_inclusion_frames_hq = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-HQ"]
                    )
                    train_inclusion_frames_lq = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-LQ"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames_hq = random.sample(
                        train_inclusion_frames_hq, k=int(len(train_inclusion_frames_hq) * self.opt.frame_perc)
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames_lq = random.sample(
                        train_inclusion_frames_lq, k=int(len(train_inclusion_frames_lq) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames_hq + train_inclusion_frames_lq
                elif self.opt.frame_quality == 'MQ-LQ':
                    train_inclusion_frames_mq = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-MQ"]
                    )
                    train_inclusion_frames_lq = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["train-frames-LQ"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames_mq = random.sample(
                        train_inclusion_frames_mq, k=int(len(train_inclusion_frames_mq) * self.opt.frame_perc)
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames_lq = random.sample(
                        train_inclusion_frames_lq, k=int(len(train_inclusion_frames_lq) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames_mq + train_inclusion_frames_lq

            val_inclusion = read_inclusion_cad2(path=self.data_dir, criteria=self.criteria["dev-images"])
            if self.opt.validation_content == 'Both':
                val_inclusion_frames = read_inclusion_cad2(
                    path=self.data_dir, criteria=self.criteria[f"dev-frames-{self.opt.frame_quality_val}"]
                )
                val_inclusion = val_inclusion + val_inclusion_frames
        elif 'UEGW' in self.opt.experimentname:
            train_inclusion = read_inclusion_cad2(path=self.data_dir, criteria=self.criteria["train-UEGW"])
            val_inclusion = read_inclusion_cad2(path=self.data_dir, criteria=self.criteria["dev-UEGW"])
        else:
            train_inclusion = read_inclusion_split(
                path=self.data_dir,
                criteria=self.criteria["train"],
                split_perc=self.opt.split_perc,
                split_seed=self.opt.split_seed,
            )
            if self.opt.training_content == 'Both':
                if self.opt.frame_quality == 'HQ':
                    train_inclusion_frames = read_inclusion(path=CACHE_PATH_FRAMES, criteria=self.criteria["train"])
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames
                elif self.opt.frame_quality == 'MQ':
                    train_inclusion_frames = read_inclusion(path=CACHE_PATH_FRAMES_MQ, criteria=self.criteria["train"])
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames
                elif self.opt.frame_quality == 'LQ':
                    train_inclusion_frames = read_inclusion(path=CACHE_PATH_FRAMES_LQ, criteria=self.criteria["train"])
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames
                elif self.opt.frame_quality == 'HQ-MQ':
                    train_inclusion_frames_hq = read_inclusion(path=CACHE_PATH_FRAMES, criteria=self.criteria["train"])
                    train_inclusion_frames_hq = random.sample(
                        train_inclusion_frames_hq, k=int(len(train_inclusion_frames_hq) * self.opt.frame_perc)
                    )
                    train_inclusion_frames_mq = read_inclusion(
                        path=CACHE_PATH_FRAMES_MQ, criteria=self.criteria["train"]
                    )
                    train_inclusion_frames_mq = random.sample(
                        train_inclusion_frames_mq, k=int(len(train_inclusion_frames_mq) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames_hq + train_inclusion_frames_mq
                elif self.opt.frame_quality == 'HQ-LQ':
                    train_inclusion_frames_hq = read_inclusion(path=CACHE_PATH_FRAMES, criteria=self.criteria["train"])
                    train_inclusion_frames_hq = random.sample(
                        train_inclusion_frames_hq, k=int(len(train_inclusion_frames_hq) * self.opt.frame_perc)
                    )
                    train_inclusion_frames_lq = read_inclusion(
                        path=CACHE_PATH_FRAMES_LQ, criteria=self.criteria["train"]
                    )
                    train_inclusion_frames_lq = random.sample(
                        train_inclusion_frames_lq, k=int(len(train_inclusion_frames_lq) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames_hq + train_inclusion_frames_lq
                elif self.opt.frame_quality == 'MQ-LQ':
                    train_inclusion_frames_mq = read_inclusion(
                        path=CACHE_PATH_FRAMES_MQ, criteria=self.criteria["train"]
                    )
                    train_inclusion_frames_mq = random.sample(
                        train_inclusion_frames_mq, k=int(len(train_inclusion_frames_mq) * self.opt.frame_perc)
                    )
                    train_inclusion_frames_lq = read_inclusion(
                        path=CACHE_PATH_FRAMES_LQ, criteria=self.criteria["train"]
                    )
                    train_inclusion_frames_lq = random.sample(
                        train_inclusion_frames_lq, k=int(len(train_inclusion_frames_lq) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames_mq + train_inclusion_frames_lq

            val_inclusion = read_inclusion(path=self.data_dir, criteria=self.criteria["dev"])
            val_inclusion_frames = read_inclusion(path=CACHE_PATH_FRAMES, criteria=self.criteria["dev"])
            val_inclusion = val_inclusion + val_inclusion_frames

        # Construct weights for the samples
        train_weights = sample_weights(train_inclusion)
        self.train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_inclusion),
            replacement=True,
        )

        # Construct datasets
        self.train_set = DATASET_TRAIN_TEST(
            opt=self.opt,
            inclusion=train_inclusion,
            transform=self.transforms["train"],
            random_noise=True,
        )
        self.val_set_test = DATASET_TRAIN_TEST(
            opt=self.opt,
            inclusion=val_inclusion,
            transform=self.transforms["test"],
            random_noise=False,
        )
        if 'CAD2' in self.opt.experimentname:
            self.val_set_train = DATASET_VAL(opt=self.opt, inclusion=val_inclusion, transform=self.transforms["val"])
        elif 'UEGW' in self.opt.experimentname:
            self.val_set_train = DATASET_TRAIN_TEST(
                opt=self.opt, inclusion=val_inclusion, transform=self.transforms["val"], random_noise=False
            )
        else:
            self.val_set_train = DATASET_VAL(opt=self.opt, inclusion=val_inclusion, transform=self.transforms["val"])

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set_train,
            batch_size=opt.batchsize,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
        )

    def test_dataloader(self):
        return DataLoader(self.val_set_test, batch_size=opt.batchsize, num_workers=4)


"""""" """""" """""" """""" """""" """""" """""" """""" ""
"""" MODEL: PYTORCH LIGHTNING & PYTORCH MODULE """
"""""" """""" """""" """""" """""" """""" """""" """""" ""
# https://www.pytorchlightning.ai/
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
# https://medium.com/aimstack/how-to-tune-hyper-params-with-fixed-seeds-using-pytorch-lightning-and-aim-c61c73f75c7c
# https://pytorch-lightning.readthedocs.io/en/1.4.3/common/weights_loading.html
# https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html


class WLEModel(pl.LightningModule):
    def __init__(self, opt, finetune):
        super(WLEModel, self).__init__()

        # Fix seed for reproducibility
        pl.seed_everything(seed=opt.seed, workers=True)

        # Define whether the stage is training or finetuning
        self.finetune = finetune

        # Define label smoothing
        self.label_smoothing = opt.label_smoothing

        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Define loss functions for classification and segmentation
        self.cls_criterion, self.seg_criterion = construct_loss_function(opt=opt)

        # Define model
        self.model = Model(opt=opt)

        # Specify metrics
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.train_aucseg = torchmetrics.AUROC(pos_label=1)
        self.train_dice = construct_metric(opt=opt)
        self.val_acc = torchmetrics.Accuracy(threshold=0.5)
        self.val_spec = torchmetrics.Specificity(threshold=0.5)
        self.val_sens = torchmetrics.Recall(threshold=0.5)
        self.val_auc = torchmetrics.AUROC(pos_label=1)
        self.val_aucseg = torchmetrics.AUROC(pos_label=1)
        self.val_dice = construct_metric(opt=opt)
        self.test_acc = torchmetrics.Accuracy(threshold=0.5)
        self.test_spec = torchmetrics.Specificity(threshold=0.5)
        self.test_sens = torchmetrics.Recall(threshold=0.5)
        self.test_auc = torchmetrics.AUROC(pos_label=1)
        self.test_aucseg = torchmetrics.AUROC(pos_label=1)
        self.test_dice = construct_metric(opt=opt)

    def forward(self, x):
        # # Extract outputs of the model
        # cls_out, mask_out = self.model(x)

        # Extract outputs of the model: Segmentation [BS, 1, h, w], Classification [BS, 1]
        out1, out2 = self.model(x)
        cls_out = out1 if out1.dim() == 2 else out2
        mask_out = out2 if out2.dim() == 4 else out1

        return cls_out, mask_out

    def configure_optimizers(self):
        # Define learning rate
        if not self.finetune:
            learning_rate = opt.train_lr
        else:
            learning_rate = opt.finetune_lr

        # Define optimizer
        optimizer = construct_optimizer(optim=opt.optimizer, parameters=self.parameters(), lr=learning_rate)

        # Define learning rate scheduler
        scheduler = construct_scheduler(schedule=opt.scheduler, optimizer=optimizer, lr=learning_rate)

        if scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        # Extract images, labels, mask and has_mask
        img, lab, mask, has_mask = train_batch

        # Extract predictions of the network
        preds, seg = self.forward(img)

        # Perform label smoothing
        lab_smooth = (1.0 - self.label_smoothing) * lab + self.label_smoothing * 0.5
        mask_smooth = (1.0 - self.label_smoothing) * mask + self.label_smoothing * 0.5

        # Compute Loss for both outputs
        cls_loss = self.cls_criterion(preds, lab_smooth)
        self.log("train_loss_cls", cls_loss.item())

        seg_loss = self.seg_criterion(seg, mask_smooth, has_mask, lab_smooth, batch_idx)
        self.log("train_loss_seg", seg_loss.item())

        summed_loss = cls_loss + seg_loss
        self.log("train_loss_combine", summed_loss.item())

        # Update metrics
        logits_cls = self.sigmoid(preds)
        logits_seg = self.sigmoid(seg)
        self.train_auc.update(logits_cls, lab.to(torch.int32))
        max_val, _ = torch.max(logits_seg.view(logits_seg.shape[0], -1), dim=1, keepdim=True)
        self.train_aucseg.update(max_val, lab.to(torch.int32))
        self.train_dice.update(logits_seg, mask, has_mask)

        return summed_loss

    def on_train_epoch_end(self):
        # Compute metrics
        train_auc = self.train_auc.compute()
        train_aucseg = self.train_aucseg.compute()
        train_dice = self.train_dice.compute()

        # Log and print metric value
        self.log("train_auc", train_auc)
        self.log("train_aucseg", train_aucseg)
        self.log("train_dice", train_dice)
        print("\n" + 120 * "=")
        print(f"Training Set:  AUC Cls: {train_auc:.4}, AUC Seg: {train_aucseg:.4}, Avg. Dice Score: {train_dice:.4}")
        print(120 * "=" + "\n")

        # Reset metric values
        self.train_auc.reset()
        self.train_aucseg.reset()
        self.train_dice.reset()

    def validation_step(self, val_batch, batch_idx):
        # Extract images, labels, mask and has_mask
        img, lab, mask, has_mask = val_batch

        # Extract predictions of the network
        preds, seg = self.forward(img)

        # Perform label smoothing
        lab_smooth = (1.0 - self.label_smoothing) * lab + self.label_smoothing * 0.5
        mask_smooth = (1.0 - self.label_smoothing) * mask + self.label_smoothing * 0.5

        # Compute Loss for both outputs
        cls_loss = self.cls_criterion(preds, lab_smooth)
        self.log("val_loss_cls", cls_loss.item())

        seg_loss = self.seg_criterion(seg, mask_smooth, has_mask, lab_smooth, batch_idx)
        self.log("val_loss_seg", seg_loss.item())

        summed_loss = cls_loss + seg_loss
        self.log("val_loss_combine", summed_loss.item())

        # Update metrics
        logits_cls = self.sigmoid(preds)
        logits_seg = self.sigmoid(seg)
        self.val_acc.update(logits_cls, lab.to(torch.int32))
        self.val_sens.update(logits_cls, lab.to(torch.int32))
        self.val_spec.update(logits_cls, lab.to(torch.int32))
        self.val_auc.update(logits_cls, lab.to(torch.int32))
        max_val, _ = torch.max(logits_seg.view(logits_seg.shape[0], -1), dim=1, keepdim=True)
        self.val_aucseg.update(max_val, lab.to(torch.int32))
        self.val_dice.update(logits_seg, mask, has_mask)

        return summed_loss

    def on_validation_epoch_end(self):
        # Compute metric values
        val_acc = self.val_acc.compute()
        val_sens = self.val_sens.compute()
        val_spec = self.val_spec.compute()
        val_auc = self.val_auc.compute()
        val_aucseg = self.val_aucseg.compute()
        val_dice = self.val_dice.compute()

        # Log and print values
        self.log("val_acc", val_acc)
        self.log("val_sens", val_sens)
        self.log("val_spec", val_spec)
        self.log("val_auc", val_auc)
        self.log("val_aucseg", val_aucseg)
        self.log("val_dice", val_dice)
        self.log("hmean_auc", (2 * val_aucseg * val_auc) / (val_aucseg + val_auc))
        print("\n\n" + 120 * "=")
        print(
            f"Validation Set: Accuracy: {val_acc:.4}, Sensitivity: {val_sens:.4}, "
            f"Specificity: {val_spec:.4}, AUC Cls: {val_auc:.4}, AUC Seg: {val_aucseg:.4}, "
            f"Avg. Dice Score: {val_dice:.4}"
        )
        print(120 * "=" + "\n")

        # Reset metric values
        self.val_acc.reset()
        self.val_sens.reset()
        self.val_spec.reset()
        self.val_auc.reset()
        self.val_aucseg.reset()
        self.val_dice.reset()

    def test_step(self, test_batch, batch_idx):
        # Extract images, labels, mask and has_mask
        img, lab, mask, has_mask = test_batch

        # Extract predictions of the network
        preds, seg = self.forward(img)

        # Update metrics
        logits_cls = self.sigmoid(preds)
        logits_seg = self.sigmoid(seg)
        self.test_acc.update(logits_cls, lab.to(torch.int32))
        self.test_sens.update(logits_cls, lab.to(torch.int32))
        self.test_spec.update(logits_cls, lab.to(torch.int32))
        self.test_auc.update(logits_cls, lab.to(torch.int32))
        max_val, _ = torch.max(logits_seg.view(logits_seg.shape[0], -1), dim=1, keepdim=True)
        self.test_aucseg.update(max_val, lab.to(torch.int32))
        self.test_dice.update(logits_seg, mask, has_mask)

    def on_test_epoch_end(self):
        # Execute metric computation
        test_acc = self.test_acc.compute()
        test_sens = self.test_sens.compute()
        test_spec = self.test_spec.compute()
        test_auc = self.test_auc.compute()
        test_aucseg = self.test_aucseg.compute()
        test_dice = self.test_dice.compute()

        # Print results
        print("\n\n" + 120 * "=")
        print(
            f"Test Set: Accuracy: {test_acc:.4}, Sensitivity: {test_sens:.4}, "
            f"Specificity: {test_spec:.4}, AUC Cls: {test_auc:.4}, AUC Seg: {test_aucseg:.4}, "
            f"Avg. Dice Score: {test_dice:.4}"
        )
        print(120 * "=" + "\n")

        # Reset metric values
        self.test_acc.reset()
        self.test_sens.reset()
        self.test_spec.reset()
        self.test_auc.reset()
        self.test_aucseg.reset()
        self.test_dice.reset()


"""""" """""" """""" """""" """"""
"""" FUNCTION FOR EXECUTION """
"""""" """""" """""" """""" """"""


def run_without_finetune(opt):
    """TEST DEVICE"""
    check_cuda()
    torch.set_float32_matmul_precision(precision="medium")

    """SETUP PYTORCH LIGHTNING DATAMODULE"""
    print("Starting PyTorch Lightning DataModule...")
    if 'CAD2' in opt.experimentname or 'UEGW' in opt.experimentname:
        criteria = get_data_inclusion_criteria_cad2()
    else:
        criteria = get_data_inclusion_criteria()

    if opt.augmentations == "domain":
        data_transforms = d_augmentations(opt)
    else:
        data_transforms = augmentations(opt)

    dm_train = WLEDataModuleTrain(
        data_dir=CACHE_PATH,
        criteria=criteria,
        transforms=data_transforms,
        opt=opt,
    )

    """SETUP PYTORCH LIGHTNING MODEL"""
    print("Starting PyTorch Lightning Model...")

    # Construct Loggers for PyTorch Lightning
    if 'CAD2' in opt.experimentname:
        wandb_logger_train = WandbLogger(
            name="{}".format(opt.experimentname),
            project="WLE CADe2.0 Development",
            save_dir=os.path.join(SAVE_DIR, opt.experimentname),
        )
    else:
        wandb_logger_train = WandbLogger(
            name="{}".format(opt.experimentname),
            project="WLE CADe Benchmark",
            save_dir=os.path.join(SAVE_DIR, opt.experimentname),
        )
    lr_monitor_train = LearningRateMonitor(logging_interval="step")

    # Construct callback used for training the model
    checkpoint_callback_train = ModelCheckpoint(
        monitor="hmean_auc",
        mode="max",
        dirpath=os.path.join(SAVE_DIR, opt.experimentname),
        filename="model-{epoch:02d}-{val_aucseg:}-{val_auc:.4f}-{hmean_auc:.4f}",
        save_top_k=3,
        save_weights_only=True,
    )

    # Construct callback for early stopping of the training
    if 'UEGW' in opt.experimentname:
        early_stopping = EarlyStopping(
            monitor='hmean_auc', min_delta=0.0005, patience=10, mode='max', check_on_train_epoch_end=False
        )
    else:
        early_stopping = EarlyStopping(
            monitor='hmean_auc', min_delta=0.0005, patience=25, mode='max', check_on_train_epoch_end=False
        )

    """TRAINING PHASE"""

    # Construct PyTorch Lightning Trainer
    pl_model = WLEModel(opt=opt, finetune=False)
    trainer = pl.Trainer(
        devices=1,
        # precision='16',
        accelerator="gpu",
        max_epochs=opt.num_epochs,
        logger=wandb_logger_train,
        callbacks=[checkpoint_callback_train, lr_monitor_train, early_stopping],
        check_val_every_n_epoch=1,
        deterministic=False,
    )

    # Start Training
    trainer.fit(model=pl_model, datamodule=dm_train)
    wandb_logger_train.experiment.finish()

    """INFERENCE PHASE"""
    best_index = find_best_model(path=os.path.join(SAVE_DIR, opt.experimentname), finetune=False)
    trainer.test(
        model=pl_model,
        datamodule=dm_train,
        ckpt_path=os.path.join(SAVE_DIR, opt.experimentname, best_index),
    )


"""""" """""" """""" """""" ""
"""EXECUTION OF FUNCTIONS"""
"""""" """""" """""" """""" ""

if __name__ == "__main__":
    """ARGUMENT PARSER"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DEFINE EXPERIMENT NAME
    parser.add_argument("--experimentname", type=str)
    parser.add_argument("--seed", type=int, default=7)

    # DEFINE OUTPUT FOLDER
    parser.add_argument("--output_folder", type=str, default=None)

    # DEFINE MODEL
    parser.add_argument("--backbone", type=str, default="MetaFormer-CAS18-FCN")
    parser.add_argument("--seg_branch", type=str, default=None)
    parser.add_argument("--weights", type=str, default="ImageNet", help="ImageNet, GastroNet, GastroNet-DSA")

    # DEFINE OPTIMIZER, CRITERION, SCHEDULER
    parser.add_argument("--optimizer", type=str, default="Adam", help="Adam, SGD")
    parser.add_argument("--scheduler", type=str, default="Plateau", help="Plateau, Step, Cosine")
    parser.add_argument("--cls_criterion", type=str, default="BCE", help="BCE, Focal")
    parser.add_argument("--cls_criterion_weight", type=float, default=1.0)
    parser.add_argument(
        "--seg_criterion",
        type=str,
        default="DiceBCE",
        help="MSE, BCE, Dice, DiceBCE, IoU, Focal, DiceFocal, "
        "MultiMaskMSE, MultiMaskBCE, MultiMaskDice, MultiMaskDiceBCE,"
        "MultiMaskDiceW, MultiMaskDiceBCEW",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument('--focal_alpha_cls', type=float, default=-1.0)
    parser.add_argument('--focal_gamma_cls', type=float, default=1.0)
    parser.add_argument('--focal_alpha_seg', type=float, default=-1.0)
    parser.add_argument('--focal_gamma_seg', type=float, default=1.0)
    parser.add_argument('--seg_metric', type=str, default='Dice', help='Dice, IoU, MultiMaskDice, MultiMaskDiceW')

    # AUGMENTATION PARAMS
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--augmentations", type=str, default="default", help="default, domain")

    # MASK PARAMS
    parser.add_argument(
        '--mask_content', type=str, default='Plausible', help='Soft, Plausible, Sweet, Hard, Random, Average, Multiple'
    )  # 'Consensus' == 'Plausible'

    # DATA PERCENTAGE PARAMS
    parser.add_argument("--split_perc", type=float, default=1.0)
    parser.add_argument("--split_seed", type=int, default=7)

    # FRAME PARAMS
    parser.add_argument("--training_content", type=str, default="Images", help="Both, Images")
    parser.add_argument('--frame_quality', type=str, default='HQ', help='HQ, MQ, LQ, HQ-MQ, HQ-LQ, MQ-LQ')
    parser.add_argument('--frame_perc', type=float, default=1.0)
    parser.add_argument("--validation_content", type=str, default="Images", help="Both, Images")
    parser.add_argument("--frame_quality_val", type=str, default="HQ", help="HQ, MQ, LQ")

    # TRAINING PARAMETERS
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--train_lr", type=float, default=1e-6)

    opt = parser.parse_args()

    """SPECIFY CACHE PATH"""
    if 'CAD2' in opt.experimentname:
        CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_dev_cad2_new_am.json')
    elif 'UEGW' in opt.experimentname:
        CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_wle_dev_cad2_uegw.json')
    else:
        CACHE_PATH = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val-test_all_masks_fixed")
        CACHE_PATH_FRAMES = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val_frames")
        CACHE_PATH_FRAMES_MQ = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val_frames_MQ")
        CACHE_PATH_FRAMES_LQ = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val_frames_LQ")

    """SPECIFY PATH FOR SAVING"""
    SAVE_DIR = os.path.join(os.getcwd(), opt.output_folder)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    # Check if direction for logging the information already exists; otherwise make direction
    if not os.path.exists(os.path.join(SAVE_DIR, opt.experimentname)):
        os.mkdir(os.path.join(SAVE_DIR, opt.experimentname))

    # Save params from opt as a dictionary in a json file 'params.json'
    with open(os.path.join(SAVE_DIR, opt.experimentname, "params.json"), "w") as fp:
        json.dump(opt.__dict__, fp, indent=4)

    # Save inclusion criteria (already dictionary) in a json file 'datacriteria.json'
    with open(os.path.join(SAVE_DIR, opt.experimentname, "datacriteria.json"), "w") as fp:
        json.dump(get_data_inclusion_criteria(), fp, indent=4)

    """SANITY CHECK FOR MULTIPLE MASKS AND SEGMENTATION LOS/CRITERION"""
    multi_mask_loss = [
        'MultiMaskMSE',
        'MultiMaskBCE',
        'MultiMaskDice',
        'MultiMaskDiceBCE',
        'MultiMaskDiceW',
        'MultiMaskDiceBCEW',
    ]
    multi_mask_metric = ['MultiMaskDice', 'MultiMaskDiceW']
    if opt.mask_content == 'Multiple':
        if opt.seg_criterion not in multi_mask_loss:
            raise Exception('For multiple masks, please select a segmentation criterion that supports multiple masks.')
        if opt.seg_metric not in multi_mask_metric:
            raise Exception('For multiple masks, please select a segmentation metric that supports multiple masks.')
    else:
        if opt.seg_criterion in multi_mask_loss:
            raise Exception('For segmention criterion that supports multiple masks, please select multiple masks.')
        if opt.seg_metric in multi_mask_metric:
            raise Exception('For segmentation metric that supports multiple masks, please select multiple masks.')

    """EXECUTE FUNCTION"""
    run_without_finetune(opt)
