"""IMPORT PACKAGES"""
import argparse
import os
import re
import json
import random
from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset_cls import (
    DATASET_TRAIN_TEST,
    DATASET_VAL,
    read_inclusion,
    read_inclusion_cad2,
    read_inclusion_split,
    sample_weights,
)

from data.dataset_cls import augmentations
from data.dataset_cls import domain_augmentations as d_augmentations
from models.model import Model_CLS as Model

from utils.loss import construct_loss_function_cls as construct_loss_function
from utils.optim import construct_optimizer, construct_scheduler

"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


# CADe1.0: Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    criteria["train"] = {
        "dataset": ["training"],
        "min_height": None,
        "min_width": None,
    }

    criteria["val"] = {
        "dataset": ["validation"],
        "min_height": None,
        "min_width": None,
    }

    return criteria


# CADe2.0: Specify function for defining inclusion criteria for training, finetuning and development set
# CADx2.0: Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria_cad2():
    criteria = dict()

    criteria["nbi-train"] = {
        "dataset": ["training"],
        "source": ["images"],
        "class": ["ndbe", "neo"],
        "cap": ["cap", "no cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-train-frames"] = {
        "dataset": ["training"],
        "source": ["frames"],
        "class": ["ndbe", "neo"],
        "cap": ["cap", "no cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-train-frames-cap"] = {
        "dataset": ["training"],
        "source": ["frames"],
        "class": ["ndbe", "neo"],
        "cap": ["cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-train-frames-no-cap"] = {
        "dataset": ["training"],
        "source": ["frames"],
        "class": ["ndbe", "neo"],
        "cap": ["no cap"],
        "type": ["focus", "overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-train-frames-no-cap-overview"] = {
        "dataset": ["training"],
        "source": ["frames"],
        "class": ["ndbe", "neo"],
        "cap": ["no cap"],
        "type": ["overview"],
        "min_height": None,
        "min_width": None,
    }

    criteria["nbi-val"] = {
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
def find_best_model(path):
    # Append all files
    files = list()
    values = list()

    # List files with certain extension
    for file in os.listdir(path):
        if file.endswith(".ckpt"):
            val = re.findall(r"\d+\.\d+", file)
            value = val[0]
            files.append(file)
            values.append(value)

    # Find file with the highest value
    max_val = max(values)
    indices = [i for i, x in enumerate(values) if x == max_val]
    max_index = indices[-1]

    return files[max_index]


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
        # Find data that satisfies the inclusion criteria regarding training data
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
                train_inclusion_frames_mq = read_inclusion(path=CACHE_PATH_FRAMES_MQ, criteria=self.criteria["train"])
                train_inclusion_frames_mq = random.sample(
                    train_inclusion_frames_mq, k=int(len(train_inclusion_frames_mq) * self.opt.frame_perc)
                )
                train_inclusion = train_inclusion + train_inclusion_frames_hq + train_inclusion_frames_mq
            elif self.opt.frame_quality == 'HQ-LQ':
                train_inclusion_frames_hq = read_inclusion(path=CACHE_PATH_FRAMES, criteria=self.criteria["train"])
                train_inclusion_frames_hq = random.sample(
                    train_inclusion_frames_hq, k=int(len(train_inclusion_frames_hq) * self.opt.frame_perc)
                )
                train_inclusion_frames_lq = read_inclusion(path=CACHE_PATH_FRAMES_LQ, criteria=self.criteria["train"])
                train_inclusion_frames_lq = random.sample(
                    train_inclusion_frames_lq, k=int(len(train_inclusion_frames_lq) * self.opt.frame_perc)
                )
                train_inclusion = train_inclusion + train_inclusion_frames_hq + train_inclusion_frames_lq
            elif self.opt.frame_quality == 'MQ-LQ':
                train_inclusion_frames_mq = read_inclusion(path=CACHE_PATH_FRAMES_MQ, criteria=self.criteria["train"])
                train_inclusion_frames_mq = random.sample(
                    train_inclusion_frames_mq, k=int(len(train_inclusion_frames_mq) * self.opt.frame_perc)
                )
                train_inclusion_frames_lq = read_inclusion(path=CACHE_PATH_FRAMES_LQ, criteria=self.criteria["train"])
                train_inclusion_frames_lq = random.sample(
                    train_inclusion_frames_lq, k=int(len(train_inclusion_frames_lq) * self.opt.frame_perc)
                )
                train_inclusion = train_inclusion + train_inclusion_frames_mq + train_inclusion_frames_lq

        # Find data that satisfies the inclusion criteria regarding validation data
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
            inclusion=train_inclusion,
            transform=self.transforms["train"],
            random_noise=True,
        )
        self.val_set_test = DATASET_TRAIN_TEST(
            inclusion=val_inclusion,
            transform=self.transforms["test"],
            random_noise=False,
        )
        self.val_set_train = DATASET_VAL(inclusion=val_inclusion, transform=self.transforms["val"])

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


class NBIDataModuleTrain(pl.LightningDataModule):
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
        # Find data that satisfies the inclusion criteria regarding training data
        train_inclusion = read_inclusion_cad2(path=self.data_dir, criteria=self.criteria["nbi-train"])

        if self.opt.training_content == 'Both':
            if self.opt.frame_cap == 'Both':
                train_inclusion_frames = read_inclusion_cad2(
                    path=self.data_dir, criteria=self.criteria["nbi-train-frames"]
                )
                random.seed(self.opt.seed)
                train_inclusion_frames = random.sample(
                    train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                )
                train_inclusion = train_inclusion + train_inclusion_frames
            elif self.opt.frame_cap == 'Cap':
                train_inclusion_frames = read_inclusion_cad2(
                    path=self.data_dir, criteria=self.criteria["nbi-train-frames-cap"]
                )
                random.seed(self.opt.seed)
                train_inclusion_frames = random.sample(
                    train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                )
                train_inclusion = train_inclusion + train_inclusion_frames
            elif self.opt.frame_cap == 'NoCap':
                if self.opt.frame_type == 'Overview':
                    train_inclusion_frames = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["nbi-train-frames-no-cap-overview"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames
                else:
                    train_inclusion_frames = read_inclusion_cad2(
                        path=self.data_dir, criteria=self.criteria["nbi-train-frames-no-cap"]
                    )
                    random.seed(self.opt.seed)
                    train_inclusion_frames = random.sample(
                        train_inclusion_frames, k=int(len(train_inclusion_frames) * self.opt.frame_perc)
                    )
                    train_inclusion = train_inclusion + train_inclusion_frames

        # Find data that satisfies the inclusion criteria regarding validation data
        val_inclusion = read_inclusion_cad2(path=self.data_dir, criteria=self.criteria["nbi-val"])
        if self.opt.validation_content == 'Both':
            val_inclusion_frames = read_inclusion_cad2(path=self.data_dir, criteria=self.criteria["nbi-val-frames"])
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
            inclusion=train_inclusion,
            transform=self.transforms["train"],
            random_noise=True,
        )
        self.val_set_test = DATASET_TRAIN_TEST(
            inclusion=val_inclusion,
            transform=self.transforms["test"],
            random_noise=False,
        )
        self.val_set_train = DATASET_VAL(inclusion=val_inclusion, transform=self.transforms["val"])

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


class WLE_NBI_Model(pl.LightningModule):
    def __init__(self, opt):
        super(WLE_NBI_Model, self).__init__()

        # Fix seed for reproducibility
        pl.seed_everything(seed=opt.seed, workers=True)

        # Define label smoothing
        self.label_smoothing = opt.label_smoothing

        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Define loss functions for classification and segmentation
        self.cls_criterion = construct_loss_function(opt=opt)

        # Define model
        self.model = Model(opt=opt)

        # Specify metrics
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.val_acc = torchmetrics.Accuracy(threshold=0.5)
        self.val_spec = torchmetrics.Specificity(threshold=0.5)
        self.val_sens = torchmetrics.Recall(threshold=0.5)
        self.val_auc = torchmetrics.AUROC(pos_label=1)
        self.test_acc = torchmetrics.Accuracy(threshold=0.5)
        self.test_spec = torchmetrics.Specificity(threshold=0.5)
        self.test_sens = torchmetrics.Recall(threshold=0.5)
        self.test_auc = torchmetrics.AUROC(pos_label=1)

    def forward(self, x):
        # Extract outputs of the model
        cls_out = self.model(x)

        return cls_out

    def configure_optimizers(self):
        # Define learning rate
        learning_rate = opt.train_lr

        # Define optimizer
        optimizer = construct_optimizer(optim=opt.optimizer, parameters=self.parameters(), lr=learning_rate)

        # Define learning rate scheduler
        scheduler = construct_scheduler(
            schedule=opt.scheduler, optimizer=optimizer, lr=learning_rate, metric='val_loss_cls'
        )

        if scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        # Extract images and labels
        img, lab = train_batch

        # Extract predictions of the network
        preds = self.forward(img)

        # Perform label smoothing
        lab_smooth = (1.0 - self.label_smoothing) * lab + self.label_smoothing * 0.5

        # Compute Loss for both outputs
        cls_loss = self.cls_criterion(preds, lab_smooth)
        self.log("train_loss_cls", cls_loss.item())

        # Update metrics
        logits_cls = self.sigmoid(preds)
        self.train_auc.update(logits_cls, lab.to(torch.int32))

        return cls_loss

    def on_train_epoch_end(self):
        # Compute metrics
        train_auc = self.train_auc.compute()

        # Log and print metric value
        self.log("train_auc", train_auc)
        print("\n" + 120 * "=")
        print(f"Training Set:  AUC Cls: {train_auc:.4}")
        print(120 * "=" + "\n")

        # Reset metric values
        self.train_auc.reset()

    def validation_step(self, val_batch, batch_idx):
        # Extract images and labels
        img, lab = val_batch

        # Extract predictions of the network
        preds = self.forward(img)

        # Perform label smoothing
        lab_smooth = (1.0 - self.label_smoothing) * lab + self.label_smoothing * 0.5

        # Compute Loss for both outputs
        cls_loss = self.cls_criterion(preds, lab_smooth)
        self.log("val_loss_cls", cls_loss.item())

        # Update metrics
        logits_cls = self.sigmoid(preds)
        self.val_acc.update(logits_cls, lab.to(torch.int32))
        self.val_sens.update(logits_cls, lab.to(torch.int32))
        self.val_spec.update(logits_cls, lab.to(torch.int32))
        self.val_auc.update(logits_cls, lab.to(torch.int32))

        return cls_loss

    def on_validation_epoch_end(self):
        # Compute metric values
        val_acc = self.val_acc.compute()
        val_sens = self.val_sens.compute()
        val_spec = self.val_spec.compute()
        val_auc = self.val_auc.compute()

        # Log and print values
        self.log("val_acc", val_acc)
        self.log("val_sens", val_sens)
        self.log("val_spec", val_spec)
        self.log("val_auc", val_auc)
        print("\n\n" + 120 * "=")
        print(
            f"Validation Set: Accuracy: {val_acc:.4}, Sensitivity: {val_sens:.4}, "
            f"Specificity: {val_spec:.4}, AUC Cls: {val_auc:.4}"
        )
        print(120 * "=" + "\n")

        # Reset metric values
        self.val_acc.reset()
        self.val_sens.reset()
        self.val_spec.reset()
        self.val_auc.reset()

    def test_step(self, test_batch, batch_idx):
        # Extract images and labels
        img, lab = test_batch

        # Extract predictions of the network
        preds = self.forward(img)

        # Update metrics
        logits_cls = self.sigmoid(preds)
        self.test_acc.update(logits_cls, lab.to(torch.int32))
        self.test_sens.update(logits_cls, lab.to(torch.int32))
        self.test_spec.update(logits_cls, lab.to(torch.int32))
        self.test_auc.update(logits_cls, lab.to(torch.int32))

    def on_test_epoch_end(self):
        # Execute metric computation
        test_acc = self.test_acc.compute()
        test_sens = self.test_sens.compute()
        test_spec = self.test_spec.compute()
        test_auc = self.test_auc.compute()

        # Print results
        print("\n\n" + 120 * "=")
        print(
            f"Test Set: Accuracy: {test_acc:.4}, Sensitivity: {test_sens:.4}, "
            f"Specificity: {test_spec:.4}, AUC Cls: {test_auc:.4}"
        )
        print(120 * "=" + "\n")

        # Reset metric values
        self.test_acc.reset()
        self.test_sens.reset()
        self.test_spec.reset()
        self.test_auc.reset()


"""""" """""" """""" """""" """"""
"""" FUNCTION FOR EXECUTION """
"""""" """""" """""" """""" """"""


def run(opt):
    """TEST DEVICE"""
    check_cuda()
    torch.set_float32_matmul_precision(precision="medium")

    """SETUP PYTORCH LIGHTNING DATAMODULE"""
    print("Starting PyTorch Lightning DataModule...")
    if 'CAD2' in opt.experimentname:
        criteria = get_data_inclusion_criteria_cad2()
    else:
        criteria = get_data_inclusion_criteria()

    if opt.augmentations == "domain":
        data_transforms = d_augmentations(opt)
    else:
        data_transforms = augmentations(opt)

    """SETUP PYTORCH LIGHTNING MODEL"""
    print("Starting PyTorch Lightning Model...")

    """DISTINGUISH BETWEEN WLE AND NBI"""
    if opt.modality == "WLE":
        dm_train = WLEDataModuleTrain(
            data_dir=CACHE_PATH,
            criteria=criteria,
            transforms=data_transforms,
            opt=opt,
        )

        # Construct Loggers for PyTorch Lightning
        wandb_logger_train = WandbLogger(
            name="{}".format(opt.experimentname),
            project="WLE CADe Benchmark",
            save_dir=os.path.join(SAVE_DIR, opt.experimentname),
        )
    elif opt.modality == "NBI":
        dm_train = NBIDataModuleTrain(
            data_dir=CACHE_PATH,
            criteria=criteria,
            transforms=data_transforms,
            opt=opt,
        )

        # Construct Loggers for PyTorch Lightning
        wandb_logger_train = WandbLogger(
            name="{}".format(opt.experimentname),
            project="NBI CADx Benchmark",
            save_dir=os.path.join(SAVE_DIR, opt.experimentname),
        )

    # Construct callback used for training the model
    checkpoint_callback_train = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        dirpath=os.path.join(SAVE_DIR, opt.experimentname),
        filename="model-{epoch:02d}-{val_auc:.4f}",
        save_top_k=3,
        save_weights_only=True,
    )

    # Construct learning rate monitor
    lr_monitor_train = LearningRateMonitor(logging_interval="step")

    # Construct callback for early stopping of the training
    early_stopping = EarlyStopping(
        monitor='val_auc', min_delta=0.0005, patience=25, mode='max', check_on_train_epoch_end=False
    )

    """TRAINING PHASE"""

    # Construct PyTorch Lightning Trainer
    pl_model = WLE_NBI_Model(opt=opt)
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
    best_index = find_best_model(path=os.path.join(SAVE_DIR, opt.experimentname))
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
    parser.add_argument("--backbone", type=str, default="CaFormer-S18")
    parser.add_argument("--weights", type=str, default="ImageNet", help="ImageNet, GastroNet, GastroNet-DSA")

    # DEFINE OPTIMIZER, CRITERION, SCHEDULER
    parser.add_argument("--optimizer", type=str, default="Adam", help="Adam, SGD")
    parser.add_argument("--scheduler", type=str, default="Plateau", help="Plateau, Step, Cosine")
    parser.add_argument("--cls_criterion", type=str, default="BCE", help="BCE, Focal")
    parser.add_argument("--cls_criterion_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument('--focal_alpha_cls', type=float, default=-1.0)
    parser.add_argument('--focal_gamma_cls', type=float, default=1.0)

    # AUGMENTATION PARAMS
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--augmentations", type=str, default="default", help="default, domain")

    # DATA PERCENTAGE PARAMS
    parser.add_argument("--split_perc", type=float, default=1.0)
    parser.add_argument("--split_seed", type=int, default=7)
    parser.add_argument("--modality", type=str, default="WLE", help="WLE, NBI")

    # FRAME PARAMS
    parser.add_argument("--training_content", type=str, default="Images", help="Both, Images")
    parser.add_argument('--frame_cap', type=str, default='Both', help='Both, Cap, NoCap')
    parser.add_argument('--frame_type', type=str, default='Both', help='Both, Focus, Overview')
    parser.add_argument('--frame_perc', type=float, default=1.0)
    parser.add_argument("--validation_content", type=str, default="Images", help="Both, Images")

    # TRAINING PARAMETERS
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--train_lr", type=float, default=1e-6)

    opt = parser.parse_args()

    """SPECIFY CACHE PATH"""
    if opt.modality == "WLE":
        CACHE_PATH = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val-test_plausible")
        CACHE_PATH_FRAMES = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val_frames")
        CACHE_PATH_FRAMES_MQ = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val_frames_MQ")
        CACHE_PATH_FRAMES_LQ = os.path.join(os.getcwd(), "cache folders", "cache_wle_train-val_frames_LQ")
    elif opt.modality == "NBI":
        CACHE_PATH = os.path.join(os.getcwd(), 'cache folders', 'cache_nbi_dev_cad2.json')
    else:
        raise ValueError("Modality not supported")

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

    """EXECUTE FUNCTION"""
    run(opt=opt)
