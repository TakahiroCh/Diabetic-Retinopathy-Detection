import collections
import os
from typing import Optional
import numpy
import pytorch_lightning as pl
import pandas as pd
import torch
from torchmetrics import Accuracy
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import save
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


# Dataset class
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self,
                 path_to_data: str,
                 path_to_labels: str,
                 picture_size: [int, int],
                 transform: transforms.Compose = None,
                 full_size: int = None):
        self.info = pd.read_csv(path_to_labels)
        self.path_to_data = path_to_data
        self.dir = os.listdir(self.path_to_data)

        self.picture_size = picture_size

        if full_size is not None:
            self.dir = self.dir[0: full_size + 1]

        self.names = [(x.image + ".jpeg") for x in self.info.iloc if (x.image + ".jpeg") in self.dir]
        self.labels = [x.level for x in self.info.iloc if (x.image + ".jpeg") in self.dir]
        print(len(self.labels), collections.Counter(self.labels))

        if transform is None:
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((self.picture_size[0], self.picture_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.size = len(self.dir)

    def cropping_black_pixels(self, current_image, tol=7):
        if current_image.ndim == 2:
            mask = current_image > tol
            return current_image[numpy.ix_(mask.any(1), mask.any(0))]
        elif current_image.ndim == 3:
            gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            mask = gray_image > tol
            check_shape = current_image[:, :, 0][numpy.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:
                return current_image
            else:
                img1 = current_image[:, :, 0][numpy.ix_(mask.any(1), mask.any(0))]
                img2 = current_image[:, :, 1][numpy.ix_(mask.any(1), mask.any(0))]
                img3 = current_image[:, :, 2][numpy.ix_(mask.any(1), mask.any(0))]
                current_image = numpy.stack([img1, img2, img3], axis=-1)
            return current_image

    def transform_circle_gray(self, path_to_image):
        current_image = cv2.imread(path_to_image)
        current_image = self.cropping_black_pixels(current_image)

        h, w, _ = current_image.shape
        largest_side = max(h, w)
        current_image = cv2.resize(current_image, (largest_side, largest_side))

        h, w, _ = current_image.shape
        w_half, h_half = int(w / 2), int(h / 2)
        r = min(w_half, h_half)
        circle_image = numpy.zeros((h, w), numpy.uint8)
        cv2.circle(circle_image, (w_half, h_half), r, 1, thickness=-1)

        current_image = cv2.bitwise_and(current_image, current_image, mask=circle_image)
        current_image = self.cropping_black_pixels(current_image)
        return current_image

    def transform_gauss(self, path_to_image):
        current_image = cv2.imread(path_to_image)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        current_image = self.cropping_black_pixels(current_image)

        h, w, _ = current_image.shape
        largest_side = max(h, w)
        current_image = cv2.resize(current_image, (largest_side, largest_side))

        h, w, _ = current_image.shape
        w_half, h_half = int(w / 2), int(h / 2)
        r = min(w_half, h_half)
        circle_image = numpy.zeros((h, w), numpy.uint8)
        cv2.circle(circle_image, (w_half, h_half), r, 1, thickness=-1)

        current_image = cv2.bitwise_and(current_image, current_image, mask=circle_image)
        current_image = self.cropping_black_pixels(current_image)

        current_image = cv2.resize(current_image, (self.picture_size[0], self.picture_size[1]))
        current_image = cv2.addWeighted(current_image, 4, cv2.GaussianBlur(current_image, (0, 0), 10), -4, 128)
        return current_image

    def transform_gauss_simple(self, path):
        current_image = cv2.imread(path)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        current_image = self.cropping_black_pixels(current_image)
        current_image = cv2.resize(current_image, (self.picture_size[0], self.picture_size[1]))
        current_image = cv2.addWeighted(current_image, 4, cv2.GaussianBlur(current_image, (0, 0), 10), -4, 128)

        return current_image

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        current_image_name = self.names[index]
        path_to_image = os.path.join(self.path_to_data, current_image_name)

        current_image = Image.open(path_to_image)
        # current_image = self.transform_gauss_simple(path_to_image)

        image = self.transform(current_image)
        return image, self.labels[index]


# Datamodule class
class DiabeticRetinopathyDataModule(pl.LightningDataModule):
    def __init__(self,
                 path_to_data: str,
                 batch_size: int = 32,
                 picture_size: [int, int] = [256, 256],
                 shuffle: bool = True,
                 size_of_data: int = None,
                 transform: transforms.Compose = None):
        super().__init__()
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.picture_size = picture_size
        self.shuffle = shuffle
        self.size = size_of_data
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train = DiabeticRetinopathyDataset(self.path_to_data + "train/", "labels/trainLabels.csv",
                                                    self.picture_size, self.transform, self.size)

            self.validation = DiabeticRetinopathyDataset(self.path_to_data + "validation/", "labels/trainLabels.csv",
                                                         self.picture_size, self.transform, self.size)

        if stage in (None, "test"):
            self.test = DiabeticRetinopathyDataset(self.path_to_data + "test/", "labels/trainLabels.csv",
                                                   self.picture_size, self.transform, self.size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=12, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size, num_workers=12)


class DiabeticRetinopathyModel(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 batch_size: int,
                 picture_size: [int, int],
                 weight_decay: float,
                 max_epochs: int,
                 shuffle_test: bool):
        super(DiabeticRetinopathyModel, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.picture_size = picture_size
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.shuffle_test = shuffle_test

        self.convnext_small = models.convnext_small(pretrained=True)

        self.accuracy_train = Accuracy(compute_on_step=False)
        self.accuracy_valid = Accuracy(compute_on_step=False)
        self.accuracy_test = Accuracy(compute_on_step=False)

    def forward(self, x):
        return self.convnext_small.forward(x)

    def training_step(self, batch, batch_nb):
        x, y = batch

        logits = self.convnext_small(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        # acc = accuracy(preds, y)

        self.log('train/val_loss', loss, prog_bar=True)
        # self.log('train/val_acc', acc, prog_bar=True)

        self.accuracy_train.update(preds, y)
        # self.log('train/val_acc', self.accuracy_train, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        avg = self.accuracy_train.compute()
        self.log('train/val_acc', avg, on_step=False, on_epoch=True, prog_bar=True)
        gc.collect()

    def validation_step(self, batch, batch_nb):
        x, y = batch

        logits = self.convnext_small(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        # acc = accuracy(preds, y)

        self.log('validation/val_loss', loss, prog_bar=True)
        # self.log('validation/val_acc', acc, prog_bar=True)

        self.accuracy_valid.update(preds, y)
        # self.log('validation/val_acc', self.accuracy_valid, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        avg = self.accuracy_valid.compute()
        self.log('validation/val_acc', avg, on_step=False, on_epoch=True, prog_bar=True)
        gc.collect()

    def test_step(self, batch, batch_nb):
        x, y = batch

        logits = self.convnext_small(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        # acc = accuracy(preds, y)

        self.log('test/val_loss', loss, prog_bar=True)
        # self.log('test/val_acc', acc, prog_bar=True)

        self.accuracy_test.update(preds, y)
        # self.log('test/val_acc', self.accuracy_test, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_epoch_end(self, outputs):
        avg = self.accuracy_test.compute()
        self.log('test/val_acc', avg, on_step=False, on_epoch=True, prog_bar=True)
        gc.collect()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


import gc

if __name__ == "__main__":
    gc.collect()
    pl.seed_everything(7)
    wandb_logger = WandbLogger(project='wandb-lightning-tune', job_type='train')

    config = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "picture_size": [256, 256],
        "weight_decay": 0,
        "max_epochs": 15,
        "shuffle_test": True
    }

    # Datamodule class
    # class DiabeticRetinopathyDataModule(pl.LightningDataModule):
    #     def __init__(self,
    #                  path_to_data: str,
    #                  batch_size: int = 32,
    #                  picture_size: int = 256,
    #                  size_of_data: int = None,
    #                  transform: transforms.Compose = None):
    DM = DiabeticRetinopathyDataModule("data/", config["batch_size"], config["picture_size"], config["shuffle_test"])

    model = DiabeticRetinopathyModel(config["learning_rate"],
                                     config["batch_size"],
                                     config["picture_size"],
                                     config["weight_decay"],
                                     config["max_epochs"],
                                     config["shuffle_test"])

    # model.load_state_dict(torch.load('E:/Dimploma/model/convnext_base.pth'))
    # model.eval()

    trainer = Trainer(max_epochs=config["max_epochs"],
                      min_epochs=1,
                      logger=wandb_logger,
                      )

    trainer.fit(model, DM)
    trainer.test(model=model, dataloaders=DM, ckpt_path="best")
    # trainer.test(model=model, dataloaders=DM)

    wandb.finish()
    save(model.state_dict(), 'E:/Dimploma/model/convnext_small(Best_params-15-without_wd-shaffle).pth')
