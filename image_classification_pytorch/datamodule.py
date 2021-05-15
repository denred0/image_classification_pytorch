import numpy as np
import pickle

# pytorch related imports
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from image_classification_pytorch.dataset import ICPDataset

import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path


class ICPDataModule(pl.LightningDataModule):
    def __init__(self, model_type,
                 batch_size,
                 data_dir,
                 input_resize,
                 input_resize_test,
                 mean,
                 std,
                 use_normalize=True,
                 half_normalize=False,
                 augment_p=0.7,
                 images_ext='jpg'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_resize = input_resize
        self.input_resize_test = input_resize_test
        self.mean = mean,
        self.std = std,
        self.augment_p = augment_p
        self.images_ext = images_ext

        transforms_composed = self._get_transforms()
        self.augments, self.preprocessing = transforms_composed

        self.dataset_train, self.dataset_val, self.dataset_test = None, None, None

    def _get_transforms(self):
        transforms = []

        if self.mean is not None:
            transforms += [A.Normalize(mean=self.mean, std=self.std)]

        transforms += [ToTensorV2(transpose_mask=True)]
        preprocessing = A.Compose(transforms)

        return self._get_train_transforms(self.augment_p), preprocessing

    def _get_train_transforms(self, p):
        return A.Compose([
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.OneOf([A.IAAAdditiveGaussianNoise(),
                     A.GaussNoise()], p=0.4),
            A.OneOf([A.MotionBlur(p=0.1),
                     A.MedianBlur(blur_limit=3, p=0.1),
                     A.Blur(blur_limit=3, p=0.1)], p=0.2),
            A.OneOf([A.CLAHE(clip_limit=2),
                     A.IAASharpen(),
                     A.IAAEmboss(),
                     A.RandomBrightnessContrast()], p=0.5),
        ], p=p)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders

        path = Path(self.data_dir)

        train_val_files = list(path.rglob('*.' + self.images_ext))
        train_val_labels = [path.parent.name for path in train_val_files]

        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(train_val_labels)
        self.num_classes = len(np.unique(encoded))

        # save labels dict to file
        with open('label_encoder/label_encoder.pkl', 'wb') as le_dump_file:
            pickle.dump(label_encoder, le_dump_file)

        train_files, val_test_files = train_test_split(train_val_files, test_size=0.3, stratify=train_val_labels)

        train_labels = [path.parent.name for path in train_files]
        train_labels = label_encoder.transform(train_labels)
        train_data = train_files, train_labels

        # without test step
        # val_labels = [path.parent.name for path in val_test_files]
        # val_labels = label_encoder.transform(val_labels)
        # val_data = val_test_files, val_labels

        # with test step
        val_test_labels = [path.parent.name for path in val_test_files]
        val_files, test_files = train_test_split(val_test_files, test_size=0.5, stratify=val_test_labels)

        val_labels = [path.parent.name for path in val_files]
        val_labels = label_encoder.transform(val_labels)

        test_labels = [path.parent.name for path in test_files]
        test_labels = label_encoder.transform(test_labels)

        val_data = val_files, val_labels
        test_data = test_files, test_labels

        if stage == 'fit' or stage is None:
            self.dataset_train = ICPDataset(
                data=train_data,
                input_resize=self.input_resize,
                augments=self.augments,
                preprocessing=self.preprocessing)

            self.dataset_val = ICPDataset(
                data=val_data,
                input_resize=self.input_resize,
                preprocessing=self.preprocessing)

            self.dims = tuple(self.dataset_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = ICPDataset(
                data=test_data,
                input_resize=self.input_resize_test,
                preprocessing=self.preprocessing)

            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
