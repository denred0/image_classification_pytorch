from project.src.model import BrickModel
from project.src.crop import get_bricket_coords

import os
from os import walk
import pickle
import shutil

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

# PyTorch - deep learning framework
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# cam1
#best_checkpoint = 'tb_logs/cam1/tf_efficientnet_b5_ns/version_2/checkpoints/tf_efficientnet_b5_ns_cam1_epoch=6_val_loss=0.132_val_acc=0.967_val_f1_epoch=0.967.ckpt'

# cam2
best_checkpoint = 'tb_logs/cam2/tf_efficientnet_b5_ns/version_4/checkpoints/tf_efficientnet_b5_ns_cam2_epoch=7_val_loss=0.226_val_acc=0.949_val_f1_epoch=0.949.ckpt'

# simpsons
#best_checkpoint = 'tb_logs/simp_data/tf_efficientnet_b5_ns/version_0/checkpoints/tf_efficientnet_b5_ns_simp_data_epoch=2_val_loss=0.013_val_acc=0.998_val_f1_epoch=0.998.ckpt'

best_model = BrickModel.load_from_checkpoint(checkpoint_path=best_checkpoint)
best_model = best_model.to("cuda")
best_model.eval()
best_model.freeze()

raw_images = False
root_directory = ''

camera_number = 'cam2'
# camera_number = 'simpsons'

if raw_images:
    root_directory = 'inference_files_raw/' + camera_number + '/'
else:
    root_directory = 'inference_files_crop/' + camera_number + '/'

IMG_SIZE = 456
threshold = 1
label_encoder = pickle.load(open("label_encoder/label_encoder.pkl", 'rb'))


class DatasetBrickets(Dataset):
    def __init__(self, path, image_ids):
        super().__init__()
        self.image_ids = image_ids
        self.path = path
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor()])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = str(self.image_ids[item])

        image = cv2.imread(self.path + image_id, cv2.IMREAD_COLOR)

        if raw_images:
            point_min, point_max = get_bricket_coords(image, 'cam1', corr_h=0, corr_w=0)
            image = image[point_min[0]:point_max[0], point_min[1]:point_max[1]]

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        image = self.transform(image=image)
        return image, image_id


all_files = 0
false_files = 0
for subdir, dirs, files in os.walk(root_directory):
    for folder in dirs:
        p = root_directory + folder + '/'
        _, _, images_list = next(walk(p))

        test_dataset = DatasetBrickets(path=root_directory + folder + '/', image_ids=images_list)
        test_loader = DataLoader(test_dataset, batch_size=1)

        if folder in label_encoder.classes_.tolist():
            gt = label_encoder.classes_.tolist().index(folder)
        else:
            gt = 999

        for data in test_loader:

            all_files += 1

            file = data[1][0]

            # list of probabilities
            y_hat = best_model(data[0].get('image').to("cuda"))

            # 2 top high probabilities
            n_top = torch.topk(y_hat, 2).values
            n_top_ind = torch.topk(y_hat, 2).indices.cpu().detach().numpy()

            class1 = label_encoder.classes_[n_top_ind[0][0]]
            class2 = label_encoder.classes_[n_top_ind[0][1]]

            min_n_top = torch.min(n_top).cpu().detach().numpy().item(0)
            max_n_top = torch.max(n_top).cpu().detach().numpy().item(0)

            if min_n_top < 0:
                difference = max_n_top + min_n_top * (-1)
            else:
                difference = max_n_top - min_n_top

            if difference != 0:
                confidence = difference / max_n_top
            else:
                confidence = 0

            y_hat = torch.argmax(y_hat, dim=1)

            y_hat = y_hat.cpu().detach().numpy()[0]
            if y_hat == gt:
                new_folder = folder + '_gt____' + folder
                path = root_directory + new_folder
                Path(path).mkdir(parents=True, exist_ok=True)

                sourcepath = root_directory + folder
                destinationpath = root_directory + new_folder

                if confidence >= threshold:
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file))
                else:
                    file_not_conf = 'not_conf___' + class1 + '__' + class2 + '___' + file
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file_not_conf))
            else:
                false_files += 1

                new_folder = folder + '_gt____' + label_encoder.classes_[y_hat]
                path = root_directory + new_folder
                Path(path).mkdir(parents=True, exist_ok=True)

                sourcepath = root_directory + folder
                destinationpath = root_directory + new_folder
                if confidence >= threshold:
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file))
                else:
                    file_not_conf = 'not_conf___' + class1 + '__' + class2 + '___' + file
                    shutil.copy(os.path.join(sourcepath, file), os.path.join(destinationpath, file_not_conf))

            if all_files % 100 == 0:
                print('Processed ' + str(all_files) + ' images...')

false_percent = false_files / all_files

print('Total images:', all_files)
print('False images:', false_files)
print("False Rate: " + "{:.4f}".format(false_percent))