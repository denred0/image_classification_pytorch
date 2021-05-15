import cv2
from torch.utils import data


class ICPDataset(data.Dataset):
    def __init__(self,
                 data,
                 input_resize,
                 augments=None,
                 preprocessing=None):
        super().__init__()
        self.imgs, self.labels = data
        self.input_resize = (input_resize, input_resize)
        self.augments = augments
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = str(self.imgs[idx])
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = cv2.resize(img, self.input_resize, interpolation=cv2.INTER_NEAREST)

        if self.augments:
            augmented = self.augments(image=img)
            img = augmented['image']

        if self.preprocessing:
            preprocessed = self.preprocessing(image=img)
            img = preprocessed['image']

        # img = rearrange(img, 'h w c -> c h w')

        return img, label
