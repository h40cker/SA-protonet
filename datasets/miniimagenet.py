import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader



class MiniImageNet(Dataset):
    def __init__(self, csv_path, img_root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.classes = list(self.data["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.class_to_idx[self.data.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label