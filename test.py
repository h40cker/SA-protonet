import os
import torch
import time
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # 新增：TensorBoard支持
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import csv
from datetime import datetime
from model.SAprotonet import  EnhancedProtoNet
from utils import argsfortest
# ====== 基础数据处理函数 ======
class MiniImageNet(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.classes = list(self.data["label"].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_n_way_k_shot_batch(dataset, n_way, k_shot, n_query):
    classes = np.random.choice(dataset.classes, n_way, replace=False)
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for idx, cls in enumerate(classes):
        samples = dataset.data[dataset.data["label"] == cls].sample(k_shot + n_query)
        images = [Image.open(os.path.join(dataset.img_root_dir, name)).convert("RGB") 
                  for name in samples.iloc[:, 0]]
        if dataset.transform:
            images = [dataset.transform(img) for img in images]

        support_images.extend(images[:k_shot])
        support_labels.extend([idx] * k_shot)
        query_images.extend(images[k_shot:])
        query_labels.extend([idx] * n_query)

    return (torch.stack(support_images), torch.tensor(support_labels),
            torch.stack(query_images), torch.tensor(query_labels))

# ====== 推理相关函数（需已有模型对象）======
def evaluate(model, dataset, n_way, k_shot, n_query, num_tasks=600):
    model.eval()
    accs = []

    with torch.no_grad():
        for _ in range(num_tasks):
            support_images, support_labels, query_images, query_labels = \
                get_n_way_k_shot_batch(dataset, n_way, k_shot, n_query)
            
            support_features = model(support_images)
            query_features = model(query_images)
            prototypes = model.compute_prototypes(support_features, support_labels)
            logits = model.classify(query_features, prototypes)

            acc = (logits.argmax(dim=1) == query_labels).float().mean().item()
            accs.append(acc)

    return np.mean(accs), np.std(accs)


# ====== 简化主程序（仅测试）======
def main():
    # 参数
    data_root = "./data/mini-imagenet/"
    model_path = "./saves/best_model.pth"  # 替换为你的路径
    n_way, k_shot, n_query = argsfortest.n_way, argsfortest.k_shot, argsfortest.n_query
    num_tasks = argsfortest.num_tasks

    # 变换
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据与模型
    test_dataset = MiniImageNet(os.path.join(data_root, "test.csv"),
                                os.path.join(data_root, "images"), transform)
    model = EnhancedProtoNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 测试开始
    start = time.time()
    mean_acc, std_acc = evaluate(model, test_dataset, n_way, k_shot, n_query, num_tasks)
    end = time.time()

    # 输出
    print(f"\nTest Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"Total time: {end - start:.2f}s | Avg per task: {(end - start)/num_tasks:.4f}s")

if __name__ == "__main__":
    main()
