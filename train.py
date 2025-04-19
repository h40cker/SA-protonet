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
from datasets.miniimagenet import MiniImageNet
from model.SAprotonet import  EnhancedProtoNet
from utils import argsfortrain
# -------------------- 0. 日志与结果保存配置 --------------------
def setup_logging():
    """创建日志和结果保存目录"""
    log_dir = f"logs/protonet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 初始化CSV日志文件
    csv_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_path, mode='w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr"])
    
    return writer, csv_path



def get_n_way_k_shot_batch(dataset, n_way, k_shot, n_query):
    """生成n-way k-shot任务"""
    classes = np.random.choice(dataset.classes, n_way, replace=False)
    support_images, support_labels = [], []
    query_images, query_labels = [], []
    
    for idx, cls in enumerate(classes):
        cls_samples = dataset.data[dataset.data["label"] == cls].sample(k_shot + n_query)
        images = [Image.open(os.path.join(dataset.img_root_dir, name)).convert("RGB") 
                  for name in cls_samples.iloc[:, 0]]
        if dataset.transform:
            images = [dataset.transform(img) for img in images]
        
        support_images.extend(images[:k_shot])
        support_labels.extend([idx] * k_shot)
        query_images.extend(images[k_shot:])
        query_labels.extend([idx] * n_query)
    
    return (torch.stack(support_images), torch.tensor(support_labels),
            torch.stack(query_images), torch.tensor(query_labels))


# -------------------- 3. 训练与验证模块 --------------------
def train_epoch(model, train_dataset, optimizer, n_way, k_shot, n_query, epoch):
    model.train()
    total_loss, total_acc = 0, 0
    
    for _ in tqdm(range(100), desc=f"Epoch {epoch+1}"):
        support_images, support_labels, query_images, query_labels = \
            get_n_way_k_shot_batch(train_dataset, n_way, k_shot, n_query)
        
        support_features = model(support_images)
        query_features = model(query_images)
        prototypes = model.compute_prototypes(support_features, support_labels)
        logits = model.classify(query_features, prototypes)
        
        loss = F.cross_entropy(logits, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (logits.argmax(dim=1) == query_labels).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / 100, total_acc / 100

def evaluate(model, dataset, n_way, k_shot, n_query, num_tasks=200):
    model.eval()
    total_acc = 0
    
    with torch.no_grad():
        for _ in tqdm(range(num_tasks), desc="Evaluating"):
            support_images, support_labels, query_images, query_labels = \
                get_n_way_k_shot_batch(dataset, n_way, k_shot, n_query)
            
            support_features = model(support_images)
            query_features = model(query_images)
            prototypes = model.compute_prototypes(support_features, support_labels)
            logits = model.classify(query_features, prototypes)
            
            acc = (logits.argmax(dim=1) == query_labels).float().mean()
            total_acc += acc.item()
    
    return total_acc / num_tasks



# -------------------- 4. 主程序（新增日志保存） --------------------
def main():
    # 初始化日志
    writer, csv_path = setup_logging()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    data_root = "./data/mini-imagenet/"
    train_dataset = MiniImageNet(os.path.join(data_root, "train.csv"), 
                                os.path.join(data_root, "images/"), transform)
    val_dataset = MiniImageNet(os.path.join(data_root, "val.csv"), 
                              os.path.join(data_root, "images/"), transform)
    test_dataset = MiniImageNet(os.path.join(data_root, "test.csv"), 
                               os.path.join(data_root, "images/"), transform)
    
    # 初始化模型
    model = EnhancedProtoNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 训练参数
    n_way, k_shot, n_query = argsfortrain.n_way, argsfortrain.k_shot, argsfortrain.n_query
    epochs = argsfortrain.epochs
    best_val_acc = 0
    
    # 训练循环
    for epoch in range(epochs):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_dataset, optimizer, n_way, k_shot, n_query, epoch
        )
        
        # 验证集评估
        val_acc = evaluate(model, val_dataset, n_way, k_shot, n_query)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(writer.log_dir, "best_model.pth"))
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 写入TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning Rate", current_lr, epoch)
        
        # 写入CSV
        with open(csv_path, mode='a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, train_loss, train_acc, val_acc, current_lr])
        
        # 打印日志
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2%}, "
              f"Val Acc = {val_acc:.2%}, LR = {current_lr:.6f}")
        
        scheduler.step()
    
    # 最终测试
    model.load_state_dict(torch.load(os.path.join(writer.log_dir, "best_model.pth")))
    test_acc = evaluate(model, test_dataset, n_way, k_shot, n_query, num_tasks=600)
    
    # 保存最终结果
    with open(os.path.join(writer.log_dir, "final_results.txt"), 'w') as f:
        f.write(f"Final Test Accuracy (5-way {k_shot}-shot): {test_acc:.2%}\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.2%}\n")
    
    print(f"\n=== Final Results ===")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Best Val Accuracy: {best_val_acc:.2%}")
    print(f"Logs saved to: {writer.log_dir}")
    
    writer.close()  # 关闭TensorBoard

if __name__ == "__main__":
    main()