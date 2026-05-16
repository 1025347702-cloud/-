#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN手写数字识别 - 模型训练与超参数对比实验
基于PyTorch实现，完全匹配实验模板要求

CNN结构: Input(1×28×28) → Conv1+ReLU+MaxPool → Conv2+ReLU+MaxPool → Flatten → FC → Output(10)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)

# 路径配置
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
MODELS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/models'
PLOTS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/plots'
SUBMISSIONS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/submissions'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# ============ 1. 加载数据 ============
def load_data():
    """加载MNIST CSV数据"""
    train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
    test_df = pd.read_csv(f'{DATA_DIR}/test.csv')
    
    X_train = train_df.iloc[:, 1:].values.astype(np.float32)
    y_train = train_df.iloc[:, 0].values.astype(np.int64)
    X_test = test_df.values.astype(np.float32)
    
    # 归一化到 [0,1] 并 reshape 为 (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28) / 255.0
    
    # 划分验证集 (20%)
    val_size = int(len(X_train) * 0.2)
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    return X_train, y_train, X_val, y_val, X_test


# ============ 2. CNN模型定义 ============
class CNN(nn.Module):
    """标准CNN模型，完全按照实验模板要求"""
    def __init__(self, dropout_rate=0.25):
        super(CNN, self).__init__()
        # Input(1×28×28) → Conv1 + ReLU + MaxPool → Conv2 + ReLU + MaxPool → Flatten → FC → Output(10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # → (32, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # → (64, 7, 7)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ============ 3. 训练函数 ============
def train_model(X_train, y_train, X_val, y_val, config, name="Exp"):
    """训练单个模型并返回结果"""
    batch_size = config['batch_size']
    lr = config['lr']
    epochs = config['epochs']
    use_early_stop = config.get('early_stop', False)
    use_augment = config.get('augment', False)
    
    # 数据增强
    if use_augment:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ])
        # 转换为张量后应用增强
        train_tensor_x = torch.FloatTensor(X_train)
        train_tensor_y = torch.LongTensor(y_train)
        # 数据增强：对每个样本随机旋转/平移
        augmented_x = []
        augmented_y = []
        for i in range(len(train_tensor_x)):
            img = train_tensor_x[i]
            # 将增强后的图像加入
            augmented_x.append(img)
            augmented_y.append(train_tensor_y[i])
            # 对部分样本进行增强
            if np.random.random() < 0.5:
                aug_img = transform(img)
                augmented_x.append(aug_img)
                augmented_y.append(train_tensor_y[i])
        
        train_tensor_x = torch.stack(augmented_x)
        train_tensor_y = torch.tensor(augmented_y)
    else:
        train_tensor_x = torch.FloatTensor(X_train)
        train_tensor_y = torch.LongTensor(y_train)
    
    val_tensor_x = torch.FloatTensor(X_val)
    val_tensor_y = torch.LongTensor(y_val)
    
    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_y),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor_x, val_tensor_y),
        batch_size=batch_size, shuffle=False
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 训练记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    best_model_state = None
    patience = 5
    no_improve = 0
    
    print(f"\n{'='*50}")
    print(f"{name}: optimizer={config['optimizer']}, lr={lr}, "
          f"batch_size={batch_size}, augment={use_augment}, early_stop={use_early_stop}")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = correct / total
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss_avg:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Early Stopping
        if use_early_stop and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 计算测试准确率（在验证集上）
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_acc = test_correct / test_total
    
    result = {
        'name': name,
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'min_loss': min(val_losses),
        'converge_epoch': next((i+1 for i, l in enumerate(val_losses) 
                                if l <= min(val_losses[:i+1]) * 1.05 and i > 3), epochs),
        'model': model,
        'best_state': best_model_state
    }
    
    print(f"  >> {name} 完成: Val Acc={best_val_acc:.4f}, Test Acc={test_acc:.4f}, "
          f"Min Loss={result['min_loss']:.4f}, Converge Epoch={result['converge_epoch']}")
    
    return result


# ============ 4. 对比实验 ============
def run_experiments(X_train, y_train, X_val, y_val):
    """运行4组必做对比实验 + 最终调优模型"""
    
    experiments = [
        {
            'name': 'Exp1',
            'config': {
                'optimizer': 'SGD', 'lr': 0.01, 'batch_size': 64,
                'epochs': 20, 'augment': False, 'early_stop': False
            }
        },
        {
            'name': 'Exp2',
            'config': {
                'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 64,
                'epochs': 20, 'augment': False, 'early_stop': False
            }
        },
        {
            'name': 'Exp3',
            'config': {
                'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 128,
                'epochs': 20, 'augment': False, 'early_stop': True
            }
        },
        {
            'name': 'Exp4',
            'config': {
                'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 64,
                'epochs': 20, 'augment': True, 'early_stop': True
            }
        },
    ]
    
    results = []
    for exp in experiments:
        result = train_model(X_train, y_train, X_val, y_val, 
                            exp['config'], name=exp['name'])
        results.append(result)
    
    return results


# ============ 5. 最终调优模型 ============
def train_final_model(X_train, y_train, X_val, y_val, X_test):
    """训练最终提交模型 — 使用最佳超参数组合"""
    print("\n" + "="*60)
    print("训练最终提交模型（最佳超参数组合）")
    print("="*60)
    
    config = {
        'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 64,
        'epochs': 30, 'augment': True, 'early_stop': True
    }
    
    # 使用所有训练数据（包括验证集）
    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    
    # 数据增强
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    ])
    
    train_tensor_x = torch.FloatTensor(X_all)
    train_tensor_y = torch.LongTensor(y_all)
    
    augmented_x, augmented_y = [], []
    for i in range(len(train_tensor_x)):
        img = train_tensor_x[i]
        augmented_x.append(img)
        augmented_y.append(train_tensor_y[i])
        if np.random.random() < 0.5:
            augmented_x.append(transform(img))
            augmented_y.append(train_tensor_y[i])
    
    train_tensor_x = torch.stack(augmented_x)
    train_tensor_y = torch.tensor(augmented_y)
    
    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_y),
        batch_size=config['batch_size'], shuffle=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_loss = float('inf')
    best_state = None
    patience = 5
    no_improve = 0
    
    print(f"训练配置: optimizer=Adam, lr=0.001, batch_size=64, "
          f"augment=True, early_stop=True, epochs=30")
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{config['epochs']} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    model.load_state_dict(best_state)
    
    # 预测测试集
    model.eval()
    test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(test_tensor)
        _, predictions = outputs.max(1)
    
    predictions = predictions.cpu().numpy()
    
    # 保存Kaggle提交文件
    submission = pd.DataFrame({
        'ImageId': np.arange(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv(f'{SUBMISSIONS_DIR}/final_submission.csv', index=False)
    print(f"\nKaggle提交文件已保存: {SUBMISSIONS_DIR}/final_submission.csv")
    print(f"训练准确率: {acc:.4f}")
    
    # 保存模型
    torch.save(best_state, f'{MODELS_DIR}/best_model.pth')
    print(f"模型已保存: {MODELS_DIR}/best_model.pth")
    
    # 保存模型信息
    model_info = {
        'architecture': 'CNN: Input(1×28×28) → Conv1+ReLU+MaxPool → Conv2+ReLU+MaxPool → Flatten → FC → Output(10)',
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs_trained': epoch + 1,
        'data_augmentation': True,
        'early_stopping': True,
        'train_accuracy': float(acc),
        'best_loss': float(best_loss),
        'dropout_rate': 0.3,
        'conv1_channels': 32,
        'conv2_channels': 64,
        'fc_hidden': 128
    }
    with open(f'{MODELS_DIR}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"模型信息已保存")
    
    return model, predictions


# ============ 6. 绘制Loss曲线 ============
def plot_loss_curves(results):
    """将4组对比实验的Loss曲线绘制在同一张图上"""
    plt.figure(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    markers = ['o', 's', '^', 'd']
    
    for i, result in enumerate(results):
        name = result['name']
        epochs = range(1, len(result['train_losses']) + 1)
        plt.plot(epochs, result['train_losses'], 
                color=colors[i], marker=markers[i], markersize=4,
                linestyle='--', label=f'{name} (Train)', alpha=0.7)
        plt.plot(epochs, result['val_losses'],
                color=colors[i], marker='', 
                linestyle='-', label=f'{name} (Val)', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('CNN Training and Validation Loss Curves - 4 Experiments Comparison', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/loss_curves.png', dpi=150)
    print(f"Loss曲线已保存: {PLOTS_DIR}/loss_curves.png")
    plt.close()
    
    # 也绘制准确率曲线
    plt.figure(figsize=(12, 6))
    for i, result in enumerate(results):
        name = result['name']
        epochs = range(1, len(result['train_accs']) + 1)
        plt.plot(epochs, result['train_accs'],
                color=colors[i], marker=markers[i], markersize=4,
                linestyle='--', label=f'{name} (Train)', alpha=0.7)
        plt.plot(epochs, result['val_accs'],
                color=colors[i], marker='',
                linestyle='-', label=f'{name} (Val)', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('CNN Training and Validation Accuracy Curves - 4 Experiments Comparison', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/accuracy_curves.png', dpi=150)
    print(f"准确率曲线已保存: {PLOTS_DIR}/accuracy_curves.png")
    plt.close()


# ============ 7. 打印实验结果表格 ============
def print_results_table(results):
    """打印对比实验结果表格"""
    print("\n" + "="*90)
    print("对比实验结果")
    print("="*90)
    print(f"{'实验编号':>8} | {'优化器':>8} | {'学习率':>8} | {'Batch Size':>10} | "
          f"{'数据增强':>8} | {'Early Stop':>10} | {'Train Acc':>9} | {'Val Acc':>9} | "
          f"{'Test Acc':>9} | {'最低Loss':>8} | {'收敛Epoch':>10}")
    print("-"*90)
    
    for result in results:
        c = result['config']
        print(f"{result['name']:>8} | {c['optimizer']:>8} | {c['lr']:>8} | "
              f"{c['batch_size']:>10} | {str(c['augment']):>8} | "
              f"{str(c['early_stop']):>10} | "
              f"{result['train_accs'][-1]:>9.4f} | {result['best_val_acc']:>9.4f} | "
              f"{result['test_acc']:>9.4f} | {result['min_loss']:>8.4f} | "
              f"{result['converge_epoch']:>10}")
    
    print("="*90)


# ============ 8. 主流程 ============
def main():
    print("="*60)
    print("CNN手写数字识别 — 模型训练与超参数对比实验")
    print("="*60)
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X_train, y_train, X_val, y_val, X_test = load_data()
    
    # 2. 运行对比实验
    print("\n[2/5] 运行4组对比实验...")
    results = run_experiments(X_train, y_train, X_val, y_val)
    
    # 3. 打印结果表格
    print_results_table(results)
    
    # 4. 绘制Loss曲线
    print("\n[3/5] 绘制Loss曲线和准确率曲线...")
    plot_loss_curves(results)
    
    # 5. 训练最终模型
    print("\n[4/5] 训练最终提交模型...")
    model, predictions = train_final_model(X_train, y_train, X_val, y_val, X_test)
    
    print("\n[5/5] 实验完成！")
    print(f"\n生成的文件:")
    print(f"  - {PLOTS_DIR}/loss_curves.png")
    print(f"  - {PLOTS_DIR}/accuracy_curves.png")
    print(f"  - {SUBMISSIONS_DIR}/final_submission.csv")
    print(f"  - {MODELS_DIR}/best_model.pth")
    print(f"  - {MODELS_DIR}/model_info.json")


if __name__ == '__main__':
    main()