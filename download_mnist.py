
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST数据集下载脚本
从torchvision下载官方MNIST数据集并转换为Kaggle格式的CSV文件
"""

import os
import pandas as pd
import torch
from torchvision import datasets, transforms

def download_mnist_dataset():
    """下载MNIST数据集并转换为CSV格式"""
    print("=" * 60)
    print("📥 MNIST数据集下载工具")
    print("=" * 60)
    
    # 创建数据目录
    os.makedirs('./data', exist_ok=True)
    print("✅ 数据目录已就绪")
    
    # 下载训练集
    print("\n📥 正在下载训练集...")
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor()
    )
    print(f"✅ 训练集下载完成 ({len(train_dataset)} 个样本)")
    
    # 下载测试集
    print("\n📥 正在下载测试集...")
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.ToTensor()
    )
    print(f"✅ 测试集下载完成 ({len(test_dataset)} 个样本)")
    
    # 转换训练集为DataFrame
    print("\n🔄 正在转换训练集格式...")
    train_data = []
    for i in range(len(train_dataset)):
        if (i + 1) % 10000 == 0:
            print(f"   处理进度: {i + 1}/{len(train_dataset)}")
        img, label = train_dataset[i]
        img_flat = img.view(-1).numpy() * 255  # 转换为0-255范围
        row = [label] + img_flat.tolist()
        train_data.append(row)
    
    columns = ['label'] + [f'pixel{i}' for i in range(784)]
    train_df = pd.DataFrame(train_data, columns=columns)
    train_df.to_csv('./data/train.csv', index=False)
    print("✅ 训练集CSV文件已保存")
    
    # 转换测试集为DataFrame
    print("\n🔄 正在转换测试集格式...")
    test_data = []
    for i in range(len(test_dataset)):
        if (i + 1) % 5000 == 0:
            print(f"   处理进度: {i + 1}/{len(test_dataset)}")
        img, _ = test_dataset[i]
        img_flat = img.view(-1).numpy() * 255
        row = img_flat.tolist()
        test_data.append(row)
    
    test_columns = [f'pixel{i}' for i in range(784)]
    test_df = pd.DataFrame(test_data, columns=test_columns)
    test_df.to_csv('./data/test.csv', index=False)
    print("✅ 测试集CSV文件已保存")
    
    # 验证文件
    print("\n🔍 验证文件...")
    train_size = os.path.getsize('./data/train.csv') / (1024 * 1024)
    test_size = os.path.getsize('./data/test.csv') / (1024 * 1024)
    
    print(f"\n🎉 下载完成！")
    print("=" * 60)
    print(f"训练集: ./data/train.csv")
    print(f"  - 样本数: {len(train_dataset)}")
    print(f"  - 文件大小: {train_size:.1f} MB")
    print(f"测试集: ./data/test.csv")
    print(f"  - 样本数: {len(test_dataset)}")
    print(f"  - 文件大小: {test_size:.1f} MB")
    print("=" * 60)
    print("\n💡 下一步：运行 train.py 重新训练模型")

if __name__ == '__main__':
    download_mnist_dataset()
