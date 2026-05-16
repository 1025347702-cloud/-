# -*- coding: utf-8 -*-
"""
Kaggle Digit Recognizer 数据下载助手
自动化下载比赛数据集
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

def create_data_structure():
    """创建必要的数据目录结构"""
    directories = ['./data', './models', './submissions', './plots']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def download_kaggle_data_manual():
    """手动下载数据的指导说明"""
    print("\n📥 Kaggle Digit Recognizer 数据下载指南")
    print("=" * 60)
    
    print("\n📋 下载步骤:")
    print("1. 访问比赛页面: https://www.kaggle.com/competitions/digit-recognizer")
    print("2. 点击右侧的 'Data' 标签页")
    print("3. 找到并下载以下文件:")
    print("   - train.csv (训练数据)")
    print("   - test.csv (测试数据)")
    print("4. 将下载的文件放入 ./data/ 目录")
    
    print("\n💡 提示:")
    print("- 需要Kaggle账号才能下载")
    print("- 首次使用可能需要安装kaggle API")
    print("- 或者可以使用下面的备选方案")

def create_sample_data():
    """创建示例数据文件以便测试"""
    print("\n🔄 创建示例数据文件用于测试...")
    
    # 创建示例训练数据
    sample_train_data = {
        'label': [1, 2, 3, 4, 5],
        'pixel0': [0] * 5,
        'pixel1': [0] * 5,
        # 这里简化，实际应该有784个像素列
    }
    
    # 添加更多列名
    columns = ['label'] + [f'pixel{i}' for i in range(10)]  # 简化版，实际784个
    
    sample_df = pd.DataFrame({col: [0] * 5 for col in columns})
    sample_df['label'] = [1, 2, 3, 4, 5]
    
    sample_df.to_csv('./data/sample_train.csv', index=False)
    sample_df.drop('label', axis=1).to_csv('./data/sample_test.csv', index=False)
    
    print("✅ 示例数据文件已创建:")
    print("   - ./data/sample_train.csv")
    print("   - ./data/sample_test.csv")

def check_kaggle_api():
    """检查Kaggle API配置"""
    print("\n🔍 检查Kaggle API配置...")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle API配置文件存在")
        
        # 检查API安装
        try:
            import kaggle
            print("✅ Kaggle Python包已安装")
            return True
        except ImportError:
            print("❌ Kaggle Python包未安装")
            print("   运行: pip install kaggle")
            return False
    else:
        print("❌ Kaggle API配置文件不存在")
        print("\n📋 配置Kaggle API:")
        print("1. 访问 https://www.kaggle.com/account")
        print("2. 创建新的API Token")
        print("3. 下载kaggle.json文件")
        print("4. 将文件放入 ~/.kaggle/ 目录")
        print("5. 设置权限: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_with_kaggle_api():
    """使用Kaggle API下载数据"""
    if not check_kaggle_api():
        return False
    
    try:
        from kaggle import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("\n📥 正在下载数据...")
        api.competition_download_files('digit-recognizer', path='./data')
        
        # 解压文件
        zip_file = './data/digit-recognizer.zip'
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('./data')
            os.remove(zip_file)
            print("✅ 数据解压完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def verify_data_files():
    """验证数据文件是否存在"""
    required_files = ['train.csv', 'test.csv']
    data_dir = './data'
    
    print("\n🔍 验证数据文件...")
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} - 存在")
            # 检查文件大小
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   大小: {file_size:.1f} MB")
        else:
            print(f"❌ {file} - 缺失")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    """主函数"""
    print("Kaggle Digit Recognizer 数据下载助手")
    print("=" * 50)
    
    # 创建目录结构
    create_data_structure()
    
    # 检查是否已有数据文件
    if verify_data_files():
        print("\n🎉 数据文件已准备就绪！")
        print("💡 可以运行 digit_recognizer_solution.py 开始训练模型")
        return
    
    # 提供下载选项
    print("\n请选择下载方式:")
    print("1. 使用Kaggle API自动下载（推荐）")
    print("2. 手动下载指导")
    print("3. 创建示例数据用于测试")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == '1':
        if download_with_kaggle_api():
            verify_data_files()
    elif choice == '2':
        download_kaggle_data_manual()
    elif choice == '3':
        create_sample_data()
        print("\n💡 示例数据可用于测试代码结构，但效果有限")
        print("   建议下载完整数据集以获得最佳效果")
    else:
        print("❌ 无效选择")
    
    print("\n📋 后续步骤:")
    print("1. 确保 ./data/ 目录下有 train.csv 和 test.csv")
    print("2. 运行: python digit_recognizer_solution.py")
    print("3. 等待模型训练完成")
    print("4. 上传生成的提交文件到Kaggle")

if __name__ == "__main__":
    main()