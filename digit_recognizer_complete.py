# -*- coding: utf-8 -*-
"""
完整的Kaggle Digit Recognizer解决方案
使用简单神经网络实现数字识别功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SimpleDigitRecognizer:
    """简化的数字识别器类"""
    
    def __init__(self, data_dir='./data'):
        """初始化识别器"""
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.img_size = 28
        self.num_classes = 10
        
        # 创建必要的目录
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./submissions', exist_ok=True)
        os.makedirs('./plots', exist_ok=True)
    
    def load_data(self, train_file='sample_train.csv', test_file='sample_test.csv'):
        """加载数据"""
        print("📥 [1/7] 加载数据...")
        
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            # 创建示例数据
            self._create_sample_data()
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"   训练数据: {train_df.shape}")
        print(f"   测试数据: {test_df.shape}")
        
        return train_df, test_df
    
    def _create_sample_data(self):
        """创建示例数据"""
        print("   创建示例数据用于演示...")
        
        # 创建更真实的示例数据（模拟MNIST格式）
        n_train = 1000
        n_test = 200
        
        # 训练数据
        train_data = []
        for i in range(n_train):
            label = i % 10
            # 简化版784像素数据
            pixels = np.random.randint(0, 256, 784).astype(np.float32)
            train_data.append([label] + list(pixels))
        
        columns = ['label'] + [f'pixel{i}' for i in range(784)]
        train_df = pd.DataFrame(train_data, columns=columns)
        train_df.to_csv('./data/sample_train.csv', index=False)
        
        # 测试数据
        test_data = []
        for i in range(n_test):
            pixels = np.random.randint(0, 256, 784).astype(np.float32)
            test_data.append(list(pixels))
        
        test_df = pd.DataFrame(test_data, columns=[f'pixel{i}' for i in range(784)])
        test_df.to_csv('./data/sample_test.csv', index=False)
        
        print(f"   已创建示例数据: {n_train}训练样本, {n_test}测试样本")
    
    def preprocess_data(self, train_df, test_df):
        """数据预处理"""
        print("🔧 [2/7] 数据预处理...")
        
        # 提取特征和标签
        y_train = train_df['label'].values
        X_train = train_df.drop('label', axis=1).values
        X_test = test_df.values
        
        # 归一化到0-1范围
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        print(f"   训练特征: {X_train.shape}")
        print(f"   测试特征: {X_test.shape}")
        
        return X_train, y_train, X_test
    
    def build_model(self):
        """构建神经网络模型"""
        print("🏗️ [3/7] 构建神经网络模型...")
        
        # 使用多层感知器作为简化版的神经网络
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 三层隐藏层
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=100,
            random_state=42,
            verbose=True
        )
        
        print("   模型架构: 784输入 → 128 → 64 → 32 → 10输出")
        return self.model
    
    def train_model(self, X_train, y_train):
        """训练模型"""
        print("🎯 [4/7] 训练模型...")
        
        # 划分训练集和验证集
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"   训练集: {X_tr.shape}, 验证集: {X_val.shape}")
        
        # 训练模型
        self.model.fit(X_tr, y_tr)
        
        # 验证集评估
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"   ✅ 验证集准确率: {val_accuracy:.3f}")
        
        return val_accuracy
    
    def evaluate_model(self, X_train, y_train):
        """完整模型评估"""
        print("📊 [5/7] 模型评估...")
        
        # 预测训练集
        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        print(f"   训练集准确率: {accuracy:.3f}")
        print("\n   📋 分类报告:")
        print(classification_report(y_train, y_pred))
        
        return accuracy
    
    def plot_results(self, X_train, y_train, accuracy):
        """可视化结果"""
        print("📈 [6/7] 生成可视化图表...")
        
        # 预测结果
        y_pred = self.model.predict(X_train)
        
        # 1. 混淆矩阵
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_train, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'混淆矩阵 (准确率: {accuracy:.3f})')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 2. 各类准确率
        plt.subplot(1, 2, 2)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        plt.bar(range(10), class_accuracy)
        plt.title('各类别准确率')
        plt.xlabel('数字类别')
        plt.ylabel('准确率')
        plt.xticks(range(10))
        
        plt.tight_layout()
        plt.savefig('./plots/model_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   ✅ 可视化图表已保存: ./plots/model_performance.png")
    
    def create_submission(self, X_test):
        """创建提交文件"""
        print("📤 [7/7] 生成提交文件...")
        
        # 预测测试集
        predictions = self.model.predict(X_test)
        
        # 创建提交格式
        submission = pd.DataFrame({
            'ImageId': range(1, len(predictions) + 1),
            'Label': predictions
        })
        
        submission.to_csv('./submissions/digit_recognizer_submission.csv', index=False)
        
        print(f"   ✅ 预测完成: {len(predictions)}个测试样本")
        print("   ✅ 提交文件已保存: ./submissions/digit_recognizer_submission.csv")
        
        return submission
    
    def run_complete_pipeline(self):
        """运行完整的处理流程"""
        print("🎯 开始Kaggle Digit Recognizer完整流程")
        print("=" * 60)
        
        # 1. 加载数据
        train_df, test_df = self.load_data()
        
        # 2. 数据预处理
        X_train, y_train, X_test = self.preprocess_data(train_df, test_df)
        
        # 3. 构建模型
        self.build_model()
        
        # 4. 训练模型
        val_accuracy = self.train_model(X_train, y_train)
        
        # 5. 评估模型
        accuracy = self.evaluate_model(X_train, y_train)
        
        # 6. 可视化结果
        self.plot_results(X_train, y_train, accuracy)
        
        # 7. 生成提交文件
        submission = self.create_submission(X_test)
        
        print("\n" + "=" * 60)
        print("🎉 数字识别器比赛解决方案完成!")
        print(f"📊 最终准确率: {accuracy:.3f}")
        print(f"📁 提交文件: ./submissions/digit_recognizer_submission.csv")
        print(f"📈 可视化图: ./plots/model_performance.png")
        print("\n💡 将提交文件上传到Kaggle进行评分:")
        print("   https://www.kaggle.com/competitions/digit-recognizer/submit")
        
        return submission, accuracy

def main():
    """主函数"""
    # 创建识别器实例
    recognizer = SimpleDigitRecognizer()
    
    # 运行完整流程
    try:
        submission, accuracy = recognizer.run_complete_pipeline()
        
        # 显示提交文件的前几行
        print("\n📋 提交文件预览:")
        print(submission.head())
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    main()