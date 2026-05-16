# -*- coding: utf-8 -*-
"""
Kaggle Digit Recognizer 比赛解决方案
基于深度学习的手写数字识别系统

比赛链接: https://www.kaggle.com/competitions/digit-recognizer
数据集: MNIST (28x28像素灰度图像)

核心功能:
1. 数据加载与预处理
2. 深度学习模型构建 (CNN)
3. 模型训练与验证
4. 预测生成与提交

作者: AI助手
日期: 2026-04-25
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

class DigitRecognizer:
    """手写数字识别器主类"""
    
    def __init__(self, data_dir='./data'):
        """
        初始化识别器
        
        参数:
            data_dir: 数据文件目录路径
        """
        self.data_dir = data_dir
        self.model = None
        self.history = None
        self.img_rows, self.img_cols = 28, 28
        self.num_classes = 10
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./submissions', exist_ok=True)
        
    def load_data(self, train_file='train.csv', test_file='test.csv'):
        """
        加载训练和测试数据
        
        参数:
            train_file: 训练数据文件名
            test_file: 测试数据文件名
            
        返回:
            X_train, y_train, X_test
        """
        print("[1/6] 正在加载数据...")
        
        # 加载训练数据
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练文件不存在: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试文件不存在: {test_path}")
        
        # 读取CSV文件
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        print(f"训练数据形状: {train_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        
        # 提取特征和标签
        y_train = train_data['label'].values
        X_train = train_data.drop('label', axis=1).values
        X_test = test_data.values
        
        # 数据预处理
        X_train = self.preprocess_images(X_train)
        X_test = self.preprocess_images(X_test)
        
        # 标签编码
        y_train = to_categorical(y_train, self.num_classes)
        
        print(f"处理后训练数据形状: {X_train.shape}")
        print(f"处理后测试数据形状: {X_test.shape}")
        print(f"标签形状: {y_train.shape}")
        
        return X_train, y_train, X_test
    
    def preprocess_images(self, images):
        """
        图像预处理
        
        参数:
            images: 原始图像数据 (n_samples, 784)
            
        返回:
            处理后的图像 (n_samples, 28, 28, 1)
        """
        # 归一化到 [0, 1]
        images = images.astype('float32') / 255.0
        
        # 重塑为图像格式
        images = images.reshape(-1, self.img_rows, self.img_cols, 1)
        
        return images
    
    def build_model(self, model_type='advanced_cnn'):
        """
        构建深度学习模型
        
        参数:
            model_type: 模型类型 ('simple_cnn', 'advanced_cnn')
            
        返回:
            编译好的模型
        """
        print(f"[2/6] 正在构建{model_type}模型...")
        
        if model_type == 'simple_cnn':
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_rows, self.img_cols, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
        else:  # advanced_cnn
            model = Sequential([
                # 第一个卷积块
                Conv2D(32, (3, 3), activation='relu', padding='same', 
                       input_shape=(self.img_rows, self.img_cols, 1)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # 第二个卷积块
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # 第三个卷积块
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Dropout(0.25),
                
                # 全连接层
                Flatten(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, validation_split=0.1, epochs=30, batch_size=128):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批次大小
        """
        print("[3/6] 正在训练模型...")
        
        # 回调函数
        callbacks = [
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, 
                             min_lr=0.00001, verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, 
                         restore_best_weights=True),
            ModelCheckpoint('./models/best_model.h5', monitor='val_accuracy', 
                          save_best_only=True, verbose=1)
        ]
        
        # 数据增强
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        
        # 训练模型
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("训练完成!")
    
    def evaluate_model(self, X_train, y_train):
        """
        评估模型性能
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
        """
        print("[4/6] 正在评估模型...")
        
        # 分割训练和验证集进行评估
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=np.argmax(y_train, axis=1)
        )
        
        # 训练集评估
        train_predictions = self.model.predict(X_train_split)
        train_accuracy = accuracy_score(np.argmax(y_train_split, axis=1), 
                                       np.argmax(train_predictions, axis=1))
        
        # 验证集评估
        val_predictions = self.model.predict(X_val_split)
        val_accuracy = accuracy_score(np.argmax(y_val_split, axis=1), 
                                     np.argmax(val_predictions, axis=1))
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"验证集准确率: {val_accuracy:.4f}")
        
        # 详细分类报告
        print("\n分类报告:")
        print(classification_report(np.argmax(y_val_split, axis=1), 
                                  np.argmax(val_predictions, axis=1)))
    
    def predict(self, X_test):
        """
        对测试集进行预测
        
        参数:
            X_test: 测试特征
            
        返回:
            预测结果
        """
        print("[5/6] 正在进行预测...")
        
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print(f"预测完成! 生成了 {len(predicted_classes)} 个预测结果")
        
        return predicted_classes
    
    def create_submission(self, predictions, submission_file='submission.csv'):
        """
        创建提交文件
        
        参数:
            predictions: 预测结果
            submission_file: 提交文件名
        """
        print("[6/6] 正在创建提交文件...")
        
        # 创建提交数据框
        submission_df = pd.DataFrame({
            'ImageId': range(1, len(predictions) + 1),
            'Label': predictions
        })
        
        # 保存到文件
        submission_path = os.path.join('./submissions', submission_file)
        submission_df.to_csv(submission_path, index=False)
        
        print(f"提交文件已保存: {submission_path}")
        print(f"文件前几行预览:")
        print(submission_df.head())
        
        return submission_path
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可显示")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 准确率曲线
        ax1.plot(self.history.history['accuracy'], label='训练准确率')
        ax1.plot(self.history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # 损失曲线
        ax2.plot(self.history.history['loss'], label='训练损失')
        ax2.plot(self.history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self, train_file='train.csv', test_file='test.csv', 
                            model_type='advanced_cnn', epochs=30):
        """
        运行完整的处理流程
        
        参数:
            train_file: 训练数据文件名
            test_file: 测试数据文件名
            model_type: 模型类型
            epochs: 训练轮数
        """
        try:
            # 1. 加载数据
            X_train, y_train, X_test = self.load_data(train_file, test_file)
            
            # 2. 构建模型
            self.build_model(model_type)
            
            # 3. 训练模型
            self.train_model(X_train, y_train, epochs=epochs)
            
            # 4. 评估模型
            self.evaluate_model(X_train, y_train)
            
            # 5. 预测
            predictions = self.predict(X_test)
            
            # 6. 创建提交文件
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            submission_file = f'submission_{timestamp}.csv'
            submission_path = self.create_submission(predictions, submission_file)
            
            # 7. 绘制训练历史
            self.plot_training_history()
            
            print(f"\n🎉 处理完成！")
            print(f"📁 提交文件: {submission_path}")
            print(f"💡 请将文件上传到Kaggle进行评分")
            
            return submission_path
            
        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
            return None

def main():
    """主函数"""
    print("=" * 60)
    print("Kaggle Digit Recognizer 比赛解决方案")
    print("=" * 60)
    
    # 创建识别器实例
    recognizer = DigitRecognizer(data_dir='./data')
    
    # 检查数据文件是否存在
    train_file = 'train.csv'
    test_file = 'test.csv'
    
    train_path = os.path.join('./data', train_file)
    test_path = os.path.join('./data', test_file)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("⚠️  数据文件未找到，请执行以下步骤:")
        print("1. 从Kaggle下载比赛数据: https://www.kaggle.com/competitions/digit-recognizer/data")
        print("2. 将train.csv和test.csv文件放置在 ./data/ 目录下")
        print("3. 重新运行此脚本")
        return
    
    # 运行完整流程
    submission_path = recognizer.run_complete_pipeline(
        train_file=train_file,
        test_file=test_file,
        model_type='advanced_cnn',  # 可以改为'simple_cnn'进行快速测试
        epochs=30
    )
    
    if submission_path:
        print("\n📋 下一步操作:")
        print("1. 访问 https://www.kaggle.com/competitions/digit-recognizer/submit")
        print("2. 点击'Submit Predictions'按钮")
        print(f"3. 上传文件: {submission_path}")
        print("4. 等待评分结果")

if __name__ == "__main__":
    main()