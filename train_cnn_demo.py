# -*- coding: utf-8 -*-
"""
CNN数字识别模型训练演示脚本
使用示例数据进行模型训练演示
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

def main():
    print('=== CNN数字识别模型训练演示 ===')
    
    # 创建必要的目录
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./submissions', exist_ok=True)
    
    # 读取示例数据 
    print('[1/5] 加载示例数据...')
    train_df = pd.read_csv('./data/sample_train.csv')
    test_df = pd.read_csv('./data/sample_test.csv')
    print(f'训练数据: {train_df.shape}')
    print(f'测试数据: {test_df.shape}')
    
    # 数据预处理
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values
    X_test = test_df.values
    
    # 归一化
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print(f'预处理后: X_train={X_train.shape}, X_test={X_test.shape}')
    
    # 创建简单的CNN模型
    print('[2/5] 构建CNN模型...')
    model = Sequential([
        layers.Reshape((10, 1, 1), input_shape=(10,)),  # 简化数据形状
        layers.Conv2D(16, (3, 1), activation='relu'),
        layers.MaxPooling2D((2, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print('[3/5] 模型架构:')
    model.summary()
    
    # 训练模型 (仅演示几轮)
    print('[4/5] 开始训练 (演示用少量轮次)...')
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print('[5/5] 模型训练完成!')
    print(f'训练准确率: {history.history["accuracy"][-1]:.3f}')
    print(f'验证准确率: {history.history["val_accuracy"][-1]:.3f}')
    
    # 保存模型
    model.save('./models/sample_cnn_model.h5')
    print('模型已保存: ./models/sample_cnn_model.h5')
    
    # 生成预测和提交文件
    print('生成预测结果...')
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })
    
    submission.to_csv('./submissions/sample_submission.csv', index=False)
    print('提交文件已保存: ./submissions/sample_submission.csv')
    
    print('=' * 50)
    print('✅ CNN数字识别模型训练演示完成!')
    print(f'📊 预测结果: {len(predicted_labels)}个样本')
    print(f'📁 模型文件: ./models/sample_cnn_model.h5')
    print(f'📤 提交文件: ./submissions/sample_submission.csv')

if __name__ == "__main__":
    main()