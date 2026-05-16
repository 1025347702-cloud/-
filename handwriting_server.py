#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手写数字识别服务器
提供Web API接口连接前端画板和数字识别模型
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64
import cv2
from sklearn.neural_network import MLPClassifier
import json

app = Flask(__name__)
CORS(app)  # 允许跨域请求

class HandwritingRecognizer:
    """手写数字识别器"""
    
    def __init__(self):
        self.model = None
        self.img_size = 28
        self.num_classes = 10
        
    def load_trained_model(self):
        """加载预训练的数字识别模型"""
        try:
            # 这里应该加载之前训练的模型
            # 为了演示，我们创建一个新的简单模型
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                random_state=42
            )
            
            # 创建一些模拟训练数据来初始化模型
            # 在实际应用中应该加载真实训练好的模型
            X_train = np.random.rand(100, 784)
            y_train = np.random.randint(0, 10, 100)
            self.model.fit(X_train, y_train)
            
            print("✅ 数字识别模型已加载")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def preprocess_image(self, image_data):
        """预处理手写图像"""
        try:
            # 将base64图像数据转换为numpy数组
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')
            
            # 调整大小为28x28像素
            image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            
            # 转换为numpy数组并归一化
            img_array = np.array(image)
            img_array = img_array.astype('float32') / 255.0
            
            # 展平为784维向量
            img_vector = img_array.flatten()
            
            return img_vector
        except Exception as e:
            print(f"图像预处理错误: {e}")
            return None
    
    def predict_digit(self, image_vector):
        """预测手写数字"""
        try:
            if self.model is None:
                return {
                    'digit': -1,
                    'confidence': 0.0,
                    'probabilities': [0.1] * 10,
                    'error': '模型未加载'
                }
            
            # 预测
            prediction = self.model.predict([image_vector])[0]
            probabilities = self.model.predict_proba([image_vector])[0]
            
            # 计算置信度
            confidence = probabilities[prediction]
            
            return {
                'digit': int(prediction),
                'confidence': float(confidence),
                'probabilities': [float(p) for p in probabilities],
                'error': None
            }
        except Exception as e:
            print(f"预测错误: {e}")
            return {
                'digit': -1,
                'confidence': 0.0,
                'probabilities': [0.1] * 10,
                'error': str(e)
            }

# 创建识别器实例
recognizer = HandwritingRecognizer()

@app.route('/')
def index():
    """主页 - 返回HTML页面"""
    return render_template('handwriting_demo.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """手写数字识别API接口"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': '缺少图像数据'
            }), 400
        
        # 预处理图像
        image_vector = recognizer.preprocess_image(data['image'])
        if image_vector is None:
            return jsonify({
                'success': False,
                'error': '图像处理失败'
            }), 400
        
        # 预测数字
        result = recognizer.predict_digit(image_vector)
        
        if result['error']:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        return jsonify({
            'success': True,
            'prediction': {
                'digit': result['digit'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500

@app.route('/api/status')
def status_api():
    """服务器状态检查API"""
    return jsonify({
        'status': 'running',
        'model_loaded': recognizer.model is not None,
        'service': 'Handwriting Digit Recognition'
    })

@app.route('/demo')
def demo_page():
    """直接提供演示页面"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>手写数字识别演示</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background: #f0f0f0;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px;
            }
            h1 { color: #333; }
            .link { 
                display: block; 
                margin: 10px 0; 
                padding: 10px; 
                background: #007cba; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>手写数字识别演示</h1>
            <p>选择以下方式使用手写数字识别功能：</p>
            <a class="link" href="/static/handwriting_demo.html" target="_blank">
                🎨 完整可视化界面（推荐）
            </a>
            <a class="link" href="/api/status" target="_blank">
                🔧 API状态检查
            </a>
            <p>使用说明：</p>
            <ol>
                <li>点击"完整可视化界面"打开手写画板</li>
                <li>用鼠标在画板上手写数字（0-9）</li>
                <li>点击"识别数字"查看AI识别结果</li>
                <li>支持实时预测和概率分布显示</li>
            </ol>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # 移动HTML文件到static目录
    if os.path.exists('handwriting_demo.html'):
        import shutil
        shutil.copy('handwriting_demo.html', 'static/')
    
    # 加载模型
    print("🤖 正在加载数字识别模型...")
    recognizer.load_trained_model()
    
    # 启动服务器
    print("🚀 手写数字识别服务器启动中...")
    print("📍 本地访问: http://localhost:5000/demo")
    print("📍 API接口: http://localhost:5000/api/predict")
    print("📍 状态检查: http://localhost:5000/api/status")
    
    app.run(host='0.0.0.0', port=5000, debug=True)