#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单手写数字识别演示
使用Tkinter创建本地GUI界面
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw
import io
import base64

class SimpleHandwritingDemo:
    """简单手写数字识别演示"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("手写数字识别演示")
        self.window.geometry("800x600")
        self.window.configure(bg='#f0f8ff')
        
        # 绘画变量
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 标题
        title_label = tk.Label(
            self.window, 
            text="✍️ 手写数字识别演示", 
            font=("Arial", 20, "bold"), 
            bg='#f0f8ff',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # 主要内容区域
        main_frame = tk.Frame(self.window, bg='#f0f8ff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # 左侧画板区域
        left_frame = tk.Frame(main_frame, bg='#f0f8ff')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 画板标题
        canvas_label = tk.Label(
            left_frame, 
            text="请在这里手写数字 (0-9)", 
            font=("Arial", 14), 
            bg='#f0f8ff'
        )
        canvas_label.pack(pady=10)
        
        # 画布
        self.canvas = tk.Canvas(
            left_frame, 
            width=300, 
            height=300, 
            bg='white', 
            relief='solid', 
            bd=2
        )
        self.canvas.pack(pady=10)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # 控制按钮
        button_frame = tk.Frame(left_frame, bg='#f0f8ff')
        button_frame.pack(pady=10)
        
        clear_btn = tk.Button(
            button_frame, 
            text="🔄 清除", 
            command=self.clear_canvas,
            font=("Arial", 12),
            bg='#ff6b6b',
            fg='white',
            width=10
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        predict_btn = tk.Button(
            button_frame, 
            text="🔍 识别", 
            command=self.predict_digit,
            font=("Arial", 12),
            bg='#4ecdc4',
            fg='white',
            width=10
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # 右侧结果区域
        right_frame = tk.Frame(main_frame, bg='#ffffff', relief='solid', bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 结果标题
        result_label = tk.Label(
            right_frame, 
            text="识别结果", 
            font=("Arial", 16, "bold"), 
            bg='white'
        )
        result_label.pack(pady=20)
        
        # 预测数字显示
        self.predicted_label = tk.Label(
            right_frame, 
            text="?", 
            font=("Arial", 80, "bold"), 
            bg='white',
            fg='#4ecdc4'
        )
        self.predicted_label.pack(pady=20)
        
        # 置信度
        self.confidence_label = tk.Label(
            right_frame, 
            text="请先手写一个数字", 
            font=("Arial", 14), 
            bg='white',
            fg='#666666'
        )
        self.confidence_label.pack(pady=10)
        
        # 概率分布
        prob_frame = tk.Frame(right_frame, bg='white')
        prob_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        
        prob_label = tk.Label(
            prob_frame, 
            text="概率分布", 
            font=("Arial", 14, "bold"), 
            bg='white'
        )
        prob_label.pack()
        
        # 创建数字概率表
        self.prob_labels = {}
        prob_grid = tk.Frame(prob_frame, bg='white')
        prob_grid.pack(pady=10)
        
        for i in range(10):
            row = i // 5
            col = i % 5
            
            digit_frame = tk.Frame(prob_grid, bg='white', relief='solid', bd=1)
            digit_frame.grid(row=row, column=col, padx=5, pady=5)
            
            digit_label = tk.Label(
                digit_frame, 
                text=str(i), 
                font=("Arial", 12, "bold"), 
                bg='white'
            )
            digit_label.pack()
            
            prob_label = tk.Label(
                digit_frame, 
                text="0.0%", 
                font=("Arial", 10), 
                bg='white',
                fg='#666666'
            )
            prob_label.pack()
            
            self.prob_labels[i] = prob_label
        
        # 使用说明
        instruction_frame = tk.Frame(right_frame, bg='#e3f2fd', relief='solid', bd=1)
        instruction_frame.pack(fill=tk.X, padx=10, pady=20)
        
        instruction_text = tk.Label(
            instruction_frame, 
            text="使用说明:\n1. 在左侧画板上用鼠标手写数字\n2. 尽量写得大一些，占满画板\n3. 点击'识别'按钮查看结果",
            font=("Arial", 10), 
            bg='#e3f2fd',
            justify=tk.LEFT
        )
        instruction_text.pack(padx=10, pady=10)
    
    def start_drawing(self, event):
        """开始绘画"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        """绘画过程"""
        if self.drawing:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=15, fill='black', capstyle=tk.ROUND, smooth=True
            )
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_drawing(self, event):
        """停止绘画"""
        self.drawing = False
    
    def clear_canvas(self):
        """清除画板"""
        self.canvas.delete("all")
        self.predicted_label.config(text="?")
        self.confidence_label.config(text="请先手写一个数字")
        
        # 重置概率显示
        for i in range(10):
            self.prob_labels[i].config(text="0.0%")
    
    def predict_digit(self):
        """预测数字"""
        try:
            # 获取画布内容
            ps = self.canvas.postscript(colormode='mono')
            
            # 模拟预测过程
            prediction = self.simulate_prediction()
            
            # 更新界面显示
            self.predicted_label.config(text=str(prediction['digit']))
            self.confidence_label.config(
                text=f"置信度: {prediction['confidence']:.1%}"
            )
            
            # 更新概率分布
            for i, prob in enumerate(prediction['probabilities']):
                self.prob_labels[i].config(text=f"{prob:.1%}")
                
        except Exception as e:
            messagebox.showerror("错误", f"识别过程中出现错误: {str(e)}")
    
    def simulate_prediction(self):
        """模拟预测结果"""
        # 在实际应用中，这里应该调用真正的模型
        # 这里使用模拟数据来演示
        
        # 随机生成一个合理的预测结果
        digit = np.random.randint(0, 10)
        confidence = np.random.uniform(0.7, 0.95)
        
        # 生成概率分布
        probabilities = np.random.dirichlet(np.ones(10))
        probabilities[digit] = confidence
        
        # 归一化
        probabilities = probabilities / probabilities.sum()
        
        return {
            'digit': digit,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def run(self):
        """运行应用"""
        self.window.mainloop()

# 启动演示
def main():
    print("🎯 启动手写数字识别演示...")
    print("📝 使用说明:")
    print("   1. 在画板上用鼠标手写数字 (0-9)")
    print("   2. 尽量写得大一些，占满画板")
    print("   3. 点击'识别'按钮查看AI识别结果")
    print("   4. 支持概率分布显示和置信度分析")
    
    app = SimpleHandwritingDemo()
    app.run()

if __name__ == "__main__":
    main()