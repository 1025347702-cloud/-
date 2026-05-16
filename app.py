#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN手写数字识别 Web应用 - 简化版
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gradio as gr

# ============ CNN模型 ============
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ============ 加载模型 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')

try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"✅ 模型加载成功: {model_path}")
except Exception as e:
    print(f"❌ 模型加载失败: {str(e)}")

model.eval()


# ============ 图像预处理 ============
def preprocess_image(img):
    """预处理图像为MNIST格式"""
    print(f"  原始图像类型: {type(img)}")
    
    # 处理None情况
    if img is None:
        print("  图像为空")
        return None
    
    # 处理字典格式（Gradio Sketchpad返回的格式）
    if isinstance(img, dict):
        print(f"  字典键: {list(img.keys())}")
        # 优先使用composite（合成后的图像）
        if 'composite' in img:
            img = img['composite']
            print(f"  使用composite图像")
        elif 'layers' in img and isinstance(img['layers'], list):
            # 从layers提取图像
            print(f"  尝试从layers提取图像")
            for layer in img['layers']:
                if isinstance(layer, np.ndarray):
                    img = layer
                    break
                elif isinstance(layer, dict) and 'image' in layer:
                    img = layer['image']
                    break
    
    # 处理列表格式
    if isinstance(img, list):
        img = np.array(img, dtype=np.float32)
        print(f"  列表转数组后形状: {img.shape}")
    
    # 转换为numpy数组（PIL图像）
    if hasattr(img, 'convert'):
        img = np.array(img.convert('L'))
        print(f"  PIL转数组后形状: {img.shape}")
    
    # 确保是numpy数组
    if not isinstance(img, np.ndarray):
        print(f"  ❌ 不是numpy数组，类型: {type(img)}")
        return None
    
    print(f"  数组形状: {img.shape}")
    
    # 处理4D张量
    if len(img.shape) == 4:
        img = img.squeeze(0)
    
    # 处理彩色图像
    if len(img.shape) == 3:
        print(f"  处理彩色图像，通道数: {img.shape[2]}")
        if img.shape[2] == 4:
            # RGBA格式：需要处理透明背景
            # 取RGB通道并考虑Alpha混合
            rgb = img[..., :3]
            alpha = img[..., 3:4] / 255.0 if img.max() > 1.0 else img[..., 3:4]
            # 白色背景上的笔画
            img = (rgb * alpha + (1 - alpha) * 255).mean(axis=2)
        elif img.shape[2] == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        elif img.shape[2] == 1:
            img = img.squeeze(2)
    
    # 确保是2D图像
    if len(img.shape) == 1:
        print(f"  展开为28x28")
        img = img.reshape(28, 28)
    
    # 调整大小
    if img.shape[:2] != (28, 28):
        from PIL import Image as PILImage
        print(f"  调整大小: {img.shape[:2]} -> (28, 28)")
        img = np.array(PILImage.fromarray(img.astype(np.uint8)).resize((28, 28)))
    
    # 预处理
    img = img.astype(np.float32)
    
    # 检查像素范围并归一化
    if img.max() > 1.0:
        img = img / 255.0
    
    # 反转颜色（MNIST是白底黑字，手写板是黑底白字）
    img = 1.0 - img
    
    img = img.reshape(1, 1, 28, 28)
    
    print(f"  ✅ 最终形状: {img.shape}, 像素范围: [{img.min():.3f}, {img.max():.3f}]")
    
    return img


# ============ 预测函数 ============
def predict(img):
    """预测手写数字"""
    print(f"\n=== 收到图片 ===")
    
    if img is None:
        print("空图片")
        return "请上传或手写一个数字", None
    
    try:
        img_array = preprocess_image(img)
        if img_array is None:
            print("预处理失败")
            return "图像预处理失败", None
        
        print(f"图像形状: {img_array.shape}")
        print(f"像素范围: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        img_tensor = torch.FloatTensor(img_array).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted].item()
        
        print(f"预测结果: {predicted}, 置信度: {confidence:.2%}")
        
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        
        result_text = f"## 预测结果: **{predicted}**\n\n"
        result_text += f"**置信度: {confidence:.2%}**\n\n"
        result_text += "### Top-3 预测:\n"
        for i in range(3):
            result_text += f"- **{top3_indices[i].item()}**: {top3_probs[i].item():.2%}\n"
        
        probs_html = "<div style='margin-top:20px'><h3>概率分布</h3>"
        probs = probabilities[0].cpu().numpy()
        for i in range(10):
            bar_width = int(probs[i] * 200)
            color = '#4ecdc4' if i == predicted else '#e0e0e0'
            probs_html += f"""
            <div style='margin:5px 0; display:flex; align-items:center;'>
                <span style='width:30px; font-weight:bold'>{i}</span>
                <div style='width:200px; height:20px; background:#f0f0f0; border-radius:3px;'>
                    <div style='width:{bar_width}px; height:20px; background:{color}; border-radius:3px;'></div>
                </div>
                <span style='margin-left:10px'>{probs[i]:.1%}</span>
            </div>
            """
        probs_html += "</div>"
        
        return result_text, probs_html
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return f"预测出错: {str(e)}", None
    


# ============ Gradio界面 ============
with gr.Blocks(title="CNN手写数字识别") as app:
    gr.Markdown("# 🧠 CNN手写数字识别系统")
    
    with gr.Tab("📤 图片上传识别"):
        upload_input = gr.Image(label="上传图片", height=300)
        upload_result = gr.Markdown()
        upload_probs = gr.HTML()
        gr.Button("🔍 识别", variant="primary").click(
            fn=predict, inputs=upload_input, outputs=[upload_result, upload_probs]
        )
    
    with gr.Tab("✍️ 手写板识别"):
        sketch_input = gr.Sketchpad(label="手写板", height=300, width=300)
        sketch_result = gr.Markdown()
        sketch_probs = gr.HTML()
        with gr.Row():
            gr.Button("🔄 清除").click(
                fn=lambda: (None, "请书写数字", ""), 
                outputs=[sketch_input, sketch_result, sketch_probs]
            )
            gr.Button("🔍 识别", variant="primary").click(
                fn=predict, inputs=sketch_input, outputs=[sketch_result, sketch_probs]
            )


if __name__ == '__main__':
    print("启动Gradio Web应用...")
    print("访问地址: http://127.0.0.1:7860")
    app.launch(share=False)
    