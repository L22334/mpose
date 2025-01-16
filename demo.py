import cv2
import numpy as np
from ultralytics import YOLO
import openpose.pyopenpose as op
from collections import deque
import torch
import time
import asyncio
import torch.cuda
from concurrent.futures import ThreadPoolExecutor
import gc
from typing import Optional, Dict
import weakref
import numba
import mmap
import cProfile
import io
import pstats
from memory_profiler import profile
import psutil
import os
import sys

# 关键点和骨架定义
OPENPOSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],  # 手臂
    [1, 8], [8, 9], [9, 10],  # 右腿
    [1, 11], [11, 12], [12, 13],  # 左腿
    [1, 0],  # 躯干到鼻子
    [0, 14], [14, 16],  # 右眼和右耳
    [0, 15], [15, 17]   # 左眼和左耳
]

YOLO_CONNECTIONS = {
    0: [1, 2, 5, 6],  # 鼻子连接到左右眼、左右肩
    1: [3],           # 左眼连接到左耳
    2: [4],           # 右眼连接到右耳
    5: [7, 11],       # 左肩连接到左肘和左髋
    6: [8, 12],       # 右肩连接到右肘和右髋
    7: [9],           # 左肘连接到左腕
    8: [10],          # 右肘连接到右腕
    11: [13],         # 左髋连接到左膝
    12: [14],         # 右髋连接到右膝
    13: [15],         # 左膝连接到左踝
    14: [16]          # 右膝连接到右踝
}

KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye", 2: "right_eye",
    3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle"
}

# 视频路径配置
VIDEO_PATH = "video/output_video.mp4"  # 视频文件路径
USE_CAMERA = False  # 设置为 False 使用视频文件
CAMERA_ID = 0     # 不使用摄像头，此项可忽略

# 确保使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化YOLOv8
yolo_model = YOLO('yolov8m-pose.pt')
yolo_model.to(device)

# OpenPose 配置
params = dict()
params["model_folder"] = "openpose\models"
params["model_pose"] = "COCO"
params["net_resolution"] = "-1x368"  
params["number_people_max"] = 1      # 只检测一个人，提高性能
params["maximize_positives"] = True  # 提高检测灵敏度

# OpenPose初始化
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 添加骨骼对定义（成对的关键点定义骨骼）
SKELETON_PAIRS = {
    'upper_arm_right': (6, 8),   # 右肩到右肘
    'lower_arm_right': (8, 10),  # 右肘到右腕
    'upper_arm_left': (5, 7),    # 左肩到左肘
    'lower_arm_left': (7, 9),    # 左肘到左腕
    'upper_leg_right': (12, 14), # 右髋到右膝
    'lower_leg_right': (14, 16), # 右膝到右踝
    'upper_leg_left': (11, 13),  # 左髋到左膝
    'lower_leg_left': (13, 15),  # 左膝到左踝
    'shoulder_width': (5, 6),    # 肩宽
    'hip_width': (11, 12),       # 髋宽
}

class SkeletonConsistencyChecker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.bone_lengths = {}
        self.bone_length_history = {bone: deque(maxlen=window_size) for bone in SKELETON_PAIRS}
        self.reference_ratios = {
            'arm_symmetry': 1.0,      # Arm length ratio
            'leg_symmetry': 1.0,      # Leg length ratio
            'arm_leg_ratio': 0.8,     # Arm to leg ratio
            'shoulder_hip_ratio': 1.2  # Shoulder to hip width ratio
        }
        self.tolerance = 0.2  # Allowed error range (20%)

    def calculate_bone_length(self, keypoints, start_idx, end_idx):
        """计算两个关键点之间的距离"""
        if (keypoints[start_idx][2] > 0.5 and 
            keypoints[end_idx][2] > 0.5):  # 检查置信度
            start = keypoints[start_idx][:2]
            end = keypoints[end_idx][:2]
            return np.sqrt(np.sum((start - end) ** 2))
        return None

    def update_bone_lengths(self, keypoints):
        """更新所有骨骼长度的历史记录"""
        current_lengths = {}
        
        for bone_name, (start_idx, end_idx) in SKELETON_PAIRS.items():
            length = self.calculate_bone_length(keypoints, start_idx, end_idx)
            if length is not None:
                current_lengths[bone_name] = length
                self.bone_length_history[bone_name].append(length)
        
        return current_lengths

    def get_stable_bone_lengths(self):
        """获取稳定的骨骼长度（使用中位数）"""
        stable_lengths = {}
        for bone_name in SKELETON_PAIRS:
            if len(self.bone_length_history[bone_name]) > 0:
                stable_lengths[bone_name] = np.median(self.bone_length_history[bone_name])
        return stable_lengths

    def check_symmetry(self, current_lengths):
        """Check skeleton symmetry"""
        issues = []
        
        # 检查手臂对称性
        if ('upper_arm_left' in current_lengths and 
            'upper_arm_right' in current_lengths):
            ratio = (current_lengths['upper_arm_left'] / 
                    current_lengths['upper_arm_right'])
            if abs(ratio - self.reference_ratios['arm_symmetry']) > self.tolerance:
                issues.append(f"Arm Asymmetry: {ratio:.2f}")

        # 检查腿部对称性
        if ('upper_leg_left' in current_lengths and 
            'upper_leg_right' in current_lengths):
            ratio = (current_lengths['upper_leg_left'] / 
                    current_lengths['upper_leg_right'])
            if abs(ratio - self.reference_ratios['leg_symmetry']) > self.tolerance:
                issues.append(f"Leg Asymmetry: {ratio:.2f}")

        return issues

    def check_proportions(self, current_lengths):
        """Check body proportions"""
        issues = []
        
        # 计算手臂总长
        left_arm = (current_lengths.get('upper_arm_left', 0) + 
                   current_lengths.get('lower_arm_left', 0))
        right_arm = (current_lengths.get('upper_arm_right', 0) + 
                    current_lengths.get('lower_arm_right', 0))
        
        # 计算腿部总长
        left_leg = (current_lengths.get('upper_leg_left', 0) + 
                   current_lengths.get('lower_leg_left', 0))
        right_leg = (current_lengths.get('upper_leg_right', 0) + 
                    current_lengths.get('lower_leg_right', 0))

        # 检查手臂与腿的比例
        if left_arm > 0 and left_leg > 0:
            ratio = left_arm / left_leg
            if abs(ratio - self.reference_ratios['arm_leg_ratio']) > self.tolerance:
                issues.append(f"Abnormal Arm-Leg Ratio: {ratio:.2f}")

        # 检查肩宽与髋宽的比例
        if ('shoulder_width' in current_lengths and 
            'hip_width' in current_lengths):
            ratio = (current_lengths['shoulder_width'] / 
                    current_lengths['hip_width'])
            if abs(ratio - self.reference_ratios['shoulder_hip_ratio']) > self.tolerance:
                issues.append(f"Abnormal Shoulder-Hip Ratio: {ratio:.2f}")

        return issues

    def check_consistency(self, keypoints):
        """Main checking function, returns all consistency issues"""
        if keypoints is None:
            return []

        current_lengths = self.update_bone_lengths(keypoints)
        if not current_lengths:
            return []

        stable_lengths = self.get_stable_bone_lengths()

        issues = []
        issues.extend(self.check_symmetry(current_lengths))
        issues.extend(self.check_proportions(current_lengths))

        # 检查骨骼长度的突变
        for bone_name, current_length in current_lengths.items():
            if bone_name in stable_lengths:
                stable_length = stable_lengths[bone_name]
                if abs(current_length - stable_length) / stable_length > self.tolerance:
                    issues.append(f"Abnormal {bone_name} length change")

        return issues

class KeypointSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []
        
    def update(self, keypoints):
        """使用滑动窗口平均进行平滑处理"""
        if keypoints is None:
            return None
            
        self.history.append(keypoints)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # 计算加权平均，最近的帧权重更大
        weights = np.linspace(0.5, 1.0, len(self.history))
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(keypoints)
        for i, kpts in enumerate(self.history):
            smoothed += kpts * weights[i]
            
        return smoothed

def calculate_angle(p1, p2, p3=None):
    """
    计算角度
    如果提供两个点，计算与水平线的夹角
    如果提供三个点，计算三点形成的角度
    """
    if p3 is None:
        # 计算与水平线的夹角
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        return angle
    else:
        # 计算三点角度
        a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle

def calculate_confidence(keypoints):
    """安全地计算关键点的平均置信度"""
    if keypoints is None:
        return 0.0
    valid_confs = [kpt[2] for kpt in keypoints if kpt[2] > 0.5]
    return np.mean(valid_confs) if valid_confs else 0.0

def analyze_pose(keypoints):
    """增强的姿态分析函数"""
    results = []
    
    try:
        # 分析躯干倾斜度
        if all(keypoints[i][2] > 0.5 for i in [5, 6, 11, 12]):  # 肩部和臀部都可见
            shoulder_angle = calculate_angle(keypoints[5], keypoints[6])
            hip_angle = calculate_angle(keypoints[11], keypoints[12])
            
            if abs(shoulder_angle) > 10:
                results.append(f"Shoulder Tilt: {shoulder_angle:.1f}°")
            if abs(hip_angle) > 10:
                results.append(f"Hip Tilt: {hip_angle:.1f}°")
        
        # 分析弯腰程度
        if all(keypoints[i][2] > 0.5 for i in [5, 11, 13]):  # 左侧身体关键点可见
            bend_angle = calculate_angle(keypoints[5], keypoints[11], keypoints[13])
            if bend_angle < 120:
                results.append(f"Significant Bending: {bend_angle:.1f}°")
            elif bend_angle < 150:
                results.append(f"Slight Bending: {bend_angle:.1f}°")
                
        # 添加稳定性分析结果
        stability_results = posture_stability_analyzer.analyze_posture_stability()
        if stability_results["warnings"]:
            results.extend(stability_results["warnings"])
        
        # 添加整体稳定性评分
        results.append(f"Stability Score: {stability_results['overall_stability']}")
                
    except Exception as e:
        print(f"姿态分析错误: {str(e)}")
        
    return results

def preprocess_frame(frame):
    """预处理帧以提高检测质量"""
    try:
        # 使用更温和的参数调整
        alpha = 1.1  # 对比度
        beta = 5     # 亮度
        
        # 创建预处理帧的副本
        processed = frame.copy()
        
        # 仅在亮度过低时进行调整
        if processed is not None and len(processed.shape) == 3:
            mean_brightness = np.mean(processed)
            if mean_brightness < 100:  # 只在画面较暗时调整
                processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
            
            # 使用更小的核进行轻微降噪
            processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
            
        return processed
        
    except Exception as e:
        print(f"预处理帧时发生错误: {str(e)}")
        return frame

def process_frame(frame):
    """处理单帧图像"""
    try:
        start_time = time.time()
        
        # 初始化所有可能用到的变量
        yolo_conf = 0.0
        openpose_conf = 0.0
        weight = 0.0
        smoothed_kpts = None
        analysis_results = []
        
        # 初始化返回值
        openpose_frame = frame.copy()
        yolo_frame = frame.copy()
        mixed_frame = frame.copy()
        
        # 预处理帧
        processed_frame = preprocess_frame(frame)
        
        # YOLOv8-pose 检测
        yolo_results = yolo_model(processed_frame)
        
        # OpenPose处理
        datum = op.Datum()
        datum.cvInputData = processed_frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        # 获取关键点数据
        openpose_kpts = None
        if hasattr(datum, "poseKeypoints") and datum.poseKeypoints is not None:
            if len(datum.poseKeypoints) > 0:
                openpose_kpts = datum.poseKeypoints[0]
                openpose_kpts = np.delete(openpose_kpts, -1, axis=0)
        
        yolo_kpts = None
        if len(yolo_results) > 0 and len(yolo_results[0].keypoints) > 0:
            yolo_kpts = yolo_results[0].keypoints.data.cpu().numpy()[0]
        
        # 计算FPS
        fps = 1.0 / (time.time() - start_time)
        
        # 初始化基本信息
        info_dict = {
            "System": "Running",
            "FPS": f"{fps:.1f}",
            "YOLO": "N/A",
            "OpenPose": "N/A",
            "Fusion": "N/A",
            "Keypoints": "0/17",
            "Analysis": []
        }
        
        # 如果两个模型都检测到关键点
        if openpose_kpts is not None and yolo_kpts is not None:
            if openpose_kpts.shape == yolo_kpts.shape:
                openpose_conf = calculate_confidence(openpose_kpts)
                yolo_conf = calculate_confidence(yolo_kpts)
                
                total_conf = openpose_conf + yolo_conf
                weight = yolo_conf / total_conf if total_conf > 0 else 0.5
                
                # 确保混合帧是原始帧的副本
                mixed_frame = frame.copy()
                
                # 融合关键点
                fused_kpts = pose_fusion.fuse_keypoints(yolo_kpts, openpose_kpts)
                
                if fused_kpts is not None:
                    # 应用平滑
                    smoothed_kpts = smoother.update(fused_kpts)
                    
                    if smoothed_kpts is not None:
                        # 在混合帧上绘制骨架
                        draw_skeleton(mixed_frame, smoothed_kpts)
                        
                        # 更新信息
                        fusion_info = pose_fusion.get_fusion_info()
                        info_dict.update({
                            "Fusion": fusion_info["Average Confidence"],
                            "Motion": fusion_info["Motion Level"],
                            "Keypoints": fusion_info["Stable Points"]
                        })
        
        # 检查骨骼一致性
        if smoothed_kpts is not None:
            # 检查骨骼一致性
            consistency_issues = skeleton_checker.check_consistency(smoothed_kpts)
            if consistency_issues:
                info_dict["Consistency"] = consistency_issues
                
        # 更新显示帧
        if hasattr(datum, "cvOutputData"):
            openpose_frame = datum.cvOutputData
        if len(yolo_results) > 0:
            yolo_frame = yolo_results[0].plot()
        
        # 存储结果用于信息面板显示
        process_frame.last_results = info_dict
        
        if smoothed_kpts is not None:
            # 更新姿态稳定性分析器
            posture_stability_analyzer.add_pose(smoothed_kpts)
            
            # 分析姿态并更新信息字典
            analysis_results = analyze_pose(smoothed_kpts)
            if analysis_results:
                info_dict["Analysis"] = analysis_results
                
        return openpose_frame, yolo_frame, mixed_frame
        
    except Exception as e:
        print(f"处理帧时发生错误: {str(e)}")
        return frame, frame, frame

def draw_skeleton(frame, keypoints, thickness=2):
    """改进的骨架绘制函数"""
    if keypoints is None:
        return
        
    # 定义骨架连接和颜色映射
    SKELETON_CONNECTIONS = {
        # 躯干
        (5, 6): (0, 255, 0),   # 肩膀连接 - 绿色
        (5, 11): (255, 0, 0),  # 左躯干 - 红色
        (6, 12): (255, 0, 0),  # 右躯干 - 红色
        (11, 12): (0, 255, 0), # 髋部连接 - 绿色
        
        # 手臂
        (5, 7): (255, 165, 0),  # 左上臂 - 橙色
        (6, 8): (255, 165, 0),  # 右上臂 - 橙色
        (7, 9): (255, 255, 0),  # 左前臂 - 黄色
        (8, 10): (255, 255, 0), # 右前臂 - 黄色
        
        # 腿部
        (11, 13): (0, 255, 255), # 左大腿 - 青色
        (12, 14): (0, 255, 255), # 右大腿 - 青色
        (13, 15): (0, 165, 255), # 左小腿 - 橙色
        (14, 16): (0, 165, 255), # 右小腿 - 橙色
    }
    
    # 绘制骨架连接
    for (start_joint, end_joint), color in SKELETON_CONNECTIONS.items():
        if (keypoints[start_joint][2] > 0.5 and 
            keypoints[end_joint][2] > 0.5):  # 只绘制置信度高的连接
            
            start_point = tuple(map(int, keypoints[start_joint][:2]))
            end_point = tuple(map(int, keypoints[end_joint][:2]))
            
            # 绘制连接线（先画粗黑线，再画细色线，提高可见性）
            cv2.line(frame, start_point, end_point, (0, 0, 0), thickness + 2)
            cv2.line(frame, start_point, end_point, color, thickness)

    # 绘制关键点
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:  # 只绘制置信度高的关键点
            x, y = int(x), int(y)
            # 绘制关键点（黑色外圈 + 彩色内圈）
            cv2.circle(frame, (x, y), 6, (0, 0, 0), -1)
            
            # 根据关键点类型选择颜色
            if i in [5, 6, 11, 12]:  # 躯干关键点
                color = (0, 255, 0)  # 绿色
            elif i in [7, 8, 9, 10]:  # 手臂关键点
                color = (255, 165, 0)  # 橙色
            else:  # 腿部关键点
                color = (0, 255, 255)  # 青色
                
            cv2.circle(frame, (x, y), 4, color, -1)

def get_confidence_color(conf):
    """根据置信度返回颜色"""
    # 从红色渐变到绿色
    return (int(255 * (1 - conf)), int(255 * conf), 0)

def add_info_overlay(frame, info_dict):
    """Add information overlay with improved visual design"""
    # 添加半透明背景
    overlay = frame.copy()
    bg_height = len(info_dict) * 20 + 30  # 背景高度
    cv2.rectangle(overlay, (5, 5), (300, bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # 透明度设置为0.3
    
    y_offset = 25  # 起始位置
    for key, value in info_dict.items():
        if isinstance(value, list):
            for item in value:
                text = f"{key}: {item}" if key else str(item)
                # 添加文字阴影效果
                cv2.putText(frame, text, (12, y_offset+1), cv2.FONT_HERSHEY_SIMPLEX,
                           0.45, (0, 0, 0), 2)  # 阴影
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.45, (50, 255, 50), 1)  # 更柔和的绿色
                y_offset += 20
        else:
            text = f"{key}: {value}"
            # 添加文字阴影效果
            cv2.putText(frame, text, (12, y_offset+1), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (0, 0, 0), 2)
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (50, 255, 50), 1)
            y_offset += 20

def create_info_panel(height, width):
    """创建信息面板，使用更专业的设计"""
    # 创建深色背景
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel.fill(20)  # 更深的背景色
    
    # 添加渐变背景
    gradient = np.linspace(0, 50, width, dtype=np.uint8)
    for i in range(3):
        panel[:, :, i] = gradient
    
    # 添加标题区域背景
    cv2.rectangle(panel, (0, 0), (width, 150), (30, 30, 30), -1)
    
    # 添加标题
    cv2.putText(panel, "System Information", 
                (40, 100), cv2.FONT_HERSHEY_DUPLEX,  # 使用更清晰的字体
                3.5, (0, 255, 0), 5)
    
    # 添加装饰线
    cv2.line(panel, (40, 130), (width-40, 130), 
             (0, 255, 0), 5)
    
    return panel

def add_info_to_panel(panel, info_dict):
    """在信息面板上添加文本"""
    y_offset = 250  # 起始位置
    
    for key, value in info_dict.items():
        if isinstance(value, list):
            for item in value:
                # 为每个分析结果添加背景框
                text = f"{item}" if key == "Analysis" else f"{key}: {item}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2.8, 4)[0]
                
                # 添加半透明背景框
                cv2.rectangle(panel, 
                            (50, y_offset-60), 
                            (50 + text_size[0] + 20, y_offset+10),  # 增加一些边距
                            (50, 50, 50), -1)
                
                # 添加文本阴影
                cv2.putText(panel, text,
                           (53, y_offset), cv2.FONT_HERSHEY_DUPLEX,
                           2.8, (0, 100, 0), 4)  # 阴影
                
                # 添加主文本
                cv2.putText(panel, text,
                           (50, y_offset), cv2.FONT_HERSHEY_DUPLEX,
                           2.8, (0, 255, 0), 4)  # 主文本
                
                y_offset += 120
        else:
            text = f"{key}: {value}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2.8, 4)[0]
            
            cv2.rectangle(panel, 
                        (50, y_offset-60), 
                        (50 + text_size[0] + 20, y_offset+10),
                        (50, 50, 50), -1)
            
            cv2.putText(panel, text,
                       (53, y_offset), cv2.FONT_HERSHEY_DUPLEX,
                       2.8, (0, 100, 0), 4)
            
            cv2.putText(panel, text,
                       (50, y_offset), cv2.FONT_HERSHEY_DUPLEX,
                       2.8, (0, 255, 0), 4)
            
            y_offset += 120

def create_display_window(openpose_frame, yolo_frame, mixed_frame, info_dict):
    """创建包含四个子窗口的主显示窗口"""
    # 获取帧的尺寸
    height, width = openpose_frame.shape[:2]
    
    # 创建信息面板
    info_panel = create_info_panel(height, width)
    add_info_to_panel(info_panel, info_dict)
    
    # 创建上下两个部分
    top_row = np.hstack((openpose_frame, yolo_frame))
    bottom_row = np.hstack((mixed_frame, info_panel))
    
    # 合并成最终显示
    final_display = np.vstack((top_row, bottom_row))
    
    return final_display

class PoseFusion:
    def __init__(self):
        # 微调权重配置
        self.joint_weights = {
            # 躯干核心关键点
            5: {'yolo': 0.85, 'openpose': 0.15},  # 左肩
            6: {'yolo': 0.85, 'openpose': 0.15},  # 右肩
            11: {'yolo': 0.85, 'openpose': 0.15}, # 左髋
            12: {'yolo': 0.85, 'openpose': 0.15}, # 右髋
            
            # 上肢关键点
            7: {'yolo': 0.8, 'openpose': 0.2},   # 左肘
            8: {'yolo': 0.8, 'openpose': 0.2},   # 右肘
            9: {'yolo': 0.8, 'openpose': 0.2},   # 左腕
            10: {'yolo': 0.8, 'openpose': 0.2},  # 右腕
            
            # 下肢关键点
            13: {'yolo': 0.8, 'openpose': 0.2},  # 左膝
            14: {'yolo': 0.8, 'openpose': 0.2},  # 右膝
            15: {'yolo': 0.8, 'openpose': 0.2},  # 左踝
            16: {'yolo': 0.8, 'openpose': 0.2},  # 右踝
        }
        
        # 调整参数
        self.confidence_threshold = 0.35  # 降低置信度阈值
        self.motion_threshold = 30       # 增加运动阈值
        self.position_threshold = 45     # 增加位置阈值
        self.history = deque(maxlen=5)    # 保持5帧历史记录
        
    def validate_keypoints(self, keypoints, reference_points):
        """验证关键点是否在合理范围内"""
        if keypoints is None or reference_points is None:
            return False
            
        # 计算参考点的中心位置（使用躯干关键点）
        ref_points = [5, 6, 11, 12]  # 肩部和髋部关键点
        ref_center = np.mean([reference_points[i][:2] for i in ref_points 
                            if reference_points[i][2] > 0.5], axis=0)
        
        # 计算待验证点的中心位置
        kpt_center = np.mean([keypoints[i][:2] for i in ref_points 
                            if keypoints[i][2] > 0.5], axis=0)
        
        # 如果两个中心点距离过大，认为是无效检测
        if np.linalg.norm(ref_center - kpt_center) > self.position_threshold:
            return False
            
        return True

    def fuse_keypoints(self, yolo_kpts, op_kpts):
        """优化的融合策略"""
        if yolo_kpts is None and op_kpts is None:
            return None
            
        # 优先使用YOLO结果作为基准
        reference_kpts = yolo_kpts if yolo_kpts is not None else op_kpts
        fused_kpts = reference_kpts.copy()
        
        if yolo_kpts is None or op_kpts is None:
            return fused_kpts
            
        # 验证关键点
        if not self.validate_keypoints(op_kpts, reference_kpts):
            return reference_kpts
            
        # 融合过程
        for joint_id in range(len(yolo_kpts)):
            yolo_conf = yolo_kpts[joint_id][2]
            op_conf = op_kpts[joint_id][2]
            
            # 对于低置信度的点，优先使用YOLO结果
            if yolo_conf < self.confidence_threshold and op_conf < self.confidence_threshold:
                fused_kpts[joint_id] = yolo_kpts[joint_id]
                continue
                
            # 获取预设权重
            weights = self.joint_weights.get(joint_id, {'yolo': 0.7, 'openpose': 0.3})
            w_yolo = weights['yolo']
            w_op = weights['openpose']
            
            # 根据置信度微调权重
            total_conf = yolo_conf + op_conf
            if total_conf > 0:
                # 保持YOLO的主导地位，但允许根据置信度小幅调整
                base_yolo_weight = weights['yolo']
                conf_ratio = yolo_conf / total_conf
                w_yolo = min(0.9, max(base_yolo_weight, conf_ratio))
                w_op = 1 - w_yolo
            
            # 计算融合位置
            fused_pos = (w_yolo * yolo_kpts[joint_id][:2] + 
                        w_op * op_kpts[joint_id][:2])
            
            # 验证融合位置是否合理
            if np.linalg.norm(fused_pos - reference_kpts[joint_id][:2]) < self.position_threshold:
                fused_kpts[joint_id][:2] = fused_pos
                # 使用较高的置信度，但给予YOLO更大权重
                fused_kpts[joint_id][2] = 0.7 * yolo_conf + 0.3 * op_conf
        
        # 应用改进的时间平滑
        if len(self.history) > 0:
            prev_kpts = self.history[-1]
            if self.validate_keypoints(prev_kpts, reference_kpts):
                alpha = 0.8  # 调整平滑强度
                for i in range(len(fused_kpts)):
                    if (prev_kpts[i][2] > 0.5 and fused_kpts[i][2] > 0.5 and
                        np.linalg.norm(fused_kpts[i][:2] - prev_kpts[i][:2]) < self.position_threshold):
                        fused_kpts[i][:2] = (alpha * fused_kpts[i][:2] + 
                                           (1-alpha) * prev_kpts[i][:2])
        
        self.history.append(fused_kpts.copy())
        return fused_kpts

    def get_fusion_info(self):
        """获取融合状态信息"""
        if len(self.history) == 0:
            return {
                "Average Confidence": "N/A",
                "Motion Level": "N/A",
                "Stable Points": "0/17"
            }
            
        latest = self.history[-1]
        avg_conf = np.mean([kpt[2] for kpt in latest if kpt[2] > 0.5])
        
        motion = 0
        if len(self.history) > 1:
            prev = self.history[-2]
            valid_points = [(i, p) for i, p in enumerate(latest) 
                          if p[2] > 0.5 and prev[i][2] > 0.5]
            if valid_points:
                motions = [np.linalg.norm(latest[i][:2] - prev[i][:2]) 
                          for i, _ in valid_points]
                motion = np.mean(motions)
        
        return {
            "Average Confidence": f"{avg_conf:.2f}",
            "Motion Level": f"{motion:.1f}",
            "Stable Points": f"{sum(1 for kpt in latest if kpt[2] > 0.5)}/17"
        }

class PostureStabilityAnalyzer:
    def __init__(self, window_size=30):  # 使用30帧窗口
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
        self.stability_thresholds = {
            'shoulder': 15.0,  # 肩部角度变化阈值
            'spine': 20.0,    # 脊柱角度变化阈值
            'hip': 15.0,      # 髋部角度变化阈值
            'position': 30.0  # 位置变化阈值(像素)
        }
        
    def add_pose(self, keypoints):
        """添加新的姿态帧到历史记录"""
        if keypoints is not None and len(keypoints) > 0:
            self.pose_history.append(keypoints)
            
    def calculate_joint_stability(self, joint_idx):
        """计算特定关节点的稳定性"""
        if len(self.pose_history) < 2:
            return 1.0, 0.0
            
        positions = []
        for pose in self.pose_history:
            if pose[joint_idx][2] > 0.5:  # 只考虑置信度高的点
                positions.append(pose[joint_idx][:2])
                
        if len(positions) < 2:
            return 1.0, 0.0
            
        positions = np.array(positions)
        # 计算位置标准差
        std_dev = np.std(positions, axis=0)
        max_deviation = np.max(std_dev)
        
        # 计算稳定性分数 (0-1)
        stability = max(0, 1 - max_deviation / self.stability_thresholds['position'])
        return stability, max_deviation
        
    def analyze_posture_stability(self):
        """分析整体姿态稳定性"""
        if len(self.pose_history) < self.window_size // 2:
            return {
                "overall_stability": "Insufficient Data",
                "details": {},
                "warnings": []
            }
            
        stability_scores = {}
        warnings = []
        
        # 分析关键关节的稳定性
        key_joints = {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14
        }
        
        for joint_name, joint_idx in key_joints.items():
            stability, deviation = self.calculate_joint_stability(joint_idx)
            stability_scores[joint_name] = stability
            
            if stability < 0.6:
                warnings.append(f"Unstable {joint_name.replace('_', ' ')}: {(1-stability)*100:.1f}% movement")
                
        # 分析躯干稳定性
        trunk_stability = self.analyze_trunk_stability()
        stability_scores['trunk'] = trunk_stability
        
        if trunk_stability < 0.6:
            warnings.append(f"Unstable trunk position: {(1-trunk_stability)*100:.1f}% movement")
            
        # 计算整体稳定性分数
        overall_stability = np.mean(list(stability_scores.values()))
        
        # 分析姿态变化趋势
        trend_analysis = self.analyze_pose_trend()
        if trend_analysis:
            warnings.extend(trend_analysis)
            
        return {
            "overall_stability": f"{overall_stability:.2f}",
            "details": stability_scores,
            "warnings": warnings
        }
        
    def analyze_trunk_stability(self):
        """分析躯干稳定性"""
        if len(self.pose_history) < 2:
            return 1.0
            
        spine_angles = []
        for pose in self.pose_history:
            # 计算脊柱角度（肩部中点到髋部中点）
            if all(pose[i][2] > 0.5 for i in [5, 6, 11, 12]):
                shoulder_mid = (pose[5][:2] + pose[6][:2]) / 2
                hip_mid = (pose[11][:2] + pose[12][:2]) / 2
                angle = calculate_angle(shoulder_mid, hip_mid)
                spine_angles.append(angle)
                
        if not spine_angles:
            return 1.0
            
        # 计算角度变化的标准差
        angle_std = np.std(spine_angles)
        stability = max(0, 1 - angle_std / self.stability_thresholds['spine'])
        return stability
        
    def analyze_pose_trend(self):
        """分析姿态变化趋势"""
        if len(self.pose_history) < self.window_size:
            return []
            
        warnings = []
        recent_poses = list(self.pose_history)[-10:]  # 分析最近10帧
        
        # 检测躯干前倾趋势
        spine_angles = []
        for pose in recent_poses:
            if all(pose[i][2] > 0.5 for i in [5, 6, 11, 12]):
                shoulder_mid = (pose[5][:2] + pose[6][:2]) / 2
                hip_mid = (pose[11][:2] + pose[12][:2]) / 2
                angle = calculate_angle(shoulder_mid, hip_mid)
                spine_angles.append(angle)
                
        if spine_angles:
            angle_trend = np.polyfit(range(len(spine_angles)), spine_angles, 1)[0]
            if abs(angle_trend) > 1.0:
                trend_direction = "forward" if angle_trend > 0 else "backward"
                warnings.append(f"Gradual {trend_direction} leaning detected")
                
        return warnings

def main():
    """主函数"""
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_ID)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 创建单个窗口
    window_name = 'Pose Detection System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    global smoother
    smoother = KeypointSmoother(window_size=5)
    
    # 初始化骨骼一致性检查器
    global skeleton_checker
    skeleton_checker = SkeletonConsistencyChecker()
    
    # 初始化姿态融合器
    global pose_fusion
    pose_fusion = PoseFusion()
    
    # 初始化姿态稳定性分析器
    global posture_stability_analyzer
    posture_stability_analyzer = PostureStabilityAnalyzer()
    
    running = True  # 添加运行状态标志
    
    try:
        while running and cap.isOpened():  # 检查运行状态
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            openpose_frame, yolo_frame, mixed_frame = process_frame(frame)
            
            # 创建显示
            display = create_display_window(openpose_frame, yolo_frame, mixed_frame, 
                                         process_frame.last_results)
            
            # 显示结果
            cv2.imshow(window_name, display)
            
            # 检查窗口状态和键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            # 检查窗口是否关闭或按下q键
            if (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1) or (key == ord('q')):
                running = False  # 设置运行状态为False
                break
                
    except Exception as e:
        print(f"Runtime error: {str(e)}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("程序已正常终止")
        sys.exit(0)  # 确保程序完全终止

if __name__ == "__main__":
    main()
