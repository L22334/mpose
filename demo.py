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
    """分析姿态并返回建议"""
    results = []
    
    try:
        # 分析躯干倾斜度
        if all(keypoints[i][2] > 0.5 for i in [5, 6, 11, 12]):  # 肩部和臀部都可见
            shoulder_angle = calculate_angle(keypoints[5], keypoints[6])
            hip_angle = calculate_angle(keypoints[11], keypoints[12])
            
            if abs(shoulder_angle) > 10:
                results.append(f"Shoulder Tilt: {shoulder_angle:.1f} deg")
            if abs(hip_angle) > 10:
                results.append(f"Hip Tilt: {hip_angle:.1f} deg")
        
        # 分析弯腰程度
        if all(keypoints[i][2] > 0.5 for i in [5, 11, 13]):  # 左侧身体关键点可见
            bend_angle = calculate_angle(keypoints[5], keypoints[11], keypoints[13])
            if bend_angle < 120:
                results.append(f"Bending Angle: {bend_angle:.1f} deg")
            elif bend_angle < 150:
                results.append(f"Slight Bending: {bend_angle:.1f} deg")
                
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
                
                combined_kpts = weight * yolo_kpts + (1 - weight) * openpose_kpts
                smoothed_kpts = smoother.update(combined_kpts)
                
                if smoothed_kpts is not None:
                    draw_skeleton(mixed_frame, smoothed_kpts)
                    analysis_results = analyze_pose(smoothed_kpts)
                    
                    info_dict.update({
                        "YOLO": f"Conf: {yolo_conf:.2f}",
                        "OpenPose": f"Conf: {openpose_conf:.2f}",
                        "Fusion": f"Weight: {weight:.2f}",
                        "Keypoints": f"{sum(1 for kpt in smoothed_kpts if kpt[2] > 0.5)}/17",
                        "Analysis": [f"{result}" for result in analysis_results[:3]]
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
        
        return openpose_frame, yolo_frame, mixed_frame
        
    except Exception as e:
        print(f"处理帧时发生错误: {str(e)}")
        return frame, frame, frame

def draw_skeleton(frame, keypoints):
    """只绘制骨架和关键点，不添加文字标注"""
    # 绘制关键点
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            x, y = int(x), int(y)
            color = get_confidence_color(conf)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)  # 黑色外圈
            cv2.circle(frame, (x, y), 3, color, -1)      # 颜色随置信度变化
    
    # 绘制骨架连接
    for i in YOLO_CONNECTIONS:
        for j in YOLO_CONNECTIONS[i]:
            if keypoints[i][2] > 0.5 and keypoints[j][2] > 0.5:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                conf = (keypoints[i][2] + keypoints[j][2]) / 2
                color = get_confidence_color(conf)
                cv2.line(frame, pt1, pt2, (0, 0, 0), 3)  # 黑色外边框
                cv2.line(frame, pt1, pt2, color, 1)      # 颜色随置信度变化

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
