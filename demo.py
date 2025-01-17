import cv2
import numpy as np
from ultralytics import YOLO
import openpose.pyopenpose as op
from collections import deque
import torch
import time
import sys
import traceback

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
VIDEO_PATH = "video/test_1.mp4"  # 视频文件路径
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
        self.confidence_threshold = 0.5
        
    def update(self, keypoints):
        """改进的平滑处理"""
        if keypoints is None:
            return None
            
        self.history.append(keypoints)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # 使用卡尔曼滤波进行平滑
        smoothed = np.zeros_like(keypoints)
        for i in range(len(keypoints)):
            valid_points = []
            valid_confs = []
            
            for hist_kpts in self.history:
                if hist_kpts[i][2] > self.confidence_threshold:
                    valid_points.append(hist_kpts[i][:2])
                    valid_confs.append(hist_kpts[i][2])
                    
            if valid_points:
                # 根据置信度加权平均
                weights = np.array(valid_confs)
                weights = weights / weights.sum()
                smoothed[i][:2] = np.average(valid_points, weights=weights, axis=0)
                smoothed[i][2] = keypoints[i][2]  # 保持原始置信度
            else:
                smoothed[i] = keypoints[i]
                
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
        
        # 创建单个帧副本，避免重复拷贝
        processed_frame = preprocess_frame(frame.copy())
        display_frames = {
            'openpose': processed_frame.copy(),
            'yolo': processed_frame.copy(),
            'mixed': processed_frame.copy()
        }
        
        # 初始化结果字典
        results = {
            'keypoints': {
                'yolo': None,
                'openpose': None,
                'fused': None,
                'smoothed': None
            },
            'info': {
                'system_status': 'Running',
                'fps': 0,
                'keypoint_count': '0/17',
                'analysis': [],
                'fusion_data': {},
                'consistency': []
            }
        }
        
        try:
            # YOLOv8-pose 检测
            yolo_results = yolo_model(processed_frame)
            if len(yolo_results) > 0 and len(yolo_results[0].keypoints) > 0:
                results['keypoints']['yolo'] = yolo_results[0].keypoints.data.cpu().numpy()[0]
                display_frames['yolo'] = yolo_results[0].plot()
            
            # OpenPose处理
            datum = op.Datum()
            datum.cvInputData = processed_frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            if hasattr(datum, "poseKeypoints") and datum.poseKeypoints is not None:
                if len(datum.poseKeypoints) > 0:
                    results['keypoints']['openpose'] = np.delete(datum.poseKeypoints[0], -1, axis=0)
                    display_frames['openpose'] = datum.cvOutputData
            
            # 融合和分析过程
            if (results['keypoints']['yolo'] is not None and 
                results['keypoints']['openpose'] is not None and
                results['keypoints']['yolo'].shape == results['keypoints']['openpose'].shape):
                process_keypoints(results, display_frames['mixed'])
            
        except Exception as e:
            print(f"处理关键点时发生错误: {str(e)}")
            traceback.print_exc()  # 打印详细的错误堆栈
        
        # 更新FPS
        results['info']['fps'] = f"{1.0 / (time.time() - start_time):.1f}"
        process_frame.last_results = results['info']
        
        return display_frames['openpose'], display_frames['yolo'], display_frames['mixed']
        
    except Exception as e:
        print(f"处理帧时发生错误: {str(e)}")
        traceback.print_exc()
        return frame, frame, frame

def process_keypoints(results, mixed_frame):
    """改进的关键点处理函数"""
    # 添加时序一致性检查
    def check_temporal_consistency(current_kpts, prev_kpts, max_speed=50):
        if prev_kpts is None:
            return current_kpts
            
        consistent_kpts = current_kpts.copy()
        for i in range(len(current_kpts)):
            if (current_kpts[i][2] > 0.5 and prev_kpts[i][2] > 0.5):
                displacement = np.linalg.norm(current_kpts[i][:2] - prev_kpts[i][:2])
                if displacement > max_speed:
                    # 如果位移过大，使用插值或降低置信度
                    consistent_kpts[i][:2] = (current_kpts[i][:2] + prev_kpts[i][:2]) / 2
                    consistent_kpts[i][2] *= 0.8
                    
        return consistent_kpts

    # 1. 融合关键点
    results['keypoints']['fused'] = pose_fusion.fuse_keypoints(
        results['keypoints']['yolo'],
        results['keypoints']['openpose']
    )
    
    if results['keypoints']['fused'] is not None:
        # 2. 应用时序一致性检查
        if hasattr(process_keypoints, 'prev_keypoints'):
            results['keypoints']['fused'] = check_temporal_consistency(
                results['keypoints']['fused'],
                process_keypoints.prev_keypoints
            )
        process_keypoints.prev_keypoints = results['keypoints']['fused'].copy()
        
        # 3. 平滑处理
        results['keypoints']['smoothed'] = smoother.update(results['keypoints']['fused'])
        
        if results['keypoints']['smoothed'] is not None:
            # 4. 一次性进行所有分析
            analyze_keypoints(results, mixed_frame)

    # 添加运动连续性检查
    if results['keypoints']['fused'] is not None:
        results['keypoints']['fused'] = motion_checker.check_motion(
            results['keypoints']['fused']
        )

    if results['keypoints']['fused'] is not None:
        # 应用轨迹分析
        results['keypoints']['fused'] = trajectory_analyzer.analyze_trajectory(
            results['keypoints']['fused']
        )
        
        # 更新检测器性能
        pose_fusion.update_performance(results['keypoints']['yolo'], 'yolo')
        pose_fusion.update_performance(results['keypoints']['openpose'], 'openpose')
        
        # 验证姿态一致性
        results['keypoints']['fused'] = pose_validator.validate_pose(
            results['keypoints']['fused']
        )

def analyze_keypoints(results, mixed_frame):
    """集中处理所有关键点分析"""
    smoothed_kpts = results['keypoints']['smoothed']
    
    # 1. 绘制骨架
    draw_skeleton(mixed_frame, smoothed_kpts)
    
    # 2. 更新融合信息
    fusion_info = pose_fusion.get_fusion_info()
    results['info'].update({
        'fusion_data': {
            'confidence': fusion_info["Average Confidence"],
            'motion': fusion_info["Motion Level"],
            'keypoint_count': fusion_info["Stable Points"]
        }
    })
    
    # 3. 检查骨骼一致性
    consistency_issues = skeleton_checker.check_consistency(smoothed_kpts)
    if consistency_issues:
        results['info']['consistency'] = consistency_issues
    
    # 4. 更新姿态稳定性
    posture_stability_analyzer.add_pose(smoothed_kpts)
    
    # 5. 分析姿态
    pose_analysis = analyze_pose(smoothed_kpts)
    if pose_analysis:
        results['info']['analysis'] = pose_analysis

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
        # 更新权重配置，根据不同关键点的特性调整
        self.joint_weights = {
            # 躯干核心关键点 - YOLO在躯干检测上更稳定
            5: {'yolo': 0.9, 'openpose': 0.1},   # 左肩
            6: {'yolo': 0.9, 'openpose': 0.1},   # 右肩
            11: {'yolo': 0.9, 'openpose': 0.1},  # 左髋
            12: {'yolo': 0.9, 'openpose': 0.1},  # 右髋
            
            # 四肢关键点 - OpenPose在四肢末端检测上较好
            9: {'yolo': 0.6, 'openpose': 0.4},   # 左腕
            10: {'yolo': 0.6, 'openpose': 0.4},  # 右腕
            15: {'yolo': 0.6, 'openpose': 0.4},  # 左踝
            16: {'yolo': 0.6, 'openpose': 0.4},  # 右踝
        }
        
        # 调整参数
        self.confidence_threshold = 0.35  # 降低置信度阈值
        self.motion_threshold = 30       # 增加运动阈值
        self.position_threshold = 45     # 增加位置阈值
        self.history = deque(maxlen=5)    # 保持5帧历史记录
        
        # 添加动态权重调整参数
        self.performance_history = {
            'yolo': deque(maxlen=30),
            'openpose': deque(maxlen=30)
        }
        
    def validate_keypoints(self, keypoints, reference_points):
        """增强的关键点验证"""
        if keypoints is None or reference_points is None:
            return False
        
        # 1. 检查躯干关键点的有效性
        trunk_points = [5, 6, 11, 12]  # 肩部和髋部关键点
        trunk_valid = all(keypoints[i][2] > 0.6 for i in trunk_points)
        if not trunk_valid:
            return False
        
        # 2. 计算身体比例
        def get_body_proportions(kpts):
            proportions = {}
            if all(kpts[i][2] > 0.5 for i in [5, 6]):
                proportions['shoulder_width'] = np.linalg.norm(kpts[5][:2] - kpts[6][:2])
            if all(kpts[i][2] > 0.5 for i in [11, 12]):
                proportions['hip_width'] = np.linalg.norm(kpts[11][:2] - kpts[12][:2])
            if all(kpts[i][2] > 0.5 for i in [5, 11]):
                proportions['torso_height'] = np.linalg.norm(kpts[5][:2] - kpts[11][:2])
            return proportions
        
        ref_proportions = get_body_proportions(reference_points)
        test_proportions = get_body_proportions(keypoints)
        
        # 3. 比较身体比例
        for key in ref_proportions:
            if key in test_proportions:
                ratio = test_proportions[key] / ref_proportions[key]
                if not (0.7 <= ratio <= 1.3):
                    return False
        
        # 4. 检查关键点的相对位置关系
        def check_relative_positions(kpts):
            # 检查肩部在头部下方
            if all(kpts[i][2] > 0.5 for i in [0, 5, 6]):
                if not (kpts[5][1] > kpts[0][1] and kpts[6][1] > kpts[0][1]):
                    return False
            # 检查髋部在肩部下方
            if all(kpts[i][2] > 0.5 for i in [5, 6, 11, 12]):
                if not (kpts[11][1] > kpts[5][1] and kpts[12][1] > kpts[6][1]):
                    return False
            return True
        
        if not check_relative_positions(keypoints):
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
        
        # 添加解剖学约束
        def apply_anatomical_constraints(kpts):
            # 检查骨骼长度约束
            def check_bone_length(p1_idx, p2_idx, min_ratio=0.4, max_ratio=2.5):
                if kpts[p1_idx][2] > 0.5 and kpts[p2_idx][2] > 0.5:
                    length = np.linalg.norm(kpts[p1_idx][:2] - kpts[p2_idx][:2])
                    # 使用肩宽作为参考
                    shoulder_width = np.linalg.norm(kpts[5][:2] - kpts[6][:2])
                    ratio = length / shoulder_width
                    return min_ratio <= ratio <= max_ratio
                return True

            # 应用解剖学约束
            for i, j in [(5,7), (7,9), (6,8), (8,10),    # 手臂
                        (11,13), (13,15), (12,14), (14,16)]: # 腿部
                if not check_bone_length(i, j):
                    # 如果违反约束，降低该点的置信度
                    kpts[j][2] *= 0.5
            
            return kpts

        # 在融合后应用解剖学约束
        fused_kpts = apply_anatomical_constraints(fused_kpts)
        
        # 应用先验知识检查
        confidence_factor = pose_prior.check_pose_validity(fused_kpts)
        if confidence_factor < 1.0:
            for i in range(len(fused_kpts)):
                fused_kpts[i][2] *= confidence_factor
        
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

    def update_weights(self, yolo_kpts, op_kpts, joint_id):
        """动态调整融合权重"""
        base_weights = self.joint_weights.get(
            joint_id, 
            {'yolo': 0.7, 'openpose': 0.3}
        )
        
        # 计算历史表现
        if len(self.performance_history['yolo']) > 0:
            yolo_stability = np.mean(self.performance_history['yolo'])
            op_stability = np.mean(self.performance_history['openpose'])
            
            # 根据历史表现调整权重
            total_stability = yolo_stability + op_stability
            if total_stability > 0:
                yolo_weight = (base_weights['yolo'] * 0.7 + 
                             (yolo_stability / total_stability) * 0.3)
                op_weight = 1 - yolo_weight
                return yolo_weight, op_weight
        
        return base_weights['yolo'], base_weights['openpose']
    
    def update_performance(self, keypoints, source):
        """更新检测器性能历史"""
        if keypoints is None:
            return
            
        # 计算关键点稳定性得分
        valid_confs = [kpt[2] for kpt in keypoints if kpt[2] > 0.5]
        if valid_confs:
            stability = np.mean(valid_confs)
            self.performance_history[source].append(stability)

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

class PosePriorKnowledge:
    def __init__(self):
        # 定义常见姿态的关键点角度范围
        self.pose_constraints = {
            'standing': {
                'spine_angle': (160, 200),  # 脊柱应该接近垂直
                'knee_angle': (160, 180),   # 站立时膝盖应该接近伸直
                'hip_width_ratio': (0.8, 1.2)  # 髋部宽度比例
            },
            'walking': {
                'step_length': (0.5, 1.5),  # 相对于身高的步长比例
                'arm_swing': (30, 60),      # 手臂摆动角度范围
                'leg_lift': (10, 45)        # 腿部抬起角度范围
            }
        }
        
        # 定义关键点对称性约束
        self.symmetry_pairs = [
            ((5,7), (6,8)),   # 左右上臂
            ((7,9), (8,10)),  # 左右前臂
            ((11,13), (12,14)), # 左右大腿
            ((13,15), (14,16))  # 左右小腿
        ]
        
    def check_pose_validity(self, keypoints):
        """检查姿态是否符合先验知识"""
        if keypoints is None:
            return 1.0  # 返回置信度调整因子
            
        confidence_factor = 1.0
        
        # 检查对称性
        for (left_pair, right_pair) in self.symmetry_pairs:
            left_length = np.linalg.norm(keypoints[left_pair[0]][:2] - keypoints[left_pair[1]][:2])
            right_length = np.linalg.norm(keypoints[right_pair[0]][:2] - keypoints[right_pair[1]][:2])
            
            if left_length > 0 and right_length > 0:
                ratio = min(left_length, right_length) / max(left_length, right_length)
                if ratio < 0.7:  # 如果不对称性过大
                    confidence_factor *= 0.8
        
        # 检查脊柱角度
        if all(keypoints[i][2] > 0.5 for i in [5, 6, 11, 12]):
            shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
            hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
            spine_angle = calculate_angle(shoulder_mid, hip_mid)
            
            # 检查是否在合理范围内
            if not (160 <= abs(spine_angle) <= 200):
                confidence_factor *= 0.9
        
        return confidence_factor

class MotionContinuityChecker:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.velocity_threshold = 50  # 最大允许速度
        self.acceleration_threshold = 25  # 最大允许加速度
        
    def check_motion(self, keypoints):
        """检查运动的连续性"""
        if keypoints is None:
            return None
            
        self.history.append(keypoints)
        if len(self.history) < 3:
            return keypoints
            
        adjusted_keypoints = keypoints.copy()
        
        # 修改这部分代码，使用列表索引而不是切片
        for i in range(len(keypoints)):
            if keypoints[i][2] < 0.5:
                continue
                
            # 获取最近三帧的位置
            positions = []
            for frame_idx in range(1, 4):  # 获取最近3帧
                if len(self.history) >= frame_idx:
                    frame = self.history[-frame_idx]
                    if frame[i][2] > 0.5:
                        positions.append(frame[i][:2])
            
            if len(positions) < 3:
                continue
                
            # 计算速度和加速度
            positions = np.array(positions)
            v1 = np.linalg.norm(positions[0] - positions[1])
            v2 = np.linalg.norm(positions[1] - positions[2])
            acceleration = abs(v1 - v2)
            
            # 如果运动不连续，调整置信度
            if v1 > self.velocity_threshold or acceleration > self.acceleration_threshold:
                adjusted_keypoints[i][2] *= 0.7
                # 使用插值来平滑运动
                adjusted_keypoints[i][:2] = (positions[1] + positions[0]) / 2
                
        return adjusted_keypoints

class KeypointTrajectoryAnalyzer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.trajectories = {i: deque(maxlen=window_size) for i in range(17)}
        self.velocity_history = {i: deque(maxlen=window_size-1) for i in range(17)}
        
    def analyze_trajectory(self, keypoints):
        """分析关键点轨迹的合理性"""
        if keypoints is None:
            return keypoints
            
        adjusted_keypoints = keypoints.copy()
        
        # 更新轨迹
        for i in range(len(keypoints)):
            if keypoints[i][2] > 0.5:
                self.trajectories[i].append(keypoints[i][:2])
                
                if len(self.trajectories[i]) > 1:
                    velocity = np.linalg.norm(
                        self.trajectories[i][-1] - self.trajectories[i][-2]
                    )
                    self.velocity_history[i].append(velocity)
        
        # 分析轨迹
        for i in range(len(keypoints)):
            if len(self.trajectories[i]) < 3:
                continue
                
            # 1. 检查速度一致性
            if len(self.velocity_history[i]) > 2:
                mean_velocity = np.mean(self.velocity_history[i])
                std_velocity = np.std(self.velocity_history[i])
                current_velocity = self.velocity_history[i][-1]
                
                if abs(current_velocity - mean_velocity) > 2 * std_velocity:
                    adjusted_keypoints[i][2] *= 0.8
            
            # 2. 检查轨迹平滑度
            if len(self.trajectories[i]) > 3:
                points = np.array(list(self.trajectories[i]))
                # 计算曲线拟合误差
                try:
                    x = np.arange(len(points))
                    coeffs_x = np.polyfit(x, points[:,0], 2)
                    coeffs_y = np.polyfit(x, points[:,1], 2)
                    
                    fitted_x = np.polyval(coeffs_x, x)
                    fitted_y = np.polyval(coeffs_y, x)
                    
                    error = np.mean(np.sqrt(
                        (fitted_x - points[:,0])**2 + 
                        (fitted_y - points[:,1])**2
                    ))
                    
                    if error > 10:  # 轨迹不平滑
                        adjusted_keypoints[i][2] *= 0.9
                except:
                    pass
            
            # 3. 检查加速度变化
            if len(self.velocity_history[i]) > 2:
                accelerations = np.diff(list(self.velocity_history[i]))
                if np.max(np.abs(accelerations)) > 30:  # 加速度突变
                    adjusted_keypoints[i][2] *= 0.85
        
        return adjusted_keypoints

class PoseConsistencyValidator:
    def __init__(self):
        self.angle_constraints = {
            # 肘部角度范围
            'elbow': {
                'left': (30, 180),
                'right': (30, 180)
            },
            # 膝盖角度范围
            'knee': {
                'left': (30, 180),
                'right': (30, 180)
            },
            # 髋部角度范围
            'hip': {
                'left': (45, 180),
                'right': (45, 180)
            }
        }
        
    def validate_pose(self, keypoints):
        """验证姿态的生理学合理性"""
        if keypoints is None:
            return keypoints
            
        adjusted_keypoints = keypoints.copy()
        
        # 检查肘部角度
        def check_elbow_angle(side):
            if side == 'left':
                pts = [5, 7, 9]  # 左肩、左肘、左腕
            else:
                pts = [6, 8, 10]  # 右肩、右肘、右腕
                
            if all(keypoints[i][2] > 0.5 for i in pts):
                angle = calculate_angle(
                    keypoints[pts[0]][:2],
                    keypoints[pts[1]][:2],
                    keypoints[pts[2]][:2]
                )
                constraints = self.angle_constraints['elbow'][side]
                if not constraints[0] <= angle <= constraints[1]:
                    adjusted_keypoints[pts[1]][2] *= 0.8
                    adjusted_keypoints[pts[2]][2] *= 0.8
        
        # 检查膝盖角度
        def check_knee_angle(side):
            if side == 'left':
                pts = [11, 13, 15]  # 左髋、左膝、左踝
            else:
                pts = [12, 14, 16]  # 右髋、右膝、右踝
                
            if all(keypoints[i][2] > 0.5 for i in pts):
                angle = calculate_angle(
                    keypoints[pts[0]][:2],
                    keypoints[pts[1]][:2],
                    keypoints[pts[2]][:2]
                )
                constraints = self.angle_constraints['knee'][side]
                if not constraints[0] <= angle <= constraints[1]:
                    adjusted_keypoints[pts[1]][2] *= 0.8
                    adjusted_keypoints[pts[2]][2] *= 0.8
        
        # 应用所有检查
        for side in ['left', 'right']:
            check_elbow_angle(side)
            check_knee_angle(side)
        
        return adjusted_keypoints

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
    
    # 初始化先验知识检查器
    global pose_prior
    pose_prior = PosePriorKnowledge()
    
    # 初始化运动连续性检查器
    global motion_checker
    motion_checker = MotionContinuityChecker()
    
    # 初始化新组件
    global trajectory_analyzer
    trajectory_analyzer = KeypointTrajectoryAnalyzer()
    
    global pose_validator
    pose_validator = PoseConsistencyValidator()
    
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
