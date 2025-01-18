from collections import deque
import numpy as np
from ..utils.geometry import calculate_angle

class PostureStabilityAnalyzer:
    def __init__(self, window_size=30):
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
            if pose[joint_idx][2] > 0.5:
                positions.append(pose[joint_idx][:2])
                
        if len(positions) < 2:
            return 1.0, 0.0
            
        positions = np.array(positions)
        std_dev = np.std(positions, axis=0)
        max_deviation = np.max(std_dev)
        
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
            if all(pose[i][2] > 0.5 for i in [5, 6, 11, 12]):
                shoulder_mid = (pose[5][:2] + pose[6][:2]) / 2
                hip_mid = (pose[11][:2] + pose[12][:2]) / 2
                angle = calculate_angle(shoulder_mid, hip_mid)
                spine_angles.append(angle)
                
        if not spine_angles:
            return 1.0
            
        angle_std = np.std(spine_angles)
        stability = max(0, 1 - angle_std / self.stability_thresholds['spine'])
        return stability
        
    def analyze_pose_trend(self):
        """分析姿态变化趋势"""
        if len(self.pose_history) < self.window_size:
            return []
            
        warnings = []
        recent_poses = list(self.pose_history)[-10:]
        
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