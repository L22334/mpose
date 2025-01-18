from collections import deque
import numpy as np
from ..utils.constants import SKELETON_PAIRS

class SkeletonConsistencyChecker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.bone_lengths = {}
        self.bone_length_history = {bone: deque(maxlen=window_size) 
                                  for bone in SKELETON_PAIRS}
        self.reference_ratios = {
            'arm_symmetry': 1.0,      # 手臂长度比例
            'leg_symmetry': 1.0,      # 腿部长度比例
            'arm_leg_ratio': 0.8,     # 手臂与腿部比例
            'shoulder_hip_ratio': 1.2  # 肩宽与髋宽比例
        }
        self.tolerance = 0.2  # 允许误差范围(20%)

    def calculate_bone_length(self, keypoints, start_idx, end_idx):
        """计算两个关键点之间的距离"""
        if (keypoints[start_idx][2] > 0.5 and 
            keypoints[end_idx][2] > 0.5):
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
        """检查骨架对称性"""
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
        """检查身体比例"""
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
        """主检查函数，返回所有一致性问题"""
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