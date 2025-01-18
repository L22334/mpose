import numpy as np
from ..utils.geometry import calculate_angle

class PoseClassifier:
    def __init__(self):
        # 角度阈值配置
        self.angle_thresholds = {
            'vertical_spine': 15,      # 脊柱垂直判定阈值
            'bend_angle': 45,          # 弯曲判定阈值
            'turn_angle': 30,          # 转身判定阈值
            'shoulder_angle': 60,      # 手臂抬高判定阈值
            'knee_angle': 150,         # 腿部伸直判定阈值
            'squat_angle': 120,        # 下蹲判定阈值
        }
        
    def classify_pose(self, keypoints):
        """主要分类函数"""
        if keypoints is None or len(keypoints) < 17:
            return {
                'posture': 'unknown',
                'arms': 'unknown',
                'legs': 'unknown'
            }
            
        results = {
            'posture': self._classify_posture(keypoints),
            'arms': self._classify_arms(keypoints),
            'legs': self._classify_legs(keypoints)
        }
        
        return results
        
    def _classify_posture(self, keypoints):
        """第一组姿势分类：站直、弯曲、转身、弯曲同时转身"""
        # 检查关键点可见性
        if not all(keypoints[i][2] > 0.5 for i in [5, 6, 11, 12]):
            return 'unknown'
            
        # 计算脊柱角度（使用肩部中点和髋部中点）
        shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
        spine_vertical_angle = abs(90 - abs(calculate_angle(shoulder_mid, hip_mid)))
        
        # 计算转身角度（使用肩部宽度变化）
        shoulder_width = np.linalg.norm(keypoints[5][:2] - keypoints[6][:2])
        hip_width = np.linalg.norm(keypoints[11][:2] - keypoints[12][:2])
        width_ratio = shoulder_width / (hip_width + 1e-6)
        is_turning = width_ratio < 0.7  # 当转身时，肩部投影宽度会显著减小
        
        # 判断弯曲
        is_bending = spine_vertical_angle > self.angle_thresholds['bend_angle']
        
        # 分类判断
        if is_bending and is_turning:
            return 'bend_and_turn'
        elif is_bending:
            return 'bending'
        elif is_turning:
            return 'turning'
        else:
            return 'standing'
            
    def _classify_arms(self, keypoints):
        """第二组姿势分类：手臂位置"""
        if not all(keypoints[i][2] > 0.5 for i in [5, 6, 7, 8, 9, 10]):
            return 'unknown'
            
        # 计算左右手臂相对于肩部的角度
        left_arm_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
        right_arm_angle = calculate_angle(keypoints[6], keypoints[8], keypoints[10])
        
        # 判断手臂是否高于肩部
        left_above = keypoints[9][1] < keypoints[5][1]  # Y坐标小表示更高
        right_above = keypoints[10][1] < keypoints[6][1]
        
        if left_above and right_above:
            return 'both_arms_up'
        elif left_above or right_above:
            return 'one_arm_up'
        else:
            return 'both_arms_down'
            
    def _classify_legs(self, keypoints):
        """第三组姿势分类：腿部姿势"""
        if not all(keypoints[i][2] > 0.5 for i in [11, 12, 13, 14, 15, 16]):
            return 'unknown'
            
        # 计算膝盖角度
        left_knee_angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        right_knee_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
        
        # 计算髋部高度与脚踝高度差
        hip_height = (keypoints[11][1] + keypoints[12][1]) / 2
        ankle_height = (keypoints[15][1] + keypoints[16][1]) / 2
        height_diff = hip_height - ankle_height
        
        # 判断是否坐姿
        is_sitting = height_diff < 0.3 * np.linalg.norm(keypoints[11][:2] - keypoints[12][:2])
        
        if is_sitting:
            return 'sitting'
        elif min(left_knee_angle, right_knee_angle) < self.angle_thresholds['squat_angle']:
            return 'kneeling_or_squatting'
        elif left_knee_angle > self.angle_thresholds['knee_angle'] and right_knee_angle > self.angle_thresholds['knee_angle']:
            return 'standing_straight'
        elif left_knee_angle > self.angle_thresholds['knee_angle'] or right_knee_angle > self.angle_thresholds['knee_angle']:
            return 'standing_one_leg'
        elif left_knee_angle < self.angle_thresholds['knee_angle'] and right_knee_angle < self.angle_thresholds['knee_angle']:
            return 'both_legs_bent'
        else:
            return 'one_leg_bent' 