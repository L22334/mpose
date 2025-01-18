import numpy as np

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
            spine_angle = np.degrees(np.arctan2(
                hip_mid[1] - shoulder_mid[1],
                hip_mid[0] - shoulder_mid[0]
            ))
            
            # 检查是否在合理范围内
            if not (160 <= abs(spine_angle) <= 200):
                confidence_factor *= 0.9
        
        return confidence_factor 