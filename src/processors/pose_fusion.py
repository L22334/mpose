from collections import deque
import numpy as np

class PoseFusion:
    def __init__(self):
        # 更新权重配置
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
        
        self.confidence_threshold = 0.35
        self.position_threshold = 45
        self.history = deque(maxlen=5)
        self.performance_history = {
            'yolo': deque(maxlen=30),
            'openpose': deque(maxlen=30)
        }
        
    def validate_keypoints(self, keypoints, reference_points):
        """验证关键点的有效性"""
        if keypoints is None or reference_points is None:
            return False
            
        # 检查躯干关键点的有效性
        trunk_points = [5, 6, 11, 12]  # 肩部和髋部关键点
        trunk_valid = all(keypoints[i][2] > 0.6 for i in trunk_points)
        if not trunk_valid:
            return False
            
        # 计算和验证身体比例
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
        
        for key in ref_proportions:
            if key in test_proportions:
                if ref_proportions[key] > 1e-6:  # 避免除以接近零的值
                    ratio = test_proportions[key] / ref_proportions[key]
                    if not (0.7 <= ratio <= 1.3):
                        return False
                else:
                    return False
                    
        return True

    def fuse_keypoints(self, yolo_kpts, op_kpts):
        """融合YOLO和OpenPose的关键点"""
        if yolo_kpts is None and op_kpts is None:
            return None
            
        reference_kpts = yolo_kpts if yolo_kpts is not None else op_kpts
        fused_kpts = reference_kpts.copy()
        
        if yolo_kpts is None or op_kpts is None:
            return fused_kpts
            
        if not self.validate_keypoints(op_kpts, reference_kpts):
            return reference_kpts
            
        # 融合过程
        for joint_id in range(len(yolo_kpts)):
            yolo_conf = yolo_kpts[joint_id][2]
            op_conf = op_kpts[joint_id][2]
            
            if yolo_conf < self.confidence_threshold and op_conf < self.confidence_threshold:
                fused_kpts[joint_id] = yolo_kpts[joint_id]
                continue
                
            weights = self.joint_weights.get(joint_id, {'yolo': 0.7, 'openpose': 0.3})
            w_yolo = weights['yolo']
            w_op = weights['openpose']
            
            total_conf = yolo_conf + op_conf
            if total_conf > 0:
                base_yolo_weight = weights['yolo']
                conf_ratio = yolo_conf / total_conf
                w_yolo = min(0.9, max(base_yolo_weight, conf_ratio))
                w_op = 1 - w_yolo
            
            fused_pos = (w_yolo * yolo_kpts[joint_id][:2] + 
                        w_op * op_kpts[joint_id][:2])
            
            if np.linalg.norm(fused_pos - reference_kpts[joint_id][:2]) < self.position_threshold:
                fused_kpts[joint_id][:2] = fused_pos
                fused_kpts[joint_id][2] = 0.7 * yolo_conf + 0.3 * op_conf
        
        # 应用时间平滑
        if len(self.history) > 0:
            prev_kpts = self.history[-1]
            if self.validate_keypoints(prev_kpts, reference_kpts):
                alpha = 0.8
                for i in range(len(fused_kpts)):
                    if (prev_kpts[i][2] > 0.5 and fused_kpts[i][2] > 0.5 and
                        np.linalg.norm(fused_kpts[i][:2] - prev_kpts[i][:2]) < self.position_threshold):
                        fused_kpts[i][:2] = (alpha * fused_kpts[i][:2] + 
                                           (1-alpha) * prev_kpts[i][:2])
        
        self.history.append(fused_kpts.copy())
        return fused_kpts

    def update_performance(self, keypoints, source):
        """更新检测器性能历史"""
        if keypoints is None:
            return
            
        valid_confs = [kpt[2] for kpt in keypoints if kpt[2] > 0.5]
        if valid_confs:
            stability = np.mean(valid_confs)
            self.performance_history[source].append(stability) 