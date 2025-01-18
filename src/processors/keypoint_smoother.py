from collections import deque
import numpy as np

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