from collections import deque
import numpy as np

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