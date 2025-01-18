from ..utils.geometry import calculate_angle, calculate_confidence
import numpy as np

class PoseAnalyzer:
    def __init__(self):
        self.last_analysis = None
        
    def analyze_pose(self, keypoints):
        """增强的姿态分析函数"""
        if keypoints is None:
            return []
            
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
                    
            # 计算整体置信度
            confidence = calculate_confidence(keypoints)
            results.append(f"Detection Confidence: {confidence:.2f}")
                    
        except Exception as e:
            print(f"姿态分析错误: {str(e)}")
            
        self.last_analysis = results
        return results

    def get_last_analysis(self):
        """获取最近的分析结果"""
        return self.last_analysis if self.last_analysis else [] 