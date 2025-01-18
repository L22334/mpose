import numpy as np

def calculate_angle(p1, p2, p3=None):
    """
    计算角度
    如果提供两个点，计算与水平线的夹角
    如果提供三个点，计算三点形成的角度
    """
    if p3 is None:
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        return angle
    else:
        a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle

def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def calculate_confidence(keypoints):
    """计算关键点的平均置信度"""
    if keypoints is None:
        return 0.0
    valid_confs = [kpt[2] for kpt in keypoints if kpt[2] > 0.5]
    return np.mean(valid_confs) if valid_confs else 0.0 