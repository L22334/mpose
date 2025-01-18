import cv2
import numpy as np
from .constants import SKELETON_PAIRS

def draw_skeleton(frame, keypoints, thickness=2):
    """绘制骨架"""
    if keypoints is None:
        return
        
    # 定义骨架连接和颜色映射
    SKELETON_CONNECTIONS = {
        (5, 6): (0, 255, 0),   # 肩膀连接 - 绿色
        (5, 11): (255, 0, 0),  # 左躯干 - 红色
        (6, 12): (255, 0, 0),  # 右躯干 - 红色
        (11, 12): (0, 255, 0), # 髋部连接 - 绿色
        (5, 7): (255, 165, 0), # 左上臂 - 橙色
        (6, 8): (255, 165, 0), # 右上臂 - 橙色
        (7, 9): (255, 255, 0), # 左前臂 - 黄色
        (8, 10): (255, 255, 0),# 右前臂 - 黄色
        (11, 13): (0, 255, 255),# 左大腿 - 青色
        (12, 14): (0, 255, 255),# 右大腿 - 青色
        (13, 15): (0, 165, 255),# 左小腿 - 橙色
        (14, 16): (0, 165, 255) # 右小腿 - 橙色
    }
    
    # 绘制骨架连接
    for (start_joint, end_joint), color in SKELETON_CONNECTIONS.items():
        if (keypoints[start_joint][2] > 0.5 and 
            keypoints[end_joint][2] > 0.5):
            
            start_point = tuple(map(int, keypoints[start_joint][:2]))
            end_point = tuple(map(int, keypoints[end_joint][:2]))
            
            cv2.line(frame, start_point, end_point, (0, 0, 0), thickness + 2)
            cv2.line(frame, start_point, end_point, color, thickness)

    # 绘制关键点
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 6, (0, 0, 0), -1)
            
            if i in [5, 6, 11, 12]:  # 躯干关键点
                color = (0, 255, 0)
            elif i in [7, 8, 9, 10]:  # 手臂关键点
                color = (255, 165, 0)
            else:  # 腿部关键点
                color = (0, 255, 255)
                
            cv2.circle(frame, (x, y), 4, color, -1)

def create_display_window(openpose_frame, yolo_frame, mixed_frame, info_dict):
    """创建显示窗口"""
    height, width = openpose_frame.shape[:2]
    
    # 创建信息面板
    info_panel = create_info_panel(height, width)
    add_info_to_panel(info_panel, info_dict)
    
    # 创建上下两个部分
    top_row = np.hstack((openpose_frame, yolo_frame))
    bottom_row = np.hstack((mixed_frame, info_panel))
    
    # 合并成最终显示
    return np.vstack((top_row, bottom_row))

def create_info_panel(height, width):
    """创建信息面板"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel.fill(20)
    
    gradient = np.linspace(0, 50, width, dtype=np.uint8)
    for i in range(3):
        panel[:, :, i] = gradient
    
    cv2.rectangle(panel, (0, 0), (width, 150), (30, 30, 30), -1)
    cv2.putText(panel, "System Information", 
                (40, 100), cv2.FONT_HERSHEY_DUPLEX,
                3.5, (0, 255, 0), 5)
    cv2.line(panel, (40, 130), (width-40, 130), 
             (0, 255, 0), 5)
    
    return panel

def add_info_to_panel(panel, info_dict):
    """在信息面板上添加文本"""
    y_offset = 250
    
    for key, value in info_dict.items():
        if isinstance(value, list):
            for item in value:
                text = f"{item}" if key == "Analysis" else f"{key}: {item}"
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