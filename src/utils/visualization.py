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
    frame_height, frame_width = openpose_frame.shape[:2]
    
    # 所有窗口使用相同的尺寸
    window_width = frame_width
    window_height = frame_height
    
    # 创建2x2布局的显示画面
    display_width = window_width * 2
    display_height = window_height * 2
    
    # 创建黑色背景
    display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # 调整所有帧的大小为统一尺寸
    openpose_resized = cv2.resize(openpose_frame, (window_width, window_height))
    yolo_resized = cv2.resize(yolo_frame, (window_width, window_height))
    mixed_resized = cv2.resize(mixed_frame, (window_width, window_height))
    
    # 创建信息面板
    info_panel = create_info_panel(window_height, window_width)
    add_info_to_panel(info_panel, info_dict)
    
    # 按2x2网格放置四个窗口
    display[0:window_height, 0:window_width] = openpose_resized                    # 左上
    display[0:window_height, window_width:] = yolo_resized                         # 右上
    display[window_height:, 0:window_width] = mixed_resized                        # 左下
    display[window_height:, window_width:] = info_panel                            # 右下
    
    return display

def create_info_panel(height, width):
    """Create info panel"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 纯黑背景
    panel.fill(0)
    
    # Add title area with dark background
    title_height = 80  # 增加标题高度
    cv2.rectangle(panel, (0, 0), (width, title_height), (20, 20, 20), -1)
    cv2.putText(panel, "POSE DETECTION", 
                (width//2 - 200, title_height-20), cv2.FONT_HERSHEY_DUPLEX,
                2.0, (0, 255, 0), 3)  # 增大标题字体
    
    return panel

def add_info_to_panel(panel, info_dict):
    """Add text to info panel"""
    if not info_dict:
        return
        
    height, width = panel.shape[:2]
    y_offset = 130  # 增加起始位置
    left_margin = 40  # 增加左边距
    line_spacing = 80  # 增加行间距
    
    # Add FPS
    if 'fps' in info_dict:
        cv2.putText(panel, f"FPS: {info_dict['fps']}", 
                   (left_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   2.0, (0, 255, 0), 2)
        y_offset += line_spacing
    
    # Add pose classification results
    if 'pose_class' in info_dict:
        pose_class = info_dict['pose_class']
        
        # Posture
        cv2.putText(panel, "POSTURE:", 
                   (left_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   2.0, (0, 255, 255), 2)
        y_offset += line_spacing - 25
        
        posture_text = pose_class['posture'].upper()
        cv2.putText(panel, f">{posture_text}", 
                   (left_margin + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   1.8, (0, 255, 0), 2)
        y_offset += line_spacing
        
        # Arms
        cv2.putText(panel, "ARMS:", 
                   (left_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   2.0, (0, 255, 255), 2)
        y_offset += line_spacing - 25
        
        arms_text = {
            'both_arms_down': 'BOTH DOWN',
            'one_arm_up': 'ONE UP',
            'both_arms_up': 'BOTH UP',
            'unknown': 'UNKNOWN'
        }.get(pose_class['arms'], pose_class['arms'])
        
        cv2.putText(panel, f">{arms_text}", 
                   (left_margin + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   1.8, (0, 255, 0), 2)
        y_offset += line_spacing
        
        # Legs
        cv2.putText(panel, "LEGS:", 
                   (left_margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   2.0, (0, 255, 255), 2)
        y_offset += line_spacing - 25
        
        legs_text = {
            'sitting': 'SITTING',
            'standing_straight': 'STANDING',
            'standing_one_leg': 'ONE LEG',
            'both_legs_bent': 'BOTH BENT',
            'one_leg_bent': 'ONE BENT',
            'kneeling_or_squatting': 'SQUAT',
            'unknown': 'UNKNOWN'
        }.get(pose_class['legs'], pose_class['legs'])
        
        cv2.putText(panel, f">{legs_text}", 
                   (left_margin + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   1.8, (0, 255, 0), 2) 