# 关键点和骨架定义
OPENPOSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],  # 手臂
    [1, 8], [8, 9], [9, 10],  # 右腿
    [1, 11], [11, 12], [12, 13],  # 左腿
    [1, 0],  # 躯干到鼻子
    [0, 14], [14, 16],  # 右眼和右耳
    [0, 15], [15, 17]   # 左眼和左耳
]

YOLO_CONNECTIONS = {
    0: [1, 2, 5, 6],  # 鼻子连接到左右眼、左右肩
    1: [3],           # 左眼连接到左耳
    2: [4],           # 右眼连接到右耳
    5: [7, 11],       # 左肩连接到左肘和左髋
    6: [8, 12],       # 右肩连接到右肘和右髋
    7: [9],           # 左肘连接到左腕
    8: [10],          # 右肘连接到右腕
    11: [13],         # 左髋连接到左膝
    12: [14],         # 右髋连接到右膝
    13: [15],         # 左膝连接到左踝
    14: [16]          # 右膝连接到右踝
}

KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye", 2: "right_eye",
    3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle"
}

SKELETON_PAIRS = {
    'upper_arm_right': (6, 8),   # 右肩到右肘
    'lower_arm_right': (8, 10),  # 右肘到右腕
    'upper_arm_left': (5, 7),    # 左肩到左肘
    'lower_arm_left': (7, 9),    # 左肘到左腕
    'upper_leg_right': (12, 14), # 右髋到右膝
    'lower_leg_right': (14, 16), # 右膝到右踝
    'upper_leg_left': (11, 13),  # 左髋到左膝
    'lower_leg_left': (13, 15),  # 左膝到左踝
    'shoulder_width': (5, 6),    # 肩宽
    'hip_width': (11, 12),       # 髋宽
} 