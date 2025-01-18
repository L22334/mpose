# 视频输入配置
VIDEO_CONFIG = {
    'USE_CAMERA': False,
    'CAMERA_ID': 0,
    'VIDEO_PATH': "video/test_1.mp4",
    'FRAME_WIDTH': 1920,
    'FRAME_HEIGHT': 1080,
    'TARGET_FPS': 30
}

# OpenPose配置
OPENPOSE_CONFIG = {
    "model_folder": "openpose/models",
    "model_pose": "COCO",
    "net_resolution": "-1x368",
    "number_people_max": 1,
    "disable_blending": False,
    "render_threshold": 0.05
}

# 分析器配置
ANALYZER_CONFIG = {
    'SMOOTHER_WINDOW_SIZE': 5,
    'STABILITY_WINDOW_SIZE': 30,
    'TRAJECTORY_WINDOW_SIZE': 30,
    'MIN_CONFIDENCE': 0.5,
    'MAX_DETECTION_AGE': 1.0,  # 秒
    'POSE_CLASSIFICATION': {
        'VERTICAL_SPINE_THRESHOLD': 15,
        'BEND_ANGLE_THRESHOLD': 45,
        'TURN_ANGLE_THRESHOLD': 30,
        'SHOULDER_ANGLE_THRESHOLD': 60,
        'KNEE_ANGLE_THRESHOLD': 150,
        'SQUAT_ANGLE_THRESHOLD': 120,
    }
}

# 显示配置
DISPLAY_CONFIG = {
    'WINDOW_WIDTH': 1920,
    'WINDOW_HEIGHT': 1080,
    'SHOW_FPS': True,
    'SHOW_DEBUG_INFO': True
} 