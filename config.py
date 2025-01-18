# 视频输入配置
USE_CAMERA = False  # 设置为False则使用视频文件
CAMERA_ID = 0     # 摄像头ID
VIDEO_PATH = "video/output_video.mp4"  # 视频文件路径

# OpenPose配置
OPENPOSE_CONFIG = {
    "model_folder": "openpose/models",
    "model_pose": "COCO",
    "net_resolution": "-1x368",
    "number_people_max": 1,
    "disable_blending": False,
    "render_threshold": 0.05
}

# 分析器参数配置
SMOOTHER_WINDOW_SIZE = 5      # 平滑器窗口大小
STABILITY_WINDOW_SIZE = 30    # 稳定性分析窗口大小
TRAJECTORY_WINDOW_SIZE = 30   # 轨迹分析窗口大小

# 显示配置
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080 