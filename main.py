import cv2
import sys
import time
import torch
from config import (
    VIDEO_CONFIG, 
    OPENPOSE_CONFIG,
    ANALYZER_CONFIG,
    DISPLAY_CONFIG
)
from src.core.pose_detector import PoseDetector
from src.core.pose_analyzer import PoseAnalyzer
from src.processors.keypoint_smoother import KeypointSmoother
from src.processors.pose_fusion import PoseFusion
from src.analyzers.stability import PostureStabilityAnalyzer
from src.analyzers.consistency import SkeletonConsistencyChecker
from src.analyzers.trajectory import KeypointTrajectoryAnalyzer
from src.analyzers.prior_knowledge import PosePriorKnowledge
from src.analyzers.pose_classifier import PoseClassifier
from src.utils.visualization import create_display_window
import os
from src.utils.logger import setup_logger
from collections import deque

class PoseDetectionSystem:
    def __init__(self):
        # 添加错误处理
        try:
            self.detector = PoseDetector(OPENPOSE_CONFIG)
            # 初始化检测器和分析器
            self.analyzer = PoseAnalyzer()
            self.smoother = KeypointSmoother(ANALYZER_CONFIG['SMOOTHER_WINDOW_SIZE'])
            self.pose_fusion = PoseFusion()
            self.stability_analyzer = PostureStabilityAnalyzer(
                ANALYZER_CONFIG['STABILITY_WINDOW_SIZE']
            )
            self.consistency_checker = SkeletonConsistencyChecker()
            self.trajectory_analyzer = KeypointTrajectoryAnalyzer(
                ANALYZER_CONFIG['TRAJECTORY_WINDOW_SIZE']
            )
            self.pose_prior = PosePriorKnowledge()
            self.pose_classifier = PoseClassifier()
            
            # 存储最后的处理结果
            self.last_results = None
            
            self.fps_history = deque(maxlen=30)  # 添加FPS历史记录
            self.processing_times = deque(maxlen=30)  # 添加处理时间历史
            
            # 添加帧跳过逻辑
            self.frame_count = 0
            self.process_every_n_frames = 2  # 每2帧处理一次
            
            # 添加缓存
            self.last_valid_results = None
            self.last_process_time = 0
            
        except Exception as e:
            print(f"初始化错误: {str(e)}")
            raise
        
    def __del__(self):
        """确保资源正确释放"""
        try:
            # 释放OpenPose资源
            if hasattr(self, 'detector'):
                self.detector.op_wrapper.stop()
        except Exception as e:
            print(f"资源释放错误: {str(e)}")
        
    def process_frame(self, frame):
        current_time = time.time()
        self.frame_count += 1
        
        # 检查是否需要处理这一帧
        if (self.frame_count % self.process_every_n_frames != 0 and 
            self.last_valid_results is not None and
            current_time - self.last_process_time < 0.1):  # 100ms内
            return self.last_valid_results
            
        # 处理新帧
        results = self._process_frame_internal(frame)
        if results is not None:
            self.last_valid_results = results
            self.last_process_time = current_time
            
        return results
        
    def _process_frame_internal(self, frame):
        """处理单帧图像"""
        try:
            start_time = time.time()
            
            # 添加帧有效性检查
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame")
            
            # 执行姿态检测
            detection_results = self.detector.detect(frame)
            
            # 使用prior knowledge增强检测结果
            if detection_results['keypoints']['yolo'] is not None:
                confidence_factor = self.pose_prior.check_pose_validity(
                    detection_results['keypoints']['yolo']
                )
                detection_results['keypoints']['yolo'][:, 2] *= confidence_factor
            
            # 初始化混合帧
            mixed_frame = frame.copy()
            
            # 融合和分析过程
            if (detection_results['keypoints']['yolo'] is not None and 
                detection_results['keypoints']['openpose'] is not None):
                
                # 融合关键点
                fused_keypoints = self.pose_fusion.fuse_keypoints(
                    detection_results['keypoints']['yolo'],
                    detection_results['keypoints']['openpose']
                )
                
                if fused_keypoints is not None:
                    # 平滑处理
                    smoothed_keypoints = self.smoother.update(fused_keypoints)
                    
                    if smoothed_keypoints is not None:
                        # 在混合帧上绘制骨架
                        from src.utils.visualization import draw_skeleton
                        draw_skeleton(mixed_frame, smoothed_keypoints, thickness=3)
                        
                        # 应用各种分析
                        smoothed_keypoints = self.trajectory_analyzer.analyze_trajectory(
                            smoothed_keypoints
                        )
                        
                        # 验证姿态一致性
                        consistency_issues = self.consistency_checker.check_consistency(
                            smoothed_keypoints
                        )
                        
                        # 更新姿态稳定性
                        self.stability_analyzer.add_pose(smoothed_keypoints)
                        stability_results = self.stability_analyzer.analyze_posture_stability()
                        
                        # 分析姿态
                        pose_analysis = self.analyzer.analyze_pose(smoothed_keypoints)
                        
                        # 在现有分析之后添加
                        pose_classification = self.pose_classifier.classify_pose(smoothed_keypoints)
                        
                        # 更新结果信息
                        self.last_results = {
                            'fps': f"{1.0 / (time.time() - start_time):.1f}",
                            'analysis': pose_analysis,
                            'consistency': consistency_issues,
                            'stability': stability_results['warnings'],
                            'pose_class': {
                                'posture': pose_classification['posture'],
                                'arms': pose_classification['arms'],
                                'legs': pose_classification['legs']
                            }
                        }
            
            # 更新性能指标
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            self.fps_history.append(1.0 / process_time)
            
            return (
                detection_results['frames']['openpose'],
                detection_results['frames']['yolo'],
                mixed_frame  # 现在返回带有骨架绘制的混合帧
            )
            
        except Exception as e:
            self.logger.error(f"处理帧时发生错误: {str(e)}")
            # 返回带有错误提示的帧
            error_frame = frame.copy()
            cv2.putText(error_frame, "Error: " + str(e),
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            return error_frame, error_frame, error_frame
            
    def get_last_results(self):
        """获取最近的处理结果"""
        return self.last_results if self.last_results else {}

def main():
    """主函数"""
    logger = setup_logger()
    logger.info("系统启动")
    
    # 检查视频输入配置
    if not VIDEO_CONFIG['USE_CAMERA'] and not os.path.exists(VIDEO_CONFIG['VIDEO_PATH']):
        print(f"错误：视频文件不存在 - {VIDEO_CONFIG['VIDEO_PATH']}")
        sys.exit(1)
        
    # 初始化视频捕获
    if VIDEO_CONFIG['USE_CAMERA']:
        cap = cv2.VideoCapture(VIDEO_CONFIG['CAMERA_ID'])
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 ID {VIDEO_CONFIG['CAMERA_ID']}")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(VIDEO_CONFIG['VIDEO_PATH'])
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {VIDEO_CONFIG['VIDEO_PATH']}")
            sys.exit(1)
            
    # 设置视频分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_CONFIG['WINDOW_WIDTH'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_CONFIG['WINDOW_HEIGHT'])
    
    # 创建窗口
    window_name = 'Pose Detection System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 创建姿态检测系统
    system = PoseDetectionSystem()
    
    running = True
    
    try:
        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            openpose_frame, yolo_frame, mixed_frame = system.process_frame(frame)
            
            # 创建显示
            display = create_display_window(
                openpose_frame, 
                yolo_frame, 
                mixed_frame, 
                system.get_last_results()
            )
            
            # 显示结果
            cv2.imshow(window_name, display)
            
            # 检查窗口状态和键盘输入
            key = cv2.waitKey(1) & 0xFF
            if (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1) or (key == ord('q')):
                running = False
                break
                
    except Exception as e:
        logger.error(f"系统错误: {str(e)}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("程序已正常终止")
        sys.exit(0)
        logger.info("系统关闭")

if __name__ == "__main__":
    main() 