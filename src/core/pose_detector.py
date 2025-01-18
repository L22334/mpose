import cv2
import torch
from ultralytics import YOLO
import openpose.pyopenpose as op
import numpy as np

class PoseDetector:
    def __init__(self, openpose_config):
        # 确保使用GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化YOLOv8
        self.yolo_model = YOLO('yolov8m-pose.pt')
        self.yolo_model.to(self.device)
        
        # OpenPose初始化
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(openpose_config)
        self.op_wrapper.start()
        
    def preprocess_frame(self, frame):
        """预处理帧以提高检测质量"""
        try:
            alpha = 1.1  # 对比度
            beta = 5     # 亮度
            
            processed = frame.copy()
            
            if processed is not None and len(processed.shape) == 3:
                mean_brightness = np.mean(processed)
                if mean_brightness < 100:
                    processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
                
                processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
                
            return processed
            
        except Exception as e:
            print(f"预处理帧时发生错误: {str(e)}")
            return frame
            
    def detect(self, frame):
        """执行姿态检测"""
        processed_frame = self.preprocess_frame(frame.copy())
        
        results = {
            'keypoints': {
                'yolo': None,
                'openpose': None
            },
            'frames': {
                'yolo': processed_frame.copy(),
                'openpose': processed_frame.copy()
            }
        }
        
        try:
            # YOLOv8-pose 检测
            yolo_results = self.yolo_model(processed_frame)
            if len(yolo_results) > 0 and len(yolo_results[0].keypoints) > 0:
                results['keypoints']['yolo'] = yolo_results[0].keypoints.data.cpu().numpy()[0]
                results['frames']['yolo'] = yolo_results[0].plot()
            
            # OpenPose处理
            datum = op.Datum()
            datum.cvInputData = processed_frame
            self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            if hasattr(datum, "poseKeypoints") and datum.poseKeypoints is not None:
                if len(datum.poseKeypoints) > 0:
                    results['keypoints']['openpose'] = np.delete(datum.poseKeypoints[0], -1, axis=0)
                    results['frames']['openpose'] = datum.cvOutputData
                    
        except Exception as e:
            print(f"姿态检测错误: {str(e)}")
            
        return results 