#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

# YOLOv8
from ultralytics import YOLO

# PyTorch 2.6 안전 로드(weights_only=True 기본값) 대응
from torch.serialization import add_safe_globals
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv

# 신뢰 가능한 체크포인트 전제: 언피클 허용 목록 등록
add_safe_globals([
    DetectionModel,
    nn.Sequential, nn.SiLU, nn.Conv2d, nn.BatchNorm2d,
    C2f, SPPF, Conv,
])


class YoloDepthMapper(Node):
    def __init__(self):
        super().__init__('yolo_depth_mapper')
        self.bridge = CvBridge()

        # ===== Parameters =====
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("conf", 0.3)
        self.declare_parameter("model_path", "yolov8n.pt")   # 경량 기본 모델
        self.declare_parameter("device", "")                 # "", "cpu", "cuda:0" 등

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.conf = float(self.get_parameter("conf").value)
        self.model_path = self.get_parameter("model_path").value
        self.param_device = (self.get_parameter("device").value or "").strip()

        # Device 결정
        if self.param_device:
            self.device = self.param_device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # ===== Model Load (YOLOv8) =====
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            # Fuse는 가능한 경우만 시도
            try:
                self.model.fuse()
            except Exception:
                pass
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model '{self.model_path}': {e}")
            raise

        self.get_logger().info(
            f"YOLOv8 model: {self.model_path}, device: {self.device}, conf: {self.conf}"
        )

        # ===== Subscriptions =====
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.on_rgb, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, 10)

        # ===== Publishers =====
        self.pub_img = self.create_publisher(Image, "/yolo/debug_image", 10)

        self.last_depth = None
        self.camera_info = None

    def on_rgb(self, msg: Image):
        # BGR8 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # YOLOv8 추론
        try:
            results = self.model(
                frame,
                conf=self.conf,
                verbose=False,
                device=self.device
            )
            res = results[0]
            annotated = res.plot()  # np.ndarray (BGR)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        # 퍼블리시
        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        self.pub_img.publish(out)

    def on_depth(self, msg: Image):
        self.last_depth = msg  # 필요 시 깊이-픽셀 매칭/3D 좌표화에 사용

    def on_info(self, msg: CameraInfo):
        self.camera_info = msg  # 필요 시 intrinsics 사용


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
