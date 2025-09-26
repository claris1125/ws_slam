#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel          # ← PyTorch 2.6 안전 로드용
from torch.serialization import add_safe_globals         # ← PyTorch 2.6 안전 로드용
import torch


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

        # ===== Device decide =====
        if self.param_device:
            self.device = self.param_device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # ===== PyTorch 2.6 safe unpickling allowlist =====
        # (Ultralytics가 내부에서 torch.load() 할 때 DetectionModel 언피클 허용)
        add_safe_globals([DetectionModel])

        # ===== Model Load (YOLOv8) =====
        try:
            self.model = YOLO(self.model_path)
            # device 고정 + half 지원 시 활성화
            self.model.to(self.device)
            if "cuda" in self.device and torch.cuda.is_available():
                try:
                    self.model.fuse()  # 가능 시 레이어 fuse
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
            # imgsz나 half를 추가로 쓰고 싶으면 아래 kwargs에 넣으면 됨.
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
