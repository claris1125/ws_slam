#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2

from ultralytics import YOLO

# --- PyTorch 2.6 안전 로드 대응 ---
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals, safe_globals

# Ultralytics가 ckpt 언피클 시 참조하는 클래스 allowlist
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv as UConvTop
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck, C3
from ultralytics.nn.modules.head import DFL

# 전역 allowlist 등록
add_safe_globals([
    # torch 기본 모듈
    nn.Sequential, nn.SiLU, nn.Conv2d, nn.BatchNorm2d,
    # ultralytics 모듈
    DetectionModel, UConvTop, Conv, C2f, SPPF, Bottleneck, C3, DFL,
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
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "")  # "", "cpu", "cuda:0"

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.conf = float(self.get_parameter("conf").value)
        self.model_path = self.get_parameter("model_path").value
        self.param_device = (self.get_parameter("device").value or "").strip()

        # Device
        self.device = self.param_device if self.param_device else (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # ===== YOLO 로드 (torch.load 패치로 weights_only=False 강제) =====
        try:
            _orig_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                # 공식 yolov8 가중치만 사용한다는 전제에서 weights_only=False
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load
            try:
                # 안전 글로벌 컨텍스트도 한 번 더 감싸서 이중 안전
                with safe_globals([
                    nn.Sequential, nn.SiLU, nn.Conv2d, nn.BatchNorm2d,
                    DetectionModel, UConvTop, Conv, C2f, SPPF, Bottleneck, C3, DFL,
                ]):
                    self.model = YOLO(self.model_path)
            finally:
                torch.load = _orig_torch_load  # 반드시 원복

            self.model.to(self.device)
            try:
                self.model.fuse()
            except Exception:
                pass

        except Exception as e:
            self.get_logger().error(
                f"Failed to load YOLO model '{self.model_path}': {e}"
            )
            raise

        self.get_logger().info(
            f"YOLOv8 model: {self.model_path}, device: {self.device}, conf: {self.conf}"
        )

        # ===== Subs / Pubs =====
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.on_rgb, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, 10)

        self.pub_img = self.create_publisher(Image, "/yolo/debug_image", 10)

        self.last_depth = None
        self.camera_info = None

    def on_rgb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        try:
            results = self.model(frame, conf=self.conf, verbose=False, device=self.device)
            res = results[0]
            annotated = res.plot()  # np.ndarray(BGR)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        self.pub_img.publish(out)

    def on_depth(self, msg: Image):
        self.last_depth = msg

    def on_info(self, msg: CameraInfo):
        self.camera_info = msg


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
