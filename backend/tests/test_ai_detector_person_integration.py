import sys
import types

import numpy as np

from backend.services.onnx_inference import OnnxBox, OnnxDetectionResult


def _make_bg_subtractor():
    return types.SimpleNamespace(apply=lambda frame: np.zeros(frame.shape[:2], dtype=np.uint8))


_cv2_stub = types.SimpleNamespace(
    createBackgroundSubtractorMOG2=lambda *args, **kwargs: _make_bg_subtractor(),
    rectangle=lambda *args, **kwargs: None,
    putText=lambda *args, **kwargs: None,
    polylines=lambda *args, **kwargs: None,
    circle=lambda *args, **kwargs: None,
    line=lambda *args, **kwargs: None,
    FONT_HERSHEY_SIMPLEX=0,
    data=types.SimpleNamespace(haarcascades=""),
    CascadeClassifier=lambda *args, **kwargs: None,
    imwrite=lambda *args, **kwargs: True,
    setNumThreads=lambda *args, **kwargs: None,
    imshow=lambda *args, **kwargs: None,
    imread=lambda *args, **kwargs: np.zeros((1, 1, 3), dtype=np.uint8),
    IMREAD_COLOR=1,
    IMREAD_GRAYSCALE=0,
    COLOR_BGR2RGB=0,
)

sys.modules.setdefault("cv2", _cv2_stub)

from backend.services.ai_detector import AIDetector
class _MockPoseResult:
    boxes = None
    keypoints = None


class _DummyDetector:
    def __init__(self, boxes, names):
        onnx_boxes = [OnnxBox(bbox, cls_idx, conf) for bbox, cls_idx, conf in boxes]
        self._result = OnnxDetectionResult(onnx_boxes)
        self.model = types.SimpleNamespace(names=names)
        self.names = names

    def __call__(self, frame, **kwargs):
        return [self._result]


class _DummyPose:
    def __init__(self, kpt_shape):
        inner_model = types.SimpleNamespace(kpt_shape=kpt_shape)
        self.model = types.SimpleNamespace(model=inner_model)

    def __call__(self, frame, **kwargs):
        return [_MockPoseResult()]


class _DummyBGSubtractor:
    def apply(self, frame):
        return np.zeros(frame.shape[:2], dtype=np.uint8)


def _create_detector(person_bbox, conf=0.9):
    detector = AIDetector.__new__(AIDetector)
    detector.camera_name = "test-camera"
    detector.detect_person = True
    detector.detect_car = False
    detector.bg_subtractor = _DummyBGSubtractor()
    detector.det_imgsz = 640
    detector.det_conf = 0.25
    detector.predict_device = None
    detector.pose_conf = 0.25
    detector.visualize = False
    detector.face_blur = False
    detector.face_cascade = None
    detector.face_detector = None
    detector.face_detector_kind = "none"
    detector.face_predict_device = None
    detector.face_conf = 0.0
    detector.phone_score_threshold = 0.5
    detector.phone_hand_dist_ratio = 0.4
    detector.phone_head_dist_ratio = 0.4
    detector.pose_only_score_threshold = 0.5
    detector.pose_only_head_ratio = 0.3
    detector.pose_wrists_dist_ratio = 0.3
    detector.pose_tilt_threshold = 0.2
    detector.score_smoothing = 0.0
    detector.employee_recognizer = None
    detector.phone_idx = None
    detector.last_car_event_time = 0.0
    detector.car_event_cooldown = 0.0
    detector.car_class_ids = set()
    detector.car_conf_threshold = 0.3
    detector.min_car_fg_ratio = 0.5
    detector._zones = []
    detector._face_warning_shown = False
    detector.device = "cpu"
    detector.face_device = "cpu"
    detector.face_detector_requested = "none"
    detector.face_device_preference = None
    detector.det_backend = "onnx"
    detector.det_names = {0: "person", 1: "cell phone"}
    detector.det = _DummyDetector([(person_bbox, 0, conf)], detector.det_names)
    detector.pose = _DummyPose((17, 3))
    return detector


def test_person_detection_without_pose_integration():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100, 120, 260, 400]
    detector = _create_detector(bbox, conf=0.8)

    phone_usage, best_conf, snapshot, vis, car_events, people_data = detector.process_frame(frame)

    assert phone_usage is False
    assert best_conf == 0.0
    assert vis is None
    assert car_events == []
    assert len(people_data) == 1

    person_info = people_data[0]
    assert person_info["pose_available"] is False
    assert person_info["detector_confidence"] == 0.8
    assert person_info["confidence"] == 0.8
    assert np.allclose(person_info["bbox"], np.array(bbox, dtype=np.float32))
    assert np.allclose(person_info["detector_bbox"], np.array(bbox, dtype=np.float32))
    assert snapshot is not None
    assert np.array_equal(snapshot, frame)
