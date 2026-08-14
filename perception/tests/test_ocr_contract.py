import importlib
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))


def _install_ros_stubs():
    class FakeNode:
        def __init__(self, name):
            self.name = name

        def create_publisher(self, *args, **kwargs):
            return mock.Mock()

        def create_subscription(self, *args, **kwargs):
            return mock.Mock()

        def destroy_subscription(self, *args, **kwargs):
            return True

        def destroy_node(self):
            return True

    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = FakeNode
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = lambda **kwargs: kwargs
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(
        RELIABLE="RELIABLE", BEST_EFFORT="BEST_EFFORT"
    )
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="KEEP_LAST")
    rclpy.qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="VOLATILE")

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs.msg.CompressedImage = type("CompressedImage", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.String = type("String", (), {})

    sys.modules.update(
        {
            "rclpy": rclpy,
            "rclpy.node": rclpy.node,
            "rclpy.qos": rclpy.qos,
            "sensor_msgs": sensor_msgs,
            "sensor_msgs.msg": sensor_msgs.msg,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs.msg,
        }
    )


class OCRContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ros_stubs()
        cls.ocr = importlib.import_module("plugins.ocr")
        cls.ocr_runtime = importlib.import_module("plugins.ocr_runtime")

    @staticmethod
    def _jpeg(width, height):
        return (
            b"\xff\xd8\xff\xc0\x00\x11\x08"
            + height.to_bytes(2, "big")
            + width.to_bytes(2, "big")
            + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
        )

    def test_tool_contract_uses_compressed_images_and_json_results(self):
        tool = self.ocr.TOOLS[0]

        self.assertEqual(tool["name"], "ocr")
        self.assertEqual(
            tool["inputSchema"]["properties"]["action"]["enum"],
            ["start", "stop", "info", "config"],
        )
        self.assertEqual(
            tool["topic_in"],
            [{"format": "image/jpeg", "desc": "camera image input"}],
        )
        self.assertEqual(
            tool["topic_out"],
            [{"format": "data/json", "desc": "OCR result with text boxes"}],
        )
        self.assertEqual(
            tool["configSchema"]["properties"]["provider"]["enum"],
            ["rapidocr"],
        )
        properties = tool["configSchema"]["properties"]
        for removed_property in (
            "url",
            "key",
            "model",
            "backend",
            "fallback_backend",
            "fallback_model_dir",
            "device",
            "num_threads",
        ):
            self.assertNotIn(removed_property, properties)
        self.assertEqual(properties["max_side_len"]["default"], 1600)
        self.assertEqual(properties["det_unclip_ratio"]["default"], 0.7)
        self.assertEqual(properties["rec_min_score"]["default"], 0.9)
        self.assertEqual(
            properties["crop_refinement"]["default"],
            {
                "enabled": True,
                "min_score": 0.9,
                "min_gain": 0.12,
                "min_text_length": 2,
                "profiles": ["prefix_65", "upper_center", "upper_tight"],
            },
        )
        self.assertEqual(
            properties["empty_result_retry"]["default"],
            {
                "enabled": True,
                "det_thresh": 0.1,
                "det_box_thresh": 0.1,
            },
        )

    def test_camera_and_result_topics_use_appropriate_reliability(self):
        self.assertEqual(self.ocr._CAMERA_QOS["reliability"], "RELIABLE")
        self.assertEqual(self.ocr._RESULT_QOS["reliability"], "RELIABLE")
        self.assertEqual(self.ocr._CAMERA_QOS["depth"], 1)
        self.assertEqual(self.ocr._RESULT_QOS["depth"], 1)

    def test_output_topic_is_derived_from_input_topic(self):
        self.assertEqual(
            self.ocr._ocr_output_topic("/robot/camera/image"),
            "/robot/camera/image/ocr",
        )

    def test_default_provider_builds_local_adapter(self):
        expected = object()
        with mock.patch(
            "plugins.ocr.RapidOCRAdapter", return_value=expected
        ) as adapter:
            result = self.ocr._build_ocr_adapter(
                {
                    "provider": "rapidocr",
                    "model_dir": "/models/ocr/ppocrv6-tiny",
                    "device_id": 0,
                    "use_angle_cls": True,
                    "max_side_len": 1600,
                }
            )

        self.assertIs(result, expected)
        adapter.assert_called_once_with(
            model_dir="/models/ocr/ppocrv6-tiny",
            device_id=0,
            use_angle_cls=True,
            max_side_len=1600,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
            crop_refinement={},
            empty_result_retry={},
        )

    def test_adapter_signature_changes_with_crop_refinement(self):
        baseline = self.ocr._adapter_signature(
            {"provider": "rapidocr", "crop_refinement": {"enabled": True}}
        )
        disabled = self.ocr._adapter_signature(
            {"provider": "rapidocr", "crop_refinement": {"enabled": False}}
        )

        self.assertNotEqual(baseline, disabled)

    def test_adapter_signature_changes_with_empty_result_retry(self):
        baseline = self.ocr._adapter_signature(
            {"provider": "rapidocr", "empty_result_retry": {"enabled": True}}
        )
        disabled = self.ocr._adapter_signature(
            {"provider": "rapidocr", "empty_result_retry": {"enabled": False}}
        )

        self.assertNotEqual(baseline, disabled)

    def test_adapter_signature_changes_with_inference_tuning(self):
        baseline = {
            "provider": "rapidocr",
            "rec_min_score": 0.9,
            "det_unclip_ratio": 0.7,
        }

        for key, value in (
            ("rec_min_score", 0.8),
            ("enable_preprocess", False),
            ("det_thresh", 0.2),
            ("det_box_thresh", 0.4),
            ("det_unclip_ratio", 1.0),
        ):
            changed = {**baseline, key: value}
            with self.subTest(key=key):
                self.assertNotEqual(
                    self.ocr._adapter_signature(baseline),
                    self.ocr._adapter_signature(changed),
                )

    def test_adapter_signature_changes_with_device_id(self):
        first = self.ocr._adapter_signature(
            {"provider": "rapidocr", "device_id": 0}
        )
        second = self.ocr._adapter_signature(
            {"provider": "rapidocr", "device_id": 1}
        )

        self.assertNotEqual(first, second)

    def test_local_adapter_initialization_is_lazy_and_start_failure_is_fatal(self):
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", side_effect=RuntimeError("load failed")
        ) as build:
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr"}, object()
            )

            plugin.dispatch("ocr", {"action": "info"})
            plugin.dispatch("ocr", {"action": "config", "language": "en"})
            self.assertEqual(build.call_count, 0)

            with self.assertRaisesRegex(RuntimeError, "load failed"):
                plugin.dispatch(
                    "ocr",
                    {"action": "start", "input_topic": "/camera"},
                )

        build.assert_called_once_with(
            {"provider": "rapidocr", "language": "en"}
        )

    def test_tensorrt_adapter_loads_external_engines(self):
        with tempfile.TemporaryDirectory() as model_tmp:
            root = Path(model_tmp)
            for filename in ("det.engine", "rec.engine", "keys.txt"):
                (root / filename).write_bytes(b"model")

            with mock.patch(
                "plugins.ocr_runtime._TensorRTPipeline"
            ) as pipeline:
                adapter = self.ocr_runtime.RapidOCRAdapter(
                    str(root),
                    device_id=0,
                    use_angle_cls=False,
                )

        pipeline.assert_called_once_with(
            root,
            device_id=0,
            use_angle_cls=False,
            max_side_len=1600,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
            crop_refinement=self.ocr_runtime.CropRefinementConfig(),
            empty_result_retry=self.ocr_runtime.EmptyResultRetryConfig(),
        )
        pipeline.return_value.warm_up.assert_called_once_with()

    def test_tensorrt_adapter_requires_classifier_engine(self):
        with tempfile.TemporaryDirectory() as model_tmp:
            root = Path(model_tmp)
            for filename in ("det.engine", "rec.engine", "keys.txt"):
                (root / filename).write_bytes(b"model")
            with self.assertRaisesRegex(FileNotFoundError, "cls.engine"):
                self.ocr_runtime.RapidOCRAdapter(str(root))

    def test_tensorrt_adapter_loads_classifier_engine(self):
        with tempfile.TemporaryDirectory() as model_tmp:
            root = Path(model_tmp)
            for filename in (
                "det.engine", "rec.engine", "cls.engine", "keys.txt"
            ):
                (root / filename).write_bytes(b"model")

            with mock.patch(
                "plugins.ocr_runtime._TensorRTPipeline"
            ) as pipeline:
                adapter = self.ocr_runtime.RapidOCRAdapter(
                    str(root),
                    device_id=0,
                    use_angle_cls=True,
                )

        pipeline.assert_called_once_with(
            root,
            device_id=0,
            use_angle_cls=True,
            max_side_len=1600,
            rec_min_score=0.9,
            enable_preprocess=True,
            det_thresh=0.3,
            det_box_thresh=0.5,
            det_unclip_ratio=0.7,
            crop_refinement=self.ocr_runtime.CropRefinementConfig(),
            empty_result_retry=self.ocr_runtime.EmptyResultRetryConfig(),
        )
        pipeline.return_value.warm_up.assert_called_once_with()

    def test_tensorrt_classifier_rotates_confident_180_crop(self):
        import numpy as np

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._cls_thresh = 0.9
        pipeline._cls = mock.Mock()
        pipeline._cls.max_batch_size.return_value = 8
        pipeline._cls.run_uint8_batch.return_value = np.asarray(
            [[0.01, 0.99], [0.95, 0.05]], dtype=np.float32
        )
        upside_down = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        upright = np.arange(36, 72, dtype=np.uint8).reshape(3, 4, 3)

        result = pipeline._orient_crops([upside_down, upright])

        np.testing.assert_array_equal(
            result[0], upside_down[::-1, ::-1]
        )
        np.testing.assert_array_equal(result[1], upright)
        call = pipeline._cls.run_uint8_batch.call_args
        self.assertEqual(call.args[0].shape, (2, 48, 192, 3))
        self.assertEqual(call.args[1], (2, 3, 48, 192))

    def test_tensorrt_session_executes_supported_dynamic_shape(self):
        import numpy as np

        state = types.SimpleNamespace(
            shape=None,
            addresses={},
            executed=False,
            synchronized=False,
            allocations=[],
            frees=[],
            host_inputs=[],
            closed=False,
        )

        class FakeCuda:
            stream_handle = 123

            def __init__(self, device_id):
                self.device_id = device_id

            @staticmethod
            def malloc(size):
                pointer = 1000 + len(state.allocations)
                state.allocations.append((pointer, size))
                return pointer

            @staticmethod
            def free(pointer):
                state.frees.append(pointer)

            @staticmethod
            def copy_host_to_device(_pointer, array):
                state.host_inputs.append(array.copy())

            @staticmethod
            def copy_device_to_host(array, _pointer):
                array.fill(0)

            @staticmethod
            def synchronize():
                state.synchronized = True

            @staticmethod
            def close():
                state.closed = True

        class FakeContext:
            @staticmethod
            def set_input_shape(_name, shape):
                state.shape = shape
                return True

            @staticmethod
            def get_tensor_shape(_name):
                return (1, 1, 32, 32)

            @staticmethod
            def set_tensor_address(name, address):
                state.addresses[name] = address

            @staticmethod
            def execute_async_v3(stream):
                state.executed = stream == 123
                return True

        class FakeEngine:
            num_io_tensors = 2
            num_optimization_profiles = 1

            @staticmethod
            def create_execution_context():
                return FakeContext()

            @staticmethod
            def get_tensor_name(index):
                return ("x", "y")[index]

            @staticmethod
            def get_tensor_mode(name):
                return "input" if name == "x" else "output"

            @staticmethod
            def get_tensor_shape(_name):
                return (-1, 3, -1, -1)

            @staticmethod
            def get_tensor_profile_shape(_name, _index):
                return (
                    (1, 3, 32, 32),
                    (1, 3, 64, 64),
                    (1, 3, 128, 128),
                )

            @staticmethod
            def get_tensor_dtype(_name):
                return "float32"

        class FakeRuntime:
            def __init__(self, _logger):
                pass

            @staticmethod
            def deserialize_cuda_engine(_data):
                return FakeEngine()

        trt = types.ModuleType("tensorrt")
        trt.Logger = type("Logger", (), {
            "WARNING": 1,
            "__init__": lambda self, _level: None,
        })
        trt.Runtime = FakeRuntime
        trt.TensorIOMode = types.SimpleNamespace(
            INPUT="input", OUTPUT="output"
        )
        trt.nptype = lambda _dtype: np.float32

        with tempfile.TemporaryDirectory() as model_tmp:
            engine_path = Path(model_tmp) / "det.engine"
            engine_path.write_bytes(b"engine")
            with mock.patch.dict(sys.modules, {"tensorrt": trt}):
                with mock.patch.object(
                    self.ocr_runtime, "_CudaRuntime", FakeCuda
                ):
                    session = self.ocr_runtime._TensorRTModelSession(
                        engine_path,
                        device_id=0,
                        mean=(127.5, 127.5, 127.5),
                        normal=(1 / 127.5, 1 / 127.5, 1 / 127.5),
                    )
                    output = session.run_uint8(
                        np.zeros((32, 32, 3), dtype=np.uint8),
                        (1, 3, 32, 32),
                    )
                    session.run_uint8(
                        np.zeros((32, 32, 3), dtype=np.uint8),
                        (1, 3, 32, 32),
                    )
                    with self.assertRaisesRegex(ValueError, "outside profiles"):
                        session.run_uint8(
                            np.zeros((160, 160, 3), dtype=np.uint8),
                            (1, 3, 160, 160),
                        )
                    session.close()

        self.assertEqual(state.shape, (1, 3, 32, 32))
        self.assertEqual(set(state.addresses), {"x", "y"})
        self.assertTrue(state.executed)
        self.assertTrue(state.synchronized)
        self.assertEqual(len(state.allocations), 2)
        self.assertEqual(len(state.frees), 2)
        self.assertTrue(state.closed)
        self.assertEqual(len(state.host_inputs), 2)
        self.assertTrue(state.host_inputs[0].flags.c_contiguous)
        self.assertAlmostEqual(float(state.host_inputs[0][0, 0, 0, 0]), -1.0)
        self.assertEqual(output.shape, (1, 1, 32, 32))

    def test_tensorrt_session_selects_nearest_profile_and_batch_limit(self):
        session = object.__new__(self.ocr_runtime._TensorRTModelSession)
        session._profiles = [
            (
                (1, 3, 48, 320),
                (8, 3, 48, 320),
                (16, 3, 48, 320),
            ),
            (
                (1, 3, 48, 320),
                (2, 3, 48, 640),
                (4, 3, 48, 1024),
            ),
        ]

        self.assertEqual(session._select_profile((8, 3, 48, 320)), 0)
        self.assertEqual(session._select_profile((2, 3, 48, 640)), 1)
        self.assertEqual(session.max_batch_size(48, 320), 16)
        self.assertEqual(session.max_batch_size(48, 640), 4)
        with self.assertRaisesRegex(
            self.ocr_runtime.TensorRTShapeError, "outside profiles"
        ):
            session.max_batch_size(48, 2048)

    def test_tensorrt_session_fits_small_image_to_profile_canvas(self):
        session = object.__new__(self.ocr_runtime._TensorRTModelSession)
        session._profiles = [
            (
                (1, 3, 704, 1216),
                (1, 3, 960, 1280),
                (1, 3, 1600, 1600),
            )
        ]

        self.assertEqual(session.fit_input_image_shape(480, 1280), (704, 1280))
        self.assertEqual(session.fit_input_image_shape(640, 640), (704, 1216))
        self.assertEqual(session.fit_input_image_shape(720, 1280), (720, 1280))
        with self.assertRaisesRegex(
            self.ocr_runtime.TensorRTShapeError, "outside profiles"
        ):
            session.fit_input_image_shape(1700, 1280)

    def test_tensorrt_pipeline_pads_short_detector_input_and_crops_output(self):
        import numpy as np

        cv2_module = types.ModuleType("cv2")
        cv2_module.INTER_AREA = 1
        cv2_module.INTER_LINEAR = 2
        cv2_module.resize = mock.Mock(
            side_effect=AssertionError("1280x480 content must not be distorted")
        )

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._max_side_len = 1600
        pipeline._det = mock.Mock()
        pipeline._det.fit_input_image_shape.return_value = (704, 1280)
        pipeline._det.run_uint8.return_value = np.zeros(
            (1, 1, 704, 1280), dtype=np.float32
        )
        image = np.full((480, 1280, 3), 255, dtype=np.uint8)

        with mock.patch.dict(sys.modules, {"cv2": cv2_module}):
            prediction, image_shape = pipeline._run_detector(image)

        detector_input, tensor_shape = pipeline._det.run_uint8.call_args.args
        self.assertEqual(tensor_shape, (1, 3, 704, 1280))
        self.assertEqual(detector_input.shape, (704, 1280, 3))
        self.assertTrue(np.all(detector_input[:480] == 255))
        self.assertTrue(np.all(detector_input[480:] == 128))
        self.assertEqual(prediction.shape, (1, 1, 480, 1280))
        self.assertEqual(image_shape, (480, 1280))

    def test_tensorrt_pipeline_upscales_then_pads_small_square(self):
        import numpy as np

        cv2_module = types.ModuleType("cv2")
        cv2_module.INTER_AREA = 1
        cv2_module.INTER_LINEAR = 2
        cv2_module.resize = mock.Mock(
            side_effect=lambda _image, size, interpolation: np.full(
                (size[1], size[0], 3), 200, dtype=np.uint8
            )
        )

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._max_side_len = 1600
        pipeline._det = mock.Mock()
        pipeline._det.fit_input_image_shape.return_value = (704, 1216)

        with mock.patch.dict(sys.modules, {"cv2": cv2_module}):
            detector_input, content_shape = pipeline._detector_input(
                np.zeros((640, 640, 3), dtype=np.uint8)
            )

        cv2_module.resize.assert_called_once()
        self.assertEqual(cv2_module.resize.call_args.args[1], (704, 704))
        self.assertEqual(
            cv2_module.resize.call_args.kwargs["interpolation"],
            cv2_module.INTER_LINEAR,
        )
        self.assertEqual(content_shape, (704, 704))
        self.assertEqual(detector_input.shape, (704, 1216, 3))
        self.assertTrue(np.all(detector_input[:, :704] == 200))
        self.assertTrue(np.all(detector_input[:, 704:] == 128))

    def test_tensorrt_pipeline_batches_equal_width_crops_in_box_order(self):
        import numpy as np

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._enable_preprocess = False
        pipeline._rec_min_score = 0.9
        pipeline._det_postprocess = object()
        boxes = np.asarray(
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 0], [30, 0], [30, 10], [20, 10]],
                [[40, 0], [50, 0], [50, 10], [40, 10]],
            ],
            dtype=np.float32,
        )
        pipeline._run_detector = mock.Mock(
            return_value=(object(), (64, 64))
        )
        pipeline._postprocess_detection = mock.Mock(
            return_value=(boxes, [1.0] * 3)
        )
        pipeline._crop = mock.Mock(side_effect=[object(), object(), object()])
        pipeline._prepare_recognition_crop = mock.Mock(side_effect=[
            (np.zeros((48, 320, 3), dtype=np.uint8), 1.0),
            (np.zeros((48, 328, 3), dtype=np.uint8), 2.0),
            (np.zeros((48, 320, 3), dtype=np.uint8), 3.0),
        ])
        pipeline._rec = mock.Mock()
        pipeline._rec.max_batch_size.return_value = 8
        pipeline._rec.run_uint8_batch.side_effect = lambda images, _shape: (
            np.zeros((len(images), 2, 2), dtype=np.float32)
        )

        def decode(_prediction, _use_space_char, *, wh_ratio_list, **_kwargs):
            return (
                [(f"text-{int(ratio)}", 0.95) for ratio in wh_ratio_list],
                [],
            )

        pipeline._rec_decode = mock.Mock(side_effect=decode)

        result = pipeline.infer(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(
            [item["text"] for item in result],
            ["text-1", "text-2", "text-3"],
        )
        calls = pipeline._rec.run_uint8_batch.call_args_list
        self.assertEqual(calls[0].args[0].shape, (2, 48, 320, 3))
        self.assertEqual(calls[0].args[1], (2, 3, 48, 320))
        self.assertEqual(calls[1].args[0].shape, (1, 48, 328, 3))

    def test_tensorrt_pipeline_refines_low_confidence_box(self):
        import numpy as np

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._enable_preprocess = False
        pipeline._rec_min_score = 0.9
        pipeline._det_postprocess = object()
        pipeline._crop_refinement = self.ocr_runtime.CropRefinementConfig()
        box = np.asarray(
            [[0, 0], [100, 0], [100, 40], [0, 40]],
            dtype=np.float32,
        )
        pipeline._run_detector = mock.Mock(
            return_value=(object(), (64, 128))
        )
        pipeline._postprocess_detection = mock.Mock(
            return_value=(np.asarray([box]), [1.0])
        )
        pipeline._recognize_boxes = mock.Mock(side_effect=[
            [("noisy-extra", 0.7)],
            [
                ("correct", 0.95),
                ("upper", 0.91),
                ("tight", 0.92),
            ],
        ])

        result = pipeline.infer(np.zeros((64, 128, 3), dtype=np.uint8))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "correct")
        self.assertAlmostEqual(result[0]["score"], 0.95)
        self.assertEqual(result[0]["bbox"], [5.0, 0.0, 70.0, 40.0])
        refined_boxes = pipeline._recognize_boxes.call_args_list[1].args[1]
        self.assertEqual(len(refined_boxes), 3)

    def test_tensorrt_pipeline_rejects_refinement_without_minimum_gain(self):
        import numpy as np

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._enable_preprocess = False
        pipeline._rec_min_score = 0.9
        pipeline._det_postprocess = object()
        pipeline._crop_refinement = self.ocr_runtime.CropRefinementConfig()
        box = np.asarray(
            [[0, 0], [100, 0], [100, 40], [0, 40]],
            dtype=np.float32,
        )
        pipeline._run_detector = mock.Mock(
            return_value=(object(), (64, 128))
        )
        pipeline._postprocess_detection = mock.Mock(
            return_value=(np.asarray([box]), [1.0])
        )
        pipeline._recognize_boxes = mock.Mock(side_effect=[
            [("primary", 0.85)],
            [("candidate", 0.96), ("other", 0.94), ("tight", 0.93)],
        ])

        result = pipeline.infer(np.zeros((64, 128, 3), dtype=np.uint8))

        self.assertEqual(result, [])

    def test_tensorrt_empty_result_retry_reuses_detector_prediction(self):
        import numpy as np

        pipeline = object.__new__(self.ocr_runtime._TensorRTPipeline)
        pipeline._enable_preprocess = False
        prediction = object()
        primary_boxes = np.asarray(
            [[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32
        )
        retry_boxes = np.asarray(
            [[[20, 20], [40, 20], [40, 30], [20, 30]]], dtype=np.float32
        )
        pipeline._det_postprocess = object()
        pipeline._empty_result_retry_postprocess = object()
        pipeline._run_detector = mock.Mock(
            return_value=(prediction, (64, 64))
        )
        pipeline._postprocess_detection = mock.Mock(
            side_effect=[
                (primary_boxes, [0.8]),
                (retry_boxes, [0.2]),
            ]
        )
        retry_item = {"text": "桌牌", "bbox": [20, 20, 40, 30], "score": 0.99}
        pipeline._recognize_with_refinement = mock.Mock(
            side_effect=[[], [retry_item]]
        )

        result = pipeline.infer(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(result, [retry_item])
        pipeline._run_detector.assert_called_once()
        self.assertEqual(pipeline._postprocess_detection.call_count, 2)
        self.assertIs(
            pipeline._postprocess_detection.call_args_list[1].args[0],
            prediction,
        )

    def test_empty_result_retry_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(
            ValueError, "empty-result retry det_thresh"
        ):
            self.ocr_runtime.EmptyResultRetryConfig.from_mapping(
                {"det_thresh": -0.1}
            )

    def test_crop_refinement_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "unknown OCR crop refinement"):
            self.ocr_runtime.CropRefinementConfig.from_mapping(
                {"profiles": ["unknown"]}
            )

    def test_rapidocr_adapter_closes_runtime_once(self):
        adapter = object.__new__(self.ocr_runtime.RapidOCRAdapter)
        adapter._request_lock = threading.Lock()
        primary = mock.Mock()
        adapter._pipeline = primary

        adapter.close()
        adapter.close()

        primary.close.assert_called_once_with()
        self.assertIsNone(adapter._pipeline)

    def test_recognition_padding_is_neutral_before_normalization(self):
        import numpy as np

        cv2_module = types.ModuleType("cv2")
        cv2_module.INTER_LINEAR = 1
        cv2_module.resize = mock.Mock(
            return_value=np.full((48, 100, 3), 200, dtype=np.uint8)
        )
        crop = np.zeros((48, 100, 3), dtype=np.uint8)

        with mock.patch.dict(sys.modules, {"cv2": cv2_module}):
            padded, ratio = self.ocr_runtime._TensorRTPipeline \
                ._prepare_recognition_crop(crop)

        self.assertEqual(ratio, 100 / 48)
        self.assertEqual(padded.shape, (48, 320, 3))
        self.assertTrue(np.all(padded[:, :100] == 200))
        self.assertTrue(np.all(padded[:, 100:] == 128))

    def test_recognition_crop_is_bounded_by_engine_profile(self):
        import numpy as np

        cv2_module = types.ModuleType("cv2")
        cv2_module.INTER_LINEAR = 1
        cv2_module.resize = mock.Mock(
            side_effect=lambda _image, size, **_kwargs: np.zeros(
                (size[1], size[0], 3), dtype=np.uint8
            )
        )
        crop = np.zeros((48, 4096, 3), dtype=np.uint8)

        with mock.patch.dict(sys.modules, {"cv2": cv2_module}):
            padded, _ = self.ocr_runtime._TensorRTPipeline \
                ._prepare_recognition_crop(crop)

        self.assertEqual(padded.shape, (48, 2048, 3))

    def test_single_pass_uses_bounded_vips_overview_for_any_source_size(self):
        adapter = object.__new__(self.ocr_runtime.RapidOCRAdapter)
        adapter._max_side_len = 960
        adapter._probe_image_header = mock.Mock(
            return_value=(12000, 9000)
        )
        adapter._infer_image = mock.Mock(
            return_value=[
                {"text": "bounded", "bbox": [10, 20, 110, 60], "score": 0.9}
            ]
        )
        overview = types.SimpleNamespace(shape=(720, 960, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview",
            return_value=overview,
        ) as decode:
            result = adapter._recognize_single_pass(b"compressed-image")

        decode.assert_called_once_with(b"compressed-image", 960)
        self.assertEqual(result[0]["bbox"], [125, 250, 1375, 750])

    def test_vips_overview_decodes_directly_to_bounded_bgr_array(self):
        import numpy as np

        state = types.SimpleNamespace(thumbnail_args=None)

        class VipsImage:
            bands = 3

            @staticmethod
            def hasalpha():
                return False

            def colourspace(self, _space):
                return self

            @staticmethod
            def numpy():
                return np.zeros((720, 960, 3), dtype=np.uint8)

        class ImageFactory:
            @staticmethod
            def thumbnail_buffer(data, width, **kwargs):
                state.thumbnail_args = (data, width, kwargs)
                return VipsImage()

        pyvips = types.ModuleType("pyvips")
        pyvips.Image = ImageFactory
        with mock.patch.dict(sys.modules, {"pyvips": pyvips}):
            decoded = self.ocr_runtime.decode_vips_overview(
                b"compressed-image", 960
            )

        self.assertEqual(
            state.thumbnail_args,
            (b"compressed-image", 960, {"size": "down", "no_rotate": False}),
        )
        self.assertEqual(decoded.shape, (720, 960, 3))
        self.assertTrue(decoded.flags.c_contiguous)

    def test_inference_error_becomes_publishable_empty_payload(self):
        adapter = mock.Mock()
        adapter.recognize.side_effect = ValueError("invalid image")

        payload = self.ocr.recognize_to_payload(
            adapter, b"not-an-image", "zh", 123.0
        )

        self.assertEqual(
            payload,
            {
                "text": "",
                "items": [],
                "error": "invalid image",
                "timestamp": 123.0,
                "language": "zh",
            },
        )

    def test_inference_uses_locked_pipeline(self):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._inference_lock = mock.MagicMock()
        adapter._pipeline = mock.Mock()
        adapter._pipeline.infer.return_value = []
        image = object()

        result = adapter._infer_image(image)

        self.assertEqual(result, [])
        adapter._inference_lock.__enter__.assert_called_once_with()
        adapter._inference_lock.__exit__.assert_called_once()
        adapter._pipeline.infer.assert_called_once_with(image)

    def test_shared_adapter_serializes_complete_requests(self):
        first_entered = threading.Event()
        both_entered = threading.Event()
        release = threading.Event()
        state_lock = threading.Lock()
        state = {"active": 0, "max_active": 0}

        def recognize_single_pass(_image_bytes):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(
                    state["max_active"], state["active"]
                )
                if state["active"] == 1:
                    first_entered.set()
                if state["active"] == 2:
                    both_entered.set()
            release.wait(timeout=2)
            with state_lock:
                state["active"] -= 1
            return []

        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._request_lock = threading.Lock()
        adapter._recognize_single_pass = recognize_single_pass
        image_bytes = self._jpeg(4000, 3000)
        first = threading.Thread(target=adapter.recognize, args=(image_bytes,))
        second = threading.Thread(target=adapter.recognize, args=(image_bytes,))

        first.start()
        self.assertTrue(first_entered.wait(timeout=1))
        second.start()
        both_entered.wait(timeout=0.1)
        release.set()
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(state["max_active"], 1)

    def _single_pass_adapter(self, source_size, output, *, max_side=1600):
        adapter = object.__new__(self.ocr.RapidOCRAdapter)
        adapter._max_side_len = max_side
        adapter._probe_image_header = mock.Mock(return_value=source_size)
        adapter._inference_lock = threading.Lock()
        adapter._pipeline = mock.Mock()
        adapter._pipeline.infer.return_value = [
            {
                "text": text,
                "bbox": [
                    min(float(point[0]) for point in polygon),
                    min(float(point[1]) for point in polygon),
                    max(float(point[0]) for point in polygon),
                    max(float(point[1]) for point in polygon),
                ],
                "score": float(score),
            }
            for polygon, text, score in zip(
                output.boxes, output.txts, output.scores
            )
        ]
        return adapter

    def test_rapidocr_adapter_uses_bounded_overview_before_inference(self):
        decoded_image = types.SimpleNamespace(shape=(100, 200, 3))
        adapter = self._single_pass_adapter(
            (200, 100),
            types.SimpleNamespace(boxes=[], txts=(), scores=()),
        )

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview",
            return_value=decoded_image,
        ) as decode:
            result = adapter.recognize(b"jpeg-bytes")

        decode.assert_called_once_with(b"jpeg-bytes", 1600)
        adapter._pipeline.infer.assert_called_once_with(decoded_image)
        self.assertEqual(result, [])

    def test_large_image_overview_restores_source_bbox(self):
        adapter = self._single_pass_adapter(
            (4000, 3000),
            types.SimpleNamespace(
                boxes=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
                txts=("large",),
                scores=(0.8,),
            ),
        )
        overview = types.SimpleNamespace(shape=(750, 1000, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"image-bytes")

        self.assertEqual(result[0]["bbox"], [40, 80, 400, 200])

    def test_single_pass_scales_float_polygon_before_rounding(self):
        adapter = self._single_pass_adapter(
            (4000, 3000),
            types.SimpleNamespace(
                boxes=[
                    [
                        [10.8, 20.8],
                        [100.2, 20.8],
                        [100.2, 50.2],
                        [10.8, 50.2],
                    ]
                ],
                txts=("precise",),
                scores=(0.8,),
            ),
        )
        overview = types.SimpleNamespace(shape=(750, 1000, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"image-bytes")

        self.assertEqual(result[0]["bbox"], [43, 83, 401, 201])

    def test_small_image_keeps_source_coordinates(self):
        adapter = self._single_pass_adapter(
            (800, 600),
            types.SimpleNamespace(
                boxes=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
                txts=("small",),
                scores=(0.8,),
            ),
        )
        overview = types.SimpleNamespace(shape=(600, 800, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"image-bytes")

        self.assertEqual(result[0]["bbox"], [10, 20, 100, 50])

    def test_non_jpeg_overview_restores_source_bbox(self):
        adapter = self._single_pass_adapter(
            (4000, 3000),
            types.SimpleNamespace(
                boxes=[[[10, 10], [100, 10], [100, 40], [10, 40]]],
                txts=("png",),
                scores=(0.7,),
            ),
        )
        adapter._probe_image_header.return_value = (4000, 3000)
        overview = types.SimpleNamespace(shape=(1200, 1600, 3))

        with mock.patch(
            "plugins.ocr_runtime.decode_vips_overview", return_value=overview
        ):
            result = adapter.recognize(b"png-bytes")

        self.assertEqual(result[0]["bbox"], [25, 25, 250, 100])

    def test_frame_queue_keeps_only_latest_image(self):
        node = self.ocr._OCRNode("/camera", object())

        self.assertEqual(node._frame_queue.maxsize, 1)

    def test_language_only_instance_config_reuses_shared_adapter(self):
        shared_adapter = object()
        executor = mock.Mock()
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", return_value=shared_adapter
        ) as build:
            plugin = self.ocr.OCRPlugin(
                {
                    "provider": "rapidocr",
                    "model_dir": "/models/ocr/ppocrv6-tiny",
                    "language": "zh",
                },
                executor,
            )
            plugin.dispatch(
                "ocr",
                {"action": "config", "instance_id": "case-1", "language": "en"},
            )
            with mock.patch.object(self.ocr, "_OCRNode") as node_type:
                node_type.return_value.start.return_value = {"state": "running"}
                plugin.dispatch(
                    "ocr",
                    {
                        "action": "start",
                        "instance_id": "case-1",
                        "input_topic": "/camera",
                    },
                )

        self.assertEqual(build.call_count, 1)
        node_type.assert_called_once_with(
            "/camera",
            shared_adapter,
            "en",
            node_suffix="case_1",
            min_interval=0.0,
        )

    def test_instance_config_rejects_shared_inference_settings(self):
        with mock.patch("plugins.ocr._build_ocr_adapter", return_value=object()):
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr"}, mock.Mock()
            )

        with self.assertRaisesRegex(ValueError, "settings are shared"):
            plugin.dispatch(
                "ocr",
                {
                    "action": "config",
                    "instance_id": "case-1",
                    "rec_min_score": 0.8,
                },
            )

    def test_empty_shared_config_does_not_eagerly_load_adapter(self):
        shared_adapter = object()
        with mock.patch(
            "plugins.ocr._build_ocr_adapter", return_value=shared_adapter
        ) as build:
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr", "language": "zh"}, mock.Mock()
            )
            result = plugin.dispatch("ocr", {"action": "config"})

        self.assertEqual(build.call_count, 0)
        self.assertIsNone(plugin._adapter)
        self.assertFalse(result["adapter_loaded"])
        self.assertFalse(result["reused"])

    def test_shared_tuning_config_is_applied_on_first_start(self):
        tuned_adapter = object()
        with mock.patch(
            "plugins.ocr._build_ocr_adapter",
            return_value=tuned_adapter,
        ) as build:
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr", "rec_min_score": 0.9}, mock.Mock()
            )
            result = plugin.dispatch(
                "ocr", {"action": "config", "rec_min_score": 0.8}
            )
            self.assertEqual(build.call_count, 0)
            with mock.patch.object(self.ocr, "_OCRNode") as node_type:
                node_type.return_value.start.return_value = {"state": "running"}
                plugin.dispatch(
                    "ocr",
                    {"action": "start", "input_topic": "/camera"},
                )

        build.assert_called_once_with(
            {"provider": "rapidocr", "rec_min_score": 0.8}
        )
        self.assertIs(plugin._adapter, tuned_adapter)
        self.assertFalse(result["adapter_loaded"])
        self.assertFalse(result["reused"])

    def test_repeated_start_stop_reuses_one_ros_node(self):
        executor = mock.Mock()
        node = mock.Mock(_input_topic="/camera", state="idle")
        node.start.return_value = {"state": "running"}
        node.stop.return_value = {"state": "idle"}

        with mock.patch("plugins.ocr._build_ocr_adapter", return_value=object()), \
                mock.patch.object(
                    self.ocr, "_OCRNode", return_value=node
                ) as node_type:
            plugin = self.ocr.OCRPlugin({"provider": "rapidocr"}, executor)
            for _ in range(250):
                plugin.dispatch(
                    "ocr",
                    {
                        "action": "start",
                        "input_topic": "/camera",
                    },
                )
                plugin.dispatch("ocr", {"action": "stop"})

        node_type.assert_called_once()
        executor.add_node.assert_called_once_with(node)
        executor.remove_node.assert_not_called()
        node.destroy_node.assert_not_called()
        self.assertIs(plugin._nodes["/camera"], node)
        self.assertEqual(node.start.call_count, 250)
        self.assertEqual(node.stop.call_count, 250)

    def test_concurrent_starts_create_one_ocr_node(self):
        executor = mock.Mock()
        first_constructor_entered = threading.Event()
        release_first_constructor = threading.Event()

        def build_node(*_args, **_kwargs):
            first_constructor_entered.set()
            release_first_constructor.wait(timeout=1)
            node = mock.Mock(_input_topic="/camera", state="idle")
            node.start.return_value = {"state": "running"}
            return node

        errors = []

        def start():
            try:
                plugin.dispatch(
                    "ocr",
                    {
                        "action": "start",
                        "input_topic": "/camera",
                    },
                )
            except Exception as exc:
                errors.append(exc)

        with mock.patch("plugins.ocr._build_ocr_adapter", return_value=object()), \
                mock.patch.object(
                    self.ocr, "_OCRNode", side_effect=build_node
                ) as node_type:
            plugin = self.ocr.OCRPlugin({"provider": "rapidocr"}, executor)
            first = threading.Thread(target=start)
            second = threading.Thread(target=start)
            first.start()
            self.assertTrue(first_constructor_entered.wait(timeout=1))
            second.start()
            time.sleep(0.05)
            release_first_constructor.set()
            first.join(timeout=1)
            second.join(timeout=1)

        self.assertEqual(errors, [])
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        node_type.assert_called_once()
        executor.add_node.assert_called_once()

    def test_stale_worker_cannot_publish_after_restart(self):
        old_inference_started = threading.Event()
        release_old_inference = threading.Event()

        def recognize(_adapter, image_bytes, _language, _timestamp):
            if image_bytes == b"old":
                old_inference_started.set()
                release_old_inference.wait(timeout=1)
            return {"text": image_bytes.decode(), "items": []}

        node = self.ocr._OCRNode("/camera", object())
        with mock.patch("plugins.ocr.recognize_to_payload", side_effect=recognize):
            node.start()
            first_stop_event = node._stop_event
            node._image_cb(types.SimpleNamespace(data=b"old", format="jpeg"))
            self.assertTrue(old_inference_started.wait(timeout=1))

            old_worker = node._worker_thread
            with mock.patch.object(old_worker, "join", return_value=None):
                node.stop()
            node.start()
            second_stop_event = node._stop_event
            release_old_inference.set()
            time.sleep(0.05)

            node._image_cb(types.SimpleNamespace(data=b"new", format="jpeg"))
            deadline = time.time() + 1
            while node._pub.publish.call_count < 1 and time.time() < deadline:
                time.sleep(0.01)
            node.stop()

        self.assertIsNot(first_stop_event, second_stop_event)
        published = [json.loads(call.args[0].data)["text"] for call in node._pub.publish.call_args_list]
        self.assertEqual(published, ["new"])

    def test_topic_change_retires_node_without_destroying_it(self):
        executor = mock.Mock()
        old_node = mock.Mock(_input_topic="/camera/old")
        new_node = mock.Mock(_input_topic="/camera/new")
        old_node.start.return_value = {"state": "running"}
        old_node.stop.return_value = {"state": "idle"}
        new_node.start.return_value = {"state": "running"}

        with mock.patch("plugins.ocr._build_ocr_adapter", return_value=object()), \
                mock.patch.object(
                    self.ocr, "_OCRNode", side_effect=[old_node, new_node]
                ):
            plugin = self.ocr.OCRPlugin({"provider": "rapidocr"}, executor)
            plugin.dispatch(
                "ocr",
                {
                    "action": "start",
                    "instance_id": "case-1",
                    "input_topic": "/camera/old",
                },
            )
            plugin.dispatch(
                "ocr",
                {
                    "action": "start",
                    "instance_id": "case-1",
                    "input_topic": "/camera/new",
                },
            )

        old_node.stop.assert_called_once_with()
        executor.remove_node.assert_called_once_with(old_node)
        old_node.destroy_node.assert_not_called()
        self.assertEqual(plugin._retired_nodes, [old_node])
        self.assertIs(plugin._nodes["case-1"], new_node)

    def test_instance_config_updates_existing_node_without_removing_it(self):
        executor = mock.Mock()
        shared_adapter = object()
        node = mock.Mock(_input_topic="/camera")
        node.start.return_value = {"state": "running"}
        node.stop.return_value = {"state": "idle"}

        with mock.patch(
            "plugins.ocr._build_ocr_adapter", return_value=shared_adapter
        ), mock.patch.object(self.ocr, "_OCRNode", return_value=node):
            plugin = self.ocr.OCRPlugin(
                {"provider": "rapidocr", "language": "zh"}, executor
            )
            plugin.dispatch(
                "ocr",
                {
                    "action": "start",
                    "instance_id": "case-1",
                    "input_topic": "/camera",
                },
            )
            plugin.dispatch(
                "ocr",
                {"action": "config", "instance_id": "case-1", "language": "en"},
            )

        node.stop.assert_called_once_with()
        executor.remove_node.assert_not_called()
        node.destroy_node.assert_not_called()
        self.assertIs(node._adapter, shared_adapter)
        self.assertEqual(node._language, "en")

    def test_nodes_are_destroyed_only_during_final_shutdown(self):
        plugin = object.__new__(self.ocr.OCRPlugin)
        plugin._lifecycle_lock = threading.RLock()
        shared_adapter = mock.Mock()
        active = mock.Mock(_adapter=shared_adapter)
        retired = mock.Mock(_adapter=shared_adapter)
        plugin._nodes = {"case-1": active}
        plugin._retired_nodes = [retired, active]
        plugin._adapter = shared_adapter

        plugin.prepare_shutdown()

        active.stop.assert_called_once_with()
        retired.stop.assert_called_once_with()
        active.destroy_node.assert_not_called()
        retired.destroy_node.assert_not_called()

        plugin.destroy_nodes()

        active.destroy_node.assert_called_once_with()
        retired.destroy_node.assert_called_once_with()
        shared_adapter.close.assert_called_once_with()
        self.assertEqual(plugin._nodes, {})
        self.assertEqual(plugin._retired_nodes, [])
        self.assertIsNone(plugin._adapter)


if __name__ == "__main__":
    unittest.main()
