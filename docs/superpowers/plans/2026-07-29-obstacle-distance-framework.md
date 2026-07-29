# 正前方最近障碍物距离框架实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用
> `superpowers:subagent-driven-development`（推荐）或
> `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）
> 语法来跟踪进度。

**目标：** 在 perception 服务中新增模型可插拔的正前方最近障碍物距离插件，完成室内
ROI P1、室外实例 mask 几何距离、离线指标、ROS/MCP 生命周期和无大文件打包约束。

**架构：** 模型侧通过 `DepthBackend` 和 `InstanceSegmentationBackend` 协议注入
Lifelong-MonoDepth NK 与 YOLO26n-seg。纯 Python core 负责场景路由、数值校验、室内和
室外后处理、兜底与指标；ROS 插件只负责队列、生命周期和 JSON 发布。正式模式不允许
自动降级为常数，诊断常数模式必须显式开启。

**技术栈：** Python 3.10+、NumPy、ROS2 `rclpy`、MCP JSON-RPC、`unittest`。

**当前工作区测试解释器：**

```text
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

系统 `python3` 缺少 NumPy。实施和验证统一使用上面的 Python 3.12.13（NumPy 2.3.5），
不得为了本任务修改既有 OCR 依赖或全局 Python 环境。

---

## 文件总览

**创建：**

- `perception/plugin_dispatch.py`：支持包含下划线的插件前缀。
- `perception/plugins/obstacle_distance_core/__init__.py`：导出稳定公共 API。
- `perception/plugins/obstacle_distance_core/contracts.py`：数据类型、协议、错误码。
- `perception/plugins/obstacle_distance_core/routing.py`：显式场景解析。
- `perception/plugins/obstacle_distance_core/postprocess.py`：室内距离后处理。
- `perception/plugins/obstacle_distance_core/geometry.py`：室外 mask 与相机几何。
- `perception/plugins/obstacle_distance_core/estimator.py`：端到端业务编排和兜底。
- `perception/plugins/obstacle_distance_core/metrics.py`：F1、RMSE、阈值扫描。
- `perception/plugins/obstacle_distance_core/backend_loader.py`：模型工厂动态加载。
- `perception/plugins/obstacle_distance.py`：ROS/MCP 插件。
- `perception/tools/evaluate_obstacle_distance.py`：离线评测 CLI。
- `perception/docs/obstacle_distance.md`：模型交接、配置、标定和部署说明。
- `perception/tests/test_plugin_dispatch.py`：带下划线插件名回归测试。
- `perception/tests/test_obstacle_distance_contracts.py`：协议和路由测试。
- `perception/tests/test_obstacle_distance_postprocess.py`：室内 P1 测试。
- `perception/tests/test_obstacle_distance_geometry.py`：室外几何测试。
- `perception/tests/test_obstacle_distance_estimator.py`：编排和兜底测试。
- `perception/tests/test_obstacle_distance_metrics.py`：指标与 CLI 测试。
- `perception/tests/test_obstacle_distance_plugin.py`：ROS/MCP 生命周期测试。
- `perception/tests/test_obstacle_distance_packaging.py`：注册、配置和大文件守卫。

**修改：**

- `perception/main.py`：使用最长前缀分发并注册 `ObstacleDistancePlugin`。
- `perception/config.yaml`：加入默认关闭的 `obstacle_distance` 配置。
- `perception/README.md`：链接障碍距离插件文档。

**不修改：**

- OCR 模型、OCR 测试和 OCR 运行时。
- `perception/Dockerfile.jetson` 的模型下载步骤；现有 `COPY perception/plugins/`
  会自动包含新插件，真实权重由模型侧交付后再配置。

---

### 任务 1：支持带下划线的插件前缀

**文件：**

- 创建：`perception/plugin_dispatch.py`
- 创建：`perception/tests/test_plugin_dispatch.py`
- 修改：`perception/main.py`

- [ ] **步骤 1：编写最长前缀分发失败测试**

在 `perception/tests/test_plugin_dispatch.py` 中创建一个最小 fake plugin：

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Plugin:
    def __init__(self, prefix):
        self.PREFIX = prefix
        self.calls = []

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return {"prefix": self.PREFIX, "name": name}


class PluginDispatchTest(unittest.TestCase):
    def test_dispatch_uses_longest_prefix(self):
        from plugin_dispatch import dispatch_plugin

        plugins = [_Plugin("obstacle"), _Plugin("obstacle_distance")]
        result = dispatch_plugin(
            plugins,
            "obstacle_distance_obstacle_distance",
            {"action": "info"},
        )

        self.assertEqual(
            result,
            {"prefix": "obstacle_distance", "name": "obstacle_distance"},
        )

    def test_full_tool_name_preserves_matching_name(self):
        from plugin_dispatch import full_tool_name

        self.assertEqual(
            full_tool_name("obstacle_distance", "obstacle_distance"),
            "obstacle_distance",
        )
        self.assertEqual(
            full_tool_name("obstacle_distance", "info"),
            "obstacle_distance_info",
        )
```

- [ ] **步骤 2：运行测试并确认正确失败**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest perception/tests/test_plugin_dispatch.py -v
```

预期：`ModuleNotFoundError: No module named 'plugin_dispatch'`。

- [ ] **步骤 3：实现最长前缀路由**

在 `perception/plugin_dispatch.py` 实现：

```python
def full_tool_name(prefix: str, tool_name: str) -> str:
    return tool_name if tool_name == prefix else f"{prefix}_{tool_name}"


def dispatch_plugin(plugins, full_name: str, args: dict):
    ordered = sorted(
        plugins,
        key=lambda plugin: len(getattr(plugin, "PREFIX", "")),
        reverse=True,
    )
    for plugin in ordered:
        prefix = getattr(plugin, "PREFIX", "")
        if not prefix:
            continue
        if full_name == prefix:
            return plugin.dispatch(prefix, args)
        marker = f"{prefix}_"
        if full_name.startswith(marker):
            return plugin.dispatch(full_name[len(marker):], args)
    return None
```

修改 `perception/main.py`：

```python
from plugin_dispatch import dispatch_plugin, full_tool_name
```

`get_all_tools()` 使用 `full_tool_name()`，`dispatch()` 使用
`dispatch_plugin(self._plugins, full_name, args)`。

- [ ] **步骤 4：运行分发与主生命周期测试**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_plugin_dispatch.py \
  perception/tests/test_main_lifecycle.py -v
```

预期：全部通过。

- [ ] **步骤 5：提交**

```bash
git add perception/plugin_dispatch.py \
  perception/tests/test_plugin_dispatch.py perception/main.py
git commit -m "fix(perception): support underscored plugin prefixes"
```

---

### 任务 2：定义模型契约和显式场景路由

**文件：**

- 创建：`perception/plugins/obstacle_distance_core/__init__.py`
- 创建：`perception/plugins/obstacle_distance_core/contracts.py`
- 创建：`perception/plugins/obstacle_distance_core/routing.py`
- 创建：`perception/tests/test_obstacle_distance_contracts.py`

- [ ] **步骤 1：编写协议和路由失败测试**

测试覆盖：

```python
def test_scene_hint_has_highest_priority():
    assert resolve_scene(
        scene_hint="vehicle",
        source_name="frame.png",
        fixed_scene="indoor",
        suffix_map={".png": "indoor"},
    ) is SceneDomain.VEHICLE


def test_suffix_routes_offline_image():
    assert resolve_scene(
        source_name="frame.PNG",
        suffix_map={".png": "indoor", ".jpg": "vehicle"},
    ) is SceneDomain.INDOOR


def test_missing_scene_is_structured_error():
    with self.assertRaises(ObstacleDistanceError) as ctx:
        resolve_scene()
    self.assertEqual(ctx.exception.code, ErrorCode.MISSING_SCENE)
```

同时构造 `DepthPrediction`、`InstanceMask` 和 `CameraCalibration`，验证：

- scene 仅允许 `indoor`、`vehicle`。
- confidence 必须在 `[0, 1]`。
- 4×4 外参必须有 16 个有限值。
- backend protocol 的方法签名包含 `deadline_monotonic`。

- [ ] **步骤 2：运行测试并确认缺少模块**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_contracts.py -v
```

预期：因 `obstacle_distance_core` 尚不存在而失败。

- [ ] **步骤 3：实现数据契约**

`contracts.py` 定义：

```python
class SceneDomain(str, Enum):
    INDOOR = "indoor"
    VEHICLE = "vehicle"


class ErrorCode(str, Enum):
    INVALID_IMAGE = "invalid_image"
    MISSING_SCENE = "missing_scene"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    INVALID_DEPTH = "invalid_depth"
    NO_VALID_DEPTH = "no_valid_depth"
    NO_TARGET_INSTANCE = "no_target_instance"
    MISSING_CALIBRATION = "missing_calibration"
    INVALID_CALIBRATION = "invalid_calibration"


@dataclass(frozen=True)
class DepthPrediction:
    depth_m: object
    source_height: int
    source_width: int
    uncertainty: object | None = None


@dataclass(frozen=True)
class InstanceMask:
    class_name: str
    confidence: float
    mask: object


@dataclass(frozen=True)
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_ego: tuple[float, ...]
    bumper_xy: tuple[float, float]
```

定义 runtime-checkable `DepthBackend` 与 `InstanceSegmentationBackend` Protocol；
两者均接受 `deadline_monotonic`。`ObstacleDistanceError` 保存稳定的
`ErrorCode` 和可读 message。

`__init__.py` 只导出模型侧需要实现的公共类型和协议。

- [ ] **步骤 4：实现显式路由**

`routing.py` 实现优先级：

1. `scene_hint`
2. `source_name` 后缀映射
3. `fixed_scene`
4. 抛出 `MISSING_SCENE`

后缀统一转小写；非法 scene 抛 `MISSING_SCENE`，错误信息不能包含整张输入内容。

- [ ] **步骤 5：运行协议测试**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_contracts.py -v
```

预期：全部通过。

- [ ] **步骤 6：提交**

```bash
git add perception/plugins/obstacle_distance_core \
  perception/tests/test_obstacle_distance_contracts.py
git commit -m "feat(obstacle-distance): define backend contracts"
```

---

### 任务 3：实现室内 ROI P1 后处理

**文件：**

- 创建：`perception/plugins/obstacle_distance_core/postprocess.py`
- 创建：`perception/tests/test_obstacle_distance_postprocess.py`

- [ ] **步骤 1：编写室内失败测试**

使用 NumPy 构造确定性的 `480×640` 深度图，测试：

```python
def test_indoor_distance_uses_roi_percentile_not_global_minimum():
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    depth[450, 10] = 0.1
    depth[0:300, 213:426] = 2.0
    depth[0:30, 213:426] = 0.8

    result = indoor_distance_m(
        depth,
        source_size=(480, 640),
        roi=(0, 300, 213, 426),
        min_depth_m=0.3,
        max_depth_m=10.0,
        percentile=1.0,
        min_valid_pixels=64,
    )

    self.assertAlmostEqual(result, 0.8, places=5)
```

其他测试：

- 输入不是 `640×480` 时按比例映射 ROI。
- NaN、Inf、零和范围外深度被过滤。
- 有效像素不足抛 `NO_VALID_DEPTH`。
- `percentile=1` 明确不等于 `min()`。

- [ ] **步骤 2：运行测试并确认函数缺失**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_postprocess.py -v
```

预期：导入 `indoor_distance_m` 失败。

- [ ] **步骤 3：实现最小室内后处理**

`postprocess.py`：

```python
def indoor_distance_m(
    depth_m,
    *,
    source_size,
    roi,
    min_depth_m,
    max_depth_m,
    percentile,
    min_valid_pixels,
) -> float:
    depth = np.asarray(depth_m, dtype=np.float32)
    validate_depth_map(depth)
    rows, cols = scaled_roi(depth.shape, source_size, roi)
    values = depth[rows, cols]
    valid = values[
        np.isfinite(values)
        & (values >= min_depth_m)
        & (values <= max_depth_m)
    ]
    if valid.size < min_valid_pixels:
        raise ObstacleDistanceError(
            ErrorCode.NO_VALID_DEPTH,
            f"indoor ROI has {valid.size} valid pixels",
        )
    return float(np.percentile(valid, percentile))
```

`validate_depth_map()` 验证二维、非空和至少一个有限值；
`scaled_roi()` 使用 source size 比例映射并钳制边界。

- [ ] **步骤 4：运行室内测试**

使用包含 NumPy 的 Python：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_postprocess.py -v
```

预期：全部通过。

- [ ] **步骤 5：提交**

```bash
git add perception/plugins/obstacle_distance_core/postprocess.py \
  perception/tests/test_obstacle_distance_postprocess.py
git commit -m "feat(obstacle-distance): add indoor ROI percentile"
```

---

### 任务 4：实现室外实例 mask 与相机几何

**文件：**

- 创建：`perception/plugins/obstacle_distance_core/geometry.py`
- 创建：`perception/tests/test_obstacle_distance_geometry.py`

- [ ] **步骤 1：编写室外几何失败测试**

测试使用单位内参和 identity 外参，使期望值可手算：

```python
def test_vehicle_distance_uses_mask_and_bumper_reference():
    depth = np.full((3, 3), 20.0, dtype=np.float32)
    depth[1, 1] = 3.0
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    calibration = CameraCalibration(
        fx=1.0,
        fy=1.0,
        cx=1.0,
        cy=1.0,
        camera_to_ego=(
            0, 0, 1, 0,
            1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, 0, 1,
        ),
        bumper_xy=(1.0, 0.0),
    )

    distance = vehicle_distance_m(
        depth,
        instances=[
            InstanceMask("car", 0.9, mask),
            InstanceMask("dog", 0.99, np.ones((3, 3), dtype=bool)),
        ],
        calibration=calibration,
        allowed_classes={"car"},
        min_confidence=0.25,
        percentile=1.0,
        min_depth_m=0.3,
        max_depth_m=80.0,
    )

    self.assertAlmostEqual(distance, 2.0)
```

这里中心像素在相机坐标为 `(right=0, down=0, forward=3)`；测试外参将它映射到
ego 水平点 `(x=3, y=0)`，到保险杠 `(x=1, y=0)` 的距离为 `2 m`。

其他测试：

- 非目标类别不参与。
- 低置信度实例不参与。
- mask 与深度尺寸不一致抛 `INVALID_DEPTH`。
- 无目标实例抛 `NO_TARGET_INSTANCE`。
- 缺失标定抛 `MISSING_CALIBRATION`。
- 显式允许近似几何时，使用光轴深度减相机到保险杠偏移，并标记 approximate。
- 外参平移和旋转影响结果。
- mask 内 NaN、Inf 和范围外深度被过滤。

- [ ] **步骤 2：运行测试并确认缺少 geometry 模块**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_geometry.py -v
```

预期：模块导入失败。

- [ ] **步骤 3：实现相机点反投影**

`geometry.py` 使用列向量约定：

```python
x_camera = (u - cx) / fx * z
y_camera = (v - cy) / fy * z
z_camera = z
camera_points = np.stack(
    [x_camera, y_camera, z_camera, np.ones_like(z)],
    axis=0,
)
ego_points = camera_to_ego @ camera_points
horizontal = np.hypot(
    ego_points[0] - bumper_x,
    ego_points[1] - bumper_y,
)
```

相机外参负责把 `[right, down, forward]` 正确转换到 ego 坐标，不在业务代码中猜轴向。

- [ ] **步骤 4：实现实例过滤与距离分位数**

`vehicle_distance_m()`：

1. 校验 calibration。
2. 筛选 allowed class 和 confidence。
3. 合并符合条件的 mask。
4. 筛选有效深度。
5. 反投影并转换到 ego。
6. 对水平距离取配置分位数。

无实例和实例存在但无有效深度使用不同错误码：

- 无实例：`NO_TARGET_INSTANCE`
- mask 内无有效深度：`NO_VALID_DEPTH`

另实现 `approximate_vehicle_distance_m()`：只在配置明确允许时，对目标 mask 内有效
光轴深度减去 `camera_to_bumper_offset_m` 后取分位数，并把负值钳制到零。该函数不伪装
成标定几何，调用方必须设置 `approximate_geometry=True`。

- [ ] **步骤 5：运行几何测试**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_geometry.py -v
```

预期：全部通过。

- [ ] **步骤 6：提交**

```bash
git add perception/plugins/obstacle_distance_core/geometry.py \
  perception/tests/test_obstacle_distance_geometry.py
git commit -m "feat(obstacle-distance): add vehicle mask geometry"
```

---

### 任务 5：实现估计器、deadline 和结构化兜底

**文件：**

- 创建：`perception/plugins/obstacle_distance_core/estimator.py`
- 创建：`perception/plugins/obstacle_distance_core/backend_loader.py`
- 创建：`perception/tests/test_obstacle_distance_estimator.py`

- [ ] **步骤 1：编写端到端失败测试**

fake backend 记录 domain 和 deadline：

```python
class _DepthBackend:
    def __init__(self, depth):
        self.depth = depth
        self.calls = []

    def predict_depth(self, image_bytes, domain, deadline_monotonic):
        self.calls.append((image_bytes, domain, deadline_monotonic))
        return DepthPrediction(
            depth_m=self.depth,
            source_height=self.depth.shape[0],
            source_width=self.depth.shape[1],
        )
```

测试：

- indoor 调用 NYU domain 路径并得到 P1。
- vehicle 同时调用 depth 与 segmentation backend。
- `near_obstacle` 使用 `decision_threshold_m`。
- 模型异常转换为 `model_error` 兜底。
- deadline 已经过期转换为 `timeout` 兜底，且不调用模型。
- 正式模式缺少 backend 初始化失败。
- 诊断常数模式必须显式配置并输出 `status=diagnostic_constant`。
- NaN/Inf 最终结果转换为兜底。
- 兜底 JSON 包含 `fallback=true` 和独立的稳定 `error_code`。

- [ ] **步骤 2：运行测试并确认估计器缺失**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_estimator.py -v
```

预期：导入 estimator 失败。

- [ ] **步骤 3：实现结果类型与估计流程**

`DistanceResult`：

```python
@dataclass(frozen=True)
class DistanceResult:
    distance_m: float
    near_obstacle: bool
    decision_threshold_m: float
    scene: str
    status: str
    error_code: str | None
    fallback: bool
    approximate_geometry: bool
    latency_ms: float
    timestamp: float
```

`ObstacleDistanceEstimator.estimate()`：

1. 验证 image bytes 非空。
2. 解析 scene。
3. 计算 `deadline_monotonic = monotonic() + soft_timeout_s`。
4. 调用 depth backend。
5. indoor 走 `indoor_distance_m()`。
6. vehicle 调用 segmentation backend 后走 `vehicle_distance_m()`；仅当配置显式允许且
   calibration 缺失时走 `approximate_vehicle_distance_m()`。
7. 校验最终标量并生成 `DistanceResult`。
8. 捕获 `ObstacleDistanceError` 和模型异常，生成显式兜底结果。

不要使用线程池强制超时。deadline 下传给 backend；backend 返回后若已超时，也必须返回
`timeout` 兜底而不是迟到的正常结果。

- [ ] **步骤 4：实现模型工厂加载**

`backend_loader.py` 接受 `module.path:function_name`：

```python
def load_backend_factory(path: str):
    module_name, separator, function_name = path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "backend_factory must use module.path:function_name"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError("backend factory is not callable")
    return factory
```

工厂签名：

```python
depth_backend, segmentation_backend = factory(plugin_config)
```

正式模式工厂为空、返回对象不满足协议或 vehicle 缺 segmentation backend 时，在插件
初始化阶段失败。诊断模式不导入工厂。

- [ ] **步骤 5：运行估计器测试**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_estimator.py -v
```

预期：全部通过。

- [ ] **步骤 6：提交**

```bash
git add perception/plugins/obstacle_distance_core/estimator.py \
  perception/plugins/obstacle_distance_core/backend_loader.py \
  perception/tests/test_obstacle_distance_estimator.py
git commit -m "feat(obstacle-distance): orchestrate inference safely"
```

---

### 任务 6：实现离线指标与评测 CLI

**文件：**

- 创建：`perception/plugins/obstacle_distance_core/metrics.py`
- 创建：`perception/tools/evaluate_obstacle_distance.py`
- 创建：`perception/tests/test_obstacle_distance_metrics.py`

- [ ] **步骤 1：编写指标失败测试**

使用小数组验证：

```python
gt = [0.5, 0.8, 1.2, 2.0]
pred = [0.4, 1.1, 0.9, 2.2]
metrics = evaluate_predictions(gt, pred, threshold_m=1.0)

self.assertEqual(metrics.tp, 1)
self.assertEqual(metrics.fp, 1)
self.assertEqual(metrics.fn, 1)
self.assertAlmostEqual(metrics.precision, 0.5)
self.assertAlmostEqual(metrics.recall, 0.5)
self.assertAlmostEqual(metrics.f1, 0.5)
```

其他测试：

- RMSE 手算一致。
- 全正例、全负例时 precision/recall 无除零。
- 空输入明确报错。
- threshold scan 能找出已知最优阈值。
- 非有限预测计入失败率，不进入 RMSE。
- 按 scene 和 status 分组。
- CLI 读取 `image_path,scene,gt_distance_m`。
- `--mode diagnostic_constant --constant-distance-m 0.5` 无模型即可输出报告。
- model CLI 测试 patch `load_backend_factory()` 返回内存 fake factory，验证工厂被调用，
  不依赖仓库外测试模块。

- [ ] **步骤 2：运行测试并确认模块缺失**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_metrics.py -v
```

预期：导入 metrics 或 CLI 失败。

- [ ] **步骤 3：实现指标**

`evaluate_predictions()` 返回 dataclass，包含：

- `samples`
- `valid_predictions`
- `failures`
- `failure_rate`
- `tp`、`fp`、`fn`
- `precision`、`recall`、`f1`
- `rmse`
- `positive_rate`

判断规则统一为 `distance_m < threshold_m`。

`scan_thresholds()` 使用有限预测的唯一值中点和边界候选，不依赖 sklearn；F1 相同时
优先选择 precision 更高，再选择更接近 `1.0 m` 的阈值。

- [ ] **步骤 4：实现 CLI**

CLI 参数：

```text
--manifest PATH
--mode diagnostic_constant|model
--constant-distance-m FLOAT
--backend-factory MODULE:FUNCTION
--config PATH
--threshold-m FLOAT
--output PATH
```

每行读取图片 bytes，调用 estimator；输出 JSON 包含 overall、by_scene、by_status、
best_threshold 和逐样本 predictions。模型模式缺工厂时退出码非零并给出明确错误。

- [ ] **步骤 5：运行指标与 CLI 测试**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_metrics.py -v
```

预期：全部通过。

- [ ] **步骤 6：提交**

```bash
git add perception/plugins/obstacle_distance_core/metrics.py \
  perception/tools/evaluate_obstacle_distance.py \
  perception/tests/test_obstacle_distance_metrics.py
git commit -m "feat(obstacle-distance): add offline evaluation"
```

---

### 任务 7：接入 ROS/MCP 插件生命周期

**文件：**

- 创建：`perception/plugins/obstacle_distance.py`
- 创建：`perception/tests/test_obstacle_distance_plugin.py`
- 修改：`perception/main.py`

- [ ] **步骤 1：编写插件契约和生命周期失败测试**

沿用 `test_ocr_contract.py` 的 ROS stub 模式，测试：

- `TOOLS[0]["name"] == "obstacle_distance"`。
- action 为 `start/stop/info/config`。
- 输入格式 `image/jpeg`，输出格式 `data/json`。
- 输出 topic 为 `{input_topic}/obstacle_distance`。
- `PerceptionBundle` 在 enabled 时注册插件。
- start 必须提供 input topic 和可解析 scene。
- 重复 start 不创建第二个 node。
- stop 幂等。
- queue 容量为 1，繁忙时保留最新帧。
- worker 发布 `DistanceResult` JSON。
- generation 变化后旧 worker 不发布。
- `prepare_shutdown()` 和 `destroy_nodes()` 清理节点。
- model mode 初始化失败不会自动切常数。

- [ ] **步骤 2：运行测试并确认插件缺失**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_plugin.py -v
```

预期：`plugins.obstacle_distance` 不存在。

- [ ] **步骤 3：实现插件工具契约**

`TOOLS` 使用：

```python
{
    "name": "obstacle_distance",
    "type": "processor",
    "multiInstance": True,
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "info", "config"],
            },
            "input_topic": {"type": "string"},
            "scene_hint": {
                "type": "string",
                "enum": ["indoor", "vehicle"],
            },
        },
        "required": ["action"],
    },
    "topic_in": [{"format": "image/jpeg", "desc": "front camera image"}],
    "topic_out": [{"format": "data/json", "desc": "nearest obstacle distance"}],
}
```

- [ ] **步骤 4：实现 node**

`_ObstacleDistanceNode`：

- QoS 与 OCR image/result QoS 一致，depth=1。
- `queue.Queue(maxsize=1)` 保存最新帧。
- 单 worker 同步调用 estimator。
- `min_interval_ms` 使用 `stop_event.wait()`。
- stop 设置 generation、清空 queue、最多等待 3 秒。
- 推理还在执行时不启动第二个 worker。
- JSON 使用 `dataclasses.asdict(result)`。

场景从 node 固定的 `scene_hint` 传给 estimator；ROS `CompressedImage` 不包含文件名，
因此 `scene_mode=metadata` 时 start 必须显式提供 scene hint。

- [ ] **步骤 5：实现 plugin 与 bundle 注册**

`ObstacleDistancePlugin` 复用 OCR 的多实例生命周期模式，但不复制 OCR 模型逻辑：

- plugin 初始化时通过 `backend_loader` 创建 estimator。
- instance config 可覆盖 scene、threshold 和 min interval。
- `dispatch()` 支持 start、stop、info、config。
- `prepare_shutdown()` 停止所有 node。
- `destroy_nodes()` 销毁并清空节点。

在 `PerceptionBundle.__init__()` 添加：

```python
if plugins_cfg.get("obstacle_distance", {}).get("enabled", False):
    from plugins.obstacle_distance import ObstacleDistancePlugin
    self._plugins.append(
        ObstacleDistancePlugin(plugins_cfg["obstacle_distance"], executor)
    )
```

- [ ] **步骤 6：运行插件与分发测试**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_plugin.py \
  perception/tests/test_plugin_dispatch.py \
  perception/tests/test_main_lifecycle.py -v
```

预期：全部通过。

- [ ] **步骤 7：提交**

```bash
git add perception/plugins/obstacle_distance.py \
  perception/tests/test_obstacle_distance_plugin.py perception/main.py
git commit -m "feat(obstacle-distance): integrate perception plugin"
```

---

### 任务 8：配置、打包守卫和模型交接文档

**文件：**

- 创建：`perception/tests/test_obstacle_distance_packaging.py`
- 创建：`perception/docs/obstacle_distance.md`
- 修改：`perception/config.yaml`
- 修改：`perception/README.md`

- [ ] **步骤 1：编写打包失败测试**

测试：

```python
def test_default_config_keeps_obstacle_distance_disabled():
    config = (PERCEPTION_ROOT / "config.yaml").read_text(encoding="utf-8")
    self.assertIn(
        "  obstacle_distance:\n    enabled: false\n    mode: model",
        config,
    )
    self.assertIn("    depth_backend: lifelong_nk", config)
    self.assertIn("    segmentation_backend: yolo26n_seg", config)


def test_git_tracks_no_obstacle_model_artifacts():
    forbidden = {".pth", ".pt", ".onnx", ".engine", ".plan"}
    model_root = REPO_ROOT / "perception" / "models" / "obstacle-distance"
    self.assertFalse(model_root.exists())
    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and "obstacle" in path.name.lower():
            self.assertNotIn(path.suffix.lower(), forbidden)
            self.assertLess(path.stat().st_size, 1024 * 1024)
```

另检查：

- main.py 注册插件。
- Dockerfile 没有 `COPY perception/models/obstacle-distance`。
- README 链接正式文档。
- 文档包含两个 backend 协议、工厂签名、相机轴约定、配置示例和大小限制。

- [ ] **步骤 2：运行测试并确认配置缺失**

运行：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_obstacle_distance_packaging.py -v
```

预期：找不到 `plugins.obstacle_distance`。

- [ ] **步骤 3：添加默认关闭配置**

把规格中的完整 `plugins.obstacle_distance` 配置加入 `perception/config.yaml`。必须满足：

- `enabled: false`
- `mode: model`
- backend 名称锁定 `lifelong_nk` 和 `yolo26n_seg`
- model dir 指向 `/models/obstacle-distance/...`
- calibration 默认空字典
- `allow_approximate_geometry: false`

- [ ] **步骤 4：编写模型交接文档**

`perception/docs/obstacle_distance.md` 必须包含：

1. 正式模型拓扑和为何不使用室外常数。
2. `DepthBackend`、`InstanceSegmentationBackend` 完整签名。
3. `module.path:function_name` 工厂示例。
4. Lifelong-MonoDepth 输出头映射：NYU index 0、KITTI index 1。
5. YOLO COCO 首版类别和挑战微调新增类别。
6. 相机 `[right, down, forward]` 到 ego 的外参约定。
7. ROS/MCP start、stop、info、config 示例。
8. 离线 CLI 示例。
9. 模型目录挂载示例。
10. 参数量限制与文件体积限制的区别；30 MB 情况要求实际验证 INT8 产物。
11. 不得把权重提交 Git。
12. 已知 OCR 基线失败与本插件验收命令。

在 `perception/README.md` 增加文档链接。

- [ ] **步骤 5：运行新增完整测试**

使用包含 NumPy 和 PyYAML 的 Python：

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest \
  perception/tests/test_plugin_dispatch.py \
  perception/tests/test_obstacle_distance_contracts.py \
  perception/tests/test_obstacle_distance_postprocess.py \
  perception/tests/test_obstacle_distance_geometry.py \
  perception/tests/test_obstacle_distance_estimator.py \
  perception/tests/test_obstacle_distance_metrics.py \
  perception/tests/test_obstacle_distance_plugin.py \
  perception/tests/test_obstacle_distance_packaging.py \
  perception/tests/test_main_lifecycle.py -v
```

预期：全部通过。

- [ ] **步骤 6：运行仓库大小和格式检查**

```bash
git diff --check
git ls-files -z | xargs -0 stat -f '%z %N' | sort -nr | head -20
```

预期：

- `git diff --check` 退出码 0。
- 本分支新增文件均小于 1 MiB。
- 没有新增模型权重文件。

- [ ] **步骤 7：提交**

```bash
git add perception/config.yaml perception/README.md \
  perception/docs/obstacle_distance.md \
  perception/tests/test_obstacle_distance_packaging.py
git commit -m "docs(obstacle-distance): document model integration"
```

---

## 最终验证

- [ ] **步骤 1：重新运行新增测试套件**

运行任务 8 步骤 5 的完整命令，必须看到 0 failures、0 errors。

- [ ] **步骤 2：运行既有 ASR 基线**

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s perception/tests -p 'test_asr*.py'
```

预期：27 个测试通过。

- [ ] **步骤 3：运行 perception 全量测试并记录既有失败**

```bash
/Users/4paradigm/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s perception/tests -p 'test_*.py'
```

预期：障碍距离新增测试全部通过；OCR 基点既有失败单独列入交付报告，不得误报为本次
回归，也不得为了全绿而修改 OCR。

- [ ] **步骤 4：检查提交与工作区**

```bash
git status --short --branch
git log --oneline fork/feat/zengzhitao..HEAD
git diff --stat fork/feat/zengzhitao...HEAD
```

预期：工作区干净，提交按任务拆分，没有模型二进制。

- [ ] **步骤 5：进行最终规格与代码质量审查**

使用独立审查子代理对照：

- `docs/superpowers/specs/2026-07-29-obstacle-distance-framework-design.md`
- 本实现计划
- `fork/feat/zengzhitao...HEAD`

所有 Critical 和 Important 问题必须修复并重新验证。
