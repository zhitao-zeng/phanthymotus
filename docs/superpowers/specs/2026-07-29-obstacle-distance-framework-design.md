# 正前方最近障碍物距离框架设计

## 1. 目标

在 PhanthyMotus perception 服务中新增一个可独立测试、可替换模型实现的
`obstacle_distance` 插件，将单张正前方 RGB 图像转换为一个以米为单位的最近障碍物距离。

正式 baseline 使用以下组合：

- 深度模型：Lifelong-MonoDepth `NK.pth.tar`
  - NYUv2 预测头负责室内 metric depth。
  - KITTI 预测头负责室外 metric depth。
- 室外实例分割：YOLO26n-seg。
- 室内后处理：固定 ROI 内有效深度的 P1 分位数。
- 室外后处理：目标类别 mask 内深度反投影到 ego 坐标系，计算目标到前保险杠的水平距离。

模型训练、微调、量化和权重产物由模型侧负责。本分支负责模型接口、业务后处理、
ROS/MCP 接入、离线评测、故障兜底、配置、测试和交接文档。

## 2. 设计边界

### 2.1 本分支负责

- 定义稳定的深度模型与实例分割模型接口。
- 提供不依赖真实权重的测试替身和常数诊断后端。
- 显式选择 NYU 或 KITTI 预测头，不使用上游原型最近邻自动路由。
- 实现室内 ROI、有效值过滤、深度钳位和 P1 统计。
- 实现室外类别过滤、mask 深度统计、相机反投影、ego 坐标变换和保险杠距离。
- 实现阈值判决、F1@1m、RMSE 和阈值扫描。
- 将插件注册到 `PerceptionBundle`，支持标准 `start`、`stop`、`info` 和 `config`
  生命周期。
- 保证 Git 中不新增大模型文件。

### 2.2 模型侧负责

- 提供 Lifelong-MonoDepth NK 的可加载产物。
- 实现框架定义的 `DepthBackend` 接口。
- 提供 YOLO26n-seg 或其挑战类别微调版本。
- 实现框架定义的 `InstanceSegmentationBackend` 接口。
- 给出模型输入尺寸、归一化方式、输出深度单位及输出插值规则。
- 若限制按文件体积计算，提供满足限制的 INT8 或其他压缩产物。

### 2.3 非目标

- 不在本分支中训练、蒸馏或微调模型。
- 不将 `.pth`、`.pt`、`.onnx` 或 TensorRT engine 提交到 Git。
- 不复用 Lifelong-MonoDepth 上游的 replay feature 原型路由。
- 不把室外常数距离作为正式 baseline。
- 不在本任务中修复既有 OCR 测试或 OCR 运行时问题。

## 3. 模块结构

新增代码按职责拆分：

```text
perception/
├── plugins/
│   ├── obstacle_distance.py
│   └── obstacle_distance_core/
│       ├── __init__.py
│       ├── contracts.py
│       ├── estimator.py
│       ├── geometry.py
│       ├── metrics.py
│       ├── postprocess.py
│       └── routing.py
├── tests/
│   ├── test_obstacle_distance_contract.py
│   ├── test_obstacle_distance_estimator.py
│   ├── test_obstacle_distance_geometry.py
│   ├── test_obstacle_distance_metrics.py
│   └── test_obstacle_distance_packaging.py
└── tools/
    └── evaluate_obstacle_distance.py
```

`obstacle_distance.py` 只负责 ROS 节点、订阅/发布和插件生命周期。数值逻辑全部位于
`obstacle_distance_core`，使离线评测和模型侧联调不依赖 ROS。

## 4. 模型接口

### 4.1 深度模型

```python
class DepthBackend(Protocol):
    def predict_depth(
        self,
        image_bytes: bytes,
        domain: SceneDomain,
        deadline_monotonic: float,
    ) -> DepthPrediction:
        ...
```

`SceneDomain` 仅允许 `indoor` 和 `vehicle`。正式 Lifelong-MonoDepth 适配器按如下映射：

- `indoor` → `outputs[0][0]`，即 NYU depth。
- `vehicle` → `outputs[1][0]`，即 KITTI depth。

`DepthPrediction` 包含：

- `depth_m`：二维浮点数组，单位为米。
- `uncertainty`：可选二维浮点数组。
- `source_height`、`source_width`：模型输出对应的图像尺寸。

框架拒绝非二维、空数组、全 NaN/Inf 或非米制输出。模型侧必须负责把 inverse depth、
disparity 或归一化输出转换成米制深度。

### 4.2 实例分割模型

```python
class InstanceSegmentationBackend(Protocol):
    def predict_instances(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> Sequence[InstanceMask]:
        ...
```

`InstanceMask` 包含：

- `class_name`：类别名称。
- `confidence`：置信度。
- `mask`：与原图对齐的二维布尔数组。

首版允许的 COCO 类别：

- `person`
- `car`
- `truck`
- `bus`
- `motorcycle`
- `bicycle`

挑战类别微调模型可进一步输出：

- `traffic_cone`
- `barrier`
- `construction_vehicle`
- `pushable_pullable`
- `debris`

类别映射通过配置传入，业务代码不硬编码 Ultralytics 数字 class id。

### 4.3 诊断后端

常数后端只用于：

- `0.5 m` 全正例提交，反解正例率。
- `5.0 m` 全负例提交，观察 RMSE 量级。
- 无真实权重时验证输入、输出、容器和评测链路。

配置必须显式设置 `mode: diagnostic_constant` 才能启用。正式配置使用
`mode: model`，模型加载失败时不得自动降级为常数。

## 5. 场景路由

场景路由按以下优先级：

1. 调用方显式提供 `scene_hint`。
2. 离线文件输入根据配置的后缀映射判断。
3. ROS 插件实例配置固定为 `indoor` 或 `vehicle`。

框架不根据图像内容猜测场景，也不读取 Lifelong-MonoDepth 上游 replay prototype。
若 `scene_mode: metadata` 但消息没有场景信息，当前帧返回结构化错误并走安全兜底；
不会静默选择预测头。

## 6. 数据流

### 6.1 室内

```text
JPEG/PNG
  → scene=indoor
  → Lifelong-MonoDepth NYU head
  → 深度图恢复到 640×480 坐标
  → ROI rows[0:300], cols[213:426]
  → 过滤非有限值与范围外深度
  → P1
  → distance_m
```

默认有效范围为 `[0.3, 10.0] m`，均可配置。ROI 按原图比例映射，避免输入尺寸变化时
把固定像素坐标直接套到错误分辨率。

若 ROI 内有效像素数不足 `min_valid_pixels`，该帧进入安全兜底。

### 6.2 室外

```text
JPEG
  ├→ Lifelong-MonoDepth KITTI head → metric depth
  └→ YOLO26n-seg → instance masks
                      ↓
          类别与置信度过滤
                      ↓
         mask 内有效深度 P1/P5
                      ↓
    使用相机内参反投影为相机坐标点
                      ↓
     使用外参变换到 ego 坐标系
                      ↓
  到前保险杠参考点的 x-y 水平距离
                      ↓
                 distance_m
```

室外不能简单地对整张深度图取最小值，否则路面、墙体、建筑和静态结构会造成误报。

`CameraCalibration` 必须提供：

- `fx`、`fy`、`cx`、`cy`
- 相机到 ego 的 4×4 刚体变换矩阵
- ego 坐标系中的前保险杠参考点，默认仅作为配置示例使用 `(3.412, 0.0)`

缺少有效标定时，正式模式拒绝室外估计。只有显式启用
`allow_approximate_vehicle_geometry` 时，才允许使用中心射线与相机到保险杠偏移的近似值；
输出必须标记 `approximate_geometry: true`。

## 7. 输出契约

ROS 输出 topic：

```text
{input_topic}/obstacle_distance
```

消息格式为 `data/json`：

```json
{
  "distance_m": 1.23,
  "near_obstacle": false,
  "decision_threshold_m": 1.0,
  "scene": "indoor",
  "status": "ok",
  "error_code": null,
  "fallback": false,
  "approximate_geometry": false,
  "latency_ms": 42.5,
  "timestamp": 1785312000.0
}
```

`near_obstacle` 使用可配置决策阈值，不要求与输出距离的 `1.0 m` 完全相同。距离值仍保持
模型和几何后处理得到的连续标量，供 RMSE 评测使用。

## 8. 错误处理与兜底

以下情况触发兜底：

- 图像为空、损坏或尺寸异常。
- 场景信息缺失或非法。
- 模型加载或推理失败。
- 推理超过软超时。
- 深度图为空、尺寸非法或没有足够有效像素。
- 室外没有目标类别实例。
- 室外缺少标定或坐标变换失败。
- 最终结果为 NaN、Inf、负数或超出配置范围。

兜底输出默认 `3.0 m`，独立的 `error_code` 必须包含机器可读错误码，例如
`invalid_image`、`missing_scene`、`model_error`、`no_valid_depth`、
`no_target_instance` 或 `missing_calibration`。兜底不会伪装成正常模型输出。

插件把单调时钟 deadline 下传给两个模型后端。模型适配器必须在预处理、推理和后处理
边界检查 deadline，并使用其运行时提供的超时或取消能力。通用 Python 层不创建无法取消
的推理线程来伪装硬超时；若底层运行时无法中断已经提交的 GPU kernel，当前调用结束后
返回 `timeout` 兜底，并在调用结束前持续拒绝新帧。插件同一实例只允许一个推理调用，
新帧在忙时按配置丢弃，避免请求堆积。

## 9. 配置

新增 `plugins.obstacle_distance` 配置，默认关闭，避免没有模型时影响现有服务：

```yaml
obstacle_distance:
  enabled: false
  mode: model
  scene_mode: metadata
  depth_backend: lifelong_nk
  segmentation_backend: yolo26n_seg
  depth_model_dir: /models/obstacle-distance/lifelong-nk
  segmentation_model_dir: /models/obstacle-distance/yolo26n-seg
  decision_threshold_m: 1.0
  fallback_distance_m: 3.0
  soft_timeout_s: 2.5
  min_interval_ms: 0
  indoor:
    roi: [0, 300, 213, 426]
    min_depth_m: 0.3
    max_depth_m: 10.0
    percentile: 1.0
    min_valid_pixels: 64
  vehicle:
    allowed_classes: [person, car, truck, bus, motorcycle, bicycle]
    min_confidence: 0.25
    percentile: 1.0
    allow_approximate_geometry: false
    calibration: {}
```

真实模型适配器未安装时，启用 `mode: model` 必须在插件初始化阶段给出明确错误。

## 10. 离线评测

`evaluate_obstacle_distance.py` 接受包含以下字段的 CSV：

```text
image_path,scene,gt_distance_m
```

输出：

- 样本数、有效预测数和失败率。
- F1@1m、precision、recall。
- RMSE。
- 正例率。
- 最优决策阈值及对应 F1。
- 按 `indoor`、`vehicle` 和错误码分组的统计。

工具不负责下载数据集，也不把验证数据加入 Git。

## 11. 测试策略

所有生产逻辑遵循 TDD：

- 协议测试：插件工具定义、topic 和输出 JSON 契约。
- 室内测试：ROI 比例映射、有效值过滤、P1、无效像素和最小有效像素数。
- 室外测试：类别过滤、mask 对齐、反投影、外参变换、保险杠距离和无目标兜底。
- 路由测试：显式 hint、后缀映射、固定实例和缺少元数据。
- 指标测试：F1、RMSE、正例率和阈值扫描。
- 生命周期测试：启动、停止、重复启动、繁忙丢帧和软超时。
- 集成测试：注入 fake depth/segmentation backend，端到端得到稳定标量。
- 打包测试：新代码不复制或追踪模型权重；新增文件均小于 1 MiB。

真实 Lifelong-MonoDepth 和 YOLO 权重的精度、速度和显存验证属于模型交接验收，不进入
不含权重的本地单元测试。

## 12. 模型与体积约束

官方 Lifelong-MonoDepth 仓库提供 `NK.pth.tar`，并说明其为 NYUv2 与 KITTI 双域模型。
官方 Ultralytics 文档列出 YOLO26n-seg 为 2.7M 参数，并支持 ONNX 与 TensorRT 导出。

组合参数量约 24.9M 的结论只有在 Lifelong-MonoDepth 参数量估算成立、且规则按参数量
计算时才适用。若规则按文件体积计算：

- FP32 和 FP16 产物很可能超过 30 MB。
- 模型侧必须提供 INT8 或其他满足限制的部署产物。
- 部署前必须测量最终下载文件和解压后 engine 的实际大小，不能只按参数量推算。

## 13. 已知基线问题

基点 `fork/feat/zengzhitao@241b72d` 在本任务开始前已有以下问题：

- `test_ocr_packaging.py` 仍期待旧的 PP-OCRv6 tiny 配置，但生产配置已切换到 small。
- `test_ocr_model_downloader.py` 仍期待 15 MiB 限制，但生产代码已改为 48 MiB。
- `ocr_runtime.py` 导入仓库中不存在的 `plugins.ocr_preprocess`。
- 当前本机 Python 3.14 环境未安装 NumPy，导致一项 OCR tiled test 无法运行。

这些问题不由本分支引入，也不在本设计范围内。新增障碍距离测试必须独立通过；最终报告
同时列出全量测试中的既有失败，避免把它们误报为本次回归。

## 14. 参考资料

- [Lifelong-MonoDepth 官方仓库](https://github.com/FreeformRobotics/Lifelong-MonoDepth)
- [Lifelong-MonoDepth 论文](https://arxiv.org/abs/2303.05050)
- [Ultralytics YOLO26 分割文档](https://github.com/ultralytics/ultralytics/blob/main/docs/en/tasks/segment.md)
- [Ultralytics YOLO26 模型表](https://github.com/ultralytics/ultralytics#segmentation-coco)
