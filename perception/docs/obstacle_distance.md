# 障碍物距离模型交接

本文是 `obstacle_distance` 插件的正式模型交接与部署说明。仓库提供稳定接口、业务后处理、
ROS/MCP 生命周期和离线评测，但**不包含**真实模型 adapter、模型权重、训练代码或数据集。
模型训练、转换、量化、INT8 校准和部署产物验证均由模型交接方另行完成。

## 1. 正式拓扑与边界

正式 `mode: model` 拓扑如下：

- 深度主干使用 Lifelong-MonoDepth 的双域模型 `NK.pth.tar`。
- `indoor` 必须选择 NYU 预测头，即 `outputs[0][0]`（head index 0）。
- `vehicle` 必须选择 KITTI 预测头，即 `outputs[1][0]`（head index 1）。
- 室内在恢复到原图坐标后的固定 ROI 中过滤有效深度，再取 P1。
- 室外由 YOLO26n-seg 产生实例 mask，并在**同一张 RGB 对应的同一坐标系深度图**
  上取目标深度；随后通过相机标定反投影到 ego 坐标，计算到前保险杠的水平距离。

首版室外 COCO 类别为 `person`、`car`、`truck`、`bus`、`motorcycle`、`bicycle`。
挑战类别微调版本可以增加 `traffic_cone`、`barrier`、`construction_vehicle`、
`pushable_pullable`、`debris`。类别名称由配置传递，不依赖 Ultralytics 数字 class id。

`diagnostic_constant` 仅用于链路诊断，例如 `0.5 m` 全正例和 `5.0 m` 全负例，从而验证
正负例、容器、输入输出与评测路径。它不是正式 baseline。`mode: model` 初始化、加载或
推理失败时绝不自动降级到常数；失败帧只产生带错误码的安全兜底结果。

## 2. Python backend 协议

模型交接方的对象必须原样满足以下两个协议签名。

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

`SceneDomain` 只有 `INDOOR = "indoor"` 和 `VEHICLE = "vehicle"`。返回的
`DepthPrediction` 包含二维米制 `depth_m`、正整数 `source_height` /
`source_width`，以及可选的二维 `uncertainty`。adapter 必须把 inverse depth、
disparity 或归一化输出转换成米，并按 `domain` 显式选取上节规定的 head。

```python
class InstanceSegmentationBackend(Protocol):
    def predict_instances(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> Sequence[InstanceMask]:
        ...
```

每个 `InstanceMask` 包含非空 `class_name`、区间 `[0, 1]` 内的 `confidence`，以及与
最终深度图坐标严格对齐的二维布尔 `mask`。室内路径不使用分割结果；室外路径会拒绝形状
不一致或非布尔 mask。

### Factory 签名与最小示例

配置项 `backend_factory` 使用 `module.path:function_name` 格式。被定位的函数应满足：

```python
def create_backends(
    config: Mapping[str, object],
) -> tuple[DepthBackend, InstanceSegmentationBackend]:
    ...
```

下面只展示交接形状，不代表仓库已有这些类或真实 adapter：

```python
from typing import Mapping

# 正式 Docker 运行时把 perception/plugins 复制为 /work/plugins。
try:
    from plugins.obstacle_distance_core.contracts import (
        DepthBackend,
        InstanceSegmentationBackend,
    )
except ModuleNotFoundError:
    # 从仓库根目录运行离线 CLI 时使用 perception.plugins 命名空间。
    from perception.plugins.obstacle_distance_core.contracts import (
        DepthBackend,
        InstanceSegmentationBackend,
    )


def create_backends(
    config: Mapping[str, object],
) -> tuple[DepthBackend, InstanceSegmentationBackend]:
    depth = LifelongNKAdapter(
        model_dir=str(config["depth_model_dir"]),
        indoor_head_index=0,
        vehicle_head_index=1,
    )
    segmentation = Yolo26nSegAdapter(
        model_dir=str(config["segmentation_model_dir"]),
    )
    return depth, segmentation
```

实际 adapter 也必须用上面选中的同一 contracts 模块构造 `DepthPrediction` 和
`InstanceMask`，不要同时加载 `plugins...` 与 `perception.plugins...` 两份模块；
estimator 会校验 `InstanceMask` 的运行时类型。生产容器和仓库离线 CLI 应分别验证
factory 可导入，并保持统一的可导入包布局。

例如把该函数放在 `company_perception.obstacle_backends` 中，配置为：

```yaml
backend_factory: "company_perception.obstacle_backends:create_backends"
```

Python 适配层会自省两个 backend 方法的可调用签名，并要求 factory 返回恰好两个
backend（深度在前、分割在后）。模块导入、factory 执行、返回值或签名验证中的异常会被
收敛为初始化错误；单帧模型异常会被 estimator 收敛为结构化 `model_error`。adapter
仍须在预处理、推理和后处理边界主动检查 `deadline_monotonic`，并利用底层运行时的取消
或超时能力。

默认 `backend_factory: ""` 之所以安全，仅因为 `enabled: false`，此时
`PerceptionBundle` 不构造插件。若把 `enabled` 改为 `true` 且保持 `mode: model`，
必须同时提供有效的 `module.path:function_name`；缺失会在插件初始化时明确失败。系统
不会用常数距离掩盖交接缺失。

## 3. 默认配置

以下字段与 `perception/config.yaml` 一致：

```yaml
plugins:
  obstacle_distance:
    enabled: false
    mode: model
    backend_factory: ""
    scene_mode: metadata
    fixed_scene: ""
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
      min_depth_m: 0.3
      max_depth_m: 80.0
      camera_to_bumper_offset_m: 1.0
      allow_approximate_geometry: false
      calibration: {}
```

室内 ROI 是源图 `rows[0:300], cols[213:426]`，会按深度输出尺寸同比缩放；有效范围为
`[0.3, 10.0] m`，至少需要 64 个有效像素。室外有效范围为 `[0.3, 80.0] m`，
置信度下限为 `0.25`，两条路径的默认统计量均为 P1。

## 4. 深度、mask 与车辆坐标

深度输出和实例 mask 必须对应同一张输入图，并在进入核心几何逻辑前对齐到相同的
`height × width` 像素坐标。任何 resize、letterbox padding 和裁剪都必须由 adapter
准确逆变换；不能把网络输入坐标的 mask 直接套到原图深度上。

相机坐标固定为 `[right, down, forward]`：

- `x_camera` 向右；
- `y_camera` 向下；
- `z_camera` 向前，且深度值就是 `z_camera` 的米数。

`camera_to_ego` 是包含 16 个有限数的 4×4 row-major 刚体齐次矩阵，最后一行必须是
`[0, 0, 0, 1]`，旋转部分必须是 proper orthogonal rotation。像素先按
`fx, fy, cx, cy` 反投影为相机点，再左乘该矩阵得到 ego 点。`bumper_xy` 是 ego
坐标系中的前保险杠参考点 `[x, y]`；输出距离是目标 ego 点与该参考点在 ego x-y
平面上的距离分位数。

下面是结构示例，数值必须由实车标定替换，不能照抄为生产标定：

```yaml
vehicle:
  calibration:
    fx: 900.0
    fy: 900.0
    cx: 640.0
    cy: 360.0
    camera_to_ego:
      [0.0, 0.0, 1.0, 1.20,
       -1.0, 0.0, 0.0, 0.00,
       0.0, -1.0, 0.0, 1.50,
       0.0, 0.0, 0.0, 1.00]
    bumper_xy: [3.412, 0.0]
```

正式 vehicle 模式缺少 `calibration` 时，结果必须为 `missing_calibration` 错误和
安全兜底距离。只有显式设置 `allow_approximate_geometry: true`，才会使用
`camera_to_bumper_offset_m` 做中心射线近似；此时成功输出也必须标记
`approximate_geometry: true`，不能冒充正式标定几何。

## 5. ROS/MCP 生命周期与输出

工具名为 `obstacle_distance`，支持 `start`、`stop`、`info`、`config`。以下为 MCP
arguments JSON 示例。

metadata 场景启动：

```json
{
  "action": "start",
  "instance_id": "front_vehicle",
  "input_topic": "/front_camera/image/compressed",
  "scene_hint": "vehicle"
}
```

停止单实例：

```json
{
  "action": "stop",
  "instance_id": "front_vehicle"
}
```

查询单实例：

```json
{
  "action": "info",
  "instance_id": "front_vehicle"
}
```

停止状态下更新实例配置：

```json
{
  "action": "config",
  "instance_id": "front_vehicle",
  "scene_hint": "vehicle",
  "decision_threshold_m": 1.2,
  "min_interval_ms": 100
}
```

运行时 `config` 只支持 `scene_hint`、`decision_threshold_m` 和
`min_interval_ms`；backend、model dir、mode 与 factory 不能热更新。已有节点会先
停止再应用配置。

当 `scene_mode: metadata` 时，每次 `start` 必须显式给出 `scene_hint`，或先为该
`instance_id` 配置 scene；缺失时启动失败，不会猜场景。使用固定实例时，把
`scene_mode` 设为 `fixed` 并把 `fixed_scene` 设为 `indoor` 或 `vehicle`；
显式 `scene_hint` 始终优先。离线 estimator 还支持 `scene_hint`、文件后缀映射、
`fixed_scene` 的顺序路由。

输入为 `sensor_msgs/CompressedImage`，输出 topic 为：

```text
{input_topic}/obstacle_distance
```

输出格式为 `data/json`，字段如下：

```json
{
  "distance_m": 1.23,
  "near_obstacle": false,
  "decision_threshold_m": 1.0,
  "scene": "vehicle",
  "status": "ok",
  "error_code": null,
  "fallback": false,
  "approximate_geometry": false,
  "latency_ms": 42.5,
  "timestamp": 1785312000.0
}
```

`near_obstacle` 使用严格小于 `decision_threshold_m`。失败帧的 `status` 为
`error`、`fallback` 为 `true`，默认 `distance_m` 为 `3.0`，并保留机器可读
`error_code`；诊断常数的 `status` 为 `diagnostic_constant`。

deadline 是单调时钟软期限。通用 Python 层不会创建不可取消线程来伪装硬超时；底层
若无法中断已提交的 GPU kernel，只能在调用返回后发布 `timeout` 兜底。一个插件对象
共享一把推理锁，因此其全部实例单插件串行推理。每个节点的队列最多保留一帧，另有一帧
可能正在等待推理锁或执行推理，所以总在途帧数有界，忙时不会形成无界堆积。`stop` 会
取消队列中尚未推理/发布的帧。若外部 ROS `publish()` 已经开始则无法撤回；发布门在
调用前做最终 generation/stop token 检查，保证 stop 边界之后不再发起新的 publish。

## 6. 离线 CLI

manifest 是 UTF-8 CSV，必需且精确包含以下字段（可有其他列）：

```text
image_path,scene,gt_distance_m
frames/000001.jpg,indoor,0.82
frames/000002.jpg,vehicle,4.30
```

相对 `image_path` 以 manifest 所在目录为基准；`scene` 只能是 `indoor` 或
`vehicle`；真值必须是有限非负米数。

下列命令使用 `${PYTHON}`，可先执行 `PYTHON=python3`。正式验收推荐把 `PYTHON`
指向团队提供的 bundled `codex-primary-runtime` Python，但文档不绑定个人机器路径。

链路正例诊断：

```bash
${PYTHON} \
  perception/tools/evaluate_obstacle_distance.py \
  --manifest /data/obstacle/manifest.csv \
  --mode diagnostic_constant \
  --constant-distance-m 0.5 \
  --output /tmp/obstacle-positive.json
```

链路负例诊断时把常数改为 `5.0`。正式模型评测：

```bash
${PYTHON} \
  perception/tools/evaluate_obstacle_distance.py \
  --manifest /data/obstacle/manifest.csv \
  --mode model \
  --backend-factory company_perception.obstacle_backends:create_backends \
  --config /data/obstacle/model-eval.json \
  --threshold-m 1.0 \
  --output /tmp/obstacle-model-report.json
```

配置文件优先使用 JSON；`.yaml` / `.yml` 需要可选 PyYAML。CLI 不下载模型或数据。

报告顶层字段为 `overall`、`by_scene`、`by_status`、`by_error_code`、
`best_threshold`、`predictions`。每组指标包含 `samples`、`valid_predictions`、
`failures`、`failure_rate`、`tp`、`fp`、`fn`、`precision`、`recall`、`f1`、
`rmse`、`positive_rate`。每条 prediction 同时含 manifest 三字段和完整输出 JSON
字段。

任何 `fallback: true` 的预测都标为失败，不进入 RMSE，也不进入有效预测的
precision/recall/F1 计数；失败率单独报告。`--threshold-m` 默认沿用配置中的
`decision_threshold_m`（默认 `1.0 m`）：`overall` 和分组指标用它同时判定真值与预测，
而 `best_threshold` 扫描始终用它固定真值正例标签，只改变预测阈值，避免扫描时改写
F1@1m 的真值定义。`best_threshold.ground_truth_threshold_m` 记录该固定真值阈值，
`best_threshold.threshold_m` 是选出的预测阈值。

`best_threshold` 只使用有效预测，阈值扫描通过排序事件实现 O(n log n)，不是对每个
候选重新全量扫描。F1 相同时依次优先 precision 更高、预测阈值更接近固定真值阈值、
预测阈值更小。

## 7. 外部模型目录与挂载

容器内预期目录为：

```text
/models/obstacle-distance/lifelong-nk
/models/obstacle-distance/yolo26n-seg
```

这些是外部只读挂载目标，不要求、也不得在仓库内创建
`perception/models/obstacle-distance`。Docker 示意：

```bash
docker run --rm \
  --mount type=bind,src=/srv/phanthy-models/obstacle-distance,dst=/models/obstacle-distance,readonly \
  phanthymotus-perception:handoff
```

Compose 示意：

```yaml
services:
  perception:
    volumes:
      - /srv/phanthy-models/obstacle-distance:/models/obstacle-distance:ro
```

制品也可以由受控模型制品仓库在部署阶段提供，但本项目不提供下载脚本。训练、模型转换、
TensorRT engine 构建和 INT8 校准均应在模型流水线中完成。

## 8. 参数量、文件体积与 Git 禁令

Lifelong-MonoDepth 与 YOLO26n-seg 的组合约 24.9M 参数只是参数量估算，不是文件体积
证明。若验收规则是单个或组合产物不得超过 30 MB，FP32 和 FP16 都可能超限；模型侧
必须提供 INT8 或其他压缩产物，并分别实测下载文件、解压后的权重以及最终 engine/plan
的字节数。不得根据参数量或 dtype 做未经测量的“满足 30 MB”保证。

以下模型制品一律不得提交 Git，包括复合后缀：

```text
*.pth
*.pth.tar
*.pt
*.onnx
*.engine
*.plan
*.safetensors
*.mnn
```

应使用只读挂载或模型制品仓库交付。不要把 Git LFS pointer 当成“未提交权重”，也不要
在本仓库新增权重下载脚本。

## 9. 已知 OCR 基线问题

基点 `fork/feat/zengzhitao@241b72d` 在本插件开始前已有以下问题，本任务没有修复：

1. `test_ocr_packaging.py` 仍期待旧的 PP-OCRv6 tiny 配置，而生产配置已经切到 small。
2. `test_ocr_model_downloader.py` 仍期待 15 MiB 限制，而生产代码已经改为 48 MiB。
3. `ocr_runtime.py` 导入仓库中不存在的 `plugins.ocr_preprocess`。
4. 本机 Python 3.14 环境没有安装 NumPy，导致一项 OCR tiled test 无法运行。

因此不要把这些旧问题描述为已修复。使用 bundled Python 运行本分支新增套件应为全绿；
仓库全量测试仍可能呈现既有 `3 failures` + `1 error`，必须在报告中与本插件回归分开。

## 10. 验收命令

任务 2–8 的障碍距离新增套件与相关生命周期测试：

```bash
${PYTHON} \
  -m unittest \
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

新增套件的验收标准是 `0 failures`、`0 errors`。仓库全量仅用于确认没有新增回归并记录
前述 OCR 基线，不能为追求全绿而修改 OCR：

```bash
${PYTHON} \
  -m unittest discover -s perception/tests -p 'test_*.py'
```

最后执行 `git diff --check`。打包守卫以基点 `241b72d` 为边界，检查分支 HEAD、
index、工作树和未跟踪新增/修改文件：每个 blob 必须小于 1 MiB，且不得是模型制品或
Git LFS 权重指针。Dockerfile 守卫同时拒绝会把模型目录带入镜像的宽范围 COPY/ADD。

CI 最好先 `fetch` 到该基点；也可以通过 `OBSTACLE_DISTANCE_BASE_REVISION` 指定可用
基点。若浅克隆中找不到基点，守卫会安全降级为从空树开始的全仓扫描，而不是跳过检查。
