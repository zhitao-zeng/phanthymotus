# 障碍物距离模型交接清单

> 完整协议见 [`obstacle_distance.md`](obstacle_distance.md)。本清单只列"要交付什么",方便直接转给模型同学。

## 要交付的 2 个模型

| 模型 | 作用 | 容器内挂载目录 | config 后端名 |
| --- | --- | --- | --- |
| Lifelong-MonoDepth 双域权重(`NK.pth.tar`) | 单目深度(室内 + 车辆都用) | `/models/obstacle-distance/lifelong-nk` | `lifelong_nk` |
| YOLO26n-seg | 实例分割(只有车辆场景用) | `/models/obstacle-distance/yolo26n-seg` | `yolo26n_seg` |

## 每个模型必须满足的接口

### 深度:Lifelong-MonoDepth
- [ ] 双域权重,室内选 **NYU 头(head 0)**、车辆选 **KITTI 头(head 1)**。
- [ ] adapter 把 inverse depth / disparity / 归一化输出统一转成**米**;输出 2D `depth_m` + 正整数 `source_height` / `source_width`。
- [ ] 实现 `DepthBackend.predict_depth(image_bytes, domain, deadline_monotonic) -> DepthPrediction`。

### 分割:YOLO26n-seg
- [ ] 每个实例输出:非空 `class_name`、`confidence ∈ [0,1]`、与最终深度图**同坐标系、同尺寸**的 2D **布尔** mask(letterbox/resize/crop 必须逆变换回去)。
- [ ] 类名用**字符串**(不依赖 Ultralytics 数字 id)。首版 COCO 6 类:`person, car, truck, bus, motorcycle, bicycle`;挑战微调版再加 `construction_vehicle, traffic_cone, barrier, pushable_pullable, debris`。
- [ ] 实现 `InstanceSegmentationBackend.predict_instances(image_bytes, deadline_monotonic) -> Sequence[InstanceMask]`。

### 工厂
- [ ] 提供 `module.path:create_backends(config) -> (depth_backend, segmentation_backend)`,配到 config 的 `backend_factory`(深度在前、分割在后)。

## 产物 / 大小 / 部署
- [ ] **实测文件字节数**:若榜单限制单个/合计 ≤ 30 MB,FP32/FP16 很可能超 → 需交 **INT8 或其它压缩产物**。参数量 ~24.9M **不算**大小证明。
- [ ] Jetson 上大概率跑 TensorRT `.engine` / `.plan`,转换 / 量化 / INT8 校准由模型侧完成。
- [ ] 放 **JuiceFS 文件服务**(`http://172.28.4.81:34567/<个人目录>/...`),文件名 / 路径稳定,**逐个验证能完整下载**(不能只看 HTTP 200,要确认不是 0 字节)。
- [ ] **不进 Git**:`*.pth / *.pth.tar / *.pt / *.onnx / *.engine / *.plan / *.safetensors / *.mnn` 一律禁止提交。

## 交接完成后由我方接手
- [ ] `perception/Dockerfile.jetson` 加这两个模型的下载步骤。
- [ ] 按检测器**实际输出的类名**填 `vehicle.allowed_classes`(对应关系见 `config.yaml` 注释)。
- [ ] 设 `backend_factory`、把 `enabled` 改为 `true`,跑通端到端链路。

---

> 注:车辆场景还需要**相机标定**(`fx/fy/cx/cy` + `camera_to_ego` + `bumper_xy`)。这来自**车端**、不属于模型交接,但缺了它车辆几何会直接走 `missing_calibration` 兜底。
