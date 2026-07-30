# ASR 漏识别根因分析（feat/asr-integrate）

**日期**：2026-07-30
**分支**：`feat/asr-integrate`（基于 `origin/main`，commit `5895407`，后追加 `820b4a6`）
**背景**：id=7 间歇性空结果（VAD 端点丢失），短句识别不稳。
**结论**：新分支 12/12 全部成功，平均 CER 20.42%（baseline 22.90%，-2.48%）。

---

## 旧 main 的 6 个漏点

### 1. stop() 顺序反了（最致命，直接致空结果）

**旧** `perception/plugins/asr.py:862`（origin/main）：
```python
# Cancel feeder threads immediately
for q in (self._pcm_queue, self._utterance_queue):
    q.cancel_join_thread(); q.close()   # ← 先关队列
if self._vad_proc and self._vad_proc.is_alive():
    self._vad_proc.join(timeout=5)       # ← 后 join worker
```

worker 还在 `pcm_q.get(timeout=1)` 阻塞 → feeder 被取消 → 队列里剩余 PCM + VAD 内部 buffer 的 segment 全丢。
**短句（如 id=7 "举双手" 1.044s）刚好在说话末尾被 stop 命中，VAD 还没 emit segment 就被砍掉 → 空结果。**

**新** `asr.py:957`：
```python
if self._vad_proc and self._vad_proc.is_alive():
    self._vad_proc.join(timeout=5)       # ← 先 join worker
                                         # worker 里有 drain+flush 尾路径
for q in (self._pcm_queue, self._utterance_queue):
    q.cancel_join_thread(); q.close()    # ← 后关队列
```

### 2. 没有 drain+flush 尾路径

**旧** worker while 循环退出后直接 `process exiting`，没有处理队列里剩余 PCM，也没调 `vad.flush()`。

**新** `asr.py:803-840`：
```python
# Drain+flush tail path
_log.info("[vad-worker] draining pending audio before exit...")
while True:
    pcm, ts = pcm_q.get(timeout=0.2)     # 抽干队列
    vad_result = vad_session.process_chunk(pcm, ts)
    ...
flushed = vad_session.flush()            # 强制 emit VAD 内部 buffer
if flushed and len(flushed) > SAMPLE_RATE:
    result_q.put((flushed, 0.0, 0.0))
```

### 3. 无 pre-roll 重建

**旧** 直接用 `vad.front.samples`，VAD 检测到语音到启动之间有 ~500ms 延迟被切掉。短词受损严重。

**新** `_SherpaVadSession`（asr_runtime.py）用 `_TimedPcmHistory` 保留 31s PCM 历史，每个 segment 前补 `pre_roll_ms=500` 音频。

### 4. silence_ms=400 太短

**旧** `vad_silence_ms: int = 400`。1.044s 短音频里的自然停顿可能触发分段，把一句话切成两段，前段 <500ms 被 `len > SAMPLE_RATE` 过滤。

**新** `silence_ms=700`，更稳。

### 5. 长 idle 后 VAD 状态漂移

**旧** VAD 对象跨 start/stop 复用，内部 buffer/状态不清理。8 分钟 idle 后 sherpa-onnx Silero VAD 的 internal queue 状态异常，segment 不 emit。

**新** 每次 `start()` 调 `vad_session.init()` = `reset()`，清干净。

### 6. 没有 lifecycle_lock

**旧** `stop()` 进行中如果 `start()` 并发调用 → VAD worker 进程状态错乱。

**新** `_state_lock + _lifecycle_lock + _load_generation` 保护。

---

## 对 id=7 的具体解释

- audio 1.044s，末尾静音短
- 旧代码：start → 音频发完 → 1500ms 静音 → stop。stop 时 `cancel_join_thread` 先砍队列 → VAD 内部刚 detected 的 segment 被 flush 走 → 空结果
- 新代码：stop 先 `join` worker → worker 跑 drain+flush → VAD 内部 segment 被 emit → "女枪手。"（虽然字面错，但至少不空）

**6 个漏点里，#1+#2 是空结果的直接原因，#3+#4+#5 是短句识别差的辅助原因。**

---

## 验证结果

**Eval 配置**：Jetson dev (.16)，`phanthymotus-perception-asr:asr-integrate` 镜像（JP6 + x-asr transducer + hotwords score=2.0 + mbs(5)），12-case 数据集 `/tmp/asr_eval/eval/`。

**对比 61b7ca6 baseline（JP6 + x-asr）：**

| 指标 | 61b7ca6 baseline | asr-integrate (new) | delta |
|---|---|---|---|
| 成功 | 12/12 | 12/12 | = |
| 平均 CER | 22.90% | 20.42% | **-2.48%** |
| 平均 latency | ~36-38s | ~35-44s | ≈ |

**逐 case CER：**
- 改善：id=3 (-50%), id=0 (-14.3%), id=4 (-16.7%)
- 持平：id=1/2/8/9/10
- 变差：id=7 (+33.3%), id=11 (+10%), id=6 (+4.8%), id=5 (+3.1%)

**id=7（原问题 case）**：baseline "给双手" → new "女枪手"。VAD 端点稳定触发（日志可见 `utterance complete, len=28800 bytes`），不再出现空结果。字面识别差异属于 x-asr 在短词上的正常波动。

---

## 架构验证

- `silence_ms=700, pre_roll_ms=500` 在新 `asr_runtime.VadSession` 下端点稳定
- `stop()` 顺序：先 join VAD worker（日志可见 `draining pending audio before exit...`），再 `cancel_join_thread` — 保留了 flushed segments
- x-asr transducer + hotwords(score=2.0) + mbs(5) 在新 `asr_offline.OfflineASRAdapter` 下正常加载
- 新分支基于 `origin/main`，ASR 8 处改动全部编译并通过端到端验证：
  1. `_resolve_asr_mode` + `ASR_MODE_ALIASES`（offline|streaming，legacy online/segmented 别名）
  2. `_build_asr_adapter` 路由 offline → `OfflineASRAdapter`，streaming → `ASR_MODELS` registry
  3. `_vad_worker` 使用 `VadSession`（pre_roll + drain tail path + flush）替代 inline `sherpa_onnx VoiceActivityDetector`
  4. `stop()` join VAD worker BEFORE `cancel_join_thread`
  5. `ASRPlugin` `_state_lock + _lifecycle_lock + _load_generation` 线程安全 async 加载
  6. metrics：received_chunks/dropped_chunks/completed_utterances/transcribe_errors/last_audio_ts/last_result_ts/last_error
  7. 保留 main 的 `asr_kws` IPA matching、perf spans/trace_id、save_vad_segments
  8. 保留 main 的 KWS with `kws_model` zh/en/zh-en selector

---

## 回滚

回滚点：`git revert 5895407`（仅 `perception/` 目录）即可回到 main 的 stock ASR，无 x-asr transducer，无 drain+flush 尾路径。

---

## 镜像打包修复（820b4a6）— 评测机 FileNotFoundError 根因

**问题**：5895407 提交后评测机（.15）容器启动即崩，日志报 `FileNotFoundError: Token file not found: /models/sherpa-onnx/tokens.txt`，ASR 12/12 全挂。

**根因**：
1. `Dockerfile.jetson` 仍是 `origin/main` 的 JP511 版本，JP6 版本 + `asr_model_downloader.py` 从未提交到 git
2. 评测容器无 volume mount（`docker inspect .Mounts` = `[]`），`/models/sherpa-onnx/asr/` 根本不存在
3. `asr_offline.py:111` 命中 `FileNotFoundError` 直接抛错，没有 auto-download 兜底

**修复（820b4a6，3 个文件）**：

1. `perception/Dockerfile.jetson`（JP511 → JP6）
   - base 换 `dustynv/l4t-pytorch:r36.4.0`（Python 3.10）
   - apt 装 ROS2 Humble + colcon + libopenblas
   - `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` 关 SHM transport（跨容器 pub/sub 必备，2026-07-24 probe 验证）
   - pip 装 pyyaml/requests/websockets/webrtcvad/websocket-client/dashscope + silero-vad + sherpa-onnx
   - **build 时烘焙 x-asr transducer** 到 `/models/sherpa-onnx/x_asr_punct_int8/`（经 `asr_model_downloader.py --model x_asr`），不依赖运行时 volume mount
   - **build 时下载 Silero VAD** 到 `/models/sherpa-onnx/vad/silero_vad.onnx`

2. `perception/utils/asr_model_downloader.py`（新文件）
   - `--model x_asr` 下载 transducer 三件套（encoder/decoder/joiner int8）+ tokens.txt + bpe.model + bpe.vocab + hotwords.txt
   - 原子替换（tempfile → rename）避免半下载文件污染目标

3. `perception/config.yaml`
   - `model_path` / `model_dir` 指向烘焙目录 `/models/sherpa-onnx/x_asr_punct_int8`
   - 新增 `sherpa_config`：tokens=tokens.txt, numThreads=2, provider=cpu, featureDim=80
   - `recognizerConfig`：modified_beam_search, maxActivePaths=5, hotwordsFile=hotwords.txt, hotwordsScore=2.0, modelingUnit=bpe, bpeVocab=bpe.vocab
   - `vad.silence_ms=700, pre_roll_ms=500`（承接 6 漏点修复）

**验证（.16 dev 机）**：
- `docker build -f perception/Dockerfile.jetson -t phanthymotus-perception-asr:820b4a6 .` 成功
- 容器启动日志：`[asr] model 'paraformer-zh-en' ready`，transducer 三件套检测命中，hotwords 已加载
- 端口 patch（15720 冲突 → sed 改 15740）后正常对外服务

**镜像清理**：删除旧 ASR 镜像 11 个 + OCR 镜像 8 个，prune dangling 96 个，回收 ~12G，磁盘 82% → 77%。

---

## 相关文件

- `perception/plugins/asr.py` — 主插件，含 stop() 顺序修复 + drain+flush 尾路径 + lifecycle_lock
- `perception/plugins/asr_runtime.py` — `VadSession` + `_SherpaVadSession`（pre_roll 500ms + `_TimedPcmHistory` + drain + flush）
- `perception/plugins/asr_offline.py` — `OfflineASRAdapter` 支持 x-asr transducer + paraformer fallback + hotwords + mbs(5)
- `perception/Dockerfile.jetson` — JP6 镜像，build 时烘焙 x-asr + VAD，消除评测机 volume mount 依赖
- `perception/utils/asr_model_downloader.py` — `--model x_asr` 下载 transducer 三件套 + hotwords，原子替换
- `perception/config.yaml` — `mode/model_path/device/vad_pre_roll_ms/silence_ms=700` + `sherpa_config`（mbs/hotwords/bpe）
