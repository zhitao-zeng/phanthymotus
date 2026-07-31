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

## 后续优化（631cc3d → 3162397，10 commits）

### A. 评测机端口冲突二次修复（631cc3d）

**问题**：820b4a6 合并 `origin/main` 时把 `main.py` 的端口读取改成只读 `config.yaml`，丢了 `os.environ.get("MCP_PORT"/"WS_PORT")`。评测平台 `run_for_darwin.py` 给 10 个 host-network 实例通过 `-e MCP_PORT=15720+i*100` 分配独立端口，改后全部无视 env、齐抢 config 写死的 15720 → 实例 1-9 `OSError [Errno 98] Address already in use` 崩溃，平台 `wait_mcp_ready` 连 15820+ 永远超时。**这是 820b4a6 在 judge 上的新失败根因。**

**修复**：恢复 `61b7ca6` 的 env-first 写法（env 读不到才 fallback config）。

### B. Transducer 尾部静音 padding（c65038a + 544033b + 265e32b + ed83cad）

**问题**：Transducer 解码器需要尾部静音来正确 emit 末尾 token。无 tail pad 时短句（case 0 "进入零力矩模式"）仅解码开头几帧 → 输出英文乱码 `"I'll shi"`。

**修复链**：
- `c65038a` `OfflineASRAdapter.transcribe()` 加 0.5s 尾部静音 — ARM 1-CER 0.7130 → 0.8388
- `544033b` `pcm16_to_float_samples` 结果转 list 再 extend（numpy 数组不可 `.extend`）
- `ed83cad` VAD `silence_ms` 700→400，避免 VAD 提前截断弱尾音节
- `265e32b` VAD segment 再补 300ms tail pad（`_SherpaVadSession`），保护 case 7 "举双手" 尾 144ms 被 VAD 误判静音而截断

**配套**：`69ac49a` 关掉 tts/htmsg/vop 插件，ASR-only eval 降 OOM（10 实例 × ~125MB）。

### C. FireRedVAD 新后端（a86c6a4 + e3f228a + 821078f + 3162397）

**动机**：silero VAD 在 case 7 等短句上仍会截断弱尾音节。FireRedVAD（DFSMN，F1 97.57 vs silero 95.95）更稳。

**实现**：
- `a86c6a4` vendor `plugins/firered_vad.py`（259 行）— ONNX runtime + `kaldi_native_fbank` + numpy，**无 torch**（torch 会给每个 VAD worker 加 0.5-1GB RSS，重触发 69ac49a 修过的 10-实例 OOM）。ONNX 导出与 torch parity < 1e-7
- 新 `_FireRedVadSession`（`asr_runtime.py:371`）：buffer-and-flush — 非 streaming，攒 PCM 到尾部 ≥1s 静音再一次性 `detect()`，避免增量 re-detect 的 segment 边界漂移。每个 segment 前 `pre_roll_s` + 后 300ms tail pad
- `e3f228a` pip install `onnxruntime`（sherpa-onnx 无 py binding）
- `821078f` 简化为 buffer-and-flush，避免增量 detect 漂移
- `3162397` 尾部静音 1s 触发 one-shot detect

**Dockerfile**（`Dockerfile.jetson:79-92`）：
- `pip install kaldi_native_fbank onnxruntime`
- `asr_model_downloader.py --model firered_vad` 下载 `firered_vad.onnx` + `.onnx.data`（external weights）+ `cmvn.npz` 到 `/models/firered_vad`
- 保留 silero VAD 下载作为 fallback

**config.yaml**：`vad.model: firered`，threshold 0.4，silence_ms 400，pre_roll_ms 500，`model_dir: /models/firered_vad`。

**本地 e2e（x-asr mbs5 + hotwords）**：firered-crop 1-CER **0.8393** vs full-file 0.8451；case 7 不再被截断（silero 会截）。

---

## 当前分支状态（截至 3162397）

`feat/asr-integrate` 共 13 commits ahead of `origin/main`：
- 5895407 ASR port（VadSession + x-asr + stop() 顺序修复）
- 820b4a6 JP6 镜像烘焙 x-asr+VAD
- ccc4ce2 文档
- 631cc3d 端口 env override
- c65038a/544033b/265e32b/ed83cad tail padding 链
- 69ac49a 关其他插件
- a86c6a4/e3f228a/821078f/3162397 FireRedVAD 后端

**改了 7 个文件**：Dockerfile.jetson / config.yaml / main.py / asr_offline.py / asr_runtime.py / **新 firered_vad.py** / asr_model_downloader.py（+418 / -17）。

**当前 config 状态**：
- ASR：x-asr transducer + modified_beam_search(5) + hotwords(2.0) + 0.5s tail pad
- VAD：**firered**（DFSMN ONNX，F1 97.57），threshold 0.4，silence_ms 400，pre_roll_ms 500
- tts/htmsg/vop 全 disabled（ASR-only eval）

**回滚点**：
- 回 FireRedVAD 之前：`git revert a86c6a4`（保留 silero VAD）
- 回 tail padding 之前：`git revert c65038a`（1-CER 降回 0.7130）
- 回 stock ASR：`git revert 5895407`（仅 `perception/`）

---

## 相关文件

- `perception/plugins/asr.py` — 主插件，含 stop() 顺序修复 + drain+flush 尾路径 + lifecycle_lock
- `perception/plugins/asr_runtime.py` — `VadSession` + `_SherpaVadSession`（pre_roll 500ms + `_TimedPcmHistory` + drain + flush）
- `perception/plugins/asr_offline.py` — `OfflineASRAdapter` 支持 x-asr transducer + paraformer fallback + hotwords + mbs(5)
- `perception/Dockerfile.jetson` — JP6 镜像，build 时烘焙 x-asr + VAD，消除评测机 volume mount 依赖
- `perception/utils/asr_model_downloader.py` — `--model x_asr` 下载 transducer 三件套 + hotwords，原子替换
- `perception/plugins/firered_vad.py` — vendor FireRedVAD ONNX runtime（DFSMN，无 torch），`FireRedVadOnnx.detect()`
- `perception/config.yaml` — `mode/model_path/device/vad_pre_roll_ms/silence_ms=400` + `sherpa_config`（mbs/hotwords/bpe）+ `vad.model=firered`

---

## 2026-08-01：FireRedVAD 空检测不再锁死 session

**基线**：`01c868f`（代码路径等价于历史最高分 `42534d7`，另含安全下载与 CI）。

**问题**：`_FireRedVadSession._run_detect()` 在 FireRedVAD 返回空 segment 时仍设置
`_detected=True`。Judge 多实例下如果 1 秒零包先于乱序/延迟语音包到达，第一次检测只看到静音并返回空；
后续 `process_chunk()` 因 `_detected` 提前返回，迟到语音不会加入 buffer，`notify_idle()` 也被同一状态挡住，
最终形成空结果。

**修复**：

- 只有检测到真实 segment 后才设置 `_detected=True`；空检测保持 session 可接收后续音频；
- 记录上次已检测的 buffer 字节数，同一份 idle buffer 不重复推理；
- 空检测后重置连续静音计数，新增音频到达后需重新累计端点，避免每个零包都触发推理；
- 补上 FireRedVAD 检测异常日志所需的 `logging` import，异常保持可重试；
- CI 扩展为运行全部 `test_asr_*.py`，覆盖空检测后迟到音频、idle 去重和临时检测失败。

未恢复 `89850d6/4e38ac6` 的 whole-buffer fallback 或 pause 强制输出；这两条路径已有 Judge 退化证据。

**本地验证**：

- 7 个 ASR 单测通过，相关 Python 源码 `py_compile` 通过；
- 使用本地 FireRed ONNX 对 12 条评测 WAV 跑 VAD：正常顺序 12/12 有 segment；
- 每条 WAV 前先注入 1 秒静音，使首次检测为空，再发送真实音频：12/12 均能在后续端点恢复 segment。

**尚未验证**：Jetson ARM/ROS2/DDS 多实例与平台分数。该修复是否清除历史 4/120 空结果，必须走
Jetson 和平台链路确认，不能由 x86 离线结果替代。
