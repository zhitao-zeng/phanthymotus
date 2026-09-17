// Stateful RNN language-model scoring for offline transducer search.
// Copyright (c) 2026. Apache-2.0.
#pragma once
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "onnxruntime_cxx_api.h"

namespace sherpa_onnx {
struct PrefixState {
  std::vector<float> h, c, next;
  double prefix_logprob = 0;
};

class PrefixLm {
 public:
  explicit PrefixLm(Ort::Session *session) : session_(session) {
    const std::vector<std::string> inputs{"x", "h0", "c0"};
    const std::vector<std::string> outputs{"log_softmax", "next_h0", "next_c0"};
    Ort::AllocatorWithDefaultOptions allocator;
    if (session_->GetInputCount() != 3 || session_->GetOutputCount() != 3)
      throw std::runtime_error("Prefix LM requires token and two recurrent states");
    for (size_t i = 0; i < 3; ++i) {
      if (inputs[i] != session_->GetInputNameAllocated(i, allocator).get() ||
          outputs[i] != session_->GetOutputNameAllocated(i, allocator).get())
        throw std::runtime_error("Unsupported prefix LM input/output names");
    }
    auto shape = session_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    auto c_shape = session_->GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
    auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != 2 || shape[2] != 256 || c_shape != shape ||
        out_shape.empty() || out_shape.back() != 5000)
      throw std::runtime_error("Prefix LM must use two 256-dimensional layers and 5000 tokens");
  }

  void Reset() { cache_.clear(); }

  const PrefixState &Get(const std::vector<int64_t> &ys, int context) {
    std::string key;
    for (size_t i = context; i < ys.size(); ++i) key += std::to_string(ys[i]) + ",";
    auto found = cache_.find(key);
    if (found != cache_.end()) return *found->second;
    std::unique_ptr<PrefixState> state;
    if (ys.size() == static_cast<size_t>(context)) {
      PrefixState zeros;
      zeros.h.assign(512, 0); zeros.c.assign(512, 0);
      state = Forward(1, zeros);  // One SOS; transducer context blanks are excluded.
    } else {
      auto parent = ys;
      int64_t token = parent.back(); parent.pop_back();
      const auto &previous = Get(parent, context);
      state = Forward(token, previous);
      state->prefix_logprob = previous.prefix_logprob + previous.next.at(token);
    }
    auto *answer = state.get();
    cache_.emplace(std::move(key), std::move(state));
    return *answer;
  }

 private:
  std::unique_ptr<PrefixState> Forward(int64_t token, const PrefixState &previous) {
    auto info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> xs{1, 1};
    std::array<int64_t, 3> hs{2, 1, 256};
    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<int64_t>(info, &token, 1, xs.data(), xs.size()),
        Ort::Value::CreateTensor<float>(info, const_cast<float *>(previous.h.data()), 512, hs.data(), hs.size()),
        Ort::Value::CreateTensor<float>(info, const_cast<float *>(previous.c.data()), 512, hs.data(), hs.size())};
    const char *in[] = {"x", "h0", "c0"};
    const char *out[] = {"log_softmax", "next_h0", "next_c0"};
    auto values = session_->Run({}, in, inputs.data(), inputs.size(), out, 3);
    if (values[0].GetTensorTypeAndShapeInfo().GetElementCount() != 5000 ||
        values[1].GetTensorTypeAndShapeInfo().GetElementCount() != 512 ||
        values[2].GetTensorTypeAndShapeInfo().GetElementCount() != 512)
      throw std::runtime_error("Unexpected prefix LM output shape");
    auto state = std::make_unique<PrefixState>();
    const float *p = values[0].GetTensorData<float>(); state->next.assign(p, p + 5000);
    p = values[1].GetTensorData<float>(); state->h.assign(p, p + 512);
    p = values[2].GetTensorData<float>(); state->c.assign(p, p + 512);
    return state;
  }

  Ort::Session *session_;  // Owned by the recognizer's OfflineRnnLM.
  std::unordered_map<std::string, std::unique_ptr<PrefixState>> cache_;
};

// State is text-only and shared across a decode batch, never across calls.
struct PrefixLmReset {
  explicit PrefixLmReset(PrefixLm *lm) : lm(lm) { if (lm) lm->Reset(); }
  ~PrefixLmReset() { if (lm) lm->Reset(); }
  PrefixLm *lm;
};
}  // namespace sherpa_onnx
