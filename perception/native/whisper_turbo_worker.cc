// Persistent CUDA Whisper context; automatic language per independent PCM16 packet.
// Optional per-language LoRA adapters: <model_dir>/lora/<lang>.wslr (e.g. ja.wslr,
// th.wslr, ru.wslr) are loaded at startup; each packet runs a base encoder pass for
// language ID, then decodes with the adapter whose language matches the detection.
// Missing or empty lora/ dir keeps the exact base behavior.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#include <dirent.h>
#include "whisper.h"
#include "json.hpp"

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

std::vector<float> Wave(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    char header[12]; f.read(header, 12);
    if (!f || std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4))
        throw std::runtime_error("Expected WAV");
    bool valid = false;
    while (f) {
        char tag[4]; uint32_t n = 0;
        f.read(tag, 4); f.read(reinterpret_cast<char *>(&n), 4);
        if (!f) break;
        auto end = f.tellg() + std::streamoff(n + (n % 2));
        if (!std::memcmp(tag, "fmt ", 4)) {
            uint16_t fmt, channels, bits, align; uint32_t rate, bytes;
            f.read(reinterpret_cast<char *>(&fmt), 2); f.read(reinterpret_cast<char *>(&channels), 2);
            f.read(reinterpret_cast<char *>(&rate), 4); f.read(reinterpret_cast<char *>(&bytes), 4);
            f.read(reinterpret_cast<char *>(&align), 2); f.read(reinterpret_cast<char *>(&bits), 2);
            valid = fmt == 1 && channels == 1 && rate == 16000 && bits == 16;
        } else if (!std::memcmp(tag, "data", 4)) {
            if (!valid || n % 2) throw std::runtime_error("Expected mono16k PCM16");
            std::vector<int16_t> raw(n / 2);
            f.read(reinterpret_cast<char *>(raw.data()), n);
            if (!f) throw std::runtime_error("Truncated PCM");
            std::vector<float> x(raw.size());
            for (size_t i = 0; i < x.size(); ++i) x[i] = raw[i] / 32768.0f;
            return x;
        }
        f.seekg(end);
    }
    throw std::runtime_error("Missing PCM data");
}

struct Deadline {
    Clock::time_point started;
    bool expired = false;
};
bool Abort(void *opaque) {
    auto *d = static_cast<Deadline *>(opaque);
    d->expired = std::chrono::duration<double>(Clock::now() - d->started).count() >= 120.0;
    return d->expired;
}

static std::map<std::string, int> LoadAdapters(whisper_context *ctx, const char *model_path) {
    std::map<std::string, int> loaded;
    std::string dir(model_path);
    auto slash = dir.find_last_of('/');
    if (slash != std::string::npos) dir.resize(slash);
    dir += "/lora";
    DIR *d = opendir(dir.c_str());
    if (!d) return loaded;
    while (dirent *e = readdir(d)) {
        std::string name(e->d_name);
        if (name.size() != 7 || name.substr(2) != ".wslr") continue;  // <2-letter-lang>.wslr
        std::string lang = name.substr(0, 2);
        int idx = whisper_lora_load(ctx, (dir + "/" + name).c_str());
        if (idx >= 0) loaded[lang] = idx;
    }
    closedir(d);
    return loaded;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    auto started = Clock::now();
    auto cp = whisper_context_default_params(); cp.use_gpu = true;
    auto *ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) return 3;
    auto adapters = LoadAdapters(ctx, argv[1]);
    whisper_lora_set(ctx, -1);
    std::cout << Json({{"ready", true}, {"worker_pid", getpid()},
        {"load_s", std::chrono::duration<double>(Clock::now() - started).count()},
        {"language", "auto"}, {"threads", 2}, {"tail_samples", 4800},
        {"no_context", true}, {"translate", false}, {"temperature", 0},
        {"temperature_inc", 0.2},
        {"sampling", "greedy"}, {"best_of", 1}, {"no_timestamps", false},
        {"lora_adapters", adapters.size()},
        {"gpu", true}, {"timeout_s", 120}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto q = Json::parse(line);
            auto x = Wave(q.at("audio"));
            const auto original_samples = x.size();
            x.resize(original_samples + 4800, 0.0f); // Exactly one 300ms tail.
            // Base encoder pass only for language ID; whisper_full re-encodes below
            // with the selected adapter, mirroring the validated routing runner.
            whisper_lora_set(ctx, -1);
            if (whisper_pcm_to_mel(ctx, x.data(), static_cast<int>(x.size()), 4) != 0)
                throw std::runtime_error("mel_failed");
            if (whisper_encode(ctx, 0, 0) != 0)
                throw std::runtime_error("encode_failed");
            const int lang_id = whisper_lang_auto_detect(ctx, 0, 0, nullptr);
            const std::string det = lang_id >= 0 ? whisper_lang_str(lang_id) : "";
            int adapter = -1;
            auto it = adapters.find(det);
            if (it != adapters.end()) adapter = it->second;
            whisper_lora_set(ctx, adapter);
            auto p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            p.n_threads = 2; p.translate = false; p.no_context = true;
            p.language = det.empty() ? "auto" : det.c_str();
            p.print_realtime = false; p.print_progress = false;
            p.print_timestamps = false; p.print_special = false;
            p.no_timestamps = false; p.temperature = 0; p.temperature_inc = 0.2;
            p.greedy.best_of = 1;
            Deadline deadline{Clock::now()};
            p.abort_callback = Abort; p.abort_callback_user_data = &deadline;
            int code = whisper_full(ctx, p, x.data(), static_cast<int>(x.size()));
            const double decode_s = std::chrono::duration<double>(Clock::now() - deadline.started).count();
            struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
            Json out{{"worker_pid", getpid()}, {"decode_s", decode_s},
                {"hwm_mib", usage.ru_maxrss / 1024.0}, {"input_samples", original_samples},
                {"padded_samples", x.size()}, {"tail_samples", 4800}, {"return_code", code},
                {"timeout", deadline.expired}, {"lora", adapter >= 0 ? det : ""},
                {"lid_s", det}};
            if (code) {
                out["error"] = deadline.expired ? "packet_timeout" : "whisper_full_failed";
                out["raw_text"] = ""; out["predicted_language"] = nullptr;
            } else {
                std::string text;
                for (int i = 0; i < whisper_full_n_segments(ctx); ++i)
                    text += whisper_full_get_segment_text(ctx, i);
                const int final_lang = whisper_full_lang_id(ctx);
                out["raw_text"] = text;
                out["predicted_language_id"] = final_lang;
                out["predicted_language"] = final_lang >= 0 ? whisper_lang_str(final_lang) : "unknown";
            }
            std::cout << out.dump(-1, ' ', false, Json::error_handler_t::replace) << std::endl;
        } catch (const std::exception &e) {
            std::cout << Json({{"error", e.what()}, {"worker_pid", getpid()},
                {"raw_text", ""}, {"predicted_language", nullptr}}).dump(-1, ' ', false,
                Json::error_handler_t::replace) << std::endl;
        }
    }
    whisper_free(ctx);
    return 0;
}
