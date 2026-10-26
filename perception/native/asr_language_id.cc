// CPU language identification only. Transcription uses the separate ASR models.
#include "whisper.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <sstream>
#include <string>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {
constexpr float kConfirmationProbability = 0.8f;

void release_context(whisper_context *&context) {
    if (context) whisper_free(context);
    context = nullptr;
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

bool identify(whisper_context *context, const std::vector<float> &samples,
              int threads, const std::vector<int> &languages,
              std::pair<int, float> &result) {
    if (whisper_pcm_to_mel(context, samples.data(), samples.size(), threads) != 0) return false;
    std::vector<float> probabilities(whisper_lang_max_id() + 1);
    if (whisper_lang_auto_detect(context, 0, threads, probabilities.data()) < 0) return false;
    const int best = *std::max_element(languages.begin(), languages.end(),
        [&probabilities](int a, int b) { return probabilities[a] < probabilities[b]; });
    result = {best, probabilities[best]};
    return true;
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) return 2;
    const int threads = std::max(1, std::atoi(argv[2]));
    std::vector<int> languages;
    std::stringstream names(argv[3]);
    std::string name;
    while (std::getline(names, name, ',')) {
        const int id = whisper_lang_id(name.c_str());
        if (id < 0) return 2;
        languages.push_back(id);
    }
    if (languages.empty()) return 2;
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    auto *context = whisper_init_from_file_with_params(argv[1], params);
    if (!context) return 3;
    std::cout << "{\"ready\":true}" << std::endl;
    uint32_t count;
    while (std::cin.read(reinterpret_cast<char *>(&count), sizeof(count))) {
        if (!count || count > 16000 * 60) break;
        std::vector<int16_t> pcm(count);
        if (!std::cin.read(reinterpret_cast<char *>(pcm.data()), count * sizeof(int16_t))) break;
        std::vector<float> samples(count);
        std::transform(pcm.begin(), pcm.end(), samples.begin(),
                       [](int16_t value) { return value / 32768.0f; });
        if (!context) context = whisper_init_from_file_with_params(argv[1], params);
        std::pair<int, float> result;
        if (!context || !identify(context, samples, threads, languages, result)) break;
        if (argc == 5 && result.second < kConfirmationProbability) {
            // Both decisions use this packet. Keep only one model resident,
            // and retain the primary result if confirmation is inconclusive.
            release_context(context);
            context = whisper_init_from_file_with_params(argv[4], params);
            std::pair<int, float> confirmation;
            if (context && identify(context, samples, threads, languages, confirmation)) {
                if (confirmation.second >= kConfirmationProbability) result = confirmation;
            } else {
                std::cerr << "Optional language verification unavailable; retaining primary result\n";
            }
            release_context(context);
        }
        std::cout << "{\"language\":\"" << whisper_lang_str(result.first)
                  << "\",\"probability\":" << result.second << "}" << std::endl;
    }
    release_context(context);
    return std::cin.eof() ? 0 : 4;
}
