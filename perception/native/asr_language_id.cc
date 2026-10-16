// CPU language identification only. Transcription uses the separate ASR models.
#include "whisper.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) return 2;
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
        if (whisper_pcm_to_mel(context, samples.data(), count, threads) != 0) break;
        std::vector<float> probabilities(whisper_lang_max_id() + 1);
        if (whisper_lang_auto_detect(context, 0, threads, probabilities.data()) < 0) break;
        const int best = *std::max_element(languages.begin(), languages.end(),
            [&probabilities](int a, int b) { return probabilities[a] < probabilities[b]; });
        std::cout << "{\"language\":\"" << whisper_lang_str(best)
                  << "\",\"probability\":" << probabilities[best] << "}" << std::endl;
    }
    whisper_free(context);
    return std::cin.eof() ? 0 : 4;
}
