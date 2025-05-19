#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class SttEngine {
    WHISPER,      // OpenAI Whisper
    GOOGLE,       // Google Speech-to-Text
    AZURE,        // Azure Speech Service
    VOSK          // Offline VOSK engine
};

struct SttParameters {
    std::string audio_path;      // Path to audio file
    std::string language = "";   // Language code (empty for auto-detection)
    bool add_timestamps = false; // Add word-level timestamps
    bool add_punctuation = true; // Add punctuation
    SttEngine engine = SttEngine::WHISPER;
};

struct SttResult {
    std::string text;                // Full transcribed text
    std::vector<std::string> segments; // Text divided into segments
    
    // Timestamp for each segment (start_time, end_time) in seconds
    std::vector<std::pair<double, double>> timestamps;
    
    // Confidence score for each segment (0.0 to 1.0)
    std::vector<float> confidence;
};

class SpeechToText {
public:
    SpeechToText();
    ~SpeechToText();

    // Initialize the STT system
    bool initialize();
    
    // Transcribe speech from audio file
    bool transcribe(
        const SttParameters& params,
        SttResult& result,
        std::function<void(bool, const SttResult&)> callback = nullptr
    );
    
    // Create a UI widget for the STT
    GtkWidget* create_widget();

private:
    // Internal implementation
    bool initialize_python_environment();
    
    void* python_state; // Opaque pointer to Python state
    bool is_initialized;
};

} // namespace AI
} // namespace BlouEdit 