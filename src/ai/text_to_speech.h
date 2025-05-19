#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class TtsVoiceModel {
    STANDARD,      // Standard TTS voice
    NEURAL,        // Neural TTS voice
    CUSTOM         // Custom trained voice
};

enum class TtsVoiceGender {
    MALE,
    FEMALE,
    NEUTRAL
};

struct TtsVoice {
    std::string id;
    std::string name;
    TtsVoiceGender gender;
    TtsVoiceModel model;
    std::string language_code;
};

struct TtsParameters {
    std::string text;
    TtsVoice voice;
    float speaking_rate = 1.0f;
    float pitch = 0.0f;
    float volume_gain_db = 0.0f;
    bool add_timestamps = false;
};

class TextToSpeech {
public:
    TextToSpeech();
    ~TextToSpeech();

    // Initialize the TTS system
    bool initialize();
    
    // Get available voices
    std::vector<TtsVoice> get_available_voices(const std::string& language_code = "");
    
    // Generate speech from text
    bool synthesize_speech(
        const TtsParameters& params,
        const std::string& output_path,
        std::function<void(bool, const std::string&)> callback = nullptr
    );
    
    // Create a UI widget for the TTS
    GtkWidget* create_widget();

private:
    // Internal implementation
    bool initialize_python_environment();
    
    void* python_state; // Opaque pointer to Python state
    bool is_initialized;
};

} // namespace AI
} // namespace BlouEdit 