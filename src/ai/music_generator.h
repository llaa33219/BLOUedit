#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

// Music genres
enum class MusicGenre {
    AMBIENT,
    CLASSICAL,
    ELECTRONIC,
    HIP_HOP,
    JAZZ,
    LO_FI,
    POP,
    ROCK,
    SOUNDTRACK,
    CUSTOM
};

// Music moods
enum class MusicMood {
    HAPPY,
    SAD,
    CALM,
    ENERGETIC,
    ROMANTIC,
    SUSPENSEFUL,
    EPIC,
    PLAYFUL,
    MYSTERIOUS,
    DARK
};

// Generation mode
enum class GenerationMode {
    FULL_TRACK,          // Generate a complete music track
    CONTINUATION,        // Continue from an existing audio clip
    ACCOMPANIMENT,       // Create accompaniment for existing melody
    STEM_SEPARATION      // Separate audio into stems (drums, bass, melody, etc.)
};

struct MusicGeneratorParameters {
    // Basic parameters
    std::string output_path;                     // Path to save the generated music
    int duration_seconds = 30;                   // Duration of the generated music
    MusicGenre genre = MusicGenre::AMBIENT;      // Music genre
    MusicMood mood = MusicMood::CALM;            // Music mood
    int tempo_bpm = 120;                         // Tempo in beats per minute
    float volume = 1.0f;                         // Volume level (0.0-1.0)
    bool normalize_audio = true;                 // Whether to normalize the audio
    
    // Advanced parameters
    GenerationMode mode = GenerationMode::FULL_TRACK;  // Generation mode
    std::string input_audio_path = "";           // Path to input audio (for continuation/accompaniment)
    std::string text_prompt = "";                // Text description for the music to generate
    std::string reference_track_path = "";       // Path to reference track to influence style
    bool loop_friendly = false;                  // Whether to make the music loop-friendly
    
    // Structure parameters
    int intro_seconds = 4;                       // Duration of intro section
    int outro_seconds = 4;                       // Duration of outro section
    bool include_drums = true;                   // Whether to include drums
    bool include_bass = true;                    // Whether to include bass
    bool include_melody = true;                  // Whether to include melody
    bool include_chords = true;                  // Whether to include chords
    
    // Export parameters
    std::string file_format = "wav";             // Output file format (wav, mp3, ogg)
    int sample_rate = 44100;                     // Sample rate
    int bit_depth = 16;                          // Bit depth
    
    // Custom parameters
    std::string custom_genre = "";               // Custom genre description
    std::vector<std::pair<float, std::string>> keypoints; // Time points with descriptions for dynamic music
};

class MusicGenerator {
public:
    MusicGenerator();
    ~MusicGenerator();
    
    // Initialize the music generator
    bool initialize();
    
    // Generate music based on parameters
    bool generate_music(
        const MusicGeneratorParameters& params,
        std::function<void(float)> progress_callback = nullptr,
        std::function<void(bool, const std::string&)> completion_callback = nullptr
    );
    
    // Cancel ongoing music generation
    void cancel();
    
    // Check if music generation is currently running
    bool is_processing() const;
    
    // Get a list of available music models
    std::vector<std::string> get_available_models() const;
    
    // Create a UI widget for the music generator
    GtkWidget* create_widget();
    
private:
    // Internal implementation
    bool initialize_python_environment();
    
    void* python_state; // Opaque pointer to Python state
    bool is_initialized;
    bool is_processing_active;
};

} // namespace AI
} // namespace BlouEdit 