#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class SubtitleFormat {
    SRT,        // SubRip Text format
    VTT,        // Web Video Text Tracks format
    ASS,        // Advanced SubStation Alpha
    TXT,        // Plain text with timestamps
    JSON        // JSON format for custom applications
};

enum class TranslationMode {
    NONE,       // No translation
    AUTO,       // Auto-detect source language and translate
    MANUAL      // User selects source and target languages
};

struct SubtitleSegment {
    int start_time_ms;          // Start time in milliseconds
    int end_time_ms;            // End time in milliseconds
    std::string text;           // Subtitle text
    float confidence;           // Recognition confidence (0.0-1.0)
};

struct AutoSubtitleParameters {
    std::string input_path;                    // Path to input video
    std::string output_path;                   // Path to output subtitle file
    SubtitleFormat format = SubtitleFormat::SRT; // Output subtitle format
    bool extract_audio_only = false;           // Whether to only extract audio without generating subtitles
    std::string extracted_audio_path = "";     // Path to save extracted audio (if extract_audio_only is true)
    std::string language = "auto";             // Language code for speech recognition (auto=auto-detect)
    bool include_confidence = false;           // Whether to include confidence scores in output
    bool filter_profanity = false;             // Whether to filter profanity
    int max_chars_per_line = 42;               // Maximum characters per subtitle line
    int max_lines_per_subtitle = 2;            // Maximum lines per subtitle
    int min_segment_duration_ms = 500;         // Minimum segment duration in milliseconds
    int max_segment_duration_ms = 5000;        // Maximum segment duration in milliseconds
    bool merge_short_segments = true;          // Whether to merge short segments
    bool split_long_segments = true;           // Whether to split long segments
    bool adjust_timing_to_scene_changes = false; // Whether to adjust timing to scene changes
    std::vector<std::pair<int, int>> segments; // Time segments to process (start/end in seconds)
    
    // Translation options
    TranslationMode translation_mode = TranslationMode::NONE; // Translation mode
    std::string source_language = "auto";      // Source language code (for translation)
    std::string target_language = "en";        // Target language code (for translation)
    
    // Styling options (for formats that support styling)
    std::string font_name = "Arial";           // Font name
    int font_size = 24;                        // Font size
    std::string font_color = "white";          // Font color
    std::string background_color = "#000000AA"; // Background color with alpha
    std::string outline_color = "black";       // Outline color
    int outline_width = 1;                     // Outline width
    std::string position = "bottom";           // Subtitle position (bottom, top, middle)
};

class AutoSubtitle {
public:
    AutoSubtitle();
    ~AutoSubtitle();

    // Initialize the automatic subtitle system
    bool initialize();
    
    // Generate subtitles for a video
    bool generate_subtitles(
        const AutoSubtitleParameters& params,
        std::function<void(float)> progress_callback = nullptr,
        std::function<void(bool, const std::string&)> completion_callback = nullptr
    );
    
    // Cancel ongoing subtitle generation
    void cancel();
    
    // Check if subtitle generation is currently running
    bool is_processing() const;
    
    // Create a UI widget for the automatic subtitle feature
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