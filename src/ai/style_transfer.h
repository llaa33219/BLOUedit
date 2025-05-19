#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class StyleCategory {
    PAINTING,       // Painting styles (e.g., Van Gogh, Monet)
    PHOTO,          // Photographic styles (e.g., HDR, Noir)
    ABSTRACT,       // Abstract art styles
    CARTOON,        // Cartoon-like styles
    CINEMATIC,      // Movie-like color grading
    CUSTOM          // Custom user-defined style
};

struct StylePreset {
    std::string id;
    std::string name;
    StyleCategory category;
    std::string preview_path;  // Path to a preview thumbnail
    float strength_default;    // Default strength (0.0-1.0)
};

struct StyleTransferParameters {
    std::string input_path;                    // Path to input video/image
    std::string output_path;                   // Path to output video/image
    std::string style_id;                      // ID of style to apply
    float strength = 0.75f;                    // Strength of style application (0.0-1.0)
    bool preserve_colors = false;              // Whether to preserve original colors
    bool process_audio = false;                // Whether to process audio along with video
    int target_fps = 0;                        // Target FPS (0 = same as input)
    std::vector<std::pair<int, int>> segments; // Time segments to process (start/end in seconds)
};

class StyleTransfer {
public:
    StyleTransfer();
    ~StyleTransfer();

    // Initialize the style transfer system
    bool initialize();
    
    // Get available style presets
    std::vector<StylePreset> get_available_styles(const StyleCategory& category = StyleCategory::PAINTING);
    
    // Apply style transfer to image or video
    bool apply_style(
        const StyleTransferParameters& params,
        std::function<void(float)> progress_callback = nullptr,
        std::function<void(bool, const std::string&)> completion_callback = nullptr
    );
    
    // Cancel ongoing style transfer
    void cancel();
    
    // Check if style transfer is currently running
    bool is_processing() const;
    
    // Create a UI widget for the style transfer feature
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