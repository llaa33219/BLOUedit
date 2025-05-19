#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class EnhancementType {
    UPSCALE,        // Increase resolution
    DENOISING,      // Reduce noise
    STABILIZATION,  // Stabilize shaky footage
    FRAME_INTERP,   // Frame interpolation
    COLOR_CORRECT,  // Color correction
    SHARPEN,        // Sharpening
    LIGHTING        // Lighting adjustment
};

struct EnhancementParameters {
    std::string input_path;                    // Path to input video/image
    std::string output_path;                   // Path to output video/image
    EnhancementType type;                      // Type of enhancement
    float strength = 1.0f;                     // Strength of enhancement (0.0-1.0)
    int target_resolution_width = 0;           // Target width in pixels (0 = auto)
    int target_resolution_height = 0;          // Target height in pixels (0 = auto)
    bool maintain_aspect_ratio = true;         // Whether to maintain aspect ratio
    bool process_audio = true;                 // Whether to process audio along with video
    std::vector<std::pair<int, int>> segments; // Time segments to process (start/end in seconds)
    
    // Enhancement-specific parameters
    // Upscale parameters
    int upscale_factor = 2;                    // Upscale factor (1, 2, 4)
    
    // Denoising parameters
    float noise_level = 0.5f;                  // Noise level (0.0-1.0)
    
    // Stabilization parameters
    float stability_level = 0.8f;              // Stability level (0.0-1.0)
    
    // Frame interpolation parameters
    int target_fps = 60;                       // Target frame rate for interpolation
    
    // Color correction parameters
    float brightness = 0.0f;                   // Brightness adjustment (-1.0 to 1.0)
    float contrast = 0.0f;                     // Contrast adjustment (-1.0 to 1.0)
    float saturation = 0.0f;                   // Saturation adjustment (-1.0 to 1.0)
    float temperature = 0.0f;                  // Color temperature adjustment (-1.0 to 1.0)
    float tint = 0.0f;                         // Tint adjustment (-1.0 to 1.0)
    
    // Sharpen parameters
    float sharpness = 0.5f;                    // Sharpness level (0.0-1.0)
    
    // Lighting parameters
    float exposure = 0.0f;                     // Exposure adjustment (-1.0 to 1.0)
    float shadows = 0.0f;                      // Shadows adjustment (-1.0 to 1.0)
    float highlights = 0.0f;                   // Highlights adjustment (-1.0 to 1.0)
};

class Enhancement {
public:
    Enhancement();
    ~Enhancement();

    // Initialize the enhancement system
    bool initialize();
    
    // Apply enhancement to image or video
    bool apply_enhancement(
        const EnhancementParameters& params,
        std::function<void(float)> progress_callback = nullptr,
        std::function<void(bool, const std::string&)> completion_callback = nullptr
    );
    
    // Cancel ongoing enhancement
    void cancel();
    
    // Check if enhancement is currently running
    bool is_processing() const;
    
    // Create a UI widget for the enhancement feature
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