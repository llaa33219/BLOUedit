#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class CutoutMethod {
    AUTOMATIC,      // Fully automatic foreground extraction
    SALIENT,        // Salient object detection
    PORTRAIT,       // Portrait mode (person extraction)
    INTERACTIVE     // Interactive (user provides initial mask or points)
};

enum class MaskType {
    BINARY,         // Binary mask (black and white)
    ALPHA,          // Alpha mask (grayscale)
    TRIMAP          // Trimap (foreground, background, unknown)
};

struct CutoutParameters {
    std::string input_path;                    // Path to input video/image
    std::string output_path;                   // Path to output video/image
    std::string mask_path = "";                // Path to save the generated mask (optional)
    CutoutMethod method = CutoutMethod::AUTOMATIC; // Method to use for cutout
    MaskType mask_type = MaskType::ALPHA;      // Type of mask to generate
    float threshold = 0.5f;                    // Threshold for binary masking (0.0-1.0)
    bool invert_mask = false;                  // Whether to invert the mask
    bool apply_feathering = false;             // Whether to apply feathering to the mask edges
    float feather_amount = 2.0f;               // Amount of feathering to apply
    bool add_shadow = false;                   // Whether to add a drop shadow
    std::string background_path = "";          // Path to background image/video to composite with
    bool process_audio = true;                 // Whether to process audio along with video
    std::vector<std::pair<int, int>> segments; // Time segments to process (start/end in seconds)
    
    // For interactive cutout
    std::vector<std::pair<int, int>> foreground_points; // Points marked as foreground
    std::vector<std::pair<int, int>> background_points; // Points marked as background
    std::string initial_mask_path = "";        // Path to initial mask for refinement
};

class SmartCutout {
public:
    SmartCutout();
    ~SmartCutout();

    // Initialize the smart cutout system
    bool initialize();
    
    // Apply smart cutout to image or video
    bool apply_cutout(
        const CutoutParameters& params,
        std::function<void(float)> progress_callback = nullptr,
        std::function<void(bool, const std::string&)> completion_callback = nullptr
    );
    
    // Cancel ongoing cutout operation
    void cancel();
    
    // Check if cutout operation is currently running
    bool is_processing() const;
    
    // Create a UI widget for the smart cutout feature
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