#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class MosaicType {
    BLUR,           // Gaussian blur
    PIXELATE,       // Pixelation
    BLACK,          // Black rectangle
    EMOJI,          // Emoji overlay
    CUSTOM_IMAGE    // Custom image overlay
};

struct FaceRect {
    int x;              // Top-left corner X
    int y;              // Top-left corner Y
    int width;          // Width of rectangle
    int height;         // Height of rectangle
    float confidence;   // Detection confidence (0.0-1.0)
    int tracking_id;    // ID for tracking across frames (-1 if not tracked)
};

struct FaceMosaicParameters {
    std::string input_path;                    // Path to input video/image
    std::string output_path;                   // Path to output video/image
    MosaicType mosaic_type = MosaicType::BLUR; // Type of mosaic effect
    float effect_intensity = 15.0f;            // Intensity of effect (blur radius, pixel size, etc.)
    bool track_faces = true;                   // Whether to track faces across frames
    bool process_audio = true;                 // Whether to process audio along with video
    bool detect_only = false;                  // Whether to only detect and not apply mosaic
    std::string custom_image_path = "";        // Path to custom overlay image (if CUSTOM_IMAGE)
    std::string emoji_type = "🙂";            // Type of emoji to use (if EMOJI)
    float detection_threshold = 0.5f;          // Confidence threshold for face detection
    bool auto_expand_rect = true;              // Whether to expand rectangles to cover entire face
    float expansion_factor = 1.2f;             // Factor to expand rectangles by
    std::vector<std::pair<int, int>> segments; // Time segments to process (start/end in seconds)
    
    // Custom rectangles to blur (if empty, detect faces automatically)
    std::vector<FaceRect> custom_rects;
};

class FaceMosaic {
public:
    FaceMosaic();
    ~FaceMosaic();

    // Initialize the face mosaic system
    bool initialize();
    
    // Detect faces in an image or video
    bool detect_faces(
        const std::string& input_path,
        std::vector<FaceRect>& faces,
        float detection_threshold = 0.5f,
        std::function<void(float)> progress_callback = nullptr
    );
    
    // Apply face mosaic to image or video
    bool apply_mosaic(
        const FaceMosaicParameters& params,
        std::function<void(float)> progress_callback = nullptr,
        std::function<void(bool, const std::string&)> completion_callback = nullptr
    );
    
    // Cancel ongoing face mosaic operation
    void cancel();
    
    // Check if face mosaic operation is currently running
    bool is_processing() const;
    
    // Create a UI widget for the face mosaic feature
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