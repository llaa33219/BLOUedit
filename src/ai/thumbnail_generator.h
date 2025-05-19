#pragma once

#include <string>
#include <functional>
#include <gtk/gtk.h>
#include "image_generator.h"

namespace BlouEdit {
namespace AI {

struct ThumbnailGenerationParams {
    std::string video_path;
    std::string prompt;
    bool use_video_frame = true;      // Use a frame from the video as base
    bool add_title = true;            // Add video title to the thumbnail
    bool add_visual_effects = true;   // Add visual effects (e.g., glow, lens flare)
    std::string title_text;           // Title to use if add_title is true
    int width = 1280;
    int height = 720;
};

class ThumbnailGenerator {
public:
    ThumbnailGenerator();
    ~ThumbnailGenerator();

    // Generate a thumbnail for a video
    bool generate_thumbnail(
        const ThumbnailGenerationParams& params,
        const std::string& output_path,
        std::function<void(bool, const std::string&)> callback = nullptr
    );

    // Create a UI widget for the thumbnail generator
    GtkWidget* create_widget();

private:
    // Get a frame from the video to use as base
    bool extract_video_frame(const std::string& video_path, const std::string& output_path);
    
    // Add title text to an image
    bool add_title_to_image(const std::string& image_path, const std::string& title);
    
    // Add visual effects to the thumbnail
    bool enhance_thumbnail(const std::string& image_path);
    
    // Get the metadata from a video file
    bool get_video_metadata(const std::string& video_path, std::string& title, int& duration);

    // Internal ImageGenerator instance for AI generation
    ImageGenerator image_generator;
    bool is_initialized;
};

} // namespace AI
} // namespace BlouEdit 