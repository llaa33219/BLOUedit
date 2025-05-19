#pragma once

#include <string>
#include <vector>
#include <functional>
#include <gtk/gtk.h>

namespace BlouEdit {
namespace AI {

enum class ImageGenerationModel {
    STABLE_DIFFUSION,
    DALL_E,
    MIDJOURNEY_COMPATIBLE
};

struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 512;
    int num_inference_steps = 50;
    float guidance_scale = 7.5f;
    ImageGenerationModel model = ImageGenerationModel::STABLE_DIFFUSION;
};

class ImageGenerator {
public:
    ImageGenerator();
    ~ImageGenerator();

    // Generate an image using AI
    bool generate_image(const ImageGenerationParams& params, 
                        const std::string& output_path,
                        std::function<void(bool, const std::string&)> callback);
    
    // Generate a thumbnail for a video
    bool generate_thumbnail(const std::string& video_path,
                           const std::string& prompt,
                           const std::string& output_path,
                           std::function<void(bool, const std::string&)> callback);

    // Create a UI widget for the image generator
    GtkWidget* create_widget();

private:
    // Internal implementation
    bool initialize_python_environment();
    bool load_model(ImageGenerationModel model);
    
    void* python_state; // Opaque pointer to Python state
    bool is_initialized;
};

} // namespace AI
} // namespace BlouEdit 