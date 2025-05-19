#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <string>
#include <vector>
#include <functional>

namespace BlouEdit {
namespace AI {

// Callback for text to video generation completion
typedef std::function<void(bool success, const std::string& output_path)> TextToVideoCallback;

// Style options for text to video generation
enum TextToVideoStyle {
    STYLE_REALISTIC,
    STYLE_ANIMATED,
    STYLE_SKETCH,
    STYLE_STYLIZED,
    STYLE_ARTISTIC
};

// Structure to define a scene in the video
typedef struct {
    std::string description;
    int duration_ms;
    std::string transition;
    std::string audio_prompt;
} VideoScene;

class TextToVideo {
public:
    TextToVideo();
    ~TextToVideo();

    // Initialize the AI model
    bool initialize();
    
    // Generate video from text description
    void generateFromText(const std::string& text_description,
                          const std::string& output_path,
                          TextToVideoCallback callback);
    
    // Generate video with specific style
    void generateWithStyle(const std::string& text_description,
                          TextToVideoStyle style,
                          const std::string& output_path,
                          TextToVideoCallback callback);
    
    // Generate video from structured scenes
    void generateFromScenes(const std::vector<VideoScene>& scenes,
                           const std::string& output_path,
                           TextToVideoCallback callback);
    
    // Add custom video style
    void addCustomStyle(const std::string& style_name,
                       const std::string& style_prompt,
                       const std::string& reference_image_path);
    
    // Set generation parameters
    void setResolution(int width, int height);
    void setFrameRate(int fps);
    void setMaxDuration(int seconds);
    
    // Check if the model is ready
    bool isModelReady() const;
    
    // Cancel ongoing generation
    void cancelGeneration();

private:
    void* model_handle;                 // Opaque handle to the AI model
    bool model_ready;                   // Flag indicating if model is initialized
    bool generation_in_progress;        // Flag indicating if generation is running
    
    // Generation parameters
    int output_width;
    int output_height;
    int output_fps;
    int max_duration_seconds;
    
    // Custom styles
    std::vector<std::string> custom_style_names;
    std::vector<std::string> custom_style_prompts;
    std::vector<std::string> custom_style_references;
    
    // Helper methods
    bool loadModel();
    void unloadModel();
    std::string processScenes(const std::vector<VideoScene>& scenes);
    std::string getStylePrompt(TextToVideoStyle style);
    GstElement* createEncodingPipeline(const std::string& output_path);
};

} // namespace AI
} // namespace BlouEdit 