#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <string>
#include <vector>
#include <functional>

namespace BlouEdit {
namespace AI {

// Callback for audio to video generation completion
typedef std::function<void(bool success, const std::string& output_path)> AudioToVideoCallback;

// Style options for audio to video visualization
enum AudioVisualizationStyle {
    VISUALIZATION_WAVEFORM,
    VISUALIZATION_SPECTRUM,
    VISUALIZATION_PARTICLES,
    VISUALIZATION_GEOMETRIC,
    VISUALIZATION_ABSTRACT
};

// Structure to define visualization parameters
typedef struct {
    AudioVisualizationStyle style;
    std::string color_scheme;
    float intensity;
    float complexity;
    bool reactive_to_beat;
    bool use_3d;
} VisualizationParams;

class AudioToVideo {
public:
    AudioToVideo();
    ~AudioToVideo();

    // Initialize the AI model
    bool initialize();
    
    // Generate video visualization from audio
    void generateFromAudio(const std::string& audio_path,
                          const std::string& output_path,
                          AudioToVideoCallback callback);
    
    // Generate video with specific style
    void generateWithStyle(const std::string& audio_path,
                          AudioVisualizationStyle style,
                          const std::string& output_path,
                          AudioToVideoCallback callback);
    
    // Generate with full parameter control
    void generateWithParams(const std::string& audio_path,
                           const VisualizationParams& params,
                           const std::string& output_path,
                           AudioToVideoCallback callback);
    
    // Generate music video based on audio and prompt
    void generateMusicVideo(const std::string& audio_path,
                           const std::string& description,
                           const std::string& output_path,
                           AudioToVideoCallback callback);
    
    // Set generation parameters
    void setResolution(int width, int height);
    void setFrameRate(int fps);
    void setColorScheme(const std::string& scheme);
    
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
    std::string color_scheme;
    
    // Helper methods
    bool loadModel();
    void unloadModel();
    std::string analyzeAudioFeatures(const std::string& audio_path);
    GstElement* createAudioProcessingPipeline(const std::string& audio_path);
};

} // namespace AI
} // namespace BlouEdit 