#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <string>
#include <vector>
#include <memory>

namespace BLOUedit {

enum class NoiseRemovalLevel {
    LIGHT,  // For minimal noise reduction
    MEDIUM, // Balanced noise reduction
    STRONG, // Aggressive noise reduction
    CUSTOM  // Custom parameters
};

class AudioNoiseRemover {
private:
    GstElement *pipeline;
    GstElement *source;
    GstElement *sink;
    GstElement *audioconvert;
    GstElement *noiseRemover;
    
    std::string inputFile;
    std::string outputFile;
    NoiseRemovalLevel removalLevel;
    
    // Custom parameters
    double threshold;  // Noise threshold (dB) - default: -30
    double reduction;  // Noise reduction factor (dB) - default: -15
    double timeConst;  // Time constant (ms) - default: 400
    
    void setupPipeline();
    void configureNoiseRemover();
    
public:
    AudioNoiseRemover();
    ~AudioNoiseRemover();
    
    void setInputFile(const std::string &input);
    void setOutputFile(const std::string &output);
    void setRemovalLevel(NoiseRemovalLevel level);
    
    // For custom level settings
    void setThreshold(double value);
    void setReduction(double value);
    void setTimeConstant(double value);
    
    bool process();
    void cancel();
    double estimateNoiseLevel(); // Analyze input file to estimate noise level
    
    // Static utility methods
    static std::vector<std::string> getPresetNames();
    
    // Signal callback
    static void onBusMessage(GstBus *bus, GstMessage *message, AudioNoiseRemover *remover);
};

} // namespace BLOUedit 