#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <string>
#include <vector>

namespace BLOUedit {

enum class VoiceModulationType {
    PITCH_SHIFT,
    ROBOT,
    ECHO,
    REVERB,
    HELIUM,
    DEEP
};

class VoiceModulator {
private:
    GstElement *pipeline;
    GstElement *source;
    GstElement *sink;
    GstElement *audioconvert;
    GstElement *effect;
    
    std::string inputFile;
    std::string outputFile;
    VoiceModulationType modulationType;
    double intensity;

    void setupPipeline();
    void applyPitchShift();
    void applyRobotEffect();
    void applyEchoEffect();
    void applyReverbEffect();
    void applyHeliumEffect();
    void applyDeepEffect();

public:
    VoiceModulator();
    ~VoiceModulator();

    void setInputFile(const std::string &input);
    void setOutputFile(const std::string &output);
    void setModulationType(VoiceModulationType type);
    void setIntensity(double value); // 0.0 - 1.0 range

    bool process();
    void cancel();
    
    static std::vector<std::string> getAvailableEffects();
    
    // Signal callback
    static void onBusMessage(GstBus *bus, GstMessage *message, VoiceModulator *modulator);
};

} // namespace BLOUedit 