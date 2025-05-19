#include "voice_modulator.h"
#include <iostream>
#include <glib.h>

namespace BLOUedit {

VoiceModulator::VoiceModulator() 
    : pipeline(nullptr), source(nullptr), sink(nullptr), audioconvert(nullptr), effect(nullptr),
      modulationType(VoiceModulationType::PITCH_SHIFT), intensity(0.5) {
    
    // Initialize GStreamer if needed
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
}

VoiceModulator::~VoiceModulator() {
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }
}

void VoiceModulator::setInputFile(const std::string &input) {
    inputFile = input;
}

void VoiceModulator::setOutputFile(const std::string &output) {
    outputFile = output;
}

void VoiceModulator::setModulationType(VoiceModulationType type) {
    modulationType = type;
}

void VoiceModulator::setIntensity(double value) {
    intensity = value;
    if (intensity < 0.0) intensity = 0.0;
    if (intensity > 1.0) intensity = 1.0;
}

bool VoiceModulator::process() {
    setupPipeline();
    
    // Set up bus messaging
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_signal_watch(bus);
    g_signal_connect(bus, "message", G_CALLBACK(onBusMessage), this);
    gst_object_unref(bus);
    
    // Start processing
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to start voice modulation pipeline" << std::endl;
        gst_object_unref(pipeline);
        pipeline = nullptr;
        return false;
    }
    
    return true;
}

void VoiceModulator::cancel() {
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline = nullptr;
    }
}

void VoiceModulator::setupPipeline() {
    // Clean up existing pipeline if any
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }
    
    // Create a new pipeline
    pipeline = gst_pipeline_new("voice-modulation-pipeline");
    
    // Create elements
    source = gst_element_factory_make("filesrc", "source");
    GstElement *decoder = gst_element_factory_make("decodebin", "decoder");
    audioconvert = gst_element_factory_make("audioconvert", "converter");
    GstElement *audioResample = gst_element_factory_make("audioresample", "resampler");
    GstElement *encoder = gst_element_factory_make("vorbisenc", "encoder");
    GstElement *muxer = gst_element_factory_make("oggmux", "muxer");
    sink = gst_element_factory_make("filesink", "sink");
    
    // Set properties
    g_object_set(G_OBJECT(source), "location", inputFile.c_str(), NULL);
    g_object_set(G_OBJECT(sink), "location", outputFile.c_str(), NULL);
    
    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, decoder, audioconvert, audioResample, NULL);
    
    // Create and configure the effect based on selected modulation type
    switch (modulationType) {
        case VoiceModulationType::PITCH_SHIFT:
            applyPitchShift();
            break;
        case VoiceModulationType::ROBOT:
            applyRobotEffect();
            break;
        case VoiceModulationType::ECHO:
            applyEchoEffect();
            break;
        case VoiceModulationType::REVERB:
            applyReverbEffect();
            break;
        case VoiceModulationType::HELIUM:
            applyHeliumEffect();
            break;
        case VoiceModulationType::DEEP:
            applyDeepEffect();
            break;
    }
    
    gst_bin_add_many(GST_BIN(pipeline), encoder, muxer, sink, NULL);
    
    // Link elements
    gst_element_link(source, decoder);
    
    // Connect pad-added signal for dynamic linking
    g_signal_connect(decoder, "pad-added", G_CALLBACK(+[](GstElement *element, GstPad *pad, gpointer data) {
        GstCaps *caps = gst_pad_get_current_caps(pad);
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        
        if (g_str_has_prefix(gst_structure_get_name(structure), "audio/x-raw")) {
            VoiceModulator *self = static_cast<VoiceModulator*>(data);
            GstPad *sinkpad = gst_element_get_static_pad(self->audioconvert, "sink");
            
            if (GST_PAD_LINK_FAILED(gst_pad_link(pad, sinkpad))) {
                std::cerr << "Failed to link decoder to converter" << std::endl;
            }
            
            gst_object_unref(sinkpad);
        }
        
        gst_caps_unref(caps);
    }), this);
    
    // Link remaining elements
    gst_element_link_many(audioconvert, audioResample, effect, encoder, muxer, sink, NULL);
}

void VoiceModulator::applyPitchShift() {
    effect = gst_element_factory_make("pitch", "pitch-effect");
    
    // Calculate pitch value: 0.5-2.0 range (0.5=low, 2.0=high)
    double pitchValue = 0.5 + (intensity * 1.5);
    g_object_set(G_OBJECT(effect), "pitch", pitchValue, NULL);
    
    gst_bin_add(GST_BIN(pipeline), effect);
}

void VoiceModulator::applyRobotEffect() {
    // Create a robot voice using audioecho and pitch
    GstElement *audioecho = gst_element_factory_make("audioecho", "echo");
    GstElement *pitch = gst_element_factory_make("pitch", "pitch");
    
    // Configure robot effect
    g_object_set(G_OBJECT(audioecho), "delay", 10000000, "intensity", 0.6, "feedback", 0.4, NULL);
    g_object_set(G_OBJECT(pitch), "pitch", 0.9, NULL);
    
    // Create a bin to hold the effect elements
    effect = gst_bin_new("robot-effect-bin");
    gst_bin_add_many(GST_BIN(effect), audioecho, pitch, NULL);
    gst_element_link(audioecho, pitch);
    
    // Add ghost pads to the bin
    GstPad *sinkpad = gst_element_get_static_pad(audioecho, "sink");
    GstPad *srcpad = gst_element_get_static_pad(pitch, "src");
    
    gst_element_add_pad(effect, gst_ghost_pad_new("sink", sinkpad));
    gst_element_add_pad(effect, gst_ghost_pad_new("src", srcpad));
    
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);
    
    gst_bin_add(GST_BIN(pipeline), effect);
}

void VoiceModulator::applyEchoEffect() {
    effect = gst_element_factory_make("audioecho", "echo-effect");
    
    // Map intensity to echo parameters: 0.0-100.0 for delay in milliseconds
    int delayMs = static_cast<int>(intensity * 500.0) * 1000000; // Convert to nanoseconds
    double echoIntensity = 0.3 + (intensity * 0.4);
    
    g_object_set(G_OBJECT(effect), 
                "delay", delayMs,
                "intensity", echoIntensity, 
                "feedback", 0.5, 
                NULL);
    
    gst_bin_add(GST_BIN(pipeline), effect);
}

void VoiceModulator::applyReverbEffect() {
    effect = gst_element_factory_make("audioecho", "reverb-effect");
    
    // Reverb is like echo but with shorter delay and higher feedback
    int delayMs = static_cast<int>((0.1 + intensity * 0.2) * 1000000000); // 100-300ms in nanoseconds
    double reverbIntensity = 0.4 + (intensity * 0.3);
    double feedback = 0.6 + (intensity * 0.3);
    
    g_object_set(G_OBJECT(effect), 
                "delay", delayMs,
                "intensity", reverbIntensity, 
                "feedback", feedback, 
                NULL);
    
    gst_bin_add(GST_BIN(pipeline), effect);
}

void VoiceModulator::applyHeliumEffect() {
    effect = gst_element_factory_make("pitch", "helium-effect");
    
    // Helium voice is high-pitched
    double pitchValue = 1.5 + (intensity * 1.0); // 1.5-2.5 range
    g_object_set(G_OBJECT(effect), "pitch", pitchValue, NULL);
    
    gst_bin_add(GST_BIN(pipeline), effect);
}

void VoiceModulator::applyDeepEffect() {
    effect = gst_element_factory_make("pitch", "deep-effect");
    
    // Deep voice is low-pitched
    double pitchValue = 0.5 + (intensity * 0.3); // 0.5-0.8 range
    g_object_set(G_OBJECT(effect), "pitch", pitchValue, NULL);
    
    gst_bin_add(GST_BIN(pipeline), effect);
}

void VoiceModulator::onBusMessage(GstBus *bus, GstMessage *message, VoiceModulator *modulator) {
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_EOS:
            // End of stream - processing complete
            gst_element_set_state(modulator->pipeline, GST_STATE_NULL);
            break;
            
        case GST_MESSAGE_ERROR: {
            GError *err = nullptr;
            gchar *debug_info = nullptr;
            
            gst_message_parse_error(message, &err, &debug_info);
            std::cerr << "Voice modulator error: " << err->message << std::endl;
            
            g_clear_error(&err);
            g_free(debug_info);
            
            gst_element_set_state(modulator->pipeline, GST_STATE_NULL);
            break;
        }
        
        default:
            break;
    }
}

std::vector<std::string> VoiceModulator::getAvailableEffects() {
    return {
        "Pitch Shift",
        "Robot Voice",
        "Echo",
        "Reverb",
        "Helium Voice",
        "Deep Voice"
    };
}

} // namespace BLOUedit 