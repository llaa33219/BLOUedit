#include "audio_noise_remover.h"
#include <iostream>
#include <glib.h>
#include <cmath>

namespace BLOUedit {

AudioNoiseRemover::AudioNoiseRemover() 
    : pipeline(nullptr), source(nullptr), sink(nullptr), audioconvert(nullptr), noiseRemover(nullptr),
      removalLevel(NoiseRemovalLevel::MEDIUM), threshold(-30.0), reduction(-15.0), timeConst(400.0) {
    
    // Initialize GStreamer if needed
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
}

AudioNoiseRemover::~AudioNoiseRemover() {
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }
}

void AudioNoiseRemover::setInputFile(const std::string &input) {
    inputFile = input;
}

void AudioNoiseRemover::setOutputFile(const std::string &output) {
    outputFile = output;
}

void AudioNoiseRemover::setRemovalLevel(NoiseRemovalLevel level) {
    removalLevel = level;
    
    // Set default parameters based on level
    switch (level) {
        case NoiseRemovalLevel::LIGHT:
            threshold = -35.0;
            reduction = -10.0;
            timeConst = 300.0;
            break;
            
        case NoiseRemovalLevel::MEDIUM:
            threshold = -30.0;
            reduction = -15.0;
            timeConst = 400.0;
            break;
            
        case NoiseRemovalLevel::STRONG:
            threshold = -25.0;
            reduction = -20.0;
            timeConst = 500.0;
            break;
            
        case NoiseRemovalLevel::CUSTOM:
            // Keep current custom values
            break;
    }
}

void AudioNoiseRemover::setThreshold(double value) {
    threshold = value;
    // Automatically switch to custom mode when manually setting parameters
    removalLevel = NoiseRemovalLevel::CUSTOM;
}

void AudioNoiseRemover::setReduction(double value) {
    reduction = value;
    removalLevel = NoiseRemovalLevel::CUSTOM;
}

void AudioNoiseRemover::setTimeConstant(double value) {
    timeConst = value;
    removalLevel = NoiseRemovalLevel::CUSTOM;
}

bool AudioNoiseRemover::process() {
    if (inputFile.empty() || outputFile.empty()) {
        std::cerr << "Input or output file not specified" << std::endl;
        return false;
    }
    
    setupPipeline();
    
    // Set up bus messaging
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_signal_watch(bus);
    g_signal_connect(bus, "message", G_CALLBACK(onBusMessage), this);
    gst_object_unref(bus);
    
    // Start processing
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to start noise removal pipeline" << std::endl;
        gst_object_unref(pipeline);
        pipeline = nullptr;
        return false;
    }
    
    return true;
}

void AudioNoiseRemover::cancel() {
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline = nullptr;
    }
}

void AudioNoiseRemover::setupPipeline() {
    // Clean up existing pipeline if any
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }
    
    // Create a new pipeline
    pipeline = gst_pipeline_new("noise-removal-pipeline");
    
    // Create elements
    source = gst_element_factory_make("filesrc", "source");
    GstElement *decoder = gst_element_factory_make("decodebin", "decoder");
    audioconvert = gst_element_factory_make("audioconvert", "converter");
    GstElement *audioResample = gst_element_factory_make("audioresample", "resampler");
    
    // Create noise removal element
    // Using GStreamer's rnnoise element which is a neural network-based noise remover
    // Alternatively, we can use audiofilter or custom noise reduction plugin
    noiseRemover = gst_element_factory_make("rnnoise", "noise-remover");
    if (!noiseRemover) {
        // Fallback to alternative noise removal filter if rnnoise is not available
        std::cout << "RNNoise plugin not available, falling back to standard noise gate" << std::endl;
        noiseRemover = gst_element_factory_make("noisegate", "noise-remover");
        
        if (!noiseRemover) {
            std::cerr << "No suitable noise removal plugin found" << std::endl;
            return;
        }
    }
    
    // Configure the noise remover based on settings
    configureNoiseRemover();
    
    // Encoding elements
    GstElement *encoder = gst_element_factory_make("vorbisenc", "encoder");
    GstElement *muxer = gst_element_factory_make("oggmux", "muxer");
    sink = gst_element_factory_make("filesink", "sink");
    
    // Set properties
    g_object_set(G_OBJECT(source), "location", inputFile.c_str(), NULL);
    g_object_set(G_OBJECT(sink), "location", outputFile.c_str(), NULL);
    
    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, decoder, audioconvert, 
                    audioResample, noiseRemover, encoder, muxer, sink, NULL);
    
    // Link elements
    gst_element_link(source, decoder);
    
    // Connect pad-added signal for dynamic linking
    g_signal_connect(decoder, "pad-added", G_CALLBACK(+[](GstElement *element, GstPad *pad, gpointer data) {
        GstCaps *caps = gst_pad_get_current_caps(pad);
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        
        if (g_str_has_prefix(gst_structure_get_name(structure), "audio/x-raw")) {
            AudioNoiseRemover *self = static_cast<AudioNoiseRemover*>(data);
            GstPad *sinkpad = gst_element_get_static_pad(self->audioconvert, "sink");
            
            if (GST_PAD_LINK_FAILED(gst_pad_link(pad, sinkpad))) {
                std::cerr << "Failed to link decoder to converter" << std::endl;
            }
            
            gst_object_unref(sinkpad);
        }
        
        gst_caps_unref(caps);
    }), this);
    
    // Link remaining elements
    gst_element_link_many(audioconvert, audioResample, noiseRemover, encoder, muxer, sink, NULL);
}

void AudioNoiseRemover::configureNoiseRemover() {
    // Check if the noise remover is rnnoise or noisegate and configure appropriately
    std::string factoryName = GST_OBJECT_NAME(GST_ELEMENT_GET_CLASS(noiseRemover));
    
    if (factoryName == "rnnoise") {
        // RNNoise has a simpler interface, mainly VAD threshold
        // VAD = Voice Activity Detection
        double vadThreshold = 0.0;
        
        switch (removalLevel) {
            case NoiseRemovalLevel::LIGHT:
                vadThreshold = 0.7;  // Higher threshold = less aggressive
                break;
            case NoiseRemovalLevel::MEDIUM:
                vadThreshold = 0.5;  // Default value
                break;
            case NoiseRemovalLevel::STRONG:
                vadThreshold = 0.3;  // Lower threshold = more aggressive
                break;
            case NoiseRemovalLevel::CUSTOM:
                // Map our parameters to rnnoise's single parameter
                // Custom threshold range is typically -60 to 0 dB
                // Map to vad threshold range 0.1 to 0.9
                vadThreshold = 0.9 - (std::fabs(threshold) / 60.0 * 0.8);
                break;
        }
        
        g_object_set(G_OBJECT(noiseRemover), "vad-threshold", vadThreshold, NULL);
    }
    else if (factoryName == "noisegate") {
        // Configure noise gate parameters
        g_object_set(G_OBJECT(noiseRemover),
                    "threshold", threshold,      // Threshold in dB
                    "reduction", reduction,      // Reduction amount in dB
                    "attack", 50.0,              // Attack time in ms
                    "release", timeConst,        // Release time in ms
                    NULL);
    }
}

double AudioNoiseRemover::estimateNoiseLevel() {
    // This is a simplified implementation that would analyze the audio file
    // to estimate the noise floor. In a real implementation, this would:
    // 1. Extract a section of audio where there's likely only background noise
    // 2. Calculate the RMS or peak levels of that section
    // 3. Return an estimated noise floor level in dB
    
    // For now, we'll just return a placeholder value
    // A real implementation would use GStreamer's analysis elements or a custom algorithm
    return -40.0;
}

std::vector<std::string> AudioNoiseRemover::getPresetNames() {
    return {
        "Light Noise Reduction",
        "Medium Noise Reduction",
        "Strong Noise Reduction",
        "Custom Settings"
    };
}

void AudioNoiseRemover::onBusMessage(GstBus *bus, GstMessage *message, AudioNoiseRemover *remover) {
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_EOS:
            // End of stream - processing complete
            gst_element_set_state(remover->pipeline, GST_STATE_NULL);
            break;
            
        case GST_MESSAGE_ERROR: {
            GError *err = nullptr;
            gchar *debug_info = nullptr;
            
            gst_message_parse_error(message, &err, &debug_info);
            std::cerr << "Noise remover error: " << err->message << std::endl;
            
            g_clear_error(&err);
            g_free(debug_info);
            
            gst_element_set_state(remover->pipeline, GST_STATE_NULL);
            break;
        }
        
        default:
            break;
    }
}

} // namespace BLOUedit 