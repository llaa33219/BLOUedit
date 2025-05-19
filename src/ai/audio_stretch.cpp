#include "audio_stretch.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <future>
#include <cmath>
#include <random>

// Constructor
AudioStretch::AudioStretch() :
    main_container_(nullptr),
    input_file_chooser_(nullptr),
    output_file_chooser_(nullptr),
    ratio_scale_(nullptr),
    preserve_pitch_check_(nullptr),
    stretch_button_(nullptr),
    progress_bar_(nullptr),
    bpm_mode_check_(nullptr),
    source_bpm_spin_(nullptr),
    target_bpm_spin_(nullptr),
    detect_bpm_button_(nullptr) {
    
    // Set default model path
    char* data_dir = g_build_filename(g_get_user_data_dir(), "blouedit", "models", "audio_stretch", NULL);
    model_path_ = std::string(data_dir);
    g_free(data_dir);
}

// Destructor
AudioStretch::~AudioStretch() {
    cleanup();
}

// Initialize audio stretch system
bool AudioStretch::initialize() {
    // Create data directory if it doesn't exist
    std::filesystem::path data_path(model_path_);
    if (!std::filesystem::exists(data_path)) {
        try {
            std::filesystem::create_directories(data_path);
        } catch (const std::exception& e) {
            last_error_ = std::string("Failed to create data directory: ") + e.what();
            return false;
        }
    }

    // Initialize UI elements
    setup_ui();
    
    // Load AI model
    return load_ai_model();
}

// Clean up resources
void AudioStretch::cleanup() {
    // Cleanup UI if needed
    if (main_container_ != nullptr) {
        // No need to destroy it here as GTK will handle widget destruction
        main_container_ = nullptr;
    }
}

// Stretch audio file with specified ratio
bool AudioStretch::stretch_audio(const std::string& input_audio_path,
                               const std::string& output_audio_path,
                               double ratio,
                               bool preserve_pitch) {
    if (input_audio_path.empty() || !std::filesystem::exists(input_audio_path)) {
        last_error_ = "Input audio file does not exist";
        return false;
    }
    
    if (output_audio_path.empty()) {
        last_error_ = "Output audio path is empty";
        return false;
    }
    
    if (ratio <= 0.0) {
        last_error_ = "Invalid stretch ratio. Must be greater than 0.";
        return false;
    }
    
    // Show progress bar activity
    gtk_widget_set_visible(progress_bar_, TRUE);
    gtk_progress_bar_pulse(GTK_PROGRESS_BAR(progress_bar_));
    
    // Run stretching in a separate thread to avoid UI freezing
    std::future<bool> result = std::async(std::launch::async, [this, input_audio_path, output_audio_path, ratio, preserve_pitch]() {
        // TODO: Implement actual audio stretching using an AI library
        // This would typically involve:
        // 1. Loading the input audio
        // 2. Analyzing the audio content
        // 3. Applying time-stretching algorithm with or without pitch preservation
        // 4. Saving the processed audio to the output path
        
        // Simulate AI processing with a delay
        for (int i = 0; i <= 100; i += 5) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Update progress bar on the main thread
            g_idle_add([](gpointer data) -> gboolean {
                int* progress = static_cast<int*>(data);
                AudioStretch* self = reinterpret_cast<AudioStretch*>(progress + 1);
                self->update_progress(*progress / 100.0);
                delete progress;
                return FALSE;
            }, new int[2] { i, reinterpret_cast<int>(this) });
        }
        
        // Create a dummy output file for demonstration
        std::ofstream audio_file(output_audio_path);
        if (!audio_file.is_open()) {
            last_error_ = "Failed to create output audio file";
            return false;
        }
        
        // Write some dummy data with information about the processing
        audio_file << "RIFF" << std::endl;
        audio_file << "# Processed with AI Audio Stretch" << std::endl;
        audio_file << "# Source: " << input_audio_path << std::endl;
        audio_file << "# Ratio: " << ratio << std::endl;
        audio_file << "# Preserve Pitch: " << (preserve_pitch ? "Yes" : "No") << std::endl;
        audio_file.close();
        
        return true;
    });
    
    // Wait for result and handle UI update in the main thread
    g_idle_add([](gpointer data) -> gboolean {
        auto* future_ptr = static_cast<std::future<bool>*>(data);
        if (future_ptr->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            bool success = future_ptr->get();
            AudioStretch* self = reinterpret_cast<AudioStretch*>(future_ptr + 1);
            
            if (success) {
                self->update_progress(1.0);
                gtk_widget_set_visible(self->progress_bar_, FALSE);
                
                // Show success dialog
                GtkWidget* dialog = gtk_message_dialog_new(NULL,
                    GTK_DIALOG_MODAL,
                    GTK_MESSAGE_INFO,
                    GTK_BUTTONS_OK,
                    "Audio stretch complete. Output saved to:\n%s",
                    output_audio_path.c_str());
                gtk_dialog_run(GTK_DIALOG(dialog));
                gtk_widget_destroy(dialog);
            } else {
                self->show_error_dialog(self->get_last_error());
                gtk_widget_set_visible(self->progress_bar_, FALSE);
            }
            
            delete future_ptr;
            return FALSE;
        }
        return TRUE;
    }, new std::future<bool>[2] { std::move(result), std::future<bool>{} });
    
    return true;
}

// Analyze audio and suggest beat-aligned stretch points
std::vector<double> AudioStretch::analyze_beats(const std::string& audio_path) {
    std::vector<double> beats;
    
    if (audio_path.empty() || !std::filesystem::exists(audio_path)) {
        last_error_ = "Input audio file does not exist";
        return beats;
    }
    
    // TODO: Implement actual beat detection using an AI library
    
    // For demonstration, generate some random beats
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.5, 1.0);
    
    double time = 0.0;
    while (time < 60.0) { // Assume 60 seconds of audio
        time += dis(gen);
        beats.push_back(time);
    }
    
    return beats;
}

// Stretch audio to match a target tempo (BPM)
bool AudioStretch::stretch_to_tempo(const std::string& input_audio_path,
                                  const std::string& output_audio_path,
                                  double source_bpm,
                                  double target_bpm,
                                  bool preserve_pitch) {
    if (source_bpm <= 0.0 || target_bpm <= 0.0) {
        last_error_ = "Invalid BPM values. Must be greater than 0.";
        return false;
    }
    
    // Calculate the stretch ratio based on BPM values
    double ratio = source_bpm / target_bpm;
    
    // Call the regular stretch method with the calculated ratio
    return stretch_audio(input_audio_path, output_audio_path, ratio, preserve_pitch);
}

// Get the last error message
std::string AudioStretch::get_last_error() const {
    return last_error_;
}

// Get the UI widget for the audio stretch interface
GtkWidget* AudioStretch::get_ui_widget() {
    if (main_container_ == nullptr) {
        setup_ui();
    }
    return main_container_;
}

// Initialize UI components
void AudioStretch::setup_ui() {
    // Create main container
    main_container_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    gtk_widget_set_margin_start(main_container_, 12);
    gtk_widget_set_margin_end(main_container_, 12);
    gtk_widget_set_margin_top(main_container_, 12);
    gtk_widget_set_margin_bottom(main_container_, 12);
    
    // Input file selection
    GtkWidget* input_label = gtk_label_new("Input Audio File:");
    gtk_widget_set_halign(input_label, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(main_container_), input_label, FALSE, FALSE, 0);
    
    input_file_chooser_ = gtk_file_chooser_button_new("Select Input Audio", 
                                                     GTK_FILE_CHOOSER_ACTION_OPEN);
    gtk_box_pack_start(GTK_BOX(main_container_), input_file_chooser_, FALSE, FALSE, 0);
    
    // Output file selection
    GtkWidget* output_label = gtk_label_new("Output Audio File:");
    gtk_widget_set_halign(output_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(output_label, 10);
    gtk_box_pack_start(GTK_BOX(main_container_), output_label, FALSE, FALSE, 0);
    
    output_file_chooser_ = gtk_file_chooser_button_new("Select Output Location", 
                                                      GTK_FILE_CHOOSER_ACTION_SAVE);
    gtk_box_pack_start(GTK_BOX(main_container_), output_file_chooser_, FALSE, FALSE, 0);
    
    // Preserve pitch checkbox
    preserve_pitch_check_ = gtk_check_button_new_with_label("Preserve Pitch");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(preserve_pitch_check_), TRUE);
    gtk_widget_set_margin_top(preserve_pitch_check_, 10);
    gtk_box_pack_start(GTK_BOX(main_container_), preserve_pitch_check_, FALSE, FALSE, 0);
    
    // Create BPM mode checkbox
    bpm_mode_check_ = gtk_check_button_new_with_label("Use BPM Mode");
    gtk_widget_set_margin_top(bpm_mode_check_, 10);
    gtk_box_pack_start(GTK_BOX(main_container_), bpm_mode_check_, FALSE, FALSE, 0);
    g_signal_connect(bpm_mode_check_, "toggled", G_CALLBACK(on_bpm_mode_toggled), this);
    
    // Create a stack for different modes (ratio vs BPM)
    GtkWidget* mode_stack = gtk_stack_new();
    gtk_box_pack_start(GTK_BOX(main_container_), mode_stack, FALSE, FALSE, 0);
    
    // --- Ratio Mode UI ---
    GtkWidget* ratio_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_stack_add_named(GTK_STACK(mode_stack), ratio_box, "ratio_mode");
    
    // Ratio slider
    GtkWidget* ratio_label = gtk_label_new("Stretch Ratio:");
    gtk_widget_set_halign(ratio_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(ratio_label, 10);
    gtk_box_pack_start(GTK_BOX(ratio_box), ratio_label, FALSE, FALSE, 0);
    
    // Create a box for the scale and its value
    GtkWidget* scale_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(ratio_box), scale_box, FALSE, FALSE, 0);
    
    // Create the scale for the ratio
    ratio_scale_ = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.5, 2.0, 0.05);
    gtk_scale_set_value_pos(GTK_SCALE(ratio_scale_), GTK_POS_RIGHT);
    gtk_range_set_value(GTK_RANGE(ratio_scale_), 1.0);
    gtk_widget_set_hexpand(ratio_scale_, TRUE);
    gtk_box_pack_start(GTK_BOX(scale_box), ratio_scale_, TRUE, TRUE, 0);
    
    // Add labels for common ratio values
    GtkWidget* ratio_info = gtk_label_new("<small>0.5x: Faster (higher pitch)\n1.0x: Original\n2.0x: Slower (lower pitch)</small>");
    gtk_label_set_use_markup(GTK_LABEL(ratio_info), TRUE);
    gtk_widget_set_halign(ratio_info, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(ratio_box), ratio_info, FALSE, FALSE, 0);
    
    // --- BPM Mode UI ---
    GtkWidget* bpm_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_stack_add_named(GTK_STACK(mode_stack), bpm_box, "bpm_mode");
    
    // Source BPM
    GtkWidget* source_bpm_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(bpm_box), source_bpm_box, FALSE, FALSE, 0);
    
    GtkWidget* source_bpm_label = gtk_label_new("Source BPM:");
    gtk_widget_set_margin_end(source_bpm_label, 10);
    gtk_box_pack_start(GTK_BOX(source_bpm_box), source_bpm_label, FALSE, FALSE, 0);
    
    source_bpm_spin_ = gtk_spin_button_new_with_range(30, 300, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(source_bpm_spin_), 120);
    gtk_box_pack_start(GTK_BOX(source_bpm_box), source_bpm_spin_, TRUE, TRUE, 0);
    
    // Auto-detect BPM button
    detect_bpm_button_ = gtk_button_new_with_label("Detect BPM");
    gtk_box_pack_start(GTK_BOX(source_bpm_box), detect_bpm_button_, FALSE, FALSE, 0);
    g_signal_connect(detect_bpm_button_, "clicked", G_CALLBACK(on_detect_bpm_clicked), this);
    
    // Target BPM
    GtkWidget* target_bpm_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(bpm_box), target_bpm_box, FALSE, FALSE, 0);
    
    GtkWidget* target_bpm_label = gtk_label_new("Target BPM:");
    gtk_widget_set_margin_end(target_bpm_label, 10);
    gtk_box_pack_start(GTK_BOX(target_bpm_box), target_bpm_label, FALSE, FALSE, 0);
    
    target_bpm_spin_ = gtk_spin_button_new_with_range(30, 300, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(target_bpm_spin_), 120);
    gtk_box_pack_start(GTK_BOX(target_bpm_box), target_bpm_spin_, TRUE, TRUE, 0);
    
    // BPM explanation
    GtkWidget* bpm_info = gtk_label_new("<small>BPM (Beats Per Minute) determines the tempo of the audio.\nAdjust source and target BPM to stretch/compress accordingly.</small>");
    gtk_label_set_use_markup(GTK_LABEL(bpm_info), TRUE);
    gtk_widget_set_halign(bpm_info, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(bpm_box), bpm_info, FALSE, FALSE, 0);
    
    // Set initial stack page
    gtk_stack_set_visible_child_name(GTK_STACK(mode_stack), "ratio_mode");
    
    // Stretch button
    stretch_button_ = gtk_button_new_with_label("Stretch Audio");
    gtk_widget_set_margin_top(stretch_button_, 20);
    gtk_box_pack_start(GTK_BOX(main_container_), stretch_button_, FALSE, FALSE, 0);
    g_signal_connect(stretch_button_, "clicked", G_CALLBACK(on_stretch_clicked), this);
    
    // Progress bar
    progress_bar_ = gtk_progress_bar_new();
    gtk_widget_set_margin_top(progress_bar_, 10);
    gtk_box_pack_start(GTK_BOX(main_container_), progress_bar_, FALSE, FALSE, 0);
    gtk_widget_set_visible(progress_bar_, FALSE);
    
    // Store stack pointer for mode switching
    g_object_set_data(G_OBJECT(bpm_mode_check_), "mode_stack", mode_stack);
    
    // Show all widgets
    gtk_widget_show_all(main_container_);
}

// UI callback functions
void AudioStretch::on_stretch_clicked(GtkButton* button, gpointer user_data) {
    AudioStretch* self = static_cast<AudioStretch*>(user_data);
    
    // Get input file
    char* input_filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(self->input_file_chooser_));
    if (input_filename == NULL) {
        self->show_error_dialog("Please select an input audio file");
        return;
    }
    
    std::string input_path(input_filename);
    g_free(input_filename);
    
    // Get output file
    char* output_filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(self->output_file_chooser_));
    if (output_filename == NULL) {
        self->show_error_dialog("Please select an output file");
        return;
    }
    
    std::string output_path(output_filename);
    g_free(output_filename);
    
    // Get preserve pitch setting
    bool preserve_pitch = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(self->preserve_pitch_check_));
    
    // Check if we're in BPM mode
    bool bpm_mode = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(self->bpm_mode_check_));
    
    if (bpm_mode) {
        // Get BPM values
        double source_bpm = gtk_spin_button_get_value(GTK_SPIN_BUTTON(self->source_bpm_spin_));
        double target_bpm = gtk_spin_button_get_value(GTK_SPIN_BUTTON(self->target_bpm_spin_));
        
        // Process using BPM values
        self->stretch_to_tempo(input_path, output_path, source_bpm, target_bpm, preserve_pitch);
    } else {
        // Get stretch ratio
        double ratio = gtk_range_get_value(GTK_RANGE(self->ratio_scale_));
        
        // Process using ratio
        self->stretch_audio(input_path, output_path, ratio, preserve_pitch);
    }
}

void AudioStretch::on_bpm_mode_toggled(GtkToggleButton* toggle_button, gpointer user_data) {
    AudioStretch* self = static_cast<AudioStretch*>(user_data);
    
    // Get the mode stack
    GtkWidget* mode_stack = GTK_WIDGET(g_object_get_data(G_OBJECT(toggle_button), "mode_stack"));
    
    // Switch between ratio and BPM mode
    bool is_bpm_mode = gtk_toggle_button_get_active(toggle_button);
    if (is_bpm_mode) {
        gtk_stack_set_visible_child_name(GTK_STACK(mode_stack), "bpm_mode");
    } else {
        gtk_stack_set_visible_child_name(GTK_STACK(mode_stack), "ratio_mode");
    }
}

void AudioStretch::on_detect_bpm_clicked(GtkButton* button, gpointer user_data) {
    AudioStretch* self = static_cast<AudioStretch*>(user_data);
    
    // Get input file
    char* input_filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(self->input_file_chooser_));
    if (input_filename == NULL) {
        self->show_error_dialog("Please select an input audio file first");
        return;
    }
    
    std::string input_path(input_filename);
    g_free(input_filename);
    
    // Show progress bar
    gtk_widget_set_visible(self->progress_bar_, TRUE);
    gtk_progress_bar_pulse(GTK_PROGRESS_BAR(self->progress_bar_));
    
    // Analyze BPM in a separate thread
    std::thread([self, input_path]() {
        // Simulate BPM detection with a delay
        for (int i = 0; i < 10; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Update progress
            g_idle_add([](gpointer data) -> gboolean {
                AudioStretch* self = static_cast<AudioStretch*>(data);
                gtk_progress_bar_pulse(GTK_PROGRESS_BAR(self->progress_bar_));
                return FALSE;
            }, self);
        }
        
        // Detect BPM (this is a simulation)
        double detected_bpm = self->analyze_bpm(input_path);
        
        // Update UI
        g_idle_add([](gpointer data) -> gboolean {
            auto* params = static_cast<std::pair<AudioStretch*, double>*>(data);
            AudioStretch* self = params->first;
            double bpm = params->second;
            
            // Hide progress bar
            gtk_widget_set_visible(self->progress_bar_, FALSE);
            
            // Set the detected BPM value
            gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->source_bpm_spin_), bpm);
            
            // Show confirmation dialog
            GtkWidget* dialog = gtk_message_dialog_new(NULL,
                GTK_DIALOG_MODAL,
                GTK_MESSAGE_INFO,
                GTK_BUTTONS_OK,
                "Detected BPM: %.1f",
                bpm);
            gtk_dialog_run(GTK_DIALOG(dialog));
            gtk_widget_destroy(dialog);
            
            delete params;
            return FALSE;
        }, new std::pair<AudioStretch*, double>(self, detected_bpm));
    }).detach();
}

// Helper methods
bool AudioStretch::load_ai_model() {
    // TODO: Implement loading of the AI audio stretch model
    // For now, we just simulate success
    return true;
}

void AudioStretch::update_progress(double progress) {
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar_), progress);
}

void AudioStretch::show_error_dialog(const std::string& message) {
    GtkWidget* dialog = gtk_message_dialog_new(NULL,
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR,
        GTK_BUTTONS_OK,
        "%s",
        message.c_str());
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
}

double AudioStretch::analyze_bpm(const std::string& audio_path) {
    // TODO: Implement actual BPM detection using AI techniques
    
    // For demonstration, generate a random BPM between 80 and 150
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(80.0, 150.0);
    
    return std::round(dis(gen) * 10.0) / 10.0; // Round to 1 decimal place
}

void AudioStretch::update_ratio_from_bpm() {
    double source_bpm = gtk_spin_button_get_value(GTK_SPIN_BUTTON(source_bpm_spin_));
    double target_bpm = gtk_spin_button_get_value(GTK_SPIN_BUTTON(target_bpm_spin_));
    
    if (source_bpm > 0 && target_bpm > 0) {
        double ratio = source_bpm / target_bpm;
        
        // Update ratio scale if in visible UI
        if (!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(bpm_mode_check_))) {
            gtk_range_set_value(GTK_RANGE(ratio_scale_), ratio);
        }
    }
} 