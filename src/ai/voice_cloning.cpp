#include "voice_cloning.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <future>

// Constructor
VoiceCloning::VoiceCloning() : 
    main_container_(nullptr),
    reference_file_chooser_(nullptr),
    extract_button_(nullptr),
    profile_combo_(nullptr),
    input_text_view_(nullptr),
    synthesis_button_(nullptr),
    source_audio_chooser_(nullptr),
    conversion_button_(nullptr),
    progress_bar_(nullptr) {
    
    // Set default model path
    char* data_dir = g_build_filename(g_get_user_data_dir(), "blouedit", "models", "voice_clone", NULL);
    model_path_ = std::string(data_dir);
    g_free(data_dir);
}

// Destructor
VoiceCloning::~VoiceCloning() {
    cleanup();
}

// Initialize voice cloning system
bool VoiceCloning::initialize() {
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

    // Create profiles directory
    std::filesystem::path profiles_path = data_path / "profiles";
    if (!std::filesystem::exists(profiles_path)) {
        try {
            std::filesystem::create_directories(profiles_path);
        } catch (const std::exception& e) {
            last_error_ = std::string("Failed to create profiles directory: ") + e.what();
            return false;
        }
    }

    // Initialize UI elements
    setup_ui();
    
    // Load AI model
    return load_ai_model();
}

// Cleanup resources
void VoiceCloning::cleanup() {
    // Cleanup UI if needed
    if (main_container_ != nullptr) {
        // No need to destroy it here as GTK will handle widget destruction
        main_container_ = nullptr;
    }
}

// Extract voice profile from reference audio
bool VoiceCloning::extract_voice_profile(const std::string& reference_audio_path, 
                                        const std::string& output_profile_path) {
    if (reference_audio_path.empty()) {
        last_error_ = "Reference audio path is empty";
        return false;
    }
    
    if (!std::filesystem::exists(reference_audio_path)) {
        last_error_ = "Reference audio file does not exist";
        return false;
    }
    
    // Show progress bar activity
    gtk_widget_set_visible(progress_bar_, TRUE);
    gtk_progress_bar_pulse(GTK_PROGRESS_BAR(progress_bar_));
    
    // Run extraction in a separate thread to avoid UI freezing
    std::future<bool> result = std::async(std::launch::async, [this, reference_audio_path, output_profile_path]() {
        // TODO: Implement the actual voice profile extraction using an AI library
        // This would typically involve:
        // 1. Loading the reference audio
        // 2. Extracting voice characteristics (pitch, timbre, etc.)
        // 3. Saving the extracted profile to the output path
        
        // Simulate AI processing with a delay
        for (int i = 0; i <= 100; i += 10) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // Update progress bar on the main thread
            g_idle_add([](gpointer data) -> gboolean {
                int* progress = static_cast<int*>(data);
                VoiceCloning* self = reinterpret_cast<VoiceCloning*>(progress + 1);
                self->update_progress(*progress / 100.0);
                delete progress;
                return FALSE;
            }, new int[2] { i, reinterpret_cast<int>(this) });
        }
        
        // Create a dummy profile file for demonstration
        std::ofstream profile_file(output_profile_path);
        if (!profile_file.is_open()) {
            last_error_ = "Failed to create profile file";
            return false;
        }
        
        profile_file << "# Voice Profile\n";
        profile_file << "source_file=" << reference_audio_path << "\n";
        profile_file << "created=" << std::time(nullptr) << "\n";
        profile_file.close();
        
        return true;
    });
    
    // Wait for result and handle UI update in the main thread
    g_idle_add([](gpointer data) -> gboolean {
        auto* future_ptr = static_cast<std::future<bool>*>(data);
        if (future_ptr->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            bool success = future_ptr->get();
            VoiceCloning* self = reinterpret_cast<VoiceCloning*>(future_ptr + 1);
            
            if (success) {
                self->update_profile_list();
                self->update_progress(1.0);
                gtk_widget_set_visible(self->progress_bar_, FALSE);
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

// Apply cloned voice to text
bool VoiceCloning::apply_voice_to_text(const std::string& profile_path,
                                      const std::string& text,
                                      const std::string& output_audio_path) {
    if (profile_path.empty() || !is_profile_valid(profile_path)) {
        last_error_ = "Invalid voice profile";
        return false;
    }
    
    if (text.empty()) {
        last_error_ = "Text input is empty";
        return false;
    }
    
    // Show progress bar activity
    gtk_widget_set_visible(progress_bar_, TRUE);
    gtk_progress_bar_pulse(GTK_PROGRESS_BAR(progress_bar_));
    
    // Run synthesis in a separate thread to avoid UI freezing
    std::future<bool> result = std::async(std::launch::async, [this, profile_path, text, output_audio_path]() {
        // TODO: Implement actual text-to-speech with voice cloning using an AI library
        // This would typically involve:
        // 1. Loading the voice profile
        // 2. Using a TTS model with the voice characteristics from the profile
        // 3. Generating audio from the text with the cloned voice
        // 4. Saving the generated audio to the output path
        
        // Simulate AI processing with a delay
        for (int i = 0; i <= 100; i += 5) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Update progress bar on the main thread
            g_idle_add([](gpointer data) -> gboolean {
                int* progress = static_cast<int*>(data);
                VoiceCloning* self = reinterpret_cast<VoiceCloning*>(progress + 1);
                self->update_progress(*progress / 100.0);
                delete progress;
                return FALSE;
            }, new int[2] { i, reinterpret_cast<int>(this) });
        }
        
        // Create a dummy audio file for demonstration
        std::ofstream audio_file(output_audio_path);
        if (!audio_file.is_open()) {
            last_error_ = "Failed to create output audio file";
            return false;
        }
        
        audio_file << "RIFF" << std::endl; // Dummy audio header
        audio_file.close();
        
        return true;
    });
    
    // Wait for result and handle UI update in the main thread
    g_idle_add([](gpointer data) -> gboolean {
        auto* future_ptr = static_cast<std::future<bool>*>(data);
        if (future_ptr->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            bool success = future_ptr->get();
            VoiceCloning* self = reinterpret_cast<VoiceCloning*>(future_ptr + 1);
            
            if (success) {
                self->update_progress(1.0);
                gtk_widget_set_visible(self->progress_bar_, FALSE);
                
                // Show success dialog
                GtkWidget* dialog = gtk_message_dialog_new(NULL,
                    GTK_DIALOG_MODAL,
                    GTK_MESSAGE_INFO,
                    GTK_BUTTONS_OK,
                    "Voice synthesis complete. Audio saved to:\n%s",
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

// Apply cloned voice to existing audio
bool VoiceCloning::apply_voice_to_audio(const std::string& profile_path,
                                       const std::string& source_audio_path,
                                       const std::string& output_audio_path) {
    if (profile_path.empty() || !is_profile_valid(profile_path)) {
        last_error_ = "Invalid voice profile";
        return false;
    }
    
    if (source_audio_path.empty() || !std::filesystem::exists(source_audio_path)) {
        last_error_ = "Source audio file does not exist";
        return false;
    }
    
    // Show progress bar activity
    gtk_widget_set_visible(progress_bar_, TRUE);
    gtk_progress_bar_pulse(GTK_PROGRESS_BAR(progress_bar_));
    
    // Run conversion in a separate thread to avoid UI freezing
    std::future<bool> result = std::async(std::launch::async, [this, profile_path, source_audio_path, output_audio_path]() {
        // TODO: Implement actual voice conversion using an AI library
        // This would typically involve:
        // 1. Loading the voice profile
        // 2. Loading the source audio
        // 3. Extracting speech content while preserving non-speech elements
        // 4. Applying the voice characteristics from the profile to the speech content
        // 5. Saving the generated audio to the output path
        
        // Simulate AI processing with a delay
        for (int i = 0; i <= 100; i += 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Update progress bar on the main thread
            g_idle_add([](gpointer data) -> gboolean {
                int* progress = static_cast<int*>(data);
                VoiceCloning* self = reinterpret_cast<VoiceCloning*>(progress + 1);
                self->update_progress(*progress / 100.0);
                delete progress;
                return FALSE;
            }, new int[2] { i, reinterpret_cast<int>(this) });
        }
        
        // Create a dummy audio file for demonstration
        std::ofstream audio_file(output_audio_path);
        if (!audio_file.is_open()) {
            last_error_ = "Failed to create output audio file";
            return false;
        }
        
        audio_file << "RIFF" << std::endl; // Dummy audio header
        audio_file.close();
        
        return true;
    });
    
    // Wait for result and handle UI update in the main thread
    g_idle_add([](gpointer data) -> gboolean {
        auto* future_ptr = static_cast<std::future<bool>*>(data);
        if (future_ptr->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            bool success = future_ptr->get();
            VoiceCloning* self = reinterpret_cast<VoiceCloning*>(future_ptr + 1);
            
            if (success) {
                self->update_progress(1.0);
                gtk_widget_set_visible(self->progress_bar_, FALSE);
                
                // Show success dialog
                GtkWidget* dialog = gtk_message_dialog_new(NULL,
                    GTK_DIALOG_MODAL,
                    GTK_MESSAGE_INFO,
                    GTK_BUTTONS_OK,
                    "Voice conversion complete. Audio saved to:\n%s",
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

// Get the last error message
std::string VoiceCloning::get_last_error() const {
    return last_error_;
}

// Check if a voice profile is valid
bool VoiceCloning::is_profile_valid(const std::string& profile_path) const {
    if (!std::filesystem::exists(profile_path)) {
        return false;
    }
    
    // TODO: Add more validation of the profile file content
    return true;
}

// List available voice profiles
std::vector<std::string> VoiceCloning::list_profiles() const {
    std::vector<std::string> profiles;
    
    std::filesystem::path profiles_dir = std::filesystem::path(model_path_) / "profiles";
    if (!std::filesystem::exists(profiles_dir)) {
        return profiles;
    }
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(profiles_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".profile") {
                profiles.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::exception& e) {
        // Log the error but return what we have so far
        std::cerr << "Error listing profiles: " << e.what() << std::endl;
    }
    
    return profiles;
}

// Get the UI widget for the voice cloning interface
GtkWidget* VoiceCloning::get_ui_widget() {
    if (main_container_ == nullptr) {
        setup_ui();
    }
    return main_container_;
}

// Initialize UI components
void VoiceCloning::setup_ui() {
    // Create main container
    main_container_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(main_container_, 12);
    gtk_widget_set_margin_end(main_container_, 12);
    gtk_widget_set_margin_top(main_container_, 12);
    gtk_widget_set_margin_bottom(main_container_, 12);
    
    // Create notebook for different operations
    GtkWidget* notebook = gtk_notebook_new();
    gtk_box_pack_start(GTK_BOX(main_container_), notebook, TRUE, TRUE, 0);
    
    // === Extract Profile Tab ===
    GtkWidget* extract_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), extract_page, 
                            gtk_label_new("Extract Voice Profile"));
    
    // File chooser for reference audio
    GtkWidget* ref_label = gtk_label_new("Select reference audio file:");
    gtk_widget_set_halign(ref_label, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(extract_page), ref_label, FALSE, FALSE, 0);
    
    reference_file_chooser_ = gtk_file_chooser_button_new("Select Reference Audio", 
                                                         GTK_FILE_CHOOSER_ACTION_OPEN);
    gtk_box_pack_start(GTK_BOX(extract_page), reference_file_chooser_, FALSE, FALSE, 0);
    
    // Profile name entry
    GtkWidget* name_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(extract_page), name_box, FALSE, FALSE, 0);
    
    GtkWidget* name_label = gtk_label_new("Profile Name:");
    gtk_box_pack_start(GTK_BOX(name_box), name_label, FALSE, FALSE, 0);
    
    GtkWidget* name_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(name_entry), "Enter profile name");
    gtk_box_pack_start(GTK_BOX(name_box), name_entry, TRUE, TRUE, 0);
    
    // Extract button
    extract_button_ = gtk_button_new_with_label("Extract Voice Profile");
    gtk_widget_set_margin_top(extract_button_, 10);
    gtk_box_pack_start(GTK_BOX(extract_page), extract_button_, FALSE, FALSE, 0);
    g_signal_connect(extract_button_, "clicked", G_CALLBACK(on_extract_clicked), this);
    
    // === Text-to-Speech Tab ===
    GtkWidget* tts_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), tts_page, 
                            gtk_label_new("Text to Speech"));
    
    // Profile selection
    GtkWidget* profile_label = gtk_label_new("Select Voice Profile:");
    gtk_widget_set_halign(profile_label, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(tts_page), profile_label, FALSE, FALSE, 0);
    
    profile_combo_ = gtk_combo_box_text_new();
    gtk_box_pack_start(GTK_BOX(tts_page), profile_combo_, FALSE, FALSE, 0);
    
    // Text input
    GtkWidget* text_label = gtk_label_new("Enter Text:");
    gtk_widget_set_halign(text_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(text_label, 10);
    gtk_box_pack_start(GTK_BOX(tts_page), text_label, FALSE, FALSE, 0);
    
    GtkWidget* scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), 
                                  GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_min_content_height(GTK_SCROLLED_WINDOW(scroll), 100);
    gtk_box_pack_start(GTK_BOX(tts_page), scroll, TRUE, TRUE, 0);
    
    input_text_view_ = gtk_text_view_new();
    gtk_container_add(GTK_CONTAINER(scroll), input_text_view_);
    
    // Output file
    GtkWidget* output_label = gtk_label_new("Output Audio File:");
    gtk_widget_set_halign(output_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(output_label, 10);
    gtk_box_pack_start(GTK_BOX(tts_page), output_label, FALSE, FALSE, 0);
    
    GtkWidget* tts_output_chooser = gtk_file_chooser_button_new("Select Output File", 
                                                             GTK_FILE_CHOOSER_ACTION_SAVE);
    gtk_box_pack_start(GTK_BOX(tts_page), tts_output_chooser, FALSE, FALSE, 0);
    
    // Synthesize button
    synthesis_button_ = gtk_button_new_with_label("Generate Speech");
    gtk_widget_set_margin_top(synthesis_button_, 10);
    gtk_box_pack_start(GTK_BOX(tts_page), synthesis_button_, FALSE, FALSE, 0);
    g_signal_connect(synthesis_button_, "clicked", G_CALLBACK(on_synthesis_clicked), this);
    
    // === Voice Conversion Tab ===
    GtkWidget* conv_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), conv_page, 
                            gtk_label_new("Voice Conversion"));
    
    // Profile selection (reuse same combo box)
    GtkWidget* conv_profile_label = gtk_label_new("Select Target Voice Profile:");
    gtk_widget_set_halign(conv_profile_label, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(conv_page), conv_profile_label, FALSE, FALSE, 0);
    
    GtkWidget* conv_profile_combo = gtk_combo_box_text_new();
    gtk_box_pack_start(GTK_BOX(conv_page), conv_profile_combo, FALSE, FALSE, 0);
    
    // Update both profile combos when profiles are updated
    g_object_set_data(G_OBJECT(profile_combo_), "alt_combo", conv_profile_combo);
    
    // Source audio
    GtkWidget* source_label = gtk_label_new("Select Source Audio:");
    gtk_widget_set_halign(source_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(source_label, 10);
    gtk_box_pack_start(GTK_BOX(conv_page), source_label, FALSE, FALSE, 0);
    
    source_audio_chooser_ = gtk_file_chooser_button_new("Select Source Audio", 
                                                       GTK_FILE_CHOOSER_ACTION_OPEN);
    gtk_box_pack_start(GTK_BOX(conv_page), source_audio_chooser_, FALSE, FALSE, 0);
    
    // Output file
    GtkWidget* conv_output_label = gtk_label_new("Output Audio File:");
    gtk_widget_set_halign(conv_output_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(conv_output_label, 10);
    gtk_box_pack_start(GTK_BOX(conv_page), conv_output_label, FALSE, FALSE, 0);
    
    GtkWidget* conv_output_chooser = gtk_file_chooser_button_new("Select Output File", 
                                                              GTK_FILE_CHOOSER_ACTION_SAVE);
    gtk_box_pack_start(GTK_BOX(conv_page), conv_output_chooser, FALSE, FALSE, 0);
    
    // Convert button
    conversion_button_ = gtk_button_new_with_label("Convert Voice");
    gtk_widget_set_margin_top(conversion_button_, 10);
    gtk_box_pack_start(GTK_BOX(conv_page), conversion_button_, FALSE, FALSE, 0);
    g_signal_connect(conversion_button_, "clicked", G_CALLBACK(on_conversion_clicked), this);
    
    // Progress bar (shared across all operations)
    progress_bar_ = gtk_progress_bar_new();
    gtk_widget_set_margin_top(progress_bar_, 15);
    gtk_box_pack_start(GTK_BOX(main_container_), progress_bar_, FALSE, FALSE, 0);
    gtk_widget_set_visible(progress_bar_, FALSE);
    
    // Update profile list
    update_profile_list();
    
    // Show all widgets
    gtk_widget_show_all(main_container_);
}

// UI callback functions
void VoiceCloning::on_extract_clicked(GtkButton* button, gpointer user_data) {
    VoiceCloning* self = static_cast<VoiceCloning*>(user_data);
    
    // Get selected file
    char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(self->reference_file_chooser_));
    if (filename == NULL) {
        self->show_error_dialog("Please select a reference audio file");
        return;
    }
    
    std::string reference_audio_path(filename);
    g_free(filename);
    
    // Get profile name
    GtkWidget* name_entry = GTK_WIDGET(g_object_get_data(G_OBJECT(self->extract_button_), "name_entry"));
    const char* profile_name = gtk_entry_get_text(GTK_ENTRY(name_entry));
    if (profile_name == NULL || strlen(profile_name) == 0) {
        self->show_error_dialog("Please enter a profile name");
        return;
    }
    
    // Create output profile path
    std::string output_profile_path = std::filesystem::path(self->model_path_) / "profiles" / 
                                      (std::string(profile_name) + ".profile");
    
    // Start extraction
    self->extract_voice_profile(reference_audio_path, output_profile_path);
}

void VoiceCloning::on_synthesis_clicked(GtkButton* button, gpointer user_data) {
    VoiceCloning* self = static_cast<VoiceCloning*>(user_data);
    
    // Get selected profile
    gchar* profile_text = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(self->profile_combo_));
    if (profile_text == NULL) {
        self->show_error_dialog("Please select a voice profile");
        return;
    }
    
    std::string profile_path = std::filesystem::path(self->model_path_) / "profiles" / profile_text;
    g_free(profile_text);
    
    // Get input text
    GtkTextBuffer* buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(self->input_text_view_));
    GtkTextIter start, end;
    gtk_text_buffer_get_bounds(buffer, &start, &end);
    char* text = gtk_text_buffer_get_text(buffer, &start, &end, FALSE);
    if (text == NULL || strlen(text) == 0) {
        self->show_error_dialog("Please enter some text");
        g_free(text);
        return;
    }
    
    std::string input_text(text);
    g_free(text);
    
    // Get output file path
    GtkWidget* output_chooser = GTK_WIDGET(g_object_get_data(G_OBJECT(self->synthesis_button_), "output_chooser"));
    char* output_filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(output_chooser));
    if (output_filename == NULL) {
        self->show_error_dialog("Please select an output file");
        return;
    }
    
    std::string output_audio_path(output_filename);
    g_free(output_filename);
    
    // Start synthesis
    self->apply_voice_to_text(profile_path, input_text, output_audio_path);
}

void VoiceCloning::on_conversion_clicked(GtkButton* button, gpointer user_data) {
    VoiceCloning* self = static_cast<VoiceCloning*>(user_data);
    
    // Get selected profile
    GtkWidget* conv_profile_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(self->profile_combo_), "alt_combo"));
    gchar* profile_text = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(conv_profile_combo));
    if (profile_text == NULL) {
        self->show_error_dialog("Please select a voice profile");
        return;
    }
    
    std::string profile_path = std::filesystem::path(self->model_path_) / "profiles" / profile_text;
    g_free(profile_text);
    
    // Get source audio file
    char* source_filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(self->source_audio_chooser_));
    if (source_filename == NULL) {
        self->show_error_dialog("Please select a source audio file");
        return;
    }
    
    std::string source_audio_path(source_filename);
    g_free(source_filename);
    
    // Get output file path
    GtkWidget* output_chooser = GTK_WIDGET(g_object_get_data(G_OBJECT(self->conversion_button_), "output_chooser"));
    char* output_filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(output_chooser));
    if (output_filename == NULL) {
        self->show_error_dialog("Please select an output file");
        return;
    }
    
    std::string output_audio_path(output_filename);
    g_free(output_filename);
    
    // Start conversion
    self->apply_voice_to_audio(profile_path, source_audio_path, output_audio_path);
}

// Helper methods
bool VoiceCloning::load_ai_model() {
    // TODO: Implement loading of the AI voice cloning model
    // For now, we just simulate success
    return true;
}

void VoiceCloning::update_profile_list() {
    // Clear existing items
    gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(profile_combo_));
    
    // Get alt combo box
    GtkWidget* alt_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(profile_combo_), "alt_combo"));
    if (alt_combo != nullptr) {
        gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(alt_combo));
    }
    
    // Get profiles
    std::vector<std::string> profiles = list_profiles();
    
    // Add profiles to combo boxes
    for (const auto& profile : profiles) {
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(profile_combo_), profile.c_str());
        
        if (alt_combo != nullptr) {
            gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(alt_combo), profile.c_str());
        }
    }
    
    // Select first item if available
    if (!profiles.empty()) {
        gtk_combo_box_set_active(GTK_COMBO_BOX(profile_combo_), 0);
        
        if (alt_combo != nullptr) {
            gtk_combo_box_set_active(GTK_COMBO_BOX(alt_combo), 0);
        }
    }
}

void VoiceCloning::show_error_dialog(const std::string& message) {
    GtkWidget* dialog = gtk_message_dialog_new(NULL,
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR,
        GTK_BUTTONS_OK,
        "%s",
        message.c_str());
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
}

void VoiceCloning::update_progress(double progress) {
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar_), progress);
} 