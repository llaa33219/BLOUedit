#ifndef VOICE_CLONING_H
#define VOICE_CLONING_H

#include <glib.h>
#include <gtk/gtk.h>
#include <string>
#include <vector>
#include <memory>

// Voice Cloning class - Provides functionality to clone a voice from sample audio
// and apply that voice to new text or audio
class VoiceCloning {
public:
    VoiceCloning();
    ~VoiceCloning();

    // Initialize the voice cloning system
    bool initialize();

    // Clean up resources
    void cleanup();

    // Extract voice characteristics from a reference audio file
    bool extract_voice_profile(const std::string& reference_audio_path, 
                               const std::string& output_profile_path);

    // Apply cloned voice to text (text-to-speech with cloned voice)
    bool apply_voice_to_text(const std::string& profile_path,
                             const std::string& text,
                             const std::string& output_audio_path);

    // Apply cloned voice to existing audio (voice conversion)
    bool apply_voice_to_audio(const std::string& profile_path,
                              const std::string& source_audio_path,
                              const std::string& output_audio_path);

    // Get the last error message
    std::string get_last_error() const;

    // Check if a voice profile is valid
    bool is_profile_valid(const std::string& profile_path) const;

    // List available voice profiles
    std::vector<std::string> list_profiles() const;

    // Get the UI widget for the voice cloning interface
    GtkWidget* get_ui_widget();

private:
    // Last error message
    std::string last_error_;

    // UI components
    GtkWidget* main_container_;
    GtkWidget* reference_file_chooser_;
    GtkWidget* extract_button_;
    GtkWidget* profile_combo_;
    GtkWidget* input_text_view_;
    GtkWidget* synthesis_button_;
    GtkWidget* source_audio_chooser_;
    GtkWidget* conversion_button_;
    GtkWidget* progress_bar_;

    // Path to the AI model
    std::string model_path_;

    // Initialize UI components
    void setup_ui();

    // UI callback functions
    static void on_extract_clicked(GtkButton* button, gpointer user_data);
    static void on_synthesis_clicked(GtkButton* button, gpointer user_data);
    static void on_conversion_clicked(GtkButton* button, gpointer user_data);

    // Helper methods for AI processing
    bool load_ai_model();
    void update_profile_list();
    void show_error_dialog(const std::string& message);
    void update_progress(double progress);
};

#endif // VOICE_CLONING_H 