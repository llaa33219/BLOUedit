#ifndef AUDIO_STRETCH_H
#define AUDIO_STRETCH_H

#include <glib.h>
#include <gtk/gtk.h>
#include <string>
#include <memory>
#include <vector>

// Audio Stretch class - Provides functionality to stretch or compress audio
// while maintaining pitch quality using AI techniques
class AudioStretch {
public:
    AudioStretch();
    ~AudioStretch();

    // Initialize the audio stretch system
    bool initialize();

    // Clean up resources
    void cleanup();

    // Stretch audio file with specified ratio
    // ratio > 1.0: stretch (slower)
    // ratio < 1.0: compress (faster)
    // preserve_pitch: whether to maintain original pitch
    bool stretch_audio(const std::string& input_audio_path,
                      const std::string& output_audio_path,
                      double ratio,
                      bool preserve_pitch);

    // Analyze audio and suggest beat-aligned stretch points
    std::vector<double> analyze_beats(const std::string& audio_path);

    // Stretch audio to match a target tempo (BPM)
    bool stretch_to_tempo(const std::string& input_audio_path,
                         const std::string& output_audio_path,
                         double source_bpm,
                         double target_bpm,
                         bool preserve_pitch);

    // Get the last error message
    std::string get_last_error() const;

    // Get the UI widget for the audio stretch interface
    GtkWidget* get_ui_widget();

private:
    // Last error message
    std::string last_error_;

    // Path to the AI model
    std::string model_path_;

    // UI components
    GtkWidget* main_container_;
    GtkWidget* input_file_chooser_;
    GtkWidget* output_file_chooser_;
    GtkWidget* ratio_scale_;
    GtkWidget* preserve_pitch_check_;
    GtkWidget* stretch_button_;
    GtkWidget* progress_bar_;
    GtkWidget* bpm_mode_check_;
    GtkWidget* source_bpm_spin_;
    GtkWidget* target_bpm_spin_;
    GtkWidget* detect_bpm_button_;

    // Initialize UI components
    void setup_ui();

    // UI callback functions
    static void on_stretch_clicked(GtkButton* button, gpointer user_data);
    static void on_bpm_mode_toggled(GtkToggleButton* toggle_button, gpointer user_data);
    static void on_detect_bpm_clicked(GtkButton* button, gpointer user_data);

    // Helper methods
    bool load_ai_model();
    void update_progress(double progress);
    void show_error_dialog(const std::string& message);
    double analyze_bpm(const std::string& audio_path);
    void update_ratio_from_bpm();
};

#endif // AUDIO_STRETCH_H 