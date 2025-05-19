#ifndef RESIZE_H
#define RESIZE_H

#include <glib.h>
#include <gtk/gtk.h>
#include <string>
#include <memory>

// Video Resize class - Provides functionality to resize video clips
// with various scaling options, aspect ratio controls, and quality settings
class VideoResize {
public:
    // Resize scaling modes
    enum class ScalingMode {
        FIT,        // Fit within dimensions while maintaining aspect ratio
        FILL,       // Fill dimensions while maintaining aspect ratio (may crop)
        STRETCH,    // Stretch to exactly match dimensions (may distort)
        CUSTOM      // Custom scaling (separate width/height scaling)
    };
    
    // Resize quality options
    enum class ResizeQuality {
        FAST,       // Fastest, lowest quality
        GOOD,       // Balance of speed and quality
        BEST        // Highest quality, slowest
    };

    VideoResize();
    ~VideoResize();

    // Initialize the resize feature
    bool initialize();

    // Clean up resources
    void cleanup();

    // Resize video file with specified settings
    bool resize_video(const std::string& input_video_path,
                     const std::string& output_video_path,
                     int target_width,
                     int target_height,
                     ScalingMode scaling_mode,
                     ResizeQuality quality);

    // Calculate new dimensions based on source dimensions and scaling mode
    std::pair<int, int> calculate_dimensions(int source_width, 
                                           int source_height,
                                           int target_width, 
                                           int target_height,
                                           ScalingMode scaling_mode);

    // Get the last error message
    std::string get_last_error() const;

    // Get the UI widget for the resize interface
    GtkWidget* get_ui_widget();

private:
    // Last error message
    std::string last_error_;

    // UI components
    GtkWidget* main_container_;
    GtkWidget* input_file_chooser_;
    GtkWidget* output_file_chooser_;
    GtkWidget* width_spin_;
    GtkWidget* height_spin_;
    GtkWidget* lock_aspect_check_;
    GtkWidget* scaling_combo_;
    GtkWidget* quality_combo_;
    GtkWidget* resize_button_;
    GtkWidget* progress_bar_;
    GtkWidget* preview_area_;
    
    // Source video dimensions
    int source_width_;
    int source_height_;
    double aspect_ratio_;

    // Initialize UI components
    void setup_ui();

    // UI callback functions
    static void on_resize_clicked(GtkButton* button, gpointer user_data);
    static void on_input_file_changed(GtkFileChooserButton* chooser, gpointer user_data);
    static void on_width_changed(GtkSpinButton* spin_button, gpointer user_data);
    static void on_height_changed(GtkSpinButton* spin_button, gpointer user_data);
    static void on_lock_aspect_toggled(GtkToggleButton* toggle_button, gpointer user_data);
    static void on_scaling_mode_changed(GtkComboBox* combo_box, gpointer user_data);

    // Helper methods
    void update_progress(double progress);
    void show_error_dialog(const std::string& message);
    bool get_video_dimensions(const std::string& video_path, int& width, int& height);
    void update_preview();
    void update_width_from_height();
    void update_height_from_width();
};

#endif // RESIZE_H 