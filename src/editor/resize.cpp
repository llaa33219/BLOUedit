#include "resize.h"
#include <iostream>
#include <filesystem>
#include <thread>
#include <future>
#include <cmath>
#include <algorithm>
#include <fstream>  // For std::ofstream

// GTK4 specific includes
#include <gtk/gtk.h>

// Constructor
VideoResize::VideoResize() :
    main_container_(nullptr),
    input_file_chooser_(nullptr),
    output_file_chooser_(nullptr),
    width_spin_(nullptr),
    height_spin_(nullptr),
    lock_aspect_check_(nullptr),
    scaling_combo_(nullptr),
    quality_combo_(nullptr),
    resize_button_(nullptr),
    progress_bar_(nullptr),
    preview_area_(nullptr),
    source_width_(0),
    source_height_(0),
    aspect_ratio_(1.0) {
}

// Destructor
VideoResize::~VideoResize() {
    cleanup();
}

// Initialize resize feature
bool VideoResize::initialize() {
    // Initialize UI elements
    setup_ui();
    return true;
}

// Clean up resources
void VideoResize::cleanup() {
    // Cleanup UI if needed
    if (main_container_ != nullptr) {
        // No need to destroy it here as GTK will handle widget destruction
        main_container_ = nullptr;
    }
}

// Resize video file with specified settings
bool VideoResize::resize_video(const std::string& input_video_path,
                             const std::string& output_video_path,
                             int target_width,
                             int target_height,
                             ScalingMode scaling_mode,
                             ResizeQuality quality) {
    if (input_video_path.empty() || !std::filesystem::exists(input_video_path)) {
        last_error_ = "Input video file does not exist";
        return false;
    }
    
    if (output_video_path.empty()) {
        last_error_ = "Output video path is empty";
        return false;
    }
    
    if (target_width <= 0 || target_height <= 0) {
        last_error_ = "Invalid dimensions. Width and height must be greater than 0.";
        return false;
    }
    
    // Get source dimensions if not already loaded
    if (source_width_ <= 0 || source_height_ <= 0) {
        if (!get_video_dimensions(input_video_path, source_width_, source_height_)) {
            return false;
        }
    }
    
    // Calculate actual dimensions based on scaling mode
    std::pair<int, int> dimensions = calculate_dimensions(
        source_width_, source_height_, target_width, target_height, scaling_mode);
    
    int final_width = dimensions.first;
    int final_height = dimensions.second;
    
    // Show progress bar activity
    gtk_widget_set_visible(progress_bar_, TRUE);
    gtk_progress_bar_set_pulse_step(GTK_PROGRESS_BAR(progress_bar_), 0.1);
    gtk_progress_bar_pulse(GTK_PROGRESS_BAR(progress_bar_));
    
    // Run resizing in a separate thread to avoid UI freezing
    std::future<bool> result = std::async(std::launch::async, [this, input_video_path, output_video_path, 
                                                             final_width, final_height, quality]() {
        // TODO: Implement actual video resizing using a video processing library
        // This would typically involve:
        // 1. Loading the input video
        // 2. Setting up a video processing pipeline with appropriate scaling filter
        // 3. Processing the video frames
        // 4. Saving the processed video to the output path
        
        // For demonstration, we'll just simulate the processing
        std::string quality_arg;
        switch (quality) {
            case ResizeQuality::FAST:
                quality_arg = "fast bilinear";
                break;
            case ResizeQuality::GOOD:
                quality_arg = "bicubic";
                break;
            case ResizeQuality::BEST:
                quality_arg = "lanczos";
                break;
        }
        
        // Simulate video processing with a delay
        int total_frames = 100; // Simulate 100 frames
        for (int i = 0; i <= total_frames; i++) {
            // Longer delay for higher quality
            int delay_ms = 30;
            if (quality == ResizeQuality::GOOD) delay_ms = 50;
            if (quality == ResizeQuality::BEST) delay_ms = 70;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            
            // Update progress bar on the main thread
            g_idle_add([](gpointer data) -> gboolean {
                int* progress_data = static_cast<int*>(data);
                int frame = progress_data[0];
                int total = progress_data[1];
                VideoResize* self = reinterpret_cast<VideoResize*>(progress_data + 2);
                
                self->update_progress(static_cast<double>(frame) / total);
                
                delete[] progress_data;
                return FALSE;
            }, new int[3] { i, total_frames, reinterpret_cast<int>(this) });
        }
        
        // Create a dummy output file for demonstration
        std::ofstream video_file(output_video_path);
        if (!video_file.is_open()) {
            last_error_ = "Failed to create output video file";
            return false;
        }
        
        // Write some dummy data
        video_file << "MOCK VIDEO FILE" << std::endl;
        video_file << "Width: " << final_width << std::endl;
        video_file << "Height: " << final_height << std::endl;
        video_file << "Quality: " << quality_arg << std::endl;
        video_file << "Source: " << input_video_path << std::endl;
        video_file.close();
        
        return true;
    });
    
    // Wait for result and handle UI update in the main thread
    g_idle_add([](gpointer data) -> gboolean {
        auto* future_ptr = static_cast<std::future<bool>*>(data);
        if (future_ptr->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            bool success = future_ptr->get();
            VideoResize* self = reinterpret_cast<VideoResize*>(future_ptr + 1);
            
            if (success) {
                self->update_progress(1.0);
                gtk_widget_set_visible(self->progress_bar_, FALSE);
                
                // Show success dialog
                GtkWidget* dialog = gtk_message_dialog_new(NULL,
                    GTK_DIALOG_MODAL,
                    GTK_MESSAGE_INFO,
                    GTK_BUTTONS_OK,
                    "Video resize complete. Output saved to:\n%s",
                    output_video_path.c_str());
                    
                g_signal_connect(dialog, "response", G_CALLBACK([](GtkDialog *dialog, gint response_id, gpointer user_data) {
                    gtk_window_destroy(GTK_WINDOW(dialog));
                }), NULL);
                
                gtk_widget_show(dialog);
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

// Calculate new dimensions based on source dimensions and scaling mode
std::pair<int, int> VideoResize::calculate_dimensions(int source_width, 
                                                   int source_height,
                                                   int target_width, 
                                                   int target_height,
                                                   ScalingMode scaling_mode) {
    switch (scaling_mode) {
        case ScalingMode::FIT: {
            // Scale to fit while maintaining aspect ratio
            double src_aspect = static_cast<double>(source_width) / source_height;
            double target_aspect = static_cast<double>(target_width) / target_height;
            
            if (src_aspect > target_aspect) {
                // Width-constrained
                int new_width = target_width;
                int new_height = static_cast<int>(new_width / src_aspect);
                return {new_width, new_height};
            } else {
                // Height-constrained
                int new_height = target_height;
                int new_width = static_cast<int>(new_height * src_aspect);
                return {new_width, new_height};
            }
        }
        case ScalingMode::FILL: {
            // Scale to fill while maintaining aspect ratio
            double src_aspect = static_cast<double>(source_width) / source_height;
            double target_aspect = static_cast<double>(target_width) / target_height;
            
            if (src_aspect < target_aspect) {
                // Width-constrained
                int new_width = target_width;
                int new_height = static_cast<int>(new_width / src_aspect);
                return {new_width, new_height};
            } else {
                // Height-constrained
                int new_height = target_height;
                int new_width = static_cast<int>(new_height * src_aspect);
                return {new_width, new_height};
            }
        }
        case ScalingMode::STRETCH:
            // Stretch to exactly match dimensions
            return {target_width, target_height};
            
        case ScalingMode::CUSTOM:
            // Custom dimensions - use as specified
            return {target_width, target_height};
            
        default:
            // Default to exact dimensions
            return {target_width, target_height};
    }
}

// Get the last error message
std::string VideoResize::get_last_error() const {
    return last_error_;
}

// Get the UI widget for the resize interface
GtkWidget* VideoResize::get_ui_widget() {
    if (main_container_ == nullptr) {
        setup_ui();
    }
    return main_container_;
}

// Initialize UI components
void VideoResize::setup_ui() {
    // Create main container
    main_container_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    gtk_widget_set_margin_start(main_container_, 12);
    gtk_widget_set_margin_end(main_container_, 12);
    gtk_widget_set_margin_top(main_container_, 12);
    gtk_widget_set_margin_bottom(main_container_, 12);
    
    // Input file selection
    GtkWidget* input_label = gtk_label_new("Input Video File:");
    gtk_widget_set_halign(input_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(main_container_), input_label);
    
    // In GTK4, GtkFileChooserButton is deprecated, using Button + FileChooserDialog
    GtkWidget* input_button = gtk_button_new_with_label("Select Input Video");
    input_file_chooser_ = input_button;
    gtk_box_append(GTK_BOX(main_container_), input_file_chooser_);
    
    g_signal_connect(input_button, "clicked", G_CALLBACK([](GtkButton *button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        
        GtkWidget *dialog = gtk_file_chooser_dialog_new("Open Video File",
                                                   NULL,
                                                   GTK_FILE_CHOOSER_ACTION_OPEN,
                                                   "_Cancel", GTK_RESPONSE_CANCEL,
                                                   "_Open", GTK_RESPONSE_ACCEPT,
                                                   NULL);
                                                   
        g_signal_connect(dialog, "response", G_CALLBACK([](GtkDialog *dialog, gint response_id, gpointer user_data) {
            if (response_id == GTK_RESPONSE_ACCEPT) {
                VideoResize* self = static_cast<VideoResize*>(user_data);
                GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
                GFile *file = gtk_file_chooser_get_file(chooser);
                
                if (file) {
                    char *filename = g_file_get_path(file);
                    
                    if (filename) {
                        std::string input_path(filename);
                        g_free(filename);
                        
                        // Update button label with file name
                        gtk_button_set_label(GTK_BUTTON(self->input_file_chooser_), 
                                            g_file_get_basename(file));
                                            
                        // Get video dimensions
                        int width, height;
                        if (self->get_video_dimensions(input_path, width, height)) {
                            self->source_width_ = width;
                            self->source_height_ = height;
                            self->aspect_ratio_ = static_cast<double>(width) / height;
                            
                            // Update UI with source dimensions
                            gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->width_spin_), width);
                            gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->height_spin_), height);
                            
                            // Update preview
                            self->update_preview();
                        }
                    }
                    
                    g_object_unref(file);
                }
            }
            
            gtk_window_destroy(GTK_WINDOW(dialog));
        }), self);
        
        gtk_widget_show(dialog);
    }), this);
    
    // Preview area
    GtkWidget* preview_label = gtk_label_new("Preview:");
    gtk_widget_set_halign(preview_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(preview_label, 10);
    gtk_box_append(GTK_BOX(main_container_), preview_label);
    
    preview_area_ = gtk_drawing_area_new();
    gtk_widget_set_size_request(preview_area_, 320, 180);
    gtk_widget_set_margin_bottom(preview_area_, 10);
    gtk_box_append(GTK_BOX(main_container_), preview_area_);
    
    // Create dimensions frame
    GtkWidget* dimensions_frame = gtk_frame_new("Dimensions");
    gtk_box_append(GTK_BOX(main_container_), dimensions_frame);
    
    GtkWidget* dimensions_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(dimensions_grid), 6);
    gtk_grid_set_column_spacing(GTK_GRID(dimensions_grid), 12);
    gtk_widget_set_margin_start(dimensions_grid, 12);
    gtk_widget_set_margin_end(dimensions_grid, 12);
    gtk_widget_set_margin_top(dimensions_grid, 12);
    gtk_widget_set_margin_bottom(dimensions_grid, 12);
    
    // In GTK4, gtk_container_add is replaced with set_child
    gtk_frame_set_child(GTK_FRAME(dimensions_frame), dimensions_grid);
    
    // Width
    GtkWidget* width_label = gtk_label_new("Width:");
    gtk_widget_set_halign(width_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(dimensions_grid), width_label, 0, 0, 1, 1);
    
    width_spin_ = gtk_spin_button_new_with_range(16, 7680, 1); // Up to 8K
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin_), 1280);
    gtk_grid_attach(GTK_GRID(dimensions_grid), width_spin_, 1, 0, 1, 1);
    
    g_signal_connect(width_spin_, "value-changed", G_CALLBACK([](GtkSpinButton* spin_button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        
        // If aspect ratio is locked, update height based on width
        if (gtk_check_button_get_active(GTK_CHECK_BUTTON(self->lock_aspect_check_)) && self->aspect_ratio_ > 0) {
            self->update_height_from_width();
        }
        
        // Update preview
        self->update_preview();
    }), this);
    
    // Height
    GtkWidget* height_label = gtk_label_new("Height:");
    gtk_widget_set_halign(height_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(dimensions_grid), height_label, 0, 1, 1, 1);
    
    height_spin_ = gtk_spin_button_new_with_range(16, 4320, 1); // Up to 8K
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin_), 720);
    gtk_grid_attach(GTK_GRID(dimensions_grid), height_spin_, 1, 1, 1, 1);
    
    g_signal_connect(height_spin_, "value-changed", G_CALLBACK([](GtkSpinButton* spin_button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        
        // If aspect ratio is locked, update width based on height
        if (gtk_check_button_get_active(GTK_CHECK_BUTTON(self->lock_aspect_check_)) && self->aspect_ratio_ > 0) {
            self->update_width_from_height();
        }
        
        // Update preview
        self->update_preview();
    }), this);
    
    // Lock aspect ratio
    lock_aspect_check_ = gtk_check_button_new_with_label("Maintain Aspect Ratio");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(lock_aspect_check_), TRUE);
    gtk_grid_attach(GTK_GRID(dimensions_grid), lock_aspect_check_, 0, 2, 2, 1);
    
    g_signal_connect(lock_aspect_check_, "toggled", G_CALLBACK([](GtkCheckButton* check_button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        
        // If toggled on and we have a valid aspect ratio, update height based on current width
        if (gtk_check_button_get_active(check_button) && self->aspect_ratio_ > 0) {
            self->update_height_from_width();
            self->update_preview();
        }
    }), this);
    
    // Presets
    GtkWidget* presets_label = gtk_label_new("Common Presets:");
    gtk_widget_set_halign(presets_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(dimensions_grid), presets_label, 0, 3, 1, 1);
    
    GtkWidget* presets_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_grid_attach(GTK_GRID(dimensions_grid), presets_box, 1, 3, 1, 1);
    
    // Preset buttons
    GtkWidget* hd_button = gtk_button_new_with_label("HD");
    gtk_box_append(GTK_BOX(presets_box), hd_button);
    
    g_signal_connect(hd_button, "clicked", G_CALLBACK([](GtkButton* button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->width_spin_), 1280);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->height_spin_), 720);
    }), this);
    
    GtkWidget* fhd_button = gtk_button_new_with_label("Full HD");
    gtk_box_append(GTK_BOX(presets_box), fhd_button);
    
    g_signal_connect(fhd_button, "clicked", G_CALLBACK([](GtkButton* button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->width_spin_), 1920);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->height_spin_), 1080);
    }), this);
    
    GtkWidget* _4k_button = gtk_button_new_with_label("4K");
    gtk_box_append(GTK_BOX(presets_box), _4k_button);
    
    g_signal_connect(_4k_button, "clicked", G_CALLBACK([](GtkButton* button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->width_spin_), 3840);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(self->height_spin_), 2160);
    }), this);
    
    // Scaling mode
    GtkWidget* scaling_label = gtk_label_new("Scaling Mode:");
    gtk_widget_set_halign(scaling_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(dimensions_grid), scaling_label, 0, 4, 1, 1);
    
    scaling_combo_ = gtk_drop_down_new_from_strings((const char*[]){
        "Fit (keep aspect ratio)",
        "Fill (keep aspect ratio)",
        "Stretch (ignore aspect ratio)",
        "Custom",
        NULL
    });
    gtk_grid_attach(GTK_GRID(dimensions_grid), scaling_combo_, 1, 4, 1, 1);
    
    g_signal_connect(scaling_combo_, "notify::selected", G_CALLBACK([](GObject* object, GParamSpec* pspec, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        
        guint selected = gtk_drop_down_get_selected(GTK_DROP_DOWN(object));
        bool is_custom = (selected == 3); // Index 3 is "Custom"
        
        // Enable/disable aspect ratio lock based on scaling mode
        gtk_widget_set_sensitive(self->lock_aspect_check_, is_custom);
        
        // If not custom, force aspect ratio lock on
        if (!is_custom) {
            gtk_check_button_set_active(GTK_CHECK_BUTTON(self->lock_aspect_check_), TRUE);
        }
        
        // Update preview
        self->update_preview();
    }), this);
    
    // Quality settings
    GtkWidget* quality_label = gtk_label_new("Quality:");
    gtk_widget_set_halign(quality_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(dimensions_grid), quality_label, 0, 5, 1, 1);
    
    quality_combo_ = gtk_drop_down_new_from_strings((const char*[]){
        "Fast (Lower quality)",
        "Good (Balanced)",
        "Best (Higher quality)",
        NULL
    });
    gtk_drop_down_set_selected(GTK_DROP_DOWN(quality_combo_), 1); // Default to Good
    gtk_grid_attach(GTK_GRID(dimensions_grid), quality_combo_, 1, 5, 1, 1);
    
    // Output file selection
    GtkWidget* output_label = gtk_label_new("Output Video File:");
    gtk_widget_set_halign(output_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(output_label, 10);
    gtk_box_append(GTK_BOX(main_container_), output_label);
    
    // In GTK4, GtkFileChooserButton is deprecated, using Button + FileChooserDialog
    GtkWidget* output_button = gtk_button_new_with_label("Select Output Location");
    output_file_chooser_ = output_button;
    gtk_box_append(GTK_BOX(main_container_), output_file_chooser_);
    
    g_signal_connect(output_button, "clicked", G_CALLBACK([](GtkButton *button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        
        GtkWidget *dialog = gtk_file_chooser_dialog_new("Save Video File",
                                                   NULL,
                                                   GTK_FILE_CHOOSER_ACTION_SAVE,
                                                   "_Cancel", GTK_RESPONSE_CANCEL,
                                                   "_Save", GTK_RESPONSE_ACCEPT,
                                                   NULL);
                                                   
        gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), "resized_video.mp4");
                                                   
        g_signal_connect(dialog, "response", G_CALLBACK([](GtkDialog *dialog, gint response_id, gpointer user_data) {
            if (response_id == GTK_RESPONSE_ACCEPT) {
                VideoResize* self = static_cast<VideoResize*>(user_data);
                GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
                GFile *file = gtk_file_chooser_get_file(chooser);
                
                if (file) {
                    char *filename = g_file_get_path(file);
                    
                    if (filename) {
                        // Update button label with file name
                        gtk_button_set_label(GTK_BUTTON(self->output_file_chooser_), 
                                            g_file_get_basename(file));
                        g_free(filename);
                    }
                    
                    g_object_unref(file);
                }
            }
            
            gtk_window_destroy(GTK_WINDOW(dialog));
        }), self);
        
        gtk_widget_show(dialog);
    }), this);
    
    // Resize button
    resize_button_ = gtk_button_new_with_label("Resize Video");
    gtk_widget_set_margin_top(resize_button_, 20);
    gtk_box_append(GTK_BOX(main_container_), resize_button_);
    
    g_signal_connect(resize_button_, "clicked", G_CALLBACK([](GtkButton* button, gpointer user_data) {
        VideoResize* self = static_cast<VideoResize*>(user_data);
        
        // Get input file and output file (would need to store paths when file dialogs are used)
        std::string input_path = "";
        std::string output_path = "";
        
        // Get dimensions
        int width = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(self->width_spin_));
        int height = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(self->height_spin_));
        
        // Get scaling mode
        guint scaling_mode_idx = gtk_drop_down_get_selected(GTK_DROP_DOWN(self->scaling_combo_));
        ScalingMode scaling_mode;
        switch (scaling_mode_idx) {
            case 0: scaling_mode = ScalingMode::FIT; break;
            case 1: scaling_mode = ScalingMode::FILL; break;
            case 2: scaling_mode = ScalingMode::STRETCH; break;
            case 3: scaling_mode = ScalingMode::CUSTOM; break;
            default: scaling_mode = ScalingMode::FIT;
        }
        
        // Get quality setting
        guint quality_idx = gtk_drop_down_get_selected(GTK_DROP_DOWN(self->quality_combo_));
        ResizeQuality quality;
        switch (quality_idx) {
            case 0: quality = ResizeQuality::FAST; break;
            case 1: quality = ResizeQuality::GOOD; break;
            case 2: quality = ResizeQuality::BEST; break;
            default: quality = ResizeQuality::GOOD;
        }
        
        // TODO: Get actual paths from stored locations when file choosers are used
        
        // Show mock error for now
        self->show_error_dialog("Please select both input and output files");
        
        // Process resize when we have actual files
        // self->resize_video(input_path, output_path, width, height, scaling_mode, quality);
    }), this);
    
    // Progress bar
    progress_bar_ = gtk_progress_bar_new();
    gtk_widget_set_margin_top(progress_bar_, 10);
    gtk_box_append(GTK_BOX(main_container_), progress_bar_);
    gtk_widget_set_visible(progress_bar_, FALSE);
    
    // Show all widgets
    gtk_widget_show(main_container_);
}

// Helper methods
void VideoResize::update_progress(double progress) {
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar_), progress);
}

void VideoResize::show_error_dialog(const std::string& message) {
    GtkWidget* dialog = gtk_message_dialog_new(NULL,
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR,
        GTK_BUTTONS_OK,
        "%s",
        message.c_str());
        
    g_signal_connect(dialog, "response", G_CALLBACK([](GtkDialog *dialog, gint response_id, gpointer user_data) {
        gtk_window_destroy(GTK_WINDOW(dialog));
    }), NULL);
    
    gtk_widget_show(dialog);
}

bool VideoResize::get_video_dimensions(const std::string& video_path, int& width, int& height) {
    // TODO: Implement actual video dimension detection using a video library
    
    // For demonstration, we'll return mock dimensions
    // In a real implementation, this would use GStreamer to get the actual dimensions
    
    // Simulate different video dimensions based on file path
    // This is just for demonstration - real implementation would inspect the video file
    const char* path = video_path.c_str();
    if (strstr(path, "720p") != nullptr) {
        width = 1280;
        height = 720;
    } else if (strstr(path, "1080p") != nullptr) {
        width = 1920;
        height = 1080;
    } else if (strstr(path, "4k") != nullptr || strstr(path, "2160p") != nullptr) {
        width = 3840;
        height = 2160;
    } else {
        // Default to 1080p if unknown
        width = 1920;
        height = 1080;
    }
    
    return true;
}

void VideoResize::update_preview() {
    // TODO: Implement actual preview rendering
    // For a real implementation, this would render a frame from the video
    // with the new dimensions applied
    
    // For now, we'll just refresh the drawing area
    if (preview_area_ != nullptr) {
        gtk_widget_queue_draw(preview_area_);
    }
}

void VideoResize::update_width_from_height() {
    if (aspect_ratio_ <= 0) return;
    
    int height = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(height_spin_));
    int new_width = static_cast<int>(height * aspect_ratio_ + 0.5); // Round to nearest int
    
    // Block the signal to prevent recursive updates
    g_signal_handlers_block_matched(width_spin_, G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, this);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin_), new_width);
    g_signal_handlers_unblock_matched(width_spin_, G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, this);
}

void VideoResize::update_height_from_width() {
    if (aspect_ratio_ <= 0) return;
    
    int width = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(width_spin_));
    int new_height = static_cast<int>(width / aspect_ratio_ + 0.5); // Round to nearest int
    
    // Block the signal to prevent recursive updates
    g_signal_handlers_block_matched(height_spin_, G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, this);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin_), new_height);
    g_signal_handlers_unblock_matched(height_spin_, G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, this);
} 