#include "thumbnail_generator.h"
#include <iostream>
#include <filesystem>
#include <cairo.h>
#include <gdk/gdk.h>

// GStreamer headers for video processing
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/pbutils/pbutils.h>

namespace BlouEdit {
namespace AI {

ThumbnailGenerator::ThumbnailGenerator() : is_initialized(true) {
    // Initialize gstreamer if needed
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
}

ThumbnailGenerator::~ThumbnailGenerator() {
    // No specific cleanup needed
}

bool ThumbnailGenerator::generate_thumbnail(
    const ThumbnailGenerationParams& params,
    const std::string& output_path,
    std::function<void(bool, const std::string&)> callback) {
    
    std::string temp_frame_path;
    std::string metadata_title;
    int duration = 0;
    
    // Create output directory if it doesn't exist
    std::filesystem::path output_dir = std::filesystem::path(output_path).parent_path();
    if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }
    
    // First approach: Extract a frame from the video if requested
    if (params.use_video_frame) {
        temp_frame_path = output_path + ".temp_frame.jpg";
        if (!extract_video_frame(params.video_path, temp_frame_path)) {
            std::cerr << "Failed to extract video frame" << std::endl;
            if (callback) callback(false, "Failed to extract video frame");
            return false;
        }
    }
    
    // Get video metadata for title if needed
    if (params.add_title && params.title_text.empty()) {
        if (get_video_metadata(params.video_path, metadata_title, duration)) {
            if (metadata_title.empty()) {
                // Use filename as fallback
                metadata_title = std::filesystem::path(params.video_path).stem().string();
            }
        } else {
            // Use filename as fallback
            metadata_title = std::filesystem::path(params.video_path).stem().string();
        }
    }
    
    // Generate AI thumbnail
    std::string prompt = params.prompt;
    if (prompt.empty()) {
        // Create a default prompt based on video metadata
        prompt = "Professional thumbnail for video titled: " + 
                (!params.title_text.empty() ? params.title_text : metadata_title);
    }
    
    // Use the frame as reference or generate from scratch
    bool success;
    if (!temp_frame_path.empty() && std::filesystem::exists(temp_frame_path)) {
        success = image_generator.generate_thumbnail(
            params.video_path, 
            prompt,
            output_path,
            [callback, this, params, output_path, metadata_title, temp_frame_path](bool success, const std::string& message) {
                // Continue processing if successful
                if (success) {
                    // Add title text if requested
                    if (params.add_title) {
                        std::string title = params.title_text.empty() ? metadata_title : params.title_text;
                        this->add_title_to_image(output_path, title);
                    }
                    
                    // Add visual effects if requested
                    if (params.add_visual_effects) {
                        this->enhance_thumbnail(output_path);
                    }
                    
                    // Clean up temporary frame
                    if (std::filesystem::exists(temp_frame_path)) {
                        std::filesystem::remove(temp_frame_path);
                    }
                    
                    if (callback) callback(true, output_path);
                } else {
                    if (callback) callback(false, message);
                }
            }
        );
    } else {
        // Configure image generation parameters
        ImageGenerationParams img_params;
        img_params.prompt = prompt;
        img_params.negative_prompt = "low quality, blurry, text, watermark";
        img_params.width = params.width;
        img_params.height = params.height;
        img_params.num_inference_steps = 50;
        img_params.guidance_scale = 7.5f;
        
        success = image_generator.generate_image(
            img_params,
            output_path,
            [callback, this, params, output_path, metadata_title](bool success, const std::string& message) {
                // Continue processing if successful
                if (success) {
                    // Add title text if requested
                    if (params.add_title) {
                        std::string title = params.title_text.empty() ? metadata_title : params.title_text;
                        this->add_title_to_image(output_path, title);
                    }
                    
                    // Add visual effects if requested
                    if (params.add_visual_effects) {
                        this->enhance_thumbnail(output_path);
                    }
                    
                    if (callback) callback(true, output_path);
                } else {
                    if (callback) callback(false, message);
                }
            }
        );
    }
    
    // If the callback is used, return true since the actual result will be handled in the callback
    if (callback) {
        return true;
    }
    
    return success;
}

bool ThumbnailGenerator::extract_video_frame(const std::string& video_path, const std::string& output_path) {
    if (!std::filesystem::exists(video_path)) {
        std::cerr << "Video file not found: " << video_path << std::endl;
        return false;
    }
    
    // Create a GStreamer pipeline to extract a frame
    GError* error = nullptr;
    std::string pipeline_str = 
        "filesrc location=\"" + video_path + "\" ! "
        "decodebin ! videoconvert ! videorate ! "
        "video/x-raw,framerate=1/1 ! videocrop ! "
        "jpegenc ! multifilesink location=\"" + output_path + "\"";
    
    GstElement* pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    
    if (error) {
        std::cerr << "Failed to create pipeline: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    // Start the pipeline and wait for EOS or error
    GstBus* bus = gst_element_get_bus(pipeline);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    // Timeout after 5 seconds
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus, 5 * GST_SECOND, 
        (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)
    );
    
    bool success = true;
    
    if (msg) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* err;
            gchar* debug_info;
            
            gst_message_parse_error(msg, &err, &debug_info);
            std::cerr << "Error: " << err->message << std::endl;
            std::cerr << "Debug info: " << (debug_info ? debug_info : "none") << std::endl;
            
            g_error_free(err);
            g_free(debug_info);
            success = false;
        }
        
        gst_message_unref(msg);
    } else {
        std::cerr << "Timeout waiting for frame extraction" << std::endl;
        success = false;
    }
    
    // Clean up
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    
    return success && std::filesystem::exists(output_path);
}

bool ThumbnailGenerator::add_title_to_image(const std::string& image_path, const std::string& title) {
    if (title.empty() || !std::filesystem::exists(image_path)) {
        return false;
    }
    
    // Load the image with GDK
    GError* error = nullptr;
    GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(image_path.c_str(), &error);
    
    if (!pixbuf) {
        std::cerr << "Failed to load image: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    int width = gdk_pixbuf_get_width(pixbuf);
    int height = gdk_pixbuf_get_height(pixbuf);
    
    // Create a Cairo surface and context
    cairo_surface_t* surface = cairo_image_surface_create(
        CAIRO_FORMAT_ARGB32, width, height
    );
    cairo_t* cr = cairo_create(surface);
    
    // Draw the image on the surface
    gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
    cairo_paint(cr);
    
    // Add a semi-transparent bar at the bottom for the title
    cairo_set_source_rgba(cr, 0, 0, 0, 0.6);
    cairo_rectangle(cr, 0, height * 0.75, width, height * 0.25);
    cairo_fill(cr);
    
    // Set up the text
    cairo_select_font_face(cr, "Arial", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    
    // Adjust font size based on title length and image width
    double font_size = std::min(height * 0.1, width * 0.05);
    cairo_set_font_size(cr, font_size);
    
    // Measure the text
    cairo_text_extents_t extents;
    cairo_text_extents(cr, title.c_str(), &extents);
    
    // Position the text
    double x = (width - extents.width) / 2;
    double y = height * 0.85 + extents.height / 2;
    
    // Draw the text with a shadow effect
    cairo_set_source_rgba(cr, 0, 0, 0, 0.5);
    cairo_move_to(cr, x + 2, y + 2);
    cairo_show_text(cr, title.c_str());
    
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_move_to(cr, x, y);
    cairo_show_text(cr, title.c_str());
    
    // Save the modified image
    cairo_surface_write_to_png(surface, image_path.c_str());
    
    // Clean up
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    g_object_unref(pixbuf);
    
    return true;
}

bool ThumbnailGenerator::enhance_thumbnail(const std::string& image_path) {
    if (!std::filesystem::exists(image_path)) {
        return false;
    }
    
    // Load the image with GDK
    GError* error = nullptr;
    GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(image_path.c_str(), &error);
    
    if (!pixbuf) {
        std::cerr << "Failed to load image: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    int width = gdk_pixbuf_get_width(pixbuf);
    int height = gdk_pixbuf_get_height(pixbuf);
    
    // Create a Cairo surface and context
    cairo_surface_t* surface = cairo_image_surface_create(
        CAIRO_FORMAT_ARGB32, width, height
    );
    cairo_t* cr = cairo_create(surface);
    
    // Draw the image on the surface
    gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
    cairo_paint(cr);
    
    // Add a subtle vignette effect
    cairo_pattern_t* pattern = cairo_pattern_create_radial(
        width / 2, height / 2, 0,
        width / 2, height / 2, width
    );
    cairo_pattern_add_color_stop_rgba(pattern, 0.7, 0, 0, 0, 0);
    cairo_pattern_add_color_stop_rgba(pattern, 1.0, 0, 0, 0, 0.5);
    
    cairo_set_source(cr, pattern);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_paint(cr);
    cairo_pattern_destroy(pattern);
    
    // Add a subtle brightness and contrast adjustment
    cairo_set_source_rgba(cr, 1, 1, 1, 0.1);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVERLAY);
    cairo_paint(cr);
    
    // Save the modified image
    cairo_surface_write_to_png(surface, image_path.c_str());
    
    // Clean up
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    g_object_unref(pixbuf);
    
    return true;
}

bool ThumbnailGenerator::get_video_metadata(const std::string& video_path, std::string& title, int& duration) {
    if (!std::filesystem::exists(video_path)) {
        return false;
    }
    
    GError* error = nullptr;
    gst_init(nullptr, nullptr);
    
    // Initialize GStreamer's discoverer
    GstDiscoverer* discoverer = gst_discoverer_new(5 * GST_SECOND, &error);
    if (!discoverer) {
        std::cerr << "Failed to create discoverer: " << error->message << std::endl;
        g_error_free(error);
        return false;
    }
    
    // Discover the URI
    std::string uri = "file://" + video_path;
    GstDiscovererInfo* info = gst_discoverer_discover_uri(discoverer, uri.c_str(), &error);
    
    if (!info) {
        std::cerr << "Failed to discover URI: " << error->message << std::endl;
        g_error_free(error);
        g_object_unref(discoverer);
        return false;
    }
    
    // Get duration
    duration = GST_TIME_AS_SECONDS(gst_discoverer_info_get_duration(info));
    
    // Get title from tags
    const GstTagList* tags = gst_discoverer_info_get_tags(info);
    if (tags) {
        gchar* title_str = nullptr;
        
        if (gst_tag_list_get_string(tags, GST_TAG_TITLE, &title_str)) {
            title = title_str;
            g_free(title_str);
        }
    }
    
    // Clean up
    gst_discoverer_info_unref(info);
    g_object_unref(discoverer);
    
    return true;
}

GtkWidget* ThumbnailGenerator::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 썸네일 생성기");
    PangoAttrList* attr_list = pango_attr_list_new();
    pango_attr_list_insert(attr_list, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attr_list, pango_attr_scale_new(1.2));
    gtk_label_set_attributes(GTK_LABEL(title), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_append(GTK_BOX(container), title);
    
    // Video selection
    GtkWidget* video_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* video_label = gtk_label_new("비디오:");
    GtkWidget* video_entry = gtk_entry_new();
    GtkWidget* video_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(video_entry, TRUE);
    gtk_box_append(GTK_BOX(video_box), video_label);
    gtk_box_append(GTK_BOX(video_box), video_entry);
    gtk_box_append(GTK_BOX(video_box), video_button);
    gtk_box_append(GTK_BOX(container), video_box);
    
    // Prompt input
    GtkWidget* prompt_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* prompt_label = gtk_label_new("프롬프트:");
    GtkWidget* prompt_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(prompt_entry), "썸네일 설명 (비워두면 자동 생성)");
    gtk_widget_set_hexpand(prompt_entry, TRUE);
    gtk_box_append(GTK_BOX(prompt_box), prompt_label);
    gtk_box_append(GTK_BOX(prompt_box), prompt_entry);
    gtk_box_append(GTK_BOX(container), prompt_box);
    
    // Title input
    GtkWidget* title_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* title_label = gtk_label_new("제목:");
    GtkWidget* title_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(title_entry), "썸네일에 표시할 제목 (비워두면 파일명 사용)");
    gtk_widget_set_hexpand(title_entry, TRUE);
    gtk_box_append(GTK_BOX(title_box), title_label);
    gtk_box_append(GTK_BOX(title_box), title_entry);
    gtk_box_append(GTK_BOX(container), title_box);
    
    // Options
    GtkWidget* options_frame = gtk_frame_new("옵션");
    GtkWidget* options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_frame_set_child(GTK_FRAME(options_frame), options_box);
    
    // Use video frame checkbox
    GtkWidget* use_frame_check = gtk_check_button_new_with_label("비디오 프레임 사용");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(use_frame_check), TRUE);
    
    // Add title checkbox
    GtkWidget* add_title_check = gtk_check_button_new_with_label("제목 추가");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(add_title_check), TRUE);
    
    // Add effects checkbox
    GtkWidget* add_effects_check = gtk_check_button_new_with_label("시각 효과 추가");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(add_effects_check), TRUE);
    
    gtk_box_append(GTK_BOX(options_box), use_frame_check);
    gtk_box_append(GTK_BOX(options_box), add_title_check);
    gtk_box_append(GTK_BOX(options_box), add_effects_check);
    gtk_box_append(GTK_BOX(container), options_frame);
    
    // Size options
    GtkWidget* size_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* size_label = gtk_label_new("크기:");
    GtkWidget* size_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "1280x720 (YouTube)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "1920x1080 (YouTube HD)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "2560x1440 (YouTube 2K)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "1200x630 (Facebook)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "1080x1080 (Instagram)");
    gtk_combo_box_set_active(GTK_COMBO_BOX(size_combo), 0);
    gtk_box_append(GTK_BOX(size_box), size_label);
    gtk_box_append(GTK_BOX(size_box), size_combo);
    gtk_box_append(GTK_BOX(container), size_box);
    
    // Generate button
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* generate_button = gtk_button_new_with_label("썸네일 생성");
    gtk_widget_set_hexpand(generate_button, TRUE);
    gtk_box_append(GTK_BOX(button_box), generate_button);
    gtk_box_append(GTK_BOX(container), button_box);
    
    // Preview area
    GtkWidget* preview_frame = gtk_frame_new("미리보기");
    GtkWidget* preview_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget* preview_image = gtk_image_new();
    gtk_widget_set_size_request(preview_image, 640, 360);
    gtk_box_append(GTK_BOX(preview_box), preview_image);
    gtk_frame_set_child(GTK_FRAME(preview_frame), preview_box);
    gtk_box_append(GTK_BOX(container), preview_frame);
    
    // TODO: Connect signals to callbacks
    
    return container;
}

} // namespace AI
} // namespace BlouEdit 