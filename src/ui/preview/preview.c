#include "preview.h"
#include <gst/gst.h>
#include <gtk/gtk.h>
#include <glib/gi18n.h>

// Private structure for preview data
typedef struct _BLOUeditPreview {
    GtkWidget *video_area;
    GtkWidget *controls_box;
    GtkWidget *playback_button;
    GtkWidget *timecode_label;
    GtkWidget *speed_button;
    GtkWidget *in_point_button;
    GtkWidget *out_point_button;
    GtkWidget *frame_forward_button;
    GtkWidget *frame_backward_button;
    GtkWidget *resolution_combo;
    GtkWidget *safe_zones_toggle;
    GtkWidget *grid_toggle;
    GtkWidget *split_screen_toggle;
    GtkWidget *scopes_button;
    
    // GStreamer pipeline components
    GstElement *pipeline;
    GstElement *playbin;
    
    // Playback state
    gboolean is_playing;
    gdouble playback_speed;
    gint64 in_point;
    gint64 out_point;
    gint64 duration;
    gint64 position;
    
    // Display options
    gboolean show_grid;
    gboolean show_safe_zones;
    gboolean split_screen_enabled;
    gint resolution_percentage;
    
    // Hardware acceleration
    gboolean hardware_acceleration_enabled;
    
    // Cache settings
    gchar *cache_type;
    gint cache_size_mb;
    
    // HDR settings
    gchar *hdr_mode;
    gboolean hdr_enabled;
    
    // LUT settings
    gchar *current_lut_path;
} BLOUeditPreview;

// Global preview instance
static BLOUeditPreview *preview = NULL;

/**
 * @brief Draw callback for the video area
 */
static gboolean on_draw_preview(GtkDrawingArea *area, cairo_t *cr, int width, int height, gpointer user_data) {
    BLOUeditPreview *preview = (BLOUeditPreview *)user_data;
    
    // Clear the area
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);
    
    // Draw video content (would be provided by GStreamer in real implementation)
    // For now, just draw a placeholder
    cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
    cairo_rectangle(cr, width * 0.1, height * 0.1, width * 0.8, height * 0.8);
    cairo_fill(cr);
    
    // Draw safe zones if enabled
    if (preview->show_safe_zones) {
        cairo_set_source_rgba(cr, 1.0, 0.0, 0.0, 0.5);
        
        // Action safe (90%)
        double action_safe_x = width * 0.05;
        double action_safe_y = height * 0.05;
        double action_safe_w = width * 0.9;
        double action_safe_h = height * 0.9;
        
        cairo_rectangle(cr, action_safe_x, action_safe_y, action_safe_w, action_safe_h);
        cairo_set_line_width(cr, 1.0);
        cairo_stroke(cr);
        
        // Title safe (80%)
        cairo_set_source_rgba(cr, 0.0, 1.0, 0.0, 0.5);
        double title_safe_x = width * 0.1;
        double title_safe_y = height * 0.1;
        double title_safe_w = width * 0.8;
        double title_safe_h = height * 0.8;
        
        cairo_rectangle(cr, title_safe_x, title_safe_y, title_safe_w, title_safe_h);
        cairo_stroke(cr);
    }
    
    // Draw grid if enabled
    if (preview->show_grid) {
        cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.3);
        cairo_set_line_width(cr, 0.5);
        
        // Draw vertical lines (rule of thirds)
        for (int i = 1; i < 3; i++) {
            double x = width * (i / 3.0);
            cairo_move_to(cr, x, 0);
            cairo_line_to(cr, x, height);
        }
        
        // Draw horizontal lines (rule of thirds)
        for (int i = 1; i < 3; i++) {
            double y = height * (i / 3.0);
            cairo_move_to(cr, 0, y);
            cairo_line_to(cr, width, y);
        }
        
        cairo_stroke(cr);
    }
    
    // Draw split screen divider if enabled
    if (preview->split_screen_enabled) {
        cairo_set_source_rgba(cr, 1.0, 1.0, 0.0, 0.8);
        cairo_set_line_width(cr, 2.0);
        
        // Draw vertical divider
        cairo_move_to(cr, width / 2.0, 0);
        cairo_line_to(cr, width / 2.0, height);
        
        cairo_stroke(cr);
        
        // Add labels
        cairo_set_source_rgba(cr, 1.0, 1.0, 0.0, 0.8);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 14);
        
        // "Original" label
        cairo_move_to(cr, width * 0.25 - 30, 20);
        cairo_show_text(cr, _("Original"));
        
        // "Edited" label
        cairo_move_to(cr, width * 0.75 - 20, 20);
        cairo_show_text(cr, _("Edited"));
    }
    
    // Draw timecode
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_select_font_face(cr, "Monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 12);
    
    // Format position as timecode
    int hours = (preview->position / 3600000000000);
    int minutes = (preview->position / 60000000000) % 60;
    int seconds = (preview->position / 1000000000) % 60;
    int frames = (preview->position / 33333333) % 30; // Assuming 30fps
    
    char timecode[32];
    g_snprintf(timecode, sizeof(timecode), "%02d:%02d:%02d:%02d", hours, minutes, seconds, frames);
    
    cairo_move_to(cr, 10, height - 10);
    cairo_show_text(cr, timecode);
    
    return TRUE;
}

/**
 * @brief Create the playback controls
 */
static GtkWidget* create_playback_controls() {
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_widget_set_margin_start(box, 12);
    gtk_widget_set_margin_end(box, 12);
    gtk_widget_set_margin_top(box, 6);
    gtk_widget_set_margin_bottom(box, 6);
    
    // Play/Pause button
    preview->playback_button = gtk_button_new_from_icon_name("media-playback-start");
    gtk_widget_set_tooltip_text(preview->playback_button, _("Play/Pause"));
    gtk_box_append(GTK_BOX(box), preview->playback_button);
    
    // Frame backward button
    preview->frame_backward_button = gtk_button_new_from_icon_name("media-skip-backward");
    gtk_widget_set_tooltip_text(preview->frame_backward_button, _("Previous Frame"));
    gtk_box_append(GTK_BOX(box), preview->frame_backward_button);
    
    // Frame forward button
    preview->frame_forward_button = gtk_button_new_from_icon_name("media-skip-forward");
    gtk_widget_set_tooltip_text(preview->frame_forward_button, _("Next Frame"));
    gtk_box_append(GTK_BOX(box), preview->frame_forward_button);
    
    // In point button
    preview->in_point_button = gtk_button_new_from_icon_name("media-record");
    gtk_widget_set_tooltip_text(preview->in_point_button, _("Set In Point"));
    gtk_box_append(GTK_BOX(box), preview->in_point_button);
    
    // Out point button
    preview->out_point_button = gtk_button_new_from_icon_name("media-playback-stop");
    gtk_widget_set_tooltip_text(preview->out_point_button, _("Set Out Point"));
    gtk_box_append(GTK_BOX(box), preview->out_point_button);
    
    // Spacer
    GtkWidget *spacer = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_hexpand(spacer, TRUE);
    gtk_box_append(GTK_BOX(box), spacer);
    
    // Speed dropdown
    preview->speed_button = gtk_drop_down_new_from_strings((const char*[]){
        "0.25x", "0.5x", "1.0x", "1.5x", "2.0x", NULL
    });
    gtk_drop_down_set_selected(GTK_DROP_DOWN(preview->speed_button), 2); // Default to 1.0x
    gtk_widget_set_tooltip_text(preview->speed_button, _("Playback Speed"));
    gtk_box_append(GTK_BOX(box), preview->speed_button);
    
    // Resolution dropdown
    preview->resolution_combo = gtk_drop_down_new_from_strings((const char*[]){
        "25%", "50%", "75%", "100%", NULL
    });
    gtk_drop_down_set_selected(GTK_DROP_DOWN(preview->resolution_combo), 3); // Default to 100%
    gtk_widget_set_tooltip_text(preview->resolution_combo, _("Preview Resolution"));
    gtk_box_append(GTK_BOX(box), preview->resolution_combo);
    
    // Safe zones toggle
    preview->safe_zones_toggle = gtk_toggle_button_new();
    gtk_button_set_icon_name(GTK_BUTTON(preview->safe_zones_toggle), "view-grid");
    gtk_widget_set_tooltip_text(preview->safe_zones_toggle, _("Safe Zones"));
    gtk_box_append(GTK_BOX(box), preview->safe_zones_toggle);
    
    // Grid toggle
    preview->grid_toggle = gtk_toggle_button_new();
    gtk_button_set_icon_name(GTK_BUTTON(preview->grid_toggle), "view-grid-symbolic");
    gtk_widget_set_tooltip_text(preview->grid_toggle, _("Grid"));
    gtk_box_append(GTK_BOX(box), preview->grid_toggle);
    
    // Split screen toggle
    preview->split_screen_toggle = gtk_toggle_button_new();
    gtk_button_set_icon_name(GTK_BUTTON(preview->split_screen_toggle), "view-dual-symbolic");
    gtk_widget_set_tooltip_text(preview->split_screen_toggle, _("Split Screen Comparison"));
    gtk_box_append(GTK_BOX(box), preview->split_screen_toggle);
    
    // Scopes button
    preview->scopes_button = gtk_menu_button_new();
    gtk_button_set_icon_name(GTK_BUTTON(preview->scopes_button), "view-more-symbolic");
    gtk_widget_set_tooltip_text(preview->scopes_button, _("Scopes and Monitoring"));
    
    // Create a popover for the scopes
    GtkWidget *scopes_popover = gtk_popover_new();
    GtkWidget *scopes_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_margin_start(scopes_box, 12);
    gtk_widget_set_margin_end(scopes_box, 12);
    gtk_widget_set_margin_top(scopes_box, 6);
    gtk_widget_set_margin_bottom(scopes_box, 6);
    
    // Add scope options
    GtkWidget *vectorscope_check = gtk_check_button_new_with_label(_("Vectorscope"));
    GtkWidget *waveform_check = gtk_check_button_new_with_label(_("Waveform"));
    GtkWidget *histogram_check = gtk_check_button_new_with_label(_("Histogram"));
    GtkWidget *audio_meter_check = gtk_check_button_new_with_label(_("Audio Meter"));
    
    gtk_box_append(GTK_BOX(scopes_box), vectorscope_check);
    gtk_box_append(GTK_BOX(scopes_box), waveform_check);
    gtk_box_append(GTK_BOX(scopes_box), histogram_check);
    gtk_box_append(GTK_BOX(scopes_box), audio_meter_check);
    
    // Add separator
    GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_append(GTK_BOX(scopes_box), separator);
    
    // Add LUT options
    GtkWidget *lut_label = gtk_label_new(_("Apply LUT:"));
    gtk_widget_set_halign(lut_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(scopes_box), lut_label);
    
    GtkWidget *lut_button = gtk_button_new_with_label(_("Choose LUT..."));
    gtk_box_append(GTK_BOX(scopes_box), lut_button);
    
    // Add HDR options
    GtkWidget *hdr_label = gtk_label_new(_("HDR Preview:"));
    gtk_widget_set_halign(hdr_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(scopes_box), hdr_label);
    
    GtkWidget *hdr_combo = gtk_drop_down_new_from_strings((const char*[]){
        _("Disabled"), "HDR10", "HLG", "Dolby Vision", NULL
    });
    gtk_box_append(GTK_BOX(scopes_box), hdr_combo);
    
    gtk_popover_set_child(GTK_POPOVER(scopes_popover), scopes_box);
    gtk_menu_button_set_popover(GTK_MENU_BUTTON(preview->scopes_button), scopes_popover);
    
    gtk_box_append(GTK_BOX(box), preview->scopes_button);
    
    // Connect signals
    g_signal_connect(preview->playback_button, "clicked", G_CALLBACK(on_play_pause_clicked), NULL);
    g_signal_connect(preview->frame_forward_button, "clicked", G_CALLBACK(on_frame_forward_clicked), NULL);
    g_signal_connect(preview->frame_backward_button, "clicked", G_CALLBACK(on_frame_backward_clicked), NULL);
    g_signal_connect(preview->in_point_button, "clicked", G_CALLBACK(on_in_point_clicked), NULL);
    g_signal_connect(preview->out_point_button, "clicked", G_CALLBACK(on_out_point_clicked), NULL);
    g_signal_connect(preview->speed_button, "notify::selected", G_CALLBACK(on_speed_changed), NULL);
    g_signal_connect(preview->resolution_combo, "notify::selected", G_CALLBACK(on_resolution_changed), NULL);
    g_signal_connect(preview->safe_zones_toggle, "toggled", G_CALLBACK(on_safe_zones_toggled), NULL);
    g_signal_connect(preview->grid_toggle, "toggled", G_CALLBACK(on_grid_toggled), NULL);
    g_signal_connect(preview->split_screen_toggle, "toggled", G_CALLBACK(on_split_screen_toggled), NULL);
    
    return box;
}

static void on_play_pause_clicked(GtkButton *button, gpointer user_data) {
    preview->is_playing = !preview->is_playing;
    
    if (preview->is_playing) {
        gtk_button_set_icon_name(GTK_BUTTON(preview->playback_button), "media-playback-pause");
    } else {
        gtk_button_set_icon_name(GTK_BUTTON(preview->playback_button), "media-playback-start");
    }
    
    // In a real implementation, we would start/stop the GStreamer pipeline here
}

static void on_frame_forward_clicked(GtkButton *button, gpointer user_data) {
    blouedit_preview_frame_step_forward();
}

static void on_frame_backward_clicked(GtkButton *button, gpointer user_data) {
    blouedit_preview_frame_step_backward();
}

static void on_in_point_clicked(GtkButton *button, gpointer user_data) {
    blouedit_preview_set_in_point();
}

static void on_out_point_clicked(GtkButton *button, gpointer user_data) {
    blouedit_preview_set_out_point();
}

static void on_speed_changed(GtkDropDown *dropdown, GParamSpec *pspec, gpointer user_data) {
    int selected = gtk_drop_down_get_selected(dropdown);
    double speed_values[] = { 0.25, 0.5, 1.0, 1.5, 2.0 };
    
    if (selected >= 0 && selected < 5) {
        blouedit_preview_set_speed(speed_values[selected]);
    }
}

static void on_resolution_changed(GtkDropDown *dropdown, GParamSpec *pspec, gpointer user_data) {
    int selected = gtk_drop_down_get_selected(dropdown);
    int resolution_values[] = { 25, 50, 75, 100 };
    
    if (selected >= 0 && selected < 4) {
        blouedit_preview_set_resolution(resolution_values[selected]);
    }
}

static void on_safe_zones_toggled(GtkToggleButton *button, gpointer user_data) {
    gboolean active = gtk_toggle_button_get_active(button);
    blouedit_preview_toggle_safe_zones(active);
}

static void on_grid_toggled(GtkToggleButton *button, gpointer user_data) {
    gboolean active = gtk_toggle_button_get_active(button);
    blouedit_preview_toggle_grid(active);
}

static void on_split_screen_toggled(GtkToggleButton *button, gpointer user_data) {
    gboolean active = gtk_toggle_button_get_active(button);
    blouedit_preview_toggle_split_screen(active);
}

// Public functions implementation

GtkWidget* blouedit_preview_create(void) {
    // Initialize preview structure if not already
    if (preview == NULL) {
        preview = g_new0(BLOUeditPreview, 1);
        preview->playback_speed = 1.0;
        preview->resolution_percentage = 100;
        preview->in_point = -1;
        preview->out_point = -1;
        preview->position = 0;
        preview->duration = 60 * 1000000000; // 60 seconds in nanoseconds
        preview->is_playing = FALSE;
        preview->show_grid = FALSE;
        preview->show_safe_zones = FALSE;
        preview->split_screen_enabled = FALSE;
        preview->hardware_acceleration_enabled = TRUE;
        preview->cache_type = g_strdup("ram");
        preview->cache_size_mb = 512;
        preview->hdr_mode = NULL;
        preview->hdr_enabled = FALSE;
        preview->current_lut_path = NULL;
    }
    
    // Create main box
    GtkWidget *main_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    
    // Create video area
    preview->video_area = gtk_drawing_area_new();
    gtk_widget_set_hexpand(preview->video_area, TRUE);
    gtk_widget_set_vexpand(preview->video_area, TRUE);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(preview->video_area), 
                                  on_draw_preview, preview, NULL);
    
    // Create controls
    GtkWidget *controls = create_playback_controls();
    
    // Pack everything
    gtk_box_append(GTK_BOX(main_box), preview->video_area);
    gtk_box_append(GTK_BOX(main_box), controls);
    
    return main_box;
}

void blouedit_preview_init(GtkWindow* window) {
    // Initialize GStreamer (would be done in main app initialization in reality)
    // gst_init(NULL, NULL);
    
    // Create a simple pipeline (would be more complex in reality)
    // preview->pipeline = gst_pipeline_new("blouedit-preview-pipeline");
    // preview->playbin = gst_element_factory_make("playbin", "playbin");
    // gst_bin_add(GST_BIN(preview->pipeline), preview->playbin);
}

void blouedit_preview_set_speed(double speed_multiplier) {
    preview->playback_speed = speed_multiplier;
    
    // In a real implementation, we would adjust the GStreamer pipeline playback rate
    g_print("Preview playback speed set to %.2fx\n", speed_multiplier);
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_frame_step_forward(void) {
    // Move position forward by one frame (assuming 30fps)
    preview->position += 33333333; // 1/30 of a second in nanoseconds
    
    if (preview->position > preview->duration) {
        preview->position = preview->duration;
    }
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
    
    g_print("Stepped forward to position %" G_GINT64_FORMAT "\n", preview->position);
}

void blouedit_preview_frame_step_backward(void) {
    // Move position backward by one frame (assuming 30fps)
    preview->position -= 33333333; // 1/30 of a second in nanoseconds
    
    if (preview->position < 0) {
        preview->position = 0;
    }
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
    
    g_print("Stepped backward to position %" G_GINT64_FORMAT "\n", preview->position);
}

void blouedit_preview_set_in_point(void) {
    preview->in_point = preview->position;
    
    // Ensure in_point is before out_point
    if (preview->out_point != -1 && preview->in_point > preview->out_point) {
        preview->out_point = -1;
    }
    
    g_print("In point set to %" G_GINT64_FORMAT "\n", preview->in_point);
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_set_out_point(void) {
    preview->out_point = preview->position;
    
    // Ensure out_point is after in_point
    if (preview->in_point != -1 && preview->out_point < preview->in_point) {
        preview->in_point = -1;
    }
    
    g_print("Out point set to %" G_GINT64_FORMAT "\n", preview->out_point);
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_toggle_grid(gboolean show_grid) {
    preview->show_grid = show_grid;
    g_print("Grid %s\n", show_grid ? "enabled" : "disabled");
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_toggle_safe_zones(gboolean show_safe_zones) {
    preview->show_safe_zones = show_safe_zones;
    g_print("Safe zones %s\n", show_safe_zones ? "enabled" : "disabled");
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_toggle_split_screen(gboolean enable_split_screen) {
    preview->split_screen_enabled = enable_split_screen;
    g_print("Split screen %s\n", enable_split_screen ? "enabled" : "disabled");
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_set_resolution(int resolution_percentage) {
    preview->resolution_percentage = resolution_percentage;
    g_print("Preview resolution set to %d%%\n", resolution_percentage);
    
    // In a real implementation, we would adjust the video sink properties
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_toggle_scope(const char* scope_type, gboolean enabled) {
    g_print("%s scope %s\n", scope_type, enabled ? "enabled" : "disabled");
    
    // In a real implementation, we would create and manage the scope windows
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_apply_lut(const char* lut_path) {
    g_free(preview->current_lut_path);
    preview->current_lut_path = g_strdup(lut_path);
    g_print("Applied LUT from %s\n", lut_path);
    
    // In a real implementation, we would apply the LUT to the video processing pipeline
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_toggle_hdr(const char* hdr_mode, gboolean enabled) {
    g_free(preview->hdr_mode);
    preview->hdr_mode = g_strdup(hdr_mode);
    preview->hdr_enabled = enabled;
    g_print("HDR mode %s %s\n", hdr_mode, enabled ? "enabled" : "disabled");
    
    // In a real implementation, we would adjust the video sink properties for HDR
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_set_hardware_acceleration(gboolean use_hardware_acceleration) {
    preview->hardware_acceleration_enabled = use_hardware_acceleration;
    g_print("Hardware acceleration %s\n", use_hardware_acceleration ? "enabled" : "disabled");
    
    // In a real implementation, we would rebuild the GStreamer pipeline with appropriate elements
    
    // Request redraw
    gtk_widget_queue_draw(preview->video_area);
}

void blouedit_preview_configure_cache(const char* cache_type, int cache_size_mb) {
    g_free(preview->cache_type);
    preview->cache_type = g_strdup(cache_type);
    preview->cache_size_mb = cache_size_mb;
    g_print("Cache configured: type=%s, size=%dMB\n", cache_type, cache_size_mb);
    
    // In a real implementation, we would adjust the caching settings
} 