#include "scopes.h"
#include <cairo.h>
#include <glib/gi18n.h>

typedef struct _BLOUeditScope {
    BLOUeditScopeType type;
    GtkWidget *drawing_area;
    guchar *data;
    int width;
    int height;
    int stride;
    float *audio_data;
    int audio_channels;
    int audio_samples;
    
    // Options
    gboolean show_grid;
    gboolean show_labels;
    double scale;
    gboolean logarithmic_scale; // For histogram
    
    // Colors for RGB parade
    GdkRGBA r_color;
    GdkRGBA g_color;
    GdkRGBA b_color;
} BLOUeditScope;

static void blouedit_scope_finalize(BLOUeditScope *scope) {
    g_free(scope->data);
    g_free(scope->audio_data);
    g_free(scope);
}

static gboolean on_draw_vectorscope(GtkDrawingArea *area, cairo_t *cr, int width, int height, BLOUeditScope *scope) {
    // Clear background
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);
    
    // Draw grid if enabled
    if (scope->show_grid) {
        cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.5);
        cairo_set_line_width(cr, 1.0);
        
        // Draw circles
        double center_x = width / 2.0;
        double center_y = height / 2.0;
        double max_radius = MIN(width, height) * 0.45;
        
        for (int i = 1; i <= 3; i++) {
            double radius = max_radius * (i / 3.0);
            cairo_arc(cr, center_x, center_y, radius, 0, 2 * G_PI);
            cairo_stroke(cr);
        }
        
        // Draw axes
        cairo_move_to(cr, center_x - max_radius, center_y);
        cairo_line_to(cr, center_x + max_radius, center_y);
        cairo_move_to(cr, center_x, center_y - max_radius);
        cairo_line_to(cr, center_x, center_y + max_radius);
        cairo_stroke(cr);
    }
    
    // Draw color points from data
    if (scope->data != NULL) {
        // This is a simplified implementation
        // Real vectorscope would plot color points based on U/V values
        
        cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.7);
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        
        double center_x = width / 2.0;
        double center_y = height / 2.0;
        double scale = MIN(width, height) * 0.45 / 128.0; // 128 is max U/V distance
        
        // Simple simulation for demonstration
        for (int y = 0; y < scope->height; y += 2) {
            for (int x = 0; x < scope->width; x += 2) {
                int offset = y * scope->stride + x * 4; // Assuming RGBA
                
                if (offset + 2 < scope->height * scope->stride) {
                    // Convert RGB to YUV (simplified)
                    int r = scope->data[offset];
                    int g = scope->data[offset + 1];
                    int b = scope->data[offset + 2];
                    
                    // Simplified U/V calculation
                    double u = (b - g) * 0.5;
                    double v = (r - g) * 0.5;
                    
                    double x_pos = center_x + u * scale;
                    double y_pos = center_y + v * scale;
                    
                    cairo_rectangle(cr, x_pos, y_pos, 1, 1);
                }
            }
        }
        cairo_fill(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    }
    
    // Draw labels if enabled
    if (scope->show_labels) {
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 12);
        
        cairo_move_to(cr, width - 70, 20);
        cairo_show_text(cr, _("Vectorscope"));
        
        if (scope->show_grid) {
            // Draw color target labels
            struct {
                const char *name;
                double angle;
                GdkRGBA color;
            } targets[] = {
                { "R", 90, {1.0, 0.0, 0.0, 1.0} },
                { "Y", 15, {1.0, 1.0, 0.0, 1.0} },
                { "G", -30, {0.0, 1.0, 0.0, 1.0} },
                { "C", -90, {0.0, 1.0, 1.0, 1.0} },
                { "B", -150, {0.0, 0.0, 1.0, 1.0} },
                { "M", -210, {1.0, 0.0, 1.0, 1.0} }
            };
            
            double center_x = width / 2.0;
            double center_y = height / 2.0;
            double radius = MIN(width, height) * 0.45;
            
            cairo_set_font_size(cr, 10);
            
            for (int i = 0; i < 6; i++) {
                double rad_angle = targets[i].angle * G_PI / 180.0;
                double x = center_x + cos(rad_angle) * radius * 0.9;
                double y = center_y + sin(rad_angle) * radius * 0.9;
                
                cairo_set_source_rgba(cr, 
                                     targets[i].color.red,
                                     targets[i].color.green,
                                     targets[i].color.blue,
                                     targets[i].color.alpha);
                
                cairo_move_to(cr, x - 5, y + 5);
                cairo_show_text(cr, targets[i].name);
            }
        }
    }
    
    return TRUE;
}

static gboolean on_draw_waveform(GtkDrawingArea *area, cairo_t *cr, int width, int height, BLOUeditScope *scope) {
    // Clear background
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);
    
    // Draw grid if enabled
    if (scope->show_grid) {
        cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.5);
        cairo_set_line_width(cr, 1.0);
        
        // Draw horizontal lines
        for (int i = 0; i <= 10; i++) {
            double y = height * (i / 10.0);
            cairo_move_to(cr, 0, y);
            cairo_line_to(cr, width, y);
        }
        
        cairo_stroke(cr);
    }
    
    // Draw waveform from data
    if (scope->data != NULL) {
        // Simple waveform implementation
        cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.7);
        cairo_set_operator(cr, CAIRO_OPERATOR_ADD);
        
        double x_scale = (double)width / scope->width;
        
        // For each column in the output
        for (int x = 0; x < width; x++) {
            // Map to input column
            int input_x = x / x_scale;
            if (input_x >= scope->width) input_x = scope->width - 1;
            
            // Collect min/max luma for this column
            int min_y = 255;
            int max_y = 0;
            
            for (int y = 0; y < scope->height; y++) {
                int offset = y * scope->stride + input_x * 4; // Assuming RGBA
                
                if (offset + 2 < scope->height * scope->stride) {
                    // Calculate luma (simplified)
                    int r = scope->data[offset];
                    int g = scope->data[offset + 1];
                    int b = scope->data[offset + 2];
                    int luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    
                    if (luma < min_y) min_y = luma;
                    if (luma > max_y) max_y = luma;
                }
            }
            
            // Draw line for this column
            if (min_y <= max_y) {
                double y1 = height - (min_y * height / 255.0);
                double y2 = height - (max_y * height / 255.0);
                cairo_move_to(cr, x, y1);
                cairo_line_to(cr, x, y2);
            }
        }
        
        cairo_stroke(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    }
    
    // Draw labels if enabled
    if (scope->show_labels) {
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 12);
        
        cairo_move_to(cr, width - 70, 20);
        cairo_show_text(cr, _("Waveform"));
        
        if (scope->show_grid) {
            cairo_set_font_size(cr, 10);
            // Draw scale
            for (int i = 0; i <= 10; i += 2) {
                double y = height * (i / 10.0);
                cairo_move_to(cr, 5, y + 12);
                char label[8];
                g_snprintf(label, sizeof(label), "%d%%", 100 - i * 10);
                cairo_show_text(cr, label);
            }
        }
    }
    
    return TRUE;
}

static gboolean on_draw_histogram(GtkDrawingArea *area, cairo_t *cr, int width, int height, BLOUeditScope *scope) {
    // Clear background
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);
    
    // Calculate histogram
    int histogram[256][3] = {0}; // R, G, B bins
    
    if (scope->data != NULL) {
        for (int y = 0; y < scope->height; y++) {
            for (int x = 0; x < scope->width; x++) {
                int offset = y * scope->stride + x * 4; // Assuming RGBA
                
                if (offset + 2 < scope->height * scope->stride) {
                    histogram[scope->data[offset]][0]++; // R
                    histogram[scope->data[offset + 1]][1]++; // G
                    histogram[scope->data[offset + 2]][2]++; // B
                }
            }
        }
    }
    
    // Find max value for scaling
    int max_value = 1; // Avoid division by zero
    for (int i = 0; i < 256; i++) {
        for (int c = 0; c < 3; c++) {
            if (histogram[i][c] > max_value) {
                max_value = histogram[i][c];
            }
        }
    }
    
    // Draw grid if enabled
    if (scope->show_grid) {
        cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.5);
        cairo_set_line_width(cr, 1.0);
        
        // Draw vertical lines
        for (int i = 0; i <= 8; i++) {
            double x = width * (i / 8.0);
            cairo_move_to(cr, x, 0);
            cairo_line_to(cr, x, height);
        }
        
        // Draw horizontal lines
        for (int i = 0; i <= 4; i++) {
            double y = height * (i / 4.0);
            cairo_move_to(cr, 0, y);
            cairo_line_to(cr, width, y);
        }
        
        cairo_stroke(cr);
    }
    
    // Draw histogram
    const double bar_width = (double)width / 256.0;
    
    // RGB combined (white)
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.3);
    for (int i = 0; i < 256; i++) {
        double x = i * bar_width;
        double combined_height = 0;
        
        for (int c = 0; c < 3; c++) {
            combined_height += histogram[i][c];
        }
        
        // Scale height to fit in the widget
        double scaled_height;
        if (scope->logarithmic_scale) {
            scaled_height = height * log10(1 + combined_height) / log10(1 + max_value * 3);
        } else {
            scaled_height = height * combined_height / (max_value * 3);
        }
        
        cairo_rectangle(cr, x, height - scaled_height, bar_width, scaled_height);
    }
    cairo_fill(cr);
    
    // R channel
    cairo_set_source_rgba(cr, 1.0, 0.0, 0.0, 0.7);
    for (int i = 0; i < 256; i++) {
        double x = i * bar_width;
        
        // Scale height to fit in the widget
        double scaled_height;
        if (scope->logarithmic_scale) {
            scaled_height = height * log10(1 + histogram[i][0]) / log10(1 + max_value);
        } else {
            scaled_height = height * histogram[i][0] / max_value;
        }
        
        cairo_rectangle(cr, x, height - scaled_height, bar_width, scaled_height);
    }
    cairo_fill(cr);
    
    // G channel
    cairo_set_source_rgba(cr, 0.0, 1.0, 0.0, 0.7);
    for (int i = 0; i < 256; i++) {
        double x = i * bar_width;
        
        // Scale height to fit in the widget
        double scaled_height;
        if (scope->logarithmic_scale) {
            scaled_height = height * log10(1 + histogram[i][1]) / log10(1 + max_value);
        } else {
            scaled_height = height * histogram[i][1] / max_value;
        }
        
        cairo_rectangle(cr, x, height - scaled_height, bar_width, scaled_height);
    }
    cairo_fill(cr);
    
    // B channel
    cairo_set_source_rgba(cr, 0.0, 0.0, 1.0, 0.7);
    for (int i = 0; i < 256; i++) {
        double x = i * bar_width;
        
        // Scale height to fit in the widget
        double scaled_height;
        if (scope->logarithmic_scale) {
            scaled_height = height * log10(1 + histogram[i][2]) / log10(1 + max_value);
        } else {
            scaled_height = height * histogram[i][2] / max_value;
        }
        
        cairo_rectangle(cr, x, height - scaled_height, bar_width, scaled_height);
    }
    cairo_fill(cr);
    
    // Draw labels if enabled
    if (scope->show_labels) {
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 12);
        
        cairo_move_to(cr, width - 70, 20);
        cairo_show_text(cr, _("Histogram"));
        
        if (scope->show_grid) {
            cairo_set_font_size(cr, 10);
            // Draw scale
            for (int i = 0; i <= 8; i += 2) {
                double x = width * (i / 8.0);
                cairo_move_to(cr, x, height - 5);
                char label[8];
                g_snprintf(label, sizeof(label), "%d", i * 32);
                cairo_show_text(cr, label);
            }
        }
    }
    
    return TRUE;
}

static gboolean on_draw_audio_meter(GtkDrawingArea *area, cairo_t *cr, int width, int height, BLOUeditScope *scope) {
    // Clear background
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);
    
    // Draw labels
    if (scope->show_labels) {
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 12);
        
        cairo_move_to(cr, width - 95, 20);
        cairo_show_text(cr, _("Audio Meter"));
    }
    
    // Calculate meter section heights
    double meter_height = height - 40; // Leave space for labels
    
    // Draw dB scales
    if (scope->show_grid) {
        cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.5);
        cairo_set_line_width(cr, 1.0);
        
        const char *db_labels[] = {"0", "-3", "-6", "-12", "-18", "-24", "-36", "-48", "-60"};
        double db_values[] = {0, -3, -6, -12, -18, -24, -36, -48, -60};
        
        for (int i = 0; i < 9; i++) {
            // Convert dB to linear scale position
            double linear_pos;
            if (db_values[i] <= -60) {
                linear_pos = 0;
            } else {
                linear_pos = 1.0 - (db_values[i] / -60.0);
            }
            
            double y = 20 + meter_height * (1.0 - linear_pos);
            
            // Draw line
            cairo_move_to(cr, 0, y);
            cairo_line_to(cr, width, y);
            
            // Draw label
            cairo_move_to(cr, 5, y - 2);
            cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.7);
            cairo_set_font_size(cr, 9);
            cairo_show_text(cr, db_labels[i]);
        }
        
        cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.5);
        cairo_stroke(cr);
    }
    
    // Draw meters
    if (scope->audio_data && scope->audio_channels > 0) {
        const char *channel_labels[] = {"L", "R", "C", "LFE", "Ls", "Rs", "Lbs", "Rbs"};
        const int num_labels = 8;
        
        // Calculate average RMS and peak for each channel
        double rms[8] = {0};
        double peak[8] = {0};
        
        for (int c = 0; c < scope->audio_channels && c < 8; c++) {
            double sum_squares = 0;
            for (int i = 0; i < scope->audio_samples; i++) {
                double sample = scope->audio_data[i * scope->audio_channels + c];
                sum_squares += sample * sample;
                if (fabs(sample) > peak[c]) peak[c] = fabs(sample);
            }
            rms[c] = sqrt(sum_squares / scope->audio_samples);
        }
        
        // Draw the meters
        double meter_width = width / (scope->audio_channels + 0.5); // Add space between meters
        
        for (int c = 0; c < scope->audio_channels && c < 8; c++) {
            double x = c * meter_width + 10;
            double peak_db = 20 * log10(peak[c] + 1e-10);
            double rms_db = 20 * log10(rms[c] + 1e-10);
            
            // Clamp to visible range
            if (peak_db < -60) peak_db = -60;
            if (rms_db < -60) rms_db = -60;
            
            // Convert to linear scale position
            double peak_pos = 1.0 - (peak_db / -60.0);
            double rms_pos = 1.0 - (rms_db / -60.0);
            
            // Draw peak meter (line)
            double peak_y = 20 + meter_height * (1.0 - peak_pos);
            cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
            cairo_set_line_width(cr, 2.0);
            cairo_move_to(cr, x, peak_y);
            cairo_line_to(cr, x + meter_width - 5, peak_y);
            cairo_stroke(cr);
            
            // Draw RMS meter (bar)
            double rms_y = 20 + meter_height * (1.0 - rms_pos);
            cairo_set_source_rgb(cr, 0.0, 0.7, 1.0);
            cairo_rectangle(cr, x, rms_y, meter_width - 5, meter_height - (rms_y - 20));
            cairo_fill(cr);
            
            // Draw channel label
            cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
            cairo_set_font_size(cr, 10);
            cairo_move_to(cr, x, height - 5);
            cairo_show_text(cr, c < num_labels ? channel_labels[c] : "?");
        }
    }
    
    return TRUE;
}

static gboolean on_draw_scope(GtkDrawingArea *area, cairo_t *cr, int width, int height, gpointer user_data) {
    BLOUeditScope *scope = (BLOUeditScope *)user_data;
    
    switch (scope->type) {
        case BLOUEDIT_SCOPE_VECTORSCOPE:
            return on_draw_vectorscope(area, cr, width, height, scope);
        case BLOUEDIT_SCOPE_WAVEFORM:
            return on_draw_waveform(area, cr, width, height, scope);
        case BLOUEDIT_SCOPE_HISTOGRAM:
            return on_draw_histogram(area, cr, width, height, scope);
        case BLOUEDIT_SCOPE_AUDIO_METER:
            return on_draw_audio_meter(area, cr, width, height, scope);
        case BLOUEDIT_SCOPE_RGB_PARADE:
            // Implementation similar to waveform but with separate R/G/B channels
            return on_draw_waveform(area, cr, width, height, scope);
        default:
            return FALSE;
    }
}

// Public functions

GtkWidget* blouedit_scope_create(BLOUeditScopeType scope_type) {
    BLOUeditScope *scope = g_new0(BLOUeditScope, 1);
    scope->type = scope_type;
    scope->show_grid = TRUE;
    scope->show_labels = TRUE;
    scope->scale = 1.0;
    scope->logarithmic_scale = TRUE;
    
    // Set default colors for RGB parade
    scope->r_color = (GdkRGBA){1.0, 0.0, 0.0, 0.7};
    scope->g_color = (GdkRGBA){0.0, 1.0, 0.0, 0.7};
    scope->b_color = (GdkRGBA){0.0, 0.0, 1.0, 0.7};
    
    // Create drawing area
    scope->drawing_area = gtk_drawing_area_new();
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(scope->drawing_area), 
                                  on_draw_scope, scope, (GDestroyNotify)blouedit_scope_finalize);
    
    gtk_widget_set_size_request(scope->drawing_area, 250, 150);
    
    return scope->drawing_area;
}

void blouedit_scope_update(GtkWidget* scope_widget, const guchar* data, int width, int height, int stride) {
    if (!GTK_IS_DRAWING_AREA(scope_widget)) return;
    
    BLOUeditScope *scope = g_object_get_data(G_OBJECT(scope_widget), "scope-data");
    if (!scope) return;
    
    // Free old data if any
    g_free(scope->data);
    
    // Copy new data
    int data_size = height * stride;
    scope->data = g_malloc(data_size);
    memcpy(scope->data, data, data_size);
    
    scope->width = width;
    scope->height = height;
    scope->stride = stride;
    
    // Request redraw
    gtk_widget_queue_draw(scope->drawing_area);
}

void blouedit_scope_update_audio(GtkWidget* scope_widget, const float* data, int channels, int samples) {
    if (!GTK_IS_DRAWING_AREA(scope_widget)) return;
    
    BLOUeditScope *scope = g_object_get_data(G_OBJECT(scope_widget), "scope-data");
    if (!scope) return;
    
    // Audio data only makes sense for audio meter
    if (scope->type != BLOUEDIT_SCOPE_AUDIO_METER) return;
    
    // Free old data if any
    g_free(scope->audio_data);
    
    // Copy new data
    int data_size = channels * samples * sizeof(float);
    scope->audio_data = g_malloc(data_size);
    memcpy(scope->audio_data, data, data_size);
    
    scope->audio_channels = channels;
    scope->audio_samples = samples;
    
    // Request redraw
    gtk_widget_queue_draw(scope->drawing_area);
}

void blouedit_scope_set_option(GtkWidget* scope_widget, const char* option_name, const GValue* option_value) {
    if (!GTK_IS_DRAWING_AREA(scope_widget)) return;
    
    BLOUeditScope *scope = g_object_get_data(G_OBJECT(scope_widget), "scope-data");
    if (!scope) return;
    
    if (g_strcmp0(option_name, "show-grid") == 0 && G_VALUE_HOLDS_BOOLEAN(option_value)) {
        scope->show_grid = g_value_get_boolean(option_value);
    } else if (g_strcmp0(option_name, "show-labels") == 0 && G_VALUE_HOLDS_BOOLEAN(option_value)) {
        scope->show_labels = g_value_get_boolean(option_value);
    } else if (g_strcmp0(option_name, "scale") == 0 && G_VALUE_HOLDS_DOUBLE(option_value)) {
        scope->scale = g_value_get_double(option_value);
    } else if (g_strcmp0(option_name, "logarithmic-scale") == 0 && G_VALUE_HOLDS_BOOLEAN(option_value)) {
        scope->logarithmic_scale = g_value_get_boolean(option_value);
    }
    
    // Request redraw
    gtk_widget_queue_draw(scope->drawing_area);
} 