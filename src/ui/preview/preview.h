/**
 * @file preview.h
 * @brief Header file for the preview component of the BLOUedit video editor
 */

#ifndef BLOUEDIT_PREVIEW_H
#define BLOUEDIT_PREVIEW_H

#include <gtk/gtk.h>

G_BEGIN_DECLS

/**
 * @brief Creates the preview widget
 * 
 * @return GtkWidget* The preview widget
 */
GtkWidget* blouedit_preview_create(void);

/**
 * @brief Initialize the preview component
 * 
 * @param window The main application window
 */
void blouedit_preview_init(GtkWindow* window);

/**
 * @brief Sets the playback speed
 * 
 * @param speed_multiplier The speed multiplier (0.25, 0.5, 1.0, 1.5, 2.0)
 */
void blouedit_preview_set_speed(double speed_multiplier);

/**
 * @brief Move one frame forward
 */
void blouedit_preview_frame_step_forward(void);

/**
 * @brief Move one frame backward
 */
void blouedit_preview_frame_step_backward(void);

/**
 * @brief Set in point at current position
 */
void blouedit_preview_set_in_point(void);

/**
 * @brief Set out point at current position
 */
void blouedit_preview_set_out_point(void);

/**
 * @brief Toggle grid and guidelines visibility
 * 
 * @param show_grid Whether to show grid
 */
void blouedit_preview_toggle_grid(gboolean show_grid);

/**
 * @brief Toggle safe zones visibility
 * 
 * @param show_safe_zones Whether to show safe zones
 */
void blouedit_preview_toggle_safe_zones(gboolean show_safe_zones);

/**
 * @brief Enable/disable split screen comparison
 * 
 * @param enable_split_screen Whether to enable split screen
 */
void blouedit_preview_toggle_split_screen(gboolean enable_split_screen);

/**
 * @brief Set preview resolution for performance optimization
 * 
 * @param resolution_percentage Percentage of original resolution (25, 50, 75, 100)
 */
void blouedit_preview_set_resolution(int resolution_percentage);

/**
 * @brief Toggle scopes (vectorscope, waveform, histogram)
 * 
 * @param scope_type The type of scope to toggle
 * @param enabled Whether to enable the scope
 */
void blouedit_preview_toggle_scope(const char* scope_type, gboolean enabled);

/**
 * @brief Apply LUT to preview
 * 
 * @param lut_path Path to the LUT file
 */
void blouedit_preview_apply_lut(const char* lut_path);

/**
 * @brief Toggle HDR preview mode
 * 
 * @param hdr_mode HDR mode (HDR10, HLG, Dolby Vision)
 * @param enabled Whether to enable HDR preview
 */
void blouedit_preview_toggle_hdr(const char* hdr_mode, gboolean enabled);

/**
 * @brief Configure hardware acceleration settings
 * 
 * @param use_hardware_acceleration Whether to use hardware acceleration
 */
void blouedit_preview_set_hardware_acceleration(gboolean use_hardware_acceleration);

/**
 * @brief Configure caching settings
 * 
 * @param cache_type Type of cache ("ram" or "disk")
 * @param cache_size_mb Cache size in megabytes
 */
void blouedit_preview_configure_cache(const char* cache_type, int cache_size_mb);

G_END_DECLS

#endif /* BLOUEDIT_PREVIEW_H */ 