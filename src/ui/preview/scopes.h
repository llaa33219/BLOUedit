/**
 * @file scopes.h
 * @brief Header file for the video scopes functionality (vectorscope, waveform, histogram, etc.)
 */

#ifndef BLOUEDIT_SCOPES_H
#define BLOUEDIT_SCOPES_H

#include <gtk/gtk.h>

G_BEGIN_DECLS

/**
 * @brief The type of video scope
 */
typedef enum {
    BLOUEDIT_SCOPE_VECTORSCOPE,
    BLOUEDIT_SCOPE_WAVEFORM,
    BLOUEDIT_SCOPE_HISTOGRAM,
    BLOUEDIT_SCOPE_RGB_PARADE,
    BLOUEDIT_SCOPE_AUDIO_METER
} BLOUeditScopeType;

/**
 * @brief Create a new scope widget
 * 
 * @param scope_type The type of scope to create
 * @return GtkWidget* The scope widget
 */
GtkWidget* blouedit_scope_create(BLOUeditScopeType scope_type);

/**
 * @brief Update the scope with new video data
 * 
 * @param scope The scope widget
 * @param data The video data
 * @param width The width of the video
 * @param height The height of the video
 * @param stride The stride of the video data
 */
void blouedit_scope_update(GtkWidget* scope, const guchar* data, int width, int height, int stride);

/**
 * @brief Update the audio meter with new audio data
 * 
 * @param scope The scope widget (must be BLOUEDIT_SCOPE_AUDIO_METER)
 * @param data The audio data
 * @param channels The number of audio channels
 * @param samples The number of audio samples
 */
void blouedit_scope_update_audio(GtkWidget* scope, const float* data, int channels, int samples);

/**
 * @brief Set scope display options
 * 
 * @param scope The scope widget
 * @param option_name The name of the option to set
 * @param option_value The value of the option
 */
void blouedit_scope_set_option(GtkWidget* scope, const char* option_name, const GValue* option_value);

G_END_DECLS

#endif /* BLOUEDIT_SCOPES_H */ 