/**
 * @file theme_manager.h
 * @brief Header file for theme management functionality
 */

#ifndef BLOUEDIT_THEME_MANAGER_H
#define BLOUEDIT_THEME_MANAGER_H

#include <gtk/gtk.h>

G_BEGIN_DECLS

/**
 * @brief Theme type (light or dark)
 */
typedef enum {
    BLOUEDIT_THEME_LIGHT,
    BLOUEDIT_THEME_DARK,
    BLOUEDIT_THEME_SYSTEM
} BLOUeditThemeType;

/**
 * @brief Initializes the theme manager
 * 
 * @param application The GtkApplication instance
 */
void blouedit_theme_manager_init(GtkApplication *application);

/**
 * @brief Sets the application theme
 * 
 * @param theme_type The theme type to set
 */
void blouedit_theme_manager_set_theme(BLOUeditThemeType theme_type);

/**
 * @brief Gets the current theme type
 * 
 * @return BLOUeditThemeType The current theme type
 */
BLOUeditThemeType blouedit_theme_manager_get_theme(void);

/**
 * @brief Sets a custom color scheme
 * 
 * @param primary_color The primary color (hex string, e.g. "#3584e4")
 * @param accent_color The accent color
 * @param background_color The background color
 * @param text_color The text color
 */
void blouedit_theme_manager_set_custom_colors(const char *primary_color, 
                                            const char *accent_color,
                                            const char *background_color,
                                            const char *text_color);

/**
 * @brief Creates a theme settings widget
 * 
 * @return GtkWidget* The theme settings widget
 */
GtkWidget* blouedit_theme_manager_create_settings_widget(void);

/**
 * @brief Applies global CSS to the application
 * 
 * @param css The CSS string to apply
 */
void blouedit_theme_manager_apply_css(const char *css);

G_END_DECLS

#endif /* BLOUEDIT_THEME_MANAGER_H */ 