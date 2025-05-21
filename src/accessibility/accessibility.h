#ifndef BLOUEDIT_ACCESSIBILITY_H
#define BLOUEDIT_ACCESSIBILITY_H

#include <gtk/gtk.h>
#include <glib.h>
#include "../core/types.h"
#include "../timeline/timeline.h"

G_BEGIN_DECLS

/**
 * BlouEditAccessibilityFeature:
 * @BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST: High contrast UI.
 * @BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT: Larger text throughout the UI.
 * @BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER: Screen reader compatibility.
 * @BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION: Enhanced keyboard navigation.
 * @BLOUEDIT_ACCESSIBILITY_FEATURE_AUDIO_VISUAL: Audio visualization for hearing-impaired users.
 * @BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION: Reduced motion effects.
 * @BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND: Color blind friendly UI.
 *
 * Accessibility features that can be enabled in BLOUedit.
 */
typedef enum {
  BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST      = 1 << 0,
  BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT         = 1 << 1,
  BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER      = 1 << 2,
  BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION = 1 << 3,
  BLOUEDIT_ACCESSIBILITY_FEATURE_AUDIO_VISUAL       = 1 << 4,
  BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION     = 1 << 5,
  BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND        = 1 << 6
} BlouEditAccessibilityFeature;

/**
 * BlouEditHighContrastTheme:
 * @BLOUEDIT_HIGH_CONTRAST_THEME_DARK: High contrast dark theme.
 * @BLOUEDIT_HIGH_CONTRAST_THEME_LIGHT: High contrast light theme.
 * @BLOUEDIT_HIGH_CONTRAST_THEME_YELLOW_ON_BLACK: Yellow on black high contrast theme.
 * @BLOUEDIT_HIGH_CONTRAST_THEME_BLACK_ON_YELLOW: Black on yellow high contrast theme.
 *
 * High contrast theme variants for accessibility.
 */
typedef enum {
  BLOUEDIT_HIGH_CONTRAST_THEME_DARK,
  BLOUEDIT_HIGH_CONTRAST_THEME_LIGHT,
  BLOUEDIT_HIGH_CONTRAST_THEME_YELLOW_ON_BLACK,
  BLOUEDIT_HIGH_CONTRAST_THEME_BLACK_ON_YELLOW
} BlouEditHighContrastTheme;

/**
 * BlouEditAccessibilitySettings:
 * @enabled_features: Bitfield of enabled accessibility features.
 * @high_contrast_theme: Current high contrast theme if enabled.
 * @text_scale_factor: Text scaling factor (1.0 is normal size).
 * @keyboard_focus_visual: Whether to show visual indicators for keyboard focus.
 * @screen_reader_verbose: Level of verbosity for screen reader announcements.
 * @audio_visualization_enabled: Whether to show audio visualization for hearing-impaired users.
 * @reduced_motion_level: Level of motion reduction (0-100, 0 is no reduction).
 * @color_blind_mode: Type of color blindness to accommodate.
 *
 * Settings for accessibility features in BLOUedit.
 */
typedef struct _BlouEditAccessibilitySettings {
  guint enabled_features;
  BlouEditHighContrastTheme high_contrast_theme;
  gdouble text_scale_factor;
  gboolean keyboard_focus_visual;
  gint screen_reader_verbose;
  gboolean audio_visualization_enabled;
  gint reduced_motion_level;
  gint color_blind_mode;
} BlouEditAccessibilitySettings;

/**
 * BlouEditKeyboardShortcut:
 * @key: The key value.
 * @modifiers: Modifier keys (Ctrl, Alt, Shift, etc.).
 * @action: Action to perform when shortcut is triggered.
 * @description: Human-readable description of the shortcut.
 *
 * Represents a keyboard shortcut for accessibility.
 */
typedef struct _BlouEditKeyboardShortcut {
  guint key;
  GdkModifierType modifiers;
  gchar *action;
  gchar *description;
} BlouEditKeyboardShortcut;

/* Function prototypes */

/**
 * blouedit_accessibility_init:
 * @application: The BLOUedit application instance.
 *
 * Initialize accessibility features.
 */
void blouedit_accessibility_init(GtkApplication *application);

/**
 * blouedit_accessibility_enable_feature:
 * @feature: The accessibility feature to enable.
 *
 * Enable a specific accessibility feature.
 */
void blouedit_accessibility_enable_feature(BlouEditAccessibilityFeature feature);

/**
 * blouedit_accessibility_disable_feature:
 * @feature: The accessibility feature to disable.
 *
 * Disable a specific accessibility feature.
 */
void blouedit_accessibility_disable_feature(BlouEditAccessibilityFeature feature);

/**
 * blouedit_accessibility_is_feature_enabled:
 * @feature: The accessibility feature to check.
 *
 * Returns: %TRUE if the feature is enabled, %FALSE otherwise.
 */
gboolean blouedit_accessibility_is_feature_enabled(BlouEditAccessibilityFeature feature);

/**
 * blouedit_accessibility_set_high_contrast_theme:
 * @theme: The high contrast theme to set.
 *
 * Set the high contrast theme to use.
 */
void blouedit_accessibility_set_high_contrast_theme(BlouEditHighContrastTheme theme);

/**
 * blouedit_accessibility_set_text_scale_factor:
 * @scale: The text scale factor (1.0 is normal).
 *
 * Set the text scale factor for all text in the application.
 */
void blouedit_accessibility_set_text_scale_factor(gdouble scale);

/**
 * blouedit_accessibility_register_keyboard_shortcut:
 * @key: The key value.
 * @modifiers: Modifier keys.
 * @action: Action to perform.
 * @description: Description of the shortcut.
 *
 * Register a new keyboard shortcut for accessibility.
 */
void blouedit_accessibility_register_keyboard_shortcut(guint key, GdkModifierType modifiers, 
                                                   const gchar *action, const gchar *description);

/**
 * blouedit_accessibility_screen_reader_announce:
 * @message: The message to announce.
 * @priority: Priority level of the announcement.
 *
 * Send a message to the screen reader.
 */
void blouedit_accessibility_screen_reader_announce(const gchar *message, gint priority);

/**
 * blouedit_accessibility_show_settings_dialog:
 * @parent: Parent window.
 *
 * Show the accessibility settings dialog.
 */
void blouedit_accessibility_show_settings_dialog(GtkWindow *parent);

/**
 * blouedit_timeline_enhance_accessibility:
 * @timeline: The timeline to enhance.
 *
 * Add accessibility features to a timeline.
 */
void blouedit_timeline_enhance_accessibility(BlouEditTimeline *timeline);

/**
 * blouedit_timeline_handle_accessibility_key_press:
 * @timeline: The timeline.
 * @event: The key press event.
 *
 * Handle keyboard navigation for the timeline.
 *
 * Returns: %TRUE if the event was handled, %FALSE otherwise.
 */
gboolean blouedit_timeline_handle_accessibility_key_press(BlouEditTimeline *timeline, GdkEventKey *event);

/**
 * blouedit_accessibility_save_settings:
 *
 * Save accessibility settings to config file.
 */
void blouedit_accessibility_save_settings(void);

/**
 * blouedit_accessibility_load_settings:
 *
 * Load accessibility settings from config file.
 */
void blouedit_accessibility_load_settings(void);

G_END_DECLS

#endif /* BLOUEDIT_ACCESSIBILITY_H */ 