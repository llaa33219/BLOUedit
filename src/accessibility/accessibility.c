#include <gtk/gtk.h>
#include <glib.h>
#include <glib/gi18n.h>
#include <json-glib/json-glib.h>
#include "accessibility.h"
#include "../config/settings.h"

/* Global accessibility settings */
static BlouEditAccessibilitySettings accessibility_settings = {
  .enabled_features = 0,
  .high_contrast_theme = BLOUEDIT_HIGH_CONTRAST_THEME_DARK,
  .text_scale_factor = 1.0,
  .keyboard_focus_visual = TRUE,
  .screen_reader_verbose = 1,
  .audio_visualization_enabled = FALSE,
  .reduced_motion_level = 0,
  .color_blind_mode = 0
};

/* List of registered keyboard shortcuts */
static GSList *keyboard_shortcuts = NULL;

/* CSS providers for different themes */
static GtkCssProvider *high_contrast_dark_provider = NULL;
static GtkCssProvider *high_contrast_light_provider = NULL;
static GtkCssProvider *yellow_on_black_provider = NULL;
static GtkCssProvider *black_on_yellow_provider = NULL;
static GtkCssProvider *current_provider = NULL;

/* Initialize accessibility features */
void 
blouedit_accessibility_init(GtkApplication *application)
{
  /* Load settings */
  blouedit_accessibility_load_settings();
  
  /* Create CSS providers for themes */
  high_contrast_dark_provider = gtk_css_provider_new();
  high_contrast_light_provider = gtk_css_provider_new();
  yellow_on_black_provider = gtk_css_provider_new();
  black_on_yellow_provider = gtk_css_provider_new();
  
  /* Load CSS data */
  gtk_css_provider_load_from_resource(high_contrast_dark_provider, 
                                     "/org/blouedit/themes/high-contrast-dark.css");
  gtk_css_provider_load_from_resource(high_contrast_light_provider, 
                                     "/org/blouedit/themes/high-contrast-light.css");
  gtk_css_provider_load_from_resource(yellow_on_black_provider, 
                                     "/org/blouedit/themes/yellow-on-black.css");
  gtk_css_provider_load_from_resource(black_on_yellow_provider, 
                                     "/org/blouedit/themes/black-on-yellow.css");
  
  /* Apply saved settings */
  if (blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST)) {
    blouedit_accessibility_set_high_contrast_theme(accessibility_settings.high_contrast_theme);
  }
  
  if (blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT)) {
    blouedit_accessibility_set_text_scale_factor(accessibility_settings.text_scale_factor);
  }
  
  /* Register default keyboard shortcuts for accessibility */
  blouedit_accessibility_register_default_shortcuts();
}

/* Enable a specific accessibility feature */
void 
blouedit_accessibility_enable_feature(BlouEditAccessibilityFeature feature)
{
  /* Check if the feature is already enabled */
  if (accessibility_settings.enabled_features & feature) {
    return;
  }
  
  /* Enable the feature */
  accessibility_settings.enabled_features |= feature;
  
  /* Apply feature-specific settings */
  switch (feature) {
    case BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST:
      blouedit_accessibility_set_high_contrast_theme(accessibility_settings.high_contrast_theme);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT:
      blouedit_accessibility_set_text_scale_factor(accessibility_settings.text_scale_factor);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER:
      /* Ensure ATK is properly set up */
      gtk_widget_set_default_direction(GTK_TEXT_DIR_LTR);
      blouedit_accessibility_screen_reader_announce(_("Screen reader support enabled"), 1);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION:
      /* Enable keyboard navigation mode */
      g_object_set(gtk_settings_get_default(), "gtk-keynav-use-caret", TRUE, NULL);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_AUDIO_VISUAL:
      accessibility_settings.audio_visualization_enabled = TRUE;
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION:
      /* Apply reduced motion settings */
      g_object_set(gtk_settings_get_default(), "gtk-enable-animations", FALSE, NULL);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND:
      /* Update UI for color blind users */
      break;
      
    default:
      g_warning("Unknown accessibility feature %d", feature);
      break;
  }
  
  /* Save settings */
  blouedit_accessibility_save_settings();
}

/* Disable a specific accessibility feature */
void 
blouedit_accessibility_disable_feature(BlouEditAccessibilityFeature feature)
{
  /* Check if the feature is already disabled */
  if (!(accessibility_settings.enabled_features & feature)) {
    return;
  }
  
  /* Disable the feature */
  accessibility_settings.enabled_features &= ~feature;
  
  /* Revert feature-specific settings */
  switch (feature) {
    case BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST:
      /* Remove the high contrast theme */
      if (current_provider != NULL) {
        gtk_style_context_remove_provider_for_screen(gdk_screen_get_default(),
                                                   GTK_STYLE_PROVIDER(current_provider));
        current_provider = NULL;
      }
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT:
      /* Reset text scale to normal */
      blouedit_accessibility_set_text_scale_factor(1.0);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER:
      blouedit_accessibility_screen_reader_announce(_("Screen reader support disabled"), 1);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION:
      /* Disable keyboard navigation mode */
      g_object_set(gtk_settings_get_default(), "gtk-keynav-use-caret", FALSE, NULL);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_AUDIO_VISUAL:
      accessibility_settings.audio_visualization_enabled = FALSE;
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION:
      /* Re-enable animations */
      g_object_set(gtk_settings_get_default(), "gtk-enable-animations", TRUE, NULL);
      break;
      
    case BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND:
      /* Reset UI color scheme */
      break;
      
    default:
      g_warning("Unknown accessibility feature %d", feature);
      break;
  }
  
  /* Save settings */
  blouedit_accessibility_save_settings();
}

/* Check if a specific accessibility feature is enabled */
gboolean 
blouedit_accessibility_is_feature_enabled(BlouEditAccessibilityFeature feature)
{
  return (accessibility_settings.enabled_features & feature) != 0;
}

/* Set the high contrast theme */
void 
blouedit_accessibility_set_high_contrast_theme(BlouEditHighContrastTheme theme)
{
  GtkCssProvider *provider = NULL;
  
  /* Remove any existing provider */
  if (current_provider != NULL) {
    gtk_style_context_remove_provider_for_screen(gdk_screen_get_default(),
                                               GTK_STYLE_PROVIDER(current_provider));
    current_provider = NULL;
  }
  
  /* Select the appropriate provider */
  switch (theme) {
    case BLOUEDIT_HIGH_CONTRAST_THEME_DARK:
      provider = high_contrast_dark_provider;
      break;
      
    case BLOUEDIT_HIGH_CONTRAST_THEME_LIGHT:
      provider = high_contrast_light_provider;
      break;
      
    case BLOUEDIT_HIGH_CONTRAST_THEME_YELLOW_ON_BLACK:
      provider = yellow_on_black_provider;
      break;
      
    case BLOUEDIT_HIGH_CONTRAST_THEME_BLACK_ON_YELLOW:
      provider = black_on_yellow_provider;
      break;
      
    default:
      g_warning("Unknown high contrast theme %d", theme);
      return;
  }
  
  /* Apply the theme */
  if (provider != NULL) {
    gtk_style_context_add_provider_for_screen(gdk_screen_get_default(),
                                            GTK_STYLE_PROVIDER(provider),
                                            GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    current_provider = provider;
  }
  
  /* Save the theme choice */
  accessibility_settings.high_contrast_theme = theme;
  blouedit_accessibility_save_settings();
}

/* Set the text scale factor */
void 
blouedit_accessibility_set_text_scale_factor(gdouble scale)
{
  /* Set the scale factor */
  accessibility_settings.text_scale_factor = scale;
  
  /* Apply to GTK */
  GtkSettings *settings = gtk_settings_get_default();
  g_object_set(settings, "gtk-xft-dpi", (gint)(scale * 96 * 1024), NULL);
  
  /* Save settings */
  blouedit_accessibility_save_settings();
}

/* Register default keyboard shortcuts for accessibility */
static void
blouedit_accessibility_register_default_shortcuts(void)
{
  /* Register common accessibility shortcuts */
  blouedit_accessibility_register_keyboard_shortcut(
    GDK_KEY_a, GDK_CONTROL_MASK | GDK_ALT_MASK,
    "accessibility-settings", _("Open accessibility settings"));
    
  blouedit_accessibility_register_keyboard_shortcut(
    GDK_KEY_plus, GDK_CONTROL_MASK,
    "increase-text-size", _("Increase text size"));
    
  blouedit_accessibility_register_keyboard_shortcut(
    GDK_KEY_minus, GDK_CONTROL_MASK,
    "decrease-text-size", _("Decrease text size"));
    
  blouedit_accessibility_register_keyboard_shortcut(
    GDK_KEY_0, GDK_CONTROL_MASK,
    "reset-text-size", _("Reset text size"));
    
  blouedit_accessibility_register_keyboard_shortcut(
    GDK_KEY_h, GDK_CONTROL_MASK | GDK_ALT_MASK,
    "toggle-high-contrast", _("Toggle high contrast mode"));
    
  blouedit_accessibility_register_keyboard_shortcut(
    GDK_KEY_r, GDK_CONTROL_MASK | GDK_ALT_MASK,
    "toggle-screen-reader", _("Toggle screen reader announcements"));
}

/* Register a keyboard shortcut for accessibility */
void 
blouedit_accessibility_register_keyboard_shortcut(guint key, GdkModifierType modifiers,
                                          const gchar *action, const gchar *description)
{
  BlouEditKeyboardShortcut *shortcut = g_new0(BlouEditKeyboardShortcut, 1);
  
  shortcut->key = key;
  shortcut->modifiers = modifiers;
  shortcut->action = g_strdup(action);
  shortcut->description = g_strdup(description);
  
  keyboard_shortcuts = g_slist_append(keyboard_shortcuts, shortcut);
}

/* Send a message to the screen reader */
void 
blouedit_accessibility_screen_reader_announce(const gchar *message, gint priority)
{
  /* Only announce if screen reader support is enabled */
  if (!blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER)) {
    return;
  }
  
  /* For verbosity level check */
  if (priority > accessibility_settings.screen_reader_verbose) {
    return;
  }
  
  /* Use ATK to make the announcement */
  AtkObject *accessible = ATK_OBJECT(gtk_accessible_new());
  atk_object_notify_state_change(accessible, ATK_STATE_SHOWING, TRUE);
  g_signal_emit_by_name(accessible, "visible-data-changed");
  
  atk_object_set_name(accessible, message);
  atk_object_set_description(accessible, message);
  
  g_object_unref(accessible);
}

/* Save accessibility settings to config file */
void 
blouedit_accessibility_save_settings(void)
{
  JsonBuilder *builder = json_builder_new();
  
  json_builder_begin_object(builder);
  
  json_builder_set_member_name(builder, "enabled_features");
  json_builder_add_int_value(builder, accessibility_settings.enabled_features);
  
  json_builder_set_member_name(builder, "high_contrast_theme");
  json_builder_add_int_value(builder, accessibility_settings.high_contrast_theme);
  
  json_builder_set_member_name(builder, "text_scale_factor");
  json_builder_add_double_value(builder, accessibility_settings.text_scale_factor);
  
  json_builder_set_member_name(builder, "keyboard_focus_visual");
  json_builder_add_boolean_value(builder, accessibility_settings.keyboard_focus_visual);
  
  json_builder_set_member_name(builder, "screen_reader_verbose");
  json_builder_add_int_value(builder, accessibility_settings.screen_reader_verbose);
  
  json_builder_set_member_name(builder, "audio_visualization_enabled");
  json_builder_add_boolean_value(builder, accessibility_settings.audio_visualization_enabled);
  
  json_builder_set_member_name(builder, "reduced_motion_level");
  json_builder_add_int_value(builder, accessibility_settings.reduced_motion_level);
  
  json_builder_set_member_name(builder, "color_blind_mode");
  json_builder_add_int_value(builder, accessibility_settings.color_blind_mode);
  
  json_builder_end_object(builder);
  
  JsonGenerator *generator = json_generator_new();
  JsonNode *root = json_builder_get_root(builder);
  json_generator_set_root(generator, root);
  
  gchar *settings_path = g_build_filename(g_get_user_config_dir(), "blouedit", "accessibility.json", NULL);
  json_generator_to_file(generator, settings_path, NULL);
  
  g_free(settings_path);
  json_node_free(root);
  g_object_unref(generator);
  g_object_unref(builder);
}

/* Load accessibility settings from config file */
void 
blouedit_accessibility_load_settings(void)
{
  gchar *settings_path = g_build_filename(g_get_user_config_dir(), "blouedit", "accessibility.json", NULL);
  
  if (!g_file_test(settings_path, G_FILE_TEST_EXISTS)) {
    g_free(settings_path);
    return;
  }
  
  JsonParser *parser = json_parser_new();
  GError *error = NULL;
  
  if (!json_parser_load_from_file(parser, settings_path, &error)) {
    g_warning("Error loading accessibility settings: %s", error->message);
    g_error_free(error);
    g_object_unref(parser);
    g_free(settings_path);
    return;
  }
  
  JsonNode *root = json_parser_get_root(parser);
  JsonObject *obj = json_node_get_object(root);
  
  if (json_object_has_member(obj, "enabled_features")) {
    accessibility_settings.enabled_features = json_object_get_int_member(obj, "enabled_features");
  }
  
  if (json_object_has_member(obj, "high_contrast_theme")) {
    accessibility_settings.high_contrast_theme = json_object_get_int_member(obj, "high_contrast_theme");
  }
  
  if (json_object_has_member(obj, "text_scale_factor")) {
    accessibility_settings.text_scale_factor = json_object_get_double_member(obj, "text_scale_factor");
  }
  
  if (json_object_has_member(obj, "keyboard_focus_visual")) {
    accessibility_settings.keyboard_focus_visual = json_object_get_boolean_member(obj, "keyboard_focus_visual");
  }
  
  if (json_object_has_member(obj, "screen_reader_verbose")) {
    accessibility_settings.screen_reader_verbose = json_object_get_int_member(obj, "screen_reader_verbose");
  }
  
  if (json_object_has_member(obj, "audio_visualization_enabled")) {
    accessibility_settings.audio_visualization_enabled = json_object_get_boolean_member(obj, "audio_visualization_enabled");
  }
  
  if (json_object_has_member(obj, "reduced_motion_level")) {
    accessibility_settings.reduced_motion_level = json_object_get_int_member(obj, "reduced_motion_level");
  }
  
  if (json_object_has_member(obj, "color_blind_mode")) {
    accessibility_settings.color_blind_mode = json_object_get_int_member(obj, "color_blind_mode");
  }
  
  g_object_unref(parser);
  g_free(settings_path);
} 