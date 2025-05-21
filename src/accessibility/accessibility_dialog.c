#include <gtk/gtk.h>
#include <glib.h>
#include <glib/gi18n.h>
#include "accessibility.h"

struct _AccessibilityDialogData {
  BlouEditAccessibilityFeature feature;
  GtkWidget *widget;
};

/* Forward declarations */
static void on_high_contrast_theme_changed(GtkComboBox *combo_box, gpointer user_data);
static void on_text_scale_changed(GtkScale *scale, gpointer user_data);
static void on_feature_toggled(GtkToggleButton *toggle, gpointer user_data);
static void on_color_blind_mode_changed(GtkComboBox *combo_box, gpointer user_data);
static void on_reduced_motion_changed(GtkScale *scale, gpointer user_data);
static void on_screen_reader_verbose_changed(GtkComboBox *combo_box, gpointer user_data);
static void update_widgets_sensitivity(GtkBuilder *builder);

/**
 * blouedit_accessibility_show_settings_dialog:
 * @parent: Parent window for the dialog.
 *
 * Shows the accessibility settings dialog.
 */
void 
blouedit_accessibility_show_settings_dialog(GtkWindow *parent)
{
  GtkBuilder *builder;
  GtkWidget *dialog;
  GtkWidget *content_area;
  GtkWidget *notebook;
  GError *error = NULL;
  
  /* Load the UI definition */
  builder = gtk_builder_new();
  if (!gtk_builder_add_from_resource(builder, "/org/blouedit/ui/accessibility-settings.ui", &error)) {
    g_warning("Error loading UI: %s", error->message);
    g_error_free(error);
    g_object_unref(builder);
    return;
  }
  
  /* Create the dialog */
  dialog = gtk_dialog_new_with_buttons(_("Accessibility Settings"),
                                     parent,
                                     GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                     _("_Close"), GTK_RESPONSE_CLOSE,
                                     NULL);
  
  /* Set dialog size */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 600, 500);
  
  /* Get the content area of the dialog */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  
  /* Create the notebook */
  notebook = gtk_builder_get_object(builder, "accessibility_notebook");
  g_object_ref(notebook);
  gtk_container_add(GTK_CONTAINER(content_area), notebook);
  
  /* Set up feature toggles */
  GtkWidget *high_contrast_toggle = GTK_WIDGET(gtk_builder_get_object(builder, "high_contrast_toggle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(high_contrast_toggle),
                            blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST));
  
  GtkWidget *large_text_toggle = GTK_WIDGET(gtk_builder_get_object(builder, "large_text_toggle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(large_text_toggle),
                            blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT));
  
  GtkWidget *screen_reader_toggle = GTK_WIDGET(gtk_builder_get_object(builder, "screen_reader_toggle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(screen_reader_toggle),
                            blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER));
  
  GtkWidget *keyboard_nav_toggle = GTK_WIDGET(gtk_builder_get_object(builder, "keyboard_nav_toggle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(keyboard_nav_toggle),
                            blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION));
  
  GtkWidget *audio_visual_toggle = GTK_WIDGET(gtk_builder_get_object(builder, "audio_visual_toggle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(audio_visual_toggle),
                            blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_AUDIO_VISUAL));
  
  GtkWidget *reduced_motion_toggle = GTK_WIDGET(gtk_builder_get_object(builder, "reduced_motion_toggle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(reduced_motion_toggle),
                            blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION));
  
  GtkWidget *color_blind_toggle = GTK_WIDGET(gtk_builder_get_object(builder, "color_blind_toggle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(color_blind_toggle),
                            blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND));
  
  /* Connect toggle signals and store feature data */
  struct _AccessibilityDialogData *high_contrast_data = g_new(struct _AccessibilityDialogData, 1);
  high_contrast_data->feature = BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST;
  high_contrast_data->widget = high_contrast_toggle;
  g_signal_connect(high_contrast_toggle, "toggled", G_CALLBACK(on_feature_toggled), high_contrast_data);
  g_object_set_data_full(G_OBJECT(high_contrast_toggle), "feature-data", high_contrast_data, g_free);
  
  struct _AccessibilityDialogData *large_text_data = g_new(struct _AccessibilityDialogData, 1);
  large_text_data->feature = BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT;
  large_text_data->widget = large_text_toggle;
  g_signal_connect(large_text_toggle, "toggled", G_CALLBACK(on_feature_toggled), large_text_data);
  g_object_set_data_full(G_OBJECT(large_text_toggle), "feature-data", large_text_data, g_free);
  
  struct _AccessibilityDialogData *screen_reader_data = g_new(struct _AccessibilityDialogData, 1);
  screen_reader_data->feature = BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER;
  screen_reader_data->widget = screen_reader_toggle;
  g_signal_connect(screen_reader_toggle, "toggled", G_CALLBACK(on_feature_toggled), screen_reader_data);
  g_object_set_data_full(G_OBJECT(screen_reader_toggle), "feature-data", screen_reader_data, g_free);
  
  struct _AccessibilityDialogData *keyboard_nav_data = g_new(struct _AccessibilityDialogData, 1);
  keyboard_nav_data->feature = BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION;
  keyboard_nav_data->widget = keyboard_nav_toggle;
  g_signal_connect(keyboard_nav_toggle, "toggled", G_CALLBACK(on_feature_toggled), keyboard_nav_data);
  g_object_set_data_full(G_OBJECT(keyboard_nav_toggle), "feature-data", keyboard_nav_data, g_free);
  
  struct _AccessibilityDialogData *audio_visual_data = g_new(struct _AccessibilityDialogData, 1);
  audio_visual_data->feature = BLOUEDIT_ACCESSIBILITY_FEATURE_AUDIO_VISUAL;
  audio_visual_data->widget = audio_visual_toggle;
  g_signal_connect(audio_visual_toggle, "toggled", G_CALLBACK(on_feature_toggled), audio_visual_data);
  g_object_set_data_full(G_OBJECT(audio_visual_toggle), "feature-data", audio_visual_data, g_free);
  
  struct _AccessibilityDialogData *reduced_motion_data = g_new(struct _AccessibilityDialogData, 1);
  reduced_motion_data->feature = BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION;
  reduced_motion_data->widget = reduced_motion_toggle;
  g_signal_connect(reduced_motion_toggle, "toggled", G_CALLBACK(on_feature_toggled), reduced_motion_data);
  g_object_set_data_full(G_OBJECT(reduced_motion_toggle), "feature-data", reduced_motion_data, g_free);
  
  struct _AccessibilityDialogData *color_blind_data = g_new(struct _AccessibilityDialogData, 1);
  color_blind_data->feature = BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND;
  color_blind_data->widget = color_blind_toggle;
  g_signal_connect(color_blind_toggle, "toggled", G_CALLBACK(on_feature_toggled), color_blind_data);
  g_object_set_data_full(G_OBJECT(color_blind_toggle), "feature-data", color_blind_data, g_free);
  
  /* Set up high contrast theme selector */
  GtkWidget *hc_theme_combo = GTK_WIDGET(gtk_builder_get_object(builder, "high_contrast_theme_combo"));
  extern BlouEditAccessibilitySettings accessibility_settings;
  gtk_combo_box_set_active(GTK_COMBO_BOX(hc_theme_combo), accessibility_settings.high_contrast_theme);
  g_signal_connect(hc_theme_combo, "changed", G_CALLBACK(on_high_contrast_theme_changed), NULL);
  
  /* Set up text scale factor slider */
  GtkWidget *text_scale_slider = GTK_WIDGET(gtk_builder_get_object(builder, "text_scale_slider"));
  gtk_range_set_value(GTK_RANGE(text_scale_slider), accessibility_settings.text_scale_factor);
  g_signal_connect(text_scale_slider, "value-changed", G_CALLBACK(on_text_scale_changed), NULL);
  
  /* Set up color blind mode combo */
  GtkWidget *color_blind_combo = GTK_WIDGET(gtk_builder_get_object(builder, "color_blind_combo"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(color_blind_combo), accessibility_settings.color_blind_mode);
  g_signal_connect(color_blind_combo, "changed", G_CALLBACK(on_color_blind_mode_changed), NULL);
  
  /* Set up reduced motion slider */
  GtkWidget *reduced_motion_slider = GTK_WIDGET(gtk_builder_get_object(builder, "reduced_motion_slider"));
  gtk_range_set_value(GTK_RANGE(reduced_motion_slider), accessibility_settings.reduced_motion_level);
  g_signal_connect(reduced_motion_slider, "value-changed", G_CALLBACK(on_reduced_motion_changed), NULL);
  
  /* Set up screen reader verbosity combo */
  GtkWidget *screen_reader_combo = GTK_WIDGET(gtk_builder_get_object(builder, "screen_reader_combo"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(screen_reader_combo), accessibility_settings.screen_reader_verbose);
  g_signal_connect(screen_reader_combo, "changed", G_CALLBACK(on_screen_reader_verbose_changed), NULL);
  
  /* Update sensitivity of widgets based on feature toggles */
  update_widgets_sensitivity(builder);
  
  /* Show the dialog and wait for response */
  gtk_widget_show_all(dialog);
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* Clean up */
  gtk_widget_destroy(dialog);
  g_object_unref(builder);
}

/* Handle high contrast theme change */
static void 
on_high_contrast_theme_changed(GtkComboBox *combo_box, gpointer user_data)
{
  gint theme_index = gtk_combo_box_get_active(combo_box);
  
  if (theme_index >= 0 && theme_index <= 3) {
    blouedit_accessibility_set_high_contrast_theme((BlouEditHighContrastTheme)theme_index);
  }
}

/* Handle text scale factor change */
static void 
on_text_scale_changed(GtkScale *scale, gpointer user_data)
{
  gdouble value = gtk_range_get_value(GTK_RANGE(scale));
  blouedit_accessibility_set_text_scale_factor(value);
}

/* Handle feature toggle */
static void 
on_feature_toggled(GtkToggleButton *toggle, gpointer user_data)
{
  struct _AccessibilityDialogData *data = (struct _AccessibilityDialogData *)user_data;
  
  if (data) {
    if (gtk_toggle_button_get_active(toggle)) {
      blouedit_accessibility_enable_feature(data->feature);
    } else {
      blouedit_accessibility_disable_feature(data->feature);
    }
    
    /* Get the parent dialog */
    GtkWidget *dialog = gtk_widget_get_toplevel(GTK_WIDGET(toggle));
    if (gtk_widget_is_toplevel(dialog)) {
      /* Find the builder */
      GtkBuilder *builder = g_object_get_data(G_OBJECT(dialog), "builder");
      if (builder) {
        update_widgets_sensitivity(builder);
      }
    }
  }
}

/* Handle color blind mode change */
static void 
on_color_blind_mode_changed(GtkComboBox *combo_box, gpointer user_data)
{
  gint mode = gtk_combo_box_get_active(combo_box);
  extern BlouEditAccessibilitySettings accessibility_settings;
  
  accessibility_settings.color_blind_mode = mode;
  blouedit_accessibility_save_settings();
  
  /* Refresh UI with new color blind settings */
  if (blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND)) {
    /* Apply color blind mode changes */
    /* This would involve updating colors used in the application */
  }
}

/* Handle reduced motion level change */
static void 
on_reduced_motion_changed(GtkScale *scale, gpointer user_data)
{
  gint value = (gint)gtk_range_get_value(GTK_RANGE(scale));
  extern BlouEditAccessibilitySettings accessibility_settings;
  
  accessibility_settings.reduced_motion_level = value;
  blouedit_accessibility_save_settings();
  
  /* Apply reduced motion settings if enabled */
  if (blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION)) {
    /* Update animation settings based on reduced motion level */
  }
}

/* Handle screen reader verbosity change */
static void 
on_screen_reader_verbose_changed(GtkComboBox *combo_box, gpointer user_data)
{
  gint level = gtk_combo_box_get_active(combo_box);
  extern BlouEditAccessibilitySettings accessibility_settings;
  
  accessibility_settings.screen_reader_verbose = level;
  blouedit_accessibility_save_settings();
  
  /* Announce the change if screen reader is enabled */
  if (blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER)) {
    const gchar *verbosity_names[] = {
      N_("Minimal"),
      N_("Standard"),
      N_("Verbose"),
      N_("Debug")
    };
    
    gchar *msg = g_strdup_printf(_("Screen reader verbosity set to %s"), _(verbosity_names[level]));
    blouedit_accessibility_screen_reader_announce(msg, 1);
    g_free(msg);
  }
}

/* Update sensitivity of widgets based on feature toggles */
static void 
update_widgets_sensitivity(GtkBuilder *builder)
{
  /* High contrast theme settings */
  GtkWidget *hc_theme_combo = GTK_WIDGET(gtk_builder_get_object(builder, "high_contrast_theme_combo"));
  gtk_widget_set_sensitive(hc_theme_combo, 
                         blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_HIGH_CONTRAST));
  
  /* Text scale settings */
  GtkWidget *text_scale_slider = GTK_WIDGET(gtk_builder_get_object(builder, "text_scale_slider"));
  gtk_widget_set_sensitive(text_scale_slider, 
                         blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_LARGE_TEXT));
  
  /* Screen reader settings */
  GtkWidget *screen_reader_combo = GTK_WIDGET(gtk_builder_get_object(builder, "screen_reader_combo"));
  gtk_widget_set_sensitive(screen_reader_combo, 
                         blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER));
  
  /* Keyboard navigation settings */
  GtkWidget *keyboard_focus_check = GTK_WIDGET(gtk_builder_get_object(builder, "keyboard_focus_check"));
  gtk_widget_set_sensitive(keyboard_focus_check, 
                         blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION));
  
  /* Audio visualization settings */
  GtkWidget *audio_visual_frame = GTK_WIDGET(gtk_builder_get_object(builder, "audio_visual_frame"));
  gtk_widget_set_sensitive(audio_visual_frame, 
                         blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_AUDIO_VISUAL));
  
  /* Reduced motion settings */
  GtkWidget *reduced_motion_slider = GTK_WIDGET(gtk_builder_get_object(builder, "reduced_motion_slider"));
  gtk_widget_set_sensitive(reduced_motion_slider, 
                         blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_REDUCED_MOTION));
  
  /* Color blind settings */
  GtkWidget *color_blind_combo = GTK_WIDGET(gtk_builder_get_object(builder, "color_blind_combo"));
  gtk_widget_set_sensitive(color_blind_combo, 
                         blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_COLOR_BLIND));
} 