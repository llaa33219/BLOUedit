#include "window.h"
#include <glib/gi18n.h>
#include <gstreamer-1.0/gst/gst.h>

// Forward declarations of components that will be implemented
typedef struct _BlouEditTimeline BlouEditTimeline;
typedef struct _BlouEditPlayer BlouEditPlayer;
typedef struct _BlouEditMediaLibrary BlouEditMediaLibrary;
typedef struct _BlouEditEffectsPanel BlouEditEffectsPanel;
typedef struct _BlouEditTextPanel BlouEditTextPanel;
typedef struct _BlouEditAIPanel BlouEditAIPanel;

struct _BlouEditWindow
{
  AdwApplicationWindow  parent_instance;

  /* Template widgets */
  GtkHeaderBar        *header_bar;
  AdwViewStack        *stack;
  GtkBox              *content_box;
  GtkPaned            *main_paned;
  GtkPaned            *side_paned;
  GtkBox              *tools_box;
  GtkButton           *new_project_button;
  GtkButton           *open_button;
  GtkButton           *save_button;
  GtkMenuButton       *menu_button;
  
  /* Project components */
  BlouEditTimeline     *timeline;
  BlouEditPlayer       *player;
  BlouEditMediaLibrary *media_library;
  BlouEditEffectsPanel *effects_panel;
  BlouEditTextPanel    *text_panel;
  BlouEditAIPanel      *ai_panel;
  
  /* State */
  GFile               *current_file;
  gboolean            modified;
  gboolean            is_playing;
};

G_DEFINE_TYPE (BlouEditWindow, blouedit_window, ADW_TYPE_APPLICATION_WINDOW)

static void
blouedit_window_class_init (BlouEditWindowClass *klass)
{
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  gtk_widget_class_set_template_from_resource (widget_class, "/com/blouedit/BLOUedit/ui/window.ui");
  
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, header_bar);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, stack);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, content_box);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, main_paned);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, side_paned);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, tools_box);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, new_project_button);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, open_button);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, save_button);
  gtk_widget_class_bind_template_child (widget_class, BlouEditWindow, menu_button);
}

// Add these function prototypes
static void on_proxy_settings_action (GSimpleAction *action, GVariant *parameter, gpointer user_data);
static void on_performance_mode_action (GSimpleAction *action, GVariant *parameter, gpointer user_data);

static void
blouedit_window_init (BlouEditWindow *self)
{
  /* Initialize GStreamer */
  gst_init(NULL, NULL);
  
  /* Load UI template */
  gtk_widget_init_template (GTK_WIDGET (self));
  
  /* Initialize state */
  self->current_file = NULL;
  self->modified = FALSE;
  self->is_playing = FALSE;
  
  /* Set up actions */
  const GActionEntry win_actions[] = {
    // Project actions
    { "new-project", NULL, NULL, NULL, NULL },
    { "open", NULL, NULL, NULL, NULL },
    { "save", NULL, NULL, NULL, NULL },
    { "save-as", NULL, NULL, NULL, NULL },
    { "export", NULL, NULL, NULL, NULL },
    // Edit actions
    { "cut", NULL, NULL, NULL, NULL },
    { "copy", NULL, NULL, NULL, NULL },
    { "paste", NULL, NULL, NULL, NULL },
    { "delete", NULL, NULL, NULL, NULL },
    { "select-all", NULL, NULL, NULL, NULL },
    { "add-transition", NULL, NULL, NULL, NULL },
    { "split-clip", NULL, NULL, NULL, NULL },
    // Playback actions
    { "play-pause", NULL, NULL, NULL, NULL },
    { "stop", NULL, NULL, NULL, NULL },
    // Timeline actions
    { "text-to-edit", NULL, NULL, NULL, NULL },
    // AI actions
    { "text-to-video", NULL, NULL, NULL, NULL },
    { "audio-to-video", NULL, NULL, NULL, NULL },
    { "image-to-video", NULL, NULL, NULL, NULL },
    { "storyboard-generator", NULL, NULL, NULL, NULL },
    { "thumbnail-generator", NULL, NULL, NULL, NULL },
    { "music-generator", NULL, NULL, NULL, NULL },
    { "vocal-remover", NULL, NULL, NULL, NULL },
    { "voice-cloning", NULL, NULL, NULL, NULL },
    { "face-mosaic", NULL, NULL, NULL, NULL },
    { "auto-caption", NULL, NULL, NULL, NULL },
    { "smart-cutout", NULL, NULL, NULL, NULL },
    { "speech-to-text", NULL, NULL, NULL, NULL },
    { "text-to-speech", NULL, NULL, NULL, NULL },
    { "sticker-generator", NULL, NULL, NULL, NULL },
    { "image-generator", NULL, NULL, NULL, NULL },
    { "ai-mate", NULL, NULL, NULL, NULL },
    { "ai-copywriting", NULL, NULL, NULL, NULL },
    { "frame-interpolation", NULL, NULL, NULL, NULL },
    { "scene-detection", NULL, NULL, NULL, NULL },
    { "style-transfer", NULL, NULL, NULL, NULL },
    { "enhance-video", NULL, NULL, NULL, NULL },
    // Advanced Video actions
    { "planar-tracking", NULL, NULL, NULL, NULL },
    { "multi-camera", NULL, NULL, NULL, NULL },
    { "image-sequence", NULL, NULL, NULL, NULL },
    { "video-compression", NULL, NULL, NULL, NULL },
    { "keyframe-curves", NULL, NULL, NULL, NULL },
    { "color-correction", NULL, NULL, NULL, NULL },
    { "speed-ramping", NULL, NULL, NULL, NULL },
    { "motion-tracking", NULL, NULL, NULL, NULL },
    { "chroma-key", NULL, NULL, NULL, NULL },
    { "auto-reframe", NULL, NULL, NULL, NULL },
    { "adjustment-layers", NULL, NULL, NULL, NULL },
    { "quick-split", NULL, NULL, NULL, NULL },
    { "keyboard-shortcuts", NULL, NULL, NULL, NULL },
    // Performance actions
    { "proxy-settings", on_proxy_settings_action, NULL, NULL, NULL },
    { "performance-mode", on_performance_mode_action, NULL, NULL, NULL },
    // Audio actions
    { "voice-modulator", NULL, NULL, NULL, NULL },
    { "beat-sync", NULL, NULL, NULL, NULL },
    { "audio-visualization", NULL, NULL, NULL, NULL },
    { "auto-sync", NULL, NULL, NULL, NULL },
    { "audio-stretch", NULL, NULL, NULL, NULL },
    { "noise-removal", NULL, NULL, NULL, NULL },
  };
  g_action_map_add_action_entries (G_ACTION_MAP (self), win_actions, G_N_ELEMENTS (win_actions), self);
  
  /* Set up CSS */
  GtkCssProvider *provider = gtk_css_provider_new ();
  gtk_css_provider_load_from_resource (provider, "/com/blouedit/BLOUedit/css/style.css");
  gtk_style_context_add_provider_for_display (gdk_display_get_default (),
                                             GTK_STYLE_PROVIDER (provider),
                                             GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
  g_object_unref (provider);
}

// Add the action implementations
static void
on_proxy_settings_action (GSimpleAction *action, GVariant *parameter, gpointer user_data)
{
  BlouEditWindow *window = BLOUEDIT_WINDOW (user_data);
  BlouEditTimeline *timeline = NULL;
  
  // Get the active timeline from the window
  // This is a simplified implementation - you might need to get the actual timeline widget
  // from your UI, depending on your application structure
  GtkWidget *editor_view = gtk_stack_get_visible_child (window->stack);
  if (editor_view) {
    // Assuming the timeline is somehow accessible from the editor view
    // This would need to be adapted to your actual application structure
    timeline = BLOUEDIT_TIMELINE (g_object_get_data (G_OBJECT (editor_view), "timeline"));
  }
  
  if (timeline) {
    blouedit_timeline_show_proxy_settings_dialog (timeline);
  } else {
    // Show an error message if the timeline is not available
    GtkWidget *dialog = gtk_message_dialog_new (GTK_WINDOW (window),
                                              GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                              GTK_MESSAGE_ERROR,
                                              GTK_BUTTONS_CLOSE,
                                              "Unable to access timeline. Please open a project first.");
    gtk_dialog_run (GTK_DIALOG (dialog));
    gtk_widget_destroy (dialog);
  }
}

static void
on_performance_mode_action (GSimpleAction *action, GVariant *parameter, gpointer user_data)
{
  BlouEditWindow *window = BLOUEDIT_WINDOW (user_data);
  BlouEditTimeline *timeline = NULL;
  
  // Get the active timeline from the window
  // This is a simplified implementation - might need to get the actual timeline widget
  // from your UI, depending on your application structure
  GtkWidget *editor_view = gtk_stack_get_visible_child (window->stack);
  if (editor_view) {
    // Assuming the timeline is somehow accessible from the editor view
    // This would need to be adapted to your actual application structure
    timeline = BLOUEDIT_TIMELINE (g_object_get_data (G_OBJECT (editor_view), "timeline"));
  }
  
  if (timeline) {
    blouedit_timeline_show_performance_settings_dialog (timeline);
  } else {
    // Show an error message if the timeline is not available
    GtkWidget *dialog = gtk_message_dialog_new (GTK_WINDOW (window),
                                              GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                              GTK_MESSAGE_ERROR,
                                              GTK_BUTTONS_CLOSE,
                                              "Unable to access timeline. Please open a project first.");
    gtk_dialog_run (GTK_DIALOG (dialog));
    gtk_widget_destroy (dialog);
  }
}

BlouEditWindow *
blouedit_window_new (BlouEditApplication *application)
{
  return g_object_new (BLOUEDIT_TYPE_WINDOW,
                       "application", application,
                       "title", "BLOUedit",
                       "default-width", 1280,
                       "default-height", 720,
                       NULL);
}

void
blouedit_window_open (BlouEditWindow *self,
                     GFile          *file)
{
  g_assert (BLOUEDIT_IS_WINDOW (self));
  g_assert (G_IS_FILE (file));

  // TODO: Implement project loading logic
  self->current_file = g_object_ref (file);
  
  // Update window title
  char *basename = g_file_get_basename (file);
  char *title = g_strdup_printf ("BLOUedit - %s", basename);
  gtk_window_set_title (GTK_WINDOW (self), title);
  g_free (basename);
  g_free (title);
} 