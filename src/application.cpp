#include "application.h"
#include "window.h"

struct _BlouEditApplication
{
  AdwApplication parent_instance;
};

G_DEFINE_TYPE (BlouEditApplication, blouedit_application, ADW_TYPE_APPLICATION)

static void
blouedit_application_finalize (GObject *object)
{
  BlouEditApplication *self = BLOUEDIT_APPLICATION (object);

  G_OBJECT_CLASS (blouedit_application_parent_class)->finalize (object);
}

static void
blouedit_application_activate (GApplication *app)
{
  GtkWindow *window;

  g_assert (BLOUEDIT_IS_APPLICATION (app));

  window = gtk_application_get_active_window (GTK_APPLICATION (app));
  if (window == NULL)
    window = GTK_WINDOW (blouedit_window_new (BLOUEDIT_APPLICATION (app)));

  gtk_window_present (window);
}

static void
blouedit_application_open (GApplication  *app,
                          GFile        **files,
                          int            n_files,
                          const char    *hint)
{
  GtkWindow *window;
  
  g_assert (BLOUEDIT_IS_APPLICATION (app));

  window = gtk_application_get_active_window (GTK_APPLICATION (app));
  if (window == NULL)
    window = GTK_WINDOW (blouedit_window_new (BLOUEDIT_APPLICATION (app)));

  for (int i = 0; i < n_files; i++)
    blouedit_window_open (BLOUEDIT_WINDOW (window), files[i]);
    
  gtk_window_present (window);
}

static void
blouedit_application_class_init (BlouEditApplicationClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GApplicationClass *app_class = G_APPLICATION_CLASS (klass);

  object_class->finalize = blouedit_application_finalize;

  app_class->activate = blouedit_application_activate;
  app_class->open = blouedit_application_open;
}

static void
blouedit_application_about_action (GSimpleAction *action,
                                  GVariant      *parameter,
                                  gpointer       user_data)
{
  static const char *developers[] = {
    "BLOUedit Team",
    NULL
  };

  static const char *artists[] = {
    "BLOUedit Team",
    NULL
  };

  BlouEditApplication *self = BLOUEDIT_APPLICATION (user_data);
  GtkWindow *window = NULL;

  window = gtk_application_get_active_window (GTK_APPLICATION (self));

  adw_show_about_window (
    window,
    "application-name", "BLOUedit",
    "application-icon", "com.blouedit.BLOUedit",
    "developer-name", "BLOUedit Team",
    "version", "0.1.0",
    "developers", developers,
    "artists", artists,
    "copyright", "© 2023 BLOUedit Team",
    "license-type", GTK_LICENSE_GPL_3_0,
    "website", "https://blouedit.com",
    "issue-url", "https://github.com/blouedit/blouedit/issues",
    NULL
  );
}

static void
blouedit_application_quit_action (GSimpleAction *action,
                                 GVariant      *parameter,
                                 gpointer       user_data)
{
  BlouEditApplication *self = BLOUEDIT_APPLICATION (user_data);

  g_assert (BLOUEDIT_IS_APPLICATION (self));

  g_application_quit (G_APPLICATION (self));
}

static void
blouedit_application_preferences_action (GSimpleAction *action,
                                        GVariant      *parameter,
                                        gpointer       user_data)
{
  BlouEditApplication *self = BLOUEDIT_APPLICATION (user_data);
  GtkWindow *window = NULL;

  g_assert (BLOUEDIT_IS_APPLICATION (self));

  window = gtk_application_get_active_window (GTK_APPLICATION (self));
  // TODO: Implement preferences dialog
}

static const GActionEntry app_actions[] = {
  { "quit", blouedit_application_quit_action },
  { "about", blouedit_application_about_action },
  { "preferences", blouedit_application_preferences_action },
};

static void
blouedit_application_init (BlouEditApplication *self)
{
  g_action_map_add_action_entries (G_ACTION_MAP (self),
                                   app_actions,
                                   G_N_ELEMENTS (app_actions),
                                   self);
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "app.quit",
                                         (const char *[]) { "<primary>q", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "app.preferences",
                                         (const char *[]) { "<primary>comma", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.new-project",
                                         (const char *[]) { "<primary>n", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.open",
                                         (const char *[]) { "<primary>o", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.save",
                                         (const char *[]) { "<primary>s", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.save-as",
                                         (const char *[]) { "<primary><shift>s", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.undo",
                                         (const char *[]) { "<primary>z", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.redo",
                                         (const char *[]) { "<primary><shift>z", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.cut",
                                         (const char *[]) { "<primary>x", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.copy",
                                         (const char *[]) { "<primary>c", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.paste",
                                         (const char *[]) { "<primary>v", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.delete",
                                         (const char *[]) { "Delete", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.select-all",
                                         (const char *[]) { "<primary>a", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.split-clip",
                                         (const char *[]) { "s", NULL });
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "win.play-pause",
                                         (const char *[]) { "space", NULL });
}

BlouEditApplication *
blouedit_application_new (const char        *application_id,
                         GApplicationFlags  flags)
{
  g_return_val_if_fail (application_id != NULL, NULL);

  return BLOUEDIT_APPLICATION (g_object_new (BLOUEDIT_TYPE_APPLICATION,
                                           "application-id", application_id,
                                           "flags", flags,
                                           NULL));
} 