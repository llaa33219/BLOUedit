#pragma once

#include <adwaita.h>
#include <gtk/gtk.h>

G_BEGIN_DECLS

#define BLOUEDIT_TYPE_APPLICATION (blouedit_application_get_type())

G_DECLARE_FINAL_TYPE (BlouEditApplication, blouedit_application, BLOUEDIT, APPLICATION, AdwApplication)

BlouEditApplication *blouedit_application_new (const char *application_id,
                                             GApplicationFlags flags);

G_END_DECLS 