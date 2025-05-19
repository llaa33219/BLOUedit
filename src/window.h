#pragma once

#include <adwaita.h>
#include "application.h"

G_BEGIN_DECLS

#define BLOUEDIT_TYPE_WINDOW (blouedit_window_get_type())

G_DECLARE_FINAL_TYPE (BlouEditWindow, blouedit_window, BLOUEDIT, WINDOW, AdwApplicationWindow)

BlouEditWindow *blouedit_window_new (BlouEditApplication *application);
void            blouedit_window_open (BlouEditWindow *window, GFile *file);

G_END_DECLS 