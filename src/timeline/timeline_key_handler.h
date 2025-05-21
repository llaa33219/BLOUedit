#pragma once

#include <gtk/gtk.h>
#include "core/timeline.h"

G_BEGIN_DECLS

/**
 * Connect key press event handler to the timeline.
 */
void blouedit_timeline_connect_key_events (BlouEditTimeline *timeline);

/**
 * Internal key press event callback handler.
 */
gboolean blouedit_timeline_key_press_event (GtkWidget *widget, GdkEventKey *event, gpointer user_data);

G_END_DECLS 