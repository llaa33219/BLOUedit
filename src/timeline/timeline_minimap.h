#pragma once

#include <gtk/gtk.h>
#include "timeline.h"

G_BEGIN_DECLS

/* Timeline minimap data structure */
typedef struct _BlouEditTimelineMinimap BlouEditTimelineMinimap;

/* Timeline minimap creation */
GtkWidget *blouedit_timeline_minimap_new (BlouEditTimeline *timeline);

/* Update minimap when timeline changes */
void blouedit_timeline_update_minimap (BlouEditTimeline *timeline);

/* Minimap visibility control */
void blouedit_timeline_show_minimap (BlouEditTimeline *timeline, gboolean show);
gboolean blouedit_timeline_get_minimap_visible (BlouEditTimeline *timeline);
gboolean blouedit_timeline_toggle_minimap (BlouEditTimeline *timeline);

/* Minimap size control */
void blouedit_timeline_set_minimap_height (BlouEditTimeline *timeline, gint height);
gint blouedit_timeline_get_minimap_height (BlouEditTimeline *timeline);

G_END_DECLS 