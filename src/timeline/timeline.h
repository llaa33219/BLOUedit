#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>

G_BEGIN_DECLS

#define BLOUEDIT_TYPE_TIMELINE (blouedit_timeline_get_type())

G_DECLARE_FINAL_TYPE (BlouEditTimeline, blouedit_timeline, BLOUEDIT, TIMELINE, GtkWidget)

BlouEditTimeline *blouedit_timeline_new (void);

void blouedit_timeline_add_file (BlouEditTimeline *timeline, GFile *file);
void blouedit_timeline_add_clip (BlouEditTimeline *timeline, const char *uri, GESTrack *track, gint64 start_time, gint64 duration);
void blouedit_timeline_split_clip (BlouEditTimeline *timeline, gint64 position);
void blouedit_timeline_delete_clip (BlouEditTimeline *timeline, GESClip *clip);
void blouedit_timeline_add_transition (BlouEditTimeline *timeline, gint64 position, const char *transition_type);

/* Timeline control */
void blouedit_timeline_set_position (BlouEditTimeline *timeline, gint64 position);
gint64 blouedit_timeline_get_position (BlouEditTimeline *timeline);
gint64 blouedit_timeline_get_duration (BlouEditTimeline *timeline);

/* Connect to player */
GstElement *blouedit_timeline_get_pipeline (BlouEditTimeline *timeline);

/* Save/load project */
gboolean blouedit_timeline_save_to_file (BlouEditTimeline *timeline, GFile *file);
gboolean blouedit_timeline_load_from_file (BlouEditTimeline *timeline, GFile *file);

G_END_DECLS 