#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 트랙 관련 함수 */
BlouEditTimelineTrack* blouedit_timeline_add_track (BlouEditTimeline *timeline, GESTrackType track_type, const gchar *name);
void                   blouedit_timeline_remove_track (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
gint                   blouedit_timeline_get_track_count (BlouEditTimeline *timeline, GESTrackType track_type);
BlouEditTimelineTrack* blouedit_timeline_get_track_by_index (BlouEditTimeline *timeline, GESTrackType track_type, gint index);
void                   blouedit_timeline_create_default_tracks (BlouEditTimeline *timeline);
gboolean               blouedit_timeline_is_track_at_max (BlouEditTimeline *timeline, GESTrackType track_type);
gint                   blouedit_timeline_get_track_layer_for_clip (BlouEditTimeline *timeline, GESClip *clip, GESTrackType track_type);
gint                   blouedit_timeline_get_max_tracks (BlouEditTimeline *timeline, GESTrackType track_type);
void                   blouedit_timeline_set_track_height (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint height);

/* 트랙 UI 관련 함수 */
void                   blouedit_timeline_show_track_controls (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gdouble x, gdouble y);
void                   blouedit_timeline_show_track_properties (BlouEditTimeline *timeline);
void                   blouedit_timeline_show_add_track_dialog (BlouEditTimeline *timeline);
void                   blouedit_timeline_show_message (BlouEditTimeline *timeline, const gchar *message);

/* 트랙 재정렬 관련 함수 */
void                   blouedit_timeline_start_track_reorder (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint y);
void                   blouedit_timeline_reorder_track_to (BlouEditTimeline *timeline, gint y);
void                   blouedit_timeline_end_track_reorder (BlouEditTimeline *timeline);
void                   blouedit_timeline_move_track_up (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
void                   blouedit_timeline_move_track_down (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);

G_END_DECLS 