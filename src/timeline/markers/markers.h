#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 마커 관련 함수 */
BlouEditTimelineMarker* blouedit_timeline_add_marker (BlouEditTimeline *timeline, gint64 position, BlouEditMarkerType type, const gchar *name, const gchar *comment);
void blouedit_timeline_remove_marker (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker);
void blouedit_timeline_remove_marker_at_position (BlouEditTimeline *timeline, gint64 position, gint64 tolerance);
void blouedit_timeline_remove_all_markers (BlouEditTimeline *timeline);
void blouedit_timeline_update_marker (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker, gint64 position, BlouEditMarkerType type, const gchar *name, const gchar *comment);
BlouEditTimelineMarker* blouedit_timeline_get_marker_at_position (BlouEditTimeline *timeline, gint64 position, gint64 tolerance);
GSList* blouedit_timeline_get_markers_in_range (BlouEditTimeline *timeline, gint64 start, gint64 end);
void blouedit_timeline_set_marker_visibility (BlouEditTimeline *timeline, gboolean visible);
gboolean blouedit_timeline_get_marker_visibility (BlouEditTimeline *timeline);
void blouedit_timeline_set_marker_color (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker, const GdkRGBA *color);
void blouedit_timeline_goto_next_marker (BlouEditTimeline *timeline);
void blouedit_timeline_goto_previous_marker (BlouEditTimeline *timeline);
void blouedit_timeline_show_marker_editor (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker);
void blouedit_timeline_show_marker_list (BlouEditTimeline *timeline);
void blouedit_timeline_import_markers (BlouEditTimeline *timeline, const gchar *filename);
void blouedit_timeline_export_markers (BlouEditTimeline *timeline, const gchar *filename);

/* 마커 상세 메모 관련 함수 */
void blouedit_timeline_set_marker_detailed_memo (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker, const gchar *detailed_memo);
const gchar* blouedit_timeline_get_marker_detailed_memo (BlouEditTimelineMarker *marker);
void blouedit_timeline_show_marker_memo_editor (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker);

G_END_DECLS 