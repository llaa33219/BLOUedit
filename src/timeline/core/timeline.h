#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include "types.h"

G_BEGIN_DECLS

#define BLOUEDIT_TYPE_TIMELINE (blouedit_timeline_get_type())
G_DECLARE_FINAL_TYPE (BlouEditTimeline, blouedit_timeline, BLOUEDIT, TIMELINE, GtkWidget)

/* 타임라인 생성 함수 */
BlouEditTimeline* blouedit_timeline_new (void);

/* 줌 기능 함수 */
void          blouedit_timeline_zoom_in (BlouEditTimeline *timeline);
void          blouedit_timeline_zoom_out (BlouEditTimeline *timeline);
void          blouedit_timeline_set_zoom_level (BlouEditTimeline *timeline, gdouble zoom_level);
gdouble       blouedit_timeline_get_zoom_level (BlouEditTimeline *timeline);
void          blouedit_timeline_zoom_fit (BlouEditTimeline *timeline);

/* 위치 및 길이 관련 함수 */
void          blouedit_timeline_set_position (BlouEditTimeline *timeline, gint64 position);
gint64        blouedit_timeline_get_position (BlouEditTimeline *timeline);
gint64        blouedit_timeline_get_duration (BlouEditTimeline *timeline);

/* 스냅 관련 함수 */
void          blouedit_timeline_set_snap_mode (BlouEditTimeline *timeline, BlouEditSnapMode mode);
BlouEditSnapMode blouedit_timeline_get_snap_mode (BlouEditTimeline *timeline);
void          blouedit_timeline_set_snap_distance (BlouEditTimeline *timeline, guint distance);
guint         blouedit_timeline_get_snap_distance (BlouEditTimeline *timeline);
gboolean      blouedit_timeline_toggle_snap (BlouEditTimeline *timeline);
gint64        blouedit_timeline_snap_position (BlouEditTimeline *timeline, gint64 position);

/* 재생헤드 관련 함수 */
void          blouedit_timeline_set_playhead_position_from_x (BlouEditTimeline *timeline, double x);
double        blouedit_timeline_get_x_from_position (BlouEditTimeline *timeline, gint64 position);

/* 스크러빙 관련 함수 */
void          blouedit_timeline_set_scrub_mode (BlouEditTimeline *timeline, BlouEditScrubMode mode);
BlouEditScrubMode blouedit_timeline_get_scrub_mode (BlouEditTimeline *timeline);
void          blouedit_timeline_start_scrubbing (BlouEditTimeline *timeline, double x);
void          blouedit_timeline_scrub_to (BlouEditTimeline *timeline, double x);
void          blouedit_timeline_end_scrubbing (BlouEditTimeline *timeline);
void          blouedit_timeline_set_scrub_sensitivity (BlouEditTimeline *timeline, gdouble sensitivity);
gdouble       blouedit_timeline_get_scrub_sensitivity (BlouEditTimeline *timeline);

/* 이벤트 핸들러 함수 */
gboolean      blouedit_timeline_handle_button_press (BlouEditTimeline *timeline, GdkEventButton *event);
gboolean      blouedit_timeline_handle_motion (BlouEditTimeline *timeline, GdkEventMotion *event);
gboolean      blouedit_timeline_handle_button_release (BlouEditTimeline *timeline, GdkEventButton *event);

/* 타임코드 관련 함수 */
gchar*        blouedit_timeline_position_to_timecode (BlouEditTimeline *timeline, gint64 position, BlouEditTimecodeFormat format);
gint64        blouedit_timeline_timecode_to_position (BlouEditTimeline *timeline, const gchar *timecode, BlouEditTimecodeFormat format);
void          blouedit_timeline_goto_timecode (BlouEditTimeline *timeline, const gchar *timecode);
void          blouedit_timeline_show_timecode_dialog (BlouEditTimeline *timeline);

/* 오토스크롤 관련 함수 */
void          blouedit_timeline_set_autoscroll_mode (BlouEditTimeline *timeline, BlouEditAutoscrollMode mode);
BlouEditAutoscrollMode blouedit_timeline_get_autoscroll_mode (BlouEditTimeline *timeline);
void          blouedit_timeline_set_horizontal_scroll (BlouEditTimeline *timeline, gdouble scroll_pos);
gdouble       blouedit_timeline_get_horizontal_scroll (BlouEditTimeline *timeline);
void          blouedit_timeline_update_scroll_for_playhead (BlouEditTimeline *timeline);
void          blouedit_timeline_handle_autoscroll (BlouEditTimeline *timeline);

/* 히스토리 관련 함수 */
void          blouedit_timeline_record_action (BlouEditTimeline *timeline, BlouEditHistoryActionType type, 
                                GESTimelineElement *element, const gchar *description,
                                const GValue *before_value, const GValue *after_value);
void          blouedit_timeline_begin_group_action (BlouEditTimeline *timeline, const gchar *description);
void          blouedit_timeline_end_group_action (BlouEditTimeline *timeline);
gboolean      blouedit_timeline_undo (BlouEditTimeline *timeline);
gboolean      blouedit_timeline_redo (BlouEditTimeline *timeline);
void          blouedit_timeline_clear_history (BlouEditTimeline *timeline);
void          blouedit_timeline_set_max_history_size (BlouEditTimeline *timeline, gint max_size);
gint          blouedit_timeline_get_max_history_size (BlouEditTimeline *timeline);
gboolean      blouedit_timeline_can_undo (BlouEditTimeline *timeline);
gboolean      blouedit_timeline_can_redo (BlouEditTimeline *timeline);
GSList*       blouedit_timeline_get_history_actions (BlouEditTimeline *timeline, gint limit);
void          blouedit_timeline_show_history_dialog (BlouEditTimeline *timeline);

G_END_DECLS 