#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 트랙 태그 관련 함수 */
void                 blouedit_timeline_set_track_tags(BlouEditTimeline *timeline, BlouEditTimelineTrack *track, const gchar **tags);
const gchar**        blouedit_timeline_get_track_tags(BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
void                 blouedit_timeline_add_track_tag(BlouEditTimeline *timeline, BlouEditTimelineTrack *track, const gchar *tag);
void                 blouedit_timeline_remove_track_tag(BlouEditTimeline *timeline, BlouEditTimelineTrack *track, const gchar *tag);
gboolean             blouedit_timeline_track_has_tag(BlouEditTimelineTrack *track, const gchar *tag);
void                 blouedit_timeline_filter_tracks_by_tag(BlouEditTimeline *timeline, const gchar *tag);
void                 blouedit_timeline_clear_track_filter(BlouEditTimeline *timeline);
void                 blouedit_timeline_show_track_tags_dialog(BlouEditTimeline *timeline, BlouEditTimelineTrack *track);

/* 트랙 프리셋 관련 함수 */
void                 blouedit_timeline_save_track_preset(BlouEditTimeline *timeline, const gchar *preset_name);
void                 blouedit_timeline_apply_track_preset(BlouEditTimeline *timeline, const gchar *preset_name);
void                 blouedit_timeline_delete_track_preset(const gchar *preset_name);
GList*               blouedit_timeline_get_track_presets(void);
void                 blouedit_timeline_show_track_presets_dialog(BlouEditTimeline *timeline);

/* 미디어 소스별 트랙 자동 구성 함수 */
void                 blouedit_timeline_auto_organize_tracks_by_media(BlouEditTimeline *timeline);
void                 blouedit_timeline_set_media_track_organization_rules(BlouEditTimeline *timeline, const gchar *rules_json);
const gchar*         blouedit_timeline_get_media_track_organization_rules(BlouEditTimeline *timeline);
void                 blouedit_timeline_show_media_track_organization_dialog(BlouEditTimeline *timeline);

G_END_DECLS 