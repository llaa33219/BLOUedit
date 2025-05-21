#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 타임라인 그룹 관련 함수 */
BlouEditTimelineGroup* blouedit_timeline_group_new (void);
void blouedit_timeline_group_free (BlouEditTimelineGroup *group);
void blouedit_timeline_group_set_active (BlouEditTimelineGroup *group, BlouEditTimeline *timeline);
BlouEditTimeline* blouedit_timeline_group_get_active (BlouEditTimelineGroup *group);
void blouedit_timeline_group_add (BlouEditTimelineGroup *group, BlouEditTimeline *timeline, const gchar *name);
void blouedit_timeline_group_remove (BlouEditTimelineGroup *group, BlouEditTimeline *timeline);
const gchar* blouedit_timeline_group_get_name (BlouEditTimelineGroup *group, BlouEditTimeline *timeline);
void blouedit_timeline_group_set_name (BlouEditTimelineGroup *group, BlouEditTimeline *timeline, const gchar *name);
gboolean blouedit_timeline_group_copy_clip (BlouEditTimelineGroup *group, BlouEditTimeline *src, BlouEditTimeline *dest, GESClip *clip);
void blouedit_timeline_group_sync_position (BlouEditTimelineGroup *group, BlouEditTimeline *src);
GtkWidget* blouedit_timeline_group_create_switcher (BlouEditTimelineGroup *group);

G_END_DECLS 