#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 애니메이션 속성 관련 함수 */
BlouEditAnimatableProperty* blouedit_timeline_register_property (BlouEditTimeline *timeline, GObject *object,
                                                         const gchar *name, const gchar *display_name,
                                                         const gchar *property_name, BlouEditPropertyType type,
                                                         gdouble min_value, gdouble max_value, gdouble default_value);
void blouedit_timeline_unregister_property (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property);
BlouEditAnimatableProperty* blouedit_timeline_get_property_by_id (BlouEditTimeline *timeline, guint id);

/* 키프레임 관련 함수 */
BlouEditKeyframe* blouedit_timeline_add_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property,
                                         gint64 position, gdouble value,
                                         BlouEditKeyframeInterpolation interpolation);
void blouedit_timeline_remove_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, BlouEditKeyframe *keyframe);
void blouedit_timeline_remove_keyframe_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position, gint64 tolerance);
void blouedit_timeline_remove_all_keyframes (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property);
void blouedit_timeline_update_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property,
                                BlouEditKeyframe *keyframe, gint64 position, gdouble value,
                                BlouEditKeyframeInterpolation interpolation);
void blouedit_timeline_update_keyframe_handles (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property,
                                       BlouEditKeyframe *keyframe,
                                       gdouble handle_left_x, gdouble handle_left_y,
                                       gdouble handle_right_x, gdouble handle_right_y);
BlouEditKeyframe* blouedit_timeline_get_keyframe_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position, gint64 tolerance);
GSList* blouedit_timeline_get_keyframes_in_range (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 start, gint64 end);
gdouble blouedit_timeline_evaluate_property_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position);
gboolean blouedit_timeline_apply_keyframes (BlouEditTimeline *timeline);

/* 키프레임 편집기 UI 관련 함수 */
void blouedit_timeline_show_keyframe_editor (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, BlouEditKeyframe *keyframe);

G_END_DECLS 