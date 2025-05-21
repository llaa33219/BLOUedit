#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 마커 타입별 기본 색상 반환 함수 */
void blouedit_marker_get_default_color_for_type(BlouEditMarkerType type, GdkRGBA *color);

/* 마커 타입에 따라 자동으로 색상 설정하는 함수 */
void blouedit_timeline_set_marker_color_by_type(BlouEditTimeline *timeline, 
                                              BlouEditTimelineMarker *marker);

/* 타임라인의 모든 마커를 타입에 따라 색상 자동 설정 */
void blouedit_timeline_recolor_all_markers_by_type(BlouEditTimeline *timeline);

/* 사용자 지정 색상 선택 대화상자 표시 */
void blouedit_timeline_show_marker_color_dialog(BlouEditTimeline *timeline, 
                                              BlouEditTimelineMarker *marker);

G_END_DECLS 