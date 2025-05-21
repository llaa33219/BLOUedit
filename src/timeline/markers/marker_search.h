#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 마커 검색 대화상자 표시 함수 */
void blouedit_timeline_show_marker_search(BlouEditTimeline *timeline);

/* 마커 간 이동 함수 - 유형별 필터링 지원 */
void blouedit_timeline_goto_next_marker_by_type(BlouEditTimeline *timeline, BlouEditMarkerType type);
void blouedit_timeline_goto_prev_marker_by_type(BlouEditTimeline *timeline, BlouEditMarkerType type);

G_END_DECLS 