#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 마커 상세 메모 설정 함수 */
void blouedit_timeline_set_marker_detailed_memo (BlouEditTimeline *timeline, 
                                              BlouEditTimelineMarker *marker, 
                                              const gchar *detailed_memo);

/* 마커 상세 메모 가져오기 함수 */
const gchar* blouedit_timeline_get_marker_detailed_memo (BlouEditTimelineMarker *marker);

/* 마커 메모 에디터 표시 함수 */
void blouedit_timeline_show_marker_memo_editor (BlouEditTimeline *timeline, 
                                             BlouEditTimelineMarker *marker);

/* 마커 편집기에 메모 버튼 추가하는 함수 */
void blouedit_timeline_add_memo_button_to_marker_editor (GtkWidget *editor, 
                                                      BlouEditTimeline *timeline, 
                                                      BlouEditTimelineMarker *marker);

G_END_DECLS 