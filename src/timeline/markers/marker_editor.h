#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 마커 편집기 표시 함수 */
void blouedit_timeline_show_marker_editor(BlouEditTimeline *timeline, BlouEditTimelineMarker *marker);

G_END_DECLS 