#pragma once

#include <gtk/gtk.h>
#include "timeline.h"
#include "clip_drawer.h"

G_BEGIN_DECLS

/**
 * 무제한 타임라인 트랙 기능
 * 이 파일은 타임라인 트랙의 무제한 지원을 위한 함수들을 정의합니다.
 */

/* 트랙 그리기 함수 */
void blouedit_timeline_draw_tracks (BlouEditTimeline *timeline, cairo_t *cr, int width, int height);

/* 스크롤 뷰 생성 함수 */
GtkWidget *blouedit_timeline_create_scrolled_view (BlouEditTimeline *timeline);

G_END_DECLS 