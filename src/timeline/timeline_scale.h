#pragma once

#include <gtk/gtk.h>
#include "core/types.h"
#include "core/timeline.h"

G_BEGIN_DECLS

/* 타임라인 눈금 모드 정의 */
typedef enum {
  BLOUEDIT_TIMELINE_SCALE_SECONDS,   /* 초 단위 눈금 */
  BLOUEDIT_TIMELINE_SCALE_MINUTES,   /* 분 단위 눈금 */
  BLOUEDIT_TIMELINE_SCALE_HOURS,     /* 시간 단위 눈금 */
  BLOUEDIT_TIMELINE_SCALE_FRAMES,    /* 프레임 단위 눈금 */
  BLOUEDIT_TIMELINE_SCALE_CUSTOM     /* 사용자 지정 눈금 */
} BlouEditTimelineScaleMode;

/* 타임라인 눈금 모드 설정 함수 */
void blouedit_timeline_set_scale_mode(BlouEditTimeline *timeline, 
                                     BlouEditTimelineScaleMode mode);

/* 타임라인 눈금 모드 가져오기 함수 */
BlouEditTimelineScaleMode blouedit_timeline_get_scale_mode(BlouEditTimeline *timeline);

/* 타임라인 사용자 지정 눈금 간격 설정 함수 (단위: 밀리초) */
void blouedit_timeline_set_custom_scale_interval(BlouEditTimeline *timeline, 
                                               guint64 interval_ms);

/* 타임라인 사용자 지정 눈금 간격 가져오기 함수 */
guint64 blouedit_timeline_get_custom_scale_interval(BlouEditTimeline *timeline);

/* 타임라인 눈금 표시 함수 - 내부 사용 */
void blouedit_timeline_draw_scale(BlouEditTimeline *timeline, 
                                cairo_t *cr, 
                                gint width, 
                                gint ruler_height);

/* 타임라인 눈금 설정 대화상자 표시 함수 */
void blouedit_timeline_show_scale_settings_dialog(BlouEditTimeline *timeline);

G_END_DECLS 