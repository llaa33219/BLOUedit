#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 타임라인 눈금 모드 */
typedef enum {
  BLOUEDIT_SCALE_MODE_SECONDS,     /* 초 단위 눈금 */
  BLOUEDIT_SCALE_MODE_MINUTES,     /* 분 단위 눈금 */
  BLOUEDIT_SCALE_MODE_HOURS,       /* 시간 단위 눈금 */
  BLOUEDIT_SCALE_MODE_FRAMES,      /* 프레임 단위 눈금 */
  BLOUEDIT_SCALE_MODE_CUSTOM       /* 사용자 지정 간격 */
} BlouEditTimelineScaleMode;

/* 타임라인 눈금 모드 설정 함수 */
void blouedit_timeline_set_scale_mode(BlouEditTimeline *timeline, BlouEditTimelineScaleMode mode);

/* 타임라인 눈금 모드 가져오기 함수 */
BlouEditTimelineScaleMode blouedit_timeline_get_scale_mode(BlouEditTimeline *timeline);

/* 타임라인 사용자 지정 눈금 간격 설정 함수 (단위: 타임라인 단위) */
void blouedit_timeline_set_custom_scale_interval(BlouEditTimeline *timeline, gint64 interval);

/* 타임라인 사용자 지정 눈금 간격 가져오기 함수 */
gint64 blouedit_timeline_get_custom_scale_interval(BlouEditTimeline *timeline);

/* 타임라인 눈금 설정 대화상자 표시 함수 */
void blouedit_timeline_show_scale_settings(BlouEditTimeline *timeline);

G_END_DECLS 