#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 타임라인 시각화 모드 정의 */
typedef enum {
  BLOUEDIT_TIMELINE_VIEW_STANDARD,   /* 표준 타임라인 표시 */
  BLOUEDIT_TIMELINE_VIEW_COMPACT,    /* 압축된 타임라인 표시 */
  BLOUEDIT_TIMELINE_VIEW_ICONIC,     /* 아이코닉 모드 (썸네일 강조) */
  BLOUEDIT_TIMELINE_VIEW_WAVEFORM,   /* 파형 강조 모드 */
  BLOUEDIT_TIMELINE_VIEW_MINIMAL     /* 최소화 모드 */
} BlouEditTimelineViewMode;

/* 타임라인 미디어 시각화 플래그 정의 */
typedef enum {
  BLOUEDIT_TIMELINE_SHOW_NONE           = 0,         /* 기본 시각화만 표시 */
  BLOUEDIT_TIMELINE_SHOW_THUMBNAILS     = 1 << 0,    /* 비디오 클립 썸네일 표시 */
  BLOUEDIT_TIMELINE_SHOW_WAVEFORMS      = 1 << 1,    /* 오디오 클립 파형 표시 */
  BLOUEDIT_TIMELINE_SHOW_LABELS         = 1 << 2,    /* 클립 레이블 표시 */
  BLOUEDIT_TIMELINE_SHOW_EFFECTS        = 1 << 3,    /* 효과 아이콘 표시 */
  BLOUEDIT_TIMELINE_SHOW_KEYFRAMES      = 1 << 4,    /* 키프레임 표시 */
  BLOUEDIT_TIMELINE_SHOW_IN_OUT_POINTS  = 1 << 5,    /* 클립 시작/끝점 표시 */
  BLOUEDIT_TIMELINE_SHOW_DURATIONS      = 1 << 6,    /* 클립 지속 시간 표시 */
  BLOUEDIT_TIMELINE_SHOW_ALL            = 0xFF       /* 모든 시각화 요소 표시 */
} BlouEditTimelineVisualizationFlags;

/* 시각화 모드 설정 함수 */
void blouedit_timeline_set_view_mode(BlouEditTimeline *timeline, BlouEditTimelineViewMode mode);
BlouEditTimelineViewMode blouedit_timeline_get_view_mode(BlouEditTimeline *timeline);

/* 시각화 옵션 설정 함수 */
void blouedit_timeline_set_visualization_flags(BlouEditTimeline *timeline, BlouEditTimelineVisualizationFlags flags);
BlouEditTimelineVisualizationFlags blouedit_timeline_get_visualization_flags(BlouEditTimeline *timeline);

/* 썸네일 크기 설정 함수 */
void blouedit_timeline_set_thumbnail_size(BlouEditTimeline *timeline, gint width, gint height);
void blouedit_timeline_get_thumbnail_size(BlouEditTimeline *timeline, gint *width, gint *height);

/* 파형 표시 옵션 설정 함수 */
void blouedit_timeline_set_waveform_resolution(BlouEditTimeline *timeline, gint resolution);
gint blouedit_timeline_get_waveform_resolution(BlouEditTimeline *timeline);
void blouedit_timeline_set_waveform_color(BlouEditTimeline *timeline, const GdkRGBA *color);
void blouedit_timeline_get_waveform_color(BlouEditTimeline *timeline, GdkRGBA *color);

/* 시각화 모드 변경 대화상자 */
void blouedit_timeline_show_view_settings_dialog(BlouEditTimeline *timeline);

G_END_DECLS 