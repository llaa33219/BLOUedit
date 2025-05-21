#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 미디어 유형별 시각적 구분 모드 */
typedef enum {
  BLOUEDIT_MEDIA_VISUAL_MODE_NONE,     /* 구분 없음 */
  BLOUEDIT_MEDIA_VISUAL_MODE_COLOR,    /* 색상으로 구분 */
  BLOUEDIT_MEDIA_VISUAL_MODE_ICON,     /* 아이콘으로 구분 */
  BLOUEDIT_MEDIA_VISUAL_MODE_BOTH      /* 색상과 아이콘 모두 사용 */
} BlouEditMediaVisualMode;

/* 미디어 유형별 시각적 구분 모드 설정 함수 */
void blouedit_timeline_set_media_visual_mode(BlouEditTimeline *timeline, BlouEditMediaVisualMode mode);

/* 미디어 유형별 시각적 구분 모드 가져오기 함수 */
BlouEditMediaVisualMode blouedit_timeline_get_media_visual_mode(BlouEditTimeline *timeline);

/* 특정 미디어 유형의 색상 설정 함수 */
void blouedit_timeline_set_media_type_color(BlouEditTimeline *timeline, BlouEditMediaFilterType type, const GdkRGBA *color);

/* 특정 미디어 유형의 색상 가져오기 함수 */
void blouedit_timeline_get_media_type_color(BlouEditTimeline *timeline, BlouEditMediaFilterType type, GdkRGBA *color);

/* 미디어 유형 시각화 설정 대화상자 표시 함수 */
void blouedit_timeline_show_media_visual_settings(BlouEditTimeline *timeline);

/* 클립에서 미디어 유형 가져오기 함수 */
BlouEditMediaFilterType blouedit_timeline_get_clip_media_type(GESClip *clip);

/* 미디어 유형에 따라 클립 색상 가져오기 */
void blouedit_timeline_get_color_for_clip(BlouEditTimeline *timeline, GESClip *clip, GdkRGBA *color);

/* 미디어 유형에 따라 아이콘 이름 가져오기 */
const gchar* blouedit_timeline_get_icon_for_media_type(BlouEditMediaFilterType type);

G_END_DECLS 