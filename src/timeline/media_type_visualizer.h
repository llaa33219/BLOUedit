#pragma once

#include <gtk/gtk.h>
#include "core/types.h"
#include "core/timeline.h"

G_BEGIN_DECLS

/* 미디어 시각화 모드 정의 */
typedef enum {
  BLOUEDIT_MEDIA_VISUAL_MODE_NONE,     /* 구분 없음 */
  BLOUEDIT_MEDIA_VISUAL_MODE_COLOR,    /* 색상으로 구분 */
  BLOUEDIT_MEDIA_VISUAL_MODE_ICON,     /* 아이콘으로 구분 */
  BLOUEDIT_MEDIA_VISUAL_MODE_BOTH      /* 색상과 아이콘 모두 사용 */
} BlouEditMediaVisualMode;

/* 미디어 시각화 모드 설정 함수 */
void blouedit_timeline_set_media_visual_mode(BlouEditTimeline *timeline, 
                                         BlouEditMediaVisualMode mode);

/* 미디어 시각화 모드 가져오기 함수 */
BlouEditMediaVisualMode blouedit_timeline_get_media_visual_mode(BlouEditTimeline *timeline);

/* 미디어 유형별 색상 설정 함수 */
void blouedit_timeline_set_media_type_color(BlouEditTimeline *timeline, 
                                        BlouEditMediaFilterType type, 
                                        const GdkRGBA *color);

/* 미디어 유형별 색상 가져오기 함수 */
void blouedit_timeline_get_media_type_color(BlouEditTimeline *timeline, 
                                        BlouEditMediaFilterType type, 
                                        GdkRGBA *color);

/* 클립의 미디어 유형 식별 함수 */
BlouEditMediaFilterType blouedit_timeline_get_clip_media_type(GESClip *clip);

/* 클립의 미디어 유형에 해당하는 색상 가져오기 함수 */
void blouedit_timeline_get_color_for_clip(BlouEditTimeline *timeline, 
                                      GESClip *clip, 
                                      GdkRGBA *color);

/* 미디어 유형별 아이콘 이름 가져오기 함수 */
const gchar* blouedit_timeline_get_icon_for_media_type(BlouEditMediaFilterType type);

/* 미디어 시각화 설정 대화상자 표시 함수 */
void blouedit_timeline_show_media_visual_settings(BlouEditTimeline *timeline);

/* 타임라인에 기본 미디어 유형 색상 초기화 함수 */
void blouedit_timeline_initialize_media_type_colors(BlouEditTimeline *timeline);

G_END_DECLS 