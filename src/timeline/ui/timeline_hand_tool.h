#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 타임라인 이동 도구 모드 */
typedef enum {
  BLOUEDIT_HAND_TOOL_MODE_OFF,     /* 도구 사용 안함 */
  BLOUEDIT_HAND_TOOL_MODE_TEMPORARY,  /* 임시 사용 (키 누르는 동안만) */
  BLOUEDIT_HAND_TOOL_MODE_LOCKED       /* 잠금 모드 (토글) */
} BlouEditHandToolMode;

/* 타임라인 이동 도구 설정 함수 */
void blouedit_timeline_set_hand_tool_mode(BlouEditTimeline *timeline, BlouEditHandToolMode mode);
BlouEditHandToolMode blouedit_timeline_get_hand_tool_mode(BlouEditTimeline *timeline);

/* 타임라인 이동 도구 키 처리 함수 */
gboolean blouedit_timeline_handle_hand_tool_key_press(BlouEditTimeline *timeline, GdkEventKey *event);
gboolean blouedit_timeline_handle_hand_tool_key_release(BlouEditTimeline *timeline, GdkEventKey *event);

/* 타임라인 이동 도구 마우스 처리 함수 */
gboolean blouedit_timeline_handle_hand_tool_button_press(BlouEditTimeline *timeline, GdkEventButton *event);
gboolean blouedit_timeline_handle_hand_tool_button_release(BlouEditTimeline *timeline, GdkEventButton *event);
gboolean blouedit_timeline_handle_hand_tool_motion(BlouEditTimeline *timeline, GdkEventMotion *event);

/* 타임라인 이동 도구 스크롤 변경 */
void blouedit_timeline_pan_to_position(BlouEditTimeline *timeline, gint64 position);
void blouedit_timeline_pan_by_pixels(BlouEditTimeline *timeline, gint delta_pixels);

/* 타임라인 핸드 툴 상태 구조체 */
typedef struct _BlouEditHandToolState BlouEditHandToolState;

/* 타임라인 핸드 툴 초기화 함수 */
BlouEditHandToolState* blouedit_timeline_init_hand_tool(BlouEditTimeline *timeline);

/* 타임라인 핸드 툴 정리 함수 */
void blouedit_timeline_cleanup_hand_tool(BlouEditTimeline *timeline);

G_END_DECLS 