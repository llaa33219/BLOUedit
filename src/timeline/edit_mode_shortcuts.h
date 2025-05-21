#pragma once

#include <gtk/gtk.h>
#include "core/types.h"
#include "core/timeline.h"

G_BEGIN_DECLS

/* 에딧 모드 단축키 관리 초기화 */
void blouedit_timeline_init_edit_mode_shortcuts(BlouEditTimeline *timeline);

/* 에딧 모드 단축키 핸들러 */
gboolean blouedit_timeline_handle_edit_mode_shortcut(BlouEditTimeline *timeline, GdkEventKey *event);

/* 에딧 모드 단축키 대화상자 표시 함수 */
void blouedit_timeline_show_edit_mode_shortcuts_dialog(BlouEditTimeline *timeline);

/* 현재 에딧 모드 이름 가져오기 */
const gchar* blouedit_timeline_get_edit_mode_name(BlouEditEditMode mode);

/* 에딧 모드 상태 표시 오버레이 그리기 함수 */
void blouedit_timeline_draw_edit_mode_overlay(BlouEditTimeline *timeline, cairo_t *cr, int width, int height);

G_END_DECLS 