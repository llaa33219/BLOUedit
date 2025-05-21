#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "timeline_hand_tool.h"

/* 핸드 툴 커서 */
#define HAND_CURSOR_OPEN  GDK_HAND1
#define HAND_CURSOR_CLOSED GDK_FLEUR

/* 키보드 단축키 정의 */
#define HAND_TOOL_TEMP_KEY     GDK_KEY_space   /* 스페이스바 누르는 동안 손 도구 사용 */
#define HAND_TOOL_TOGGLE_KEY   GDK_KEY_h       /* 'H' 키로 손 도구 토글 */

/* 타임라인 핸드 툴 상태 구조체 */
struct _BlouEditHandToolState {
  BlouEditHandToolMode mode;         /* 현재 모드 */
  gboolean is_panning;               /* 현재 패닝 중인지 여부 */
  gdouble start_x;                   /* 패닝 시작 X 좌표 */
  gdouble start_y;                   /* 패닝 시작 Y 좌표 */
  gdouble last_x;                    /* 마지막 X 좌표 */
  gdouble last_y;                    /* 마지막 Y 좌표 */
  GdkCursor *open_hand_cursor;       /* 열린 손 커서 */
  GdkCursor *closed_hand_cursor;     /* 닫힌 손 커서 */
  gint original_scroll_position;     /* 패닝 시작 시 스크롤 위치 */
  guint scroll_timeout_id;           /* 스크롤 애니메이션 타임아웃 ID */
  gdouble momentum_x;                /* 패닝 모멘텀 X */
  gdouble previous_delta_x;          /* 이전 X 변화량 */
  gint64 scroll_time;                /* 마지막 스크롤 시간 */
  gboolean momentum_enabled;         /* 모멘텀 활성화 여부 */
};

/* 타임라인 객체에서 핸드 툴 상태 가져오기 */
static const gchar *HAND_TOOL_KEY = "blouedit-timeline-hand-tool";

/* 핸드 툴 상태 가져오기 함수 */
static BlouEditHandToolState*
get_hand_tool_state(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  return g_object_get_data(G_OBJECT(timeline), HAND_TOOL_KEY);
}

/* 커서 설정 함수 */
static void
set_cursor_for_mode(BlouEditTimeline *timeline, BlouEditHandToolState *state)
{
  GdkWindow *window = gtk_widget_get_window(GTK_WIDGET(timeline));
  if (!window)
    return;
    
  if (state->mode == BLOUEDIT_HAND_TOOL_MODE_OFF) {
    /* 기본 커서로 복원 */
    gdk_window_set_cursor(window, NULL);
  } else {
    /* 현재 패닝 중인지 여부에 따라 적절한 커서 설정 */
    if (state->is_panning)
      gdk_window_set_cursor(window, state->closed_hand_cursor);
    else
      gdk_window_set_cursor(window, state->open_hand_cursor);
  }
}

/* 핸드 툴 모드 설정 함수 */
void 
blouedit_timeline_set_hand_tool_mode(BlouEditTimeline *timeline, BlouEditHandToolMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  if (!state)
    return;
    
  if (state->mode != mode) {
    state->mode = mode;
    
    /* 커서 업데이트 */
    set_cursor_for_mode(timeline, state);
    
    /* 위젯 다시 그리기 (도구 상태 표시용) */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
}

/* 핸드 툴 모드 가져오기 함수 */
BlouEditHandToolMode 
blouedit_timeline_get_hand_tool_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_HAND_TOOL_MODE_OFF);
  
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  if (!state)
    return BLOUEDIT_HAND_TOOL_MODE_OFF;
    
  return state->mode;
}

/* 타임라인 특정 위치로 이동 */
void 
blouedit_timeline_pan_to_position(BlouEditTimeline *timeline, gint64 position)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 타임라인의 해당 위치가 화면에 보이도록 스크롤 조정 */
  gdouble zoom_level = blouedit_timeline_get_zoom_level(timeline);
  gint timeline_start_x = 80; /* 트랙 레이블 영역 너비 */
  
  /* 시간선 위치를 픽셀 위치로 변환 */
  gint px_position = timeline_start_x + (position * zoom_level) / GST_SECOND;
  
  /* 타임라인 스크롤 조정 
   * 실제 구현에서는 타임라인 스크롤 조정 함수를 호출해야 합니다.
   */
  GtkAdjustment *hadj = gtk_scrollable_get_hadjustment(GTK_SCROLLABLE(timeline));
  if (hadj) {
    gtk_adjustment_set_value(hadj, px_position - gtk_widget_get_allocated_width(GTK_WIDGET(timeline)) / 2);
  }
}

/* 타임라인 픽셀 단위로 이동 */
void 
blouedit_timeline_pan_by_pixels(BlouEditTimeline *timeline, gint delta_pixels)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 현재 스크롤 위치에서 지정된 픽셀만큼 이동 */
  GtkAdjustment *hadj = gtk_scrollable_get_hadjustment(GTK_SCROLLABLE(timeline));
  if (hadj) {
    gdouble current = gtk_adjustment_get_value(hadj);
    gtk_adjustment_set_value(hadj, current + delta_pixels);
  }
}

/* 패닝 시작 함수 */
static void
start_panning(BlouEditTimeline *timeline, BlouEditHandToolState *state, gdouble x, gdouble y)
{
  state->is_panning = TRUE;
  state->start_x = x;
  state->start_y = y;
  state->last_x = x;
  state->last_y = y;
  state->previous_delta_x = 0;
  
  /* 현재 스크롤 위치 저장 */
  GtkAdjustment *hadj = gtk_scrollable_get_hadjustment(GTK_SCROLLABLE(timeline));
  if (hadj) {
    state->original_scroll_position = (gint)gtk_adjustment_get_value(hadj);
  } else {
    state->original_scroll_position = 0;
  }
  
  /* 현재 시간 기록 */
  state->scroll_time = g_get_monotonic_time();
  
  /* 커서 업데이트 */
  set_cursor_for_mode(timeline, state);
}

/* 패닝 종료 함수 */
static void
stop_panning(BlouEditTimeline *timeline, BlouEditHandToolState *state, gdouble x, gdouble y)
{
  if (!state->is_panning)
    return;
    
  state->is_panning = FALSE;
  
  /* 모멘텀 계산 */
  if (state->momentum_enabled) {
    gint64 current_time = g_get_monotonic_time();
    gint64 time_delta = current_time - state->scroll_time;
    
    /* 최근 움직임 속도를 기준으로 모멘텀 설정 */
    if (time_delta > 0 && time_delta < 100000) { /* 100ms 이내의 움직임만 고려 */
      state->momentum_x = state->previous_delta_x * 0.8; /* 감쇠 계수 */
      
      /* 모멘텀이 너무 작으면 무시 */
      if (fabs(state->momentum_x) < 0.5)
        state->momentum_x = 0;
    }
  }
  
  /* 커서 업데이트 */
  set_cursor_for_mode(timeline, state);
}

/* 패닝 업데이트 함수 */
static void
update_panning(BlouEditTimeline *timeline, BlouEditHandToolState *state, gdouble x, gdouble y)
{
  if (!state->is_panning)
    return;
    
  /* X 방향 변화량 계산 */
  gdouble delta_x = state->last_x - x;
  
  /* 스크롤 적용 */
  blouedit_timeline_pan_by_pixels(timeline, (gint)delta_x);
  
  /* 현재 위치 저장 */
  state->last_x = x;
  state->last_y = y;
  
  /* 모멘텀 계산을 위한 정보 업데이트 */
  state->previous_delta_x = delta_x;
  state->scroll_time = g_get_monotonic_time();
}

/* 모멘텀 스크롤 타임아웃 콜백 */
static gboolean
momentum_scroll_timeout(gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  
  if (!state || !state->momentum_x)
    return G_SOURCE_REMOVE;
    
  /* 감쇠된 모멘텀으로 스크롤 */
  blouedit_timeline_pan_by_pixels(timeline, (gint)state->momentum_x);
  
  /* 모멘텀 감쇠 */
  state->momentum_x *= 0.95;
  
  /* 모멘텀이 충분히 작아지면 중지 */
  if (fabs(state->momentum_x) < 0.5) {
    state->momentum_x = 0;
    state->scroll_timeout_id = 0;
    return G_SOURCE_REMOVE;
  }
  
  return G_SOURCE_CONTINUE;
}

/* 핸드 툴 키 누름 처리 함수 */
gboolean 
blouedit_timeline_handle_hand_tool_key_press(BlouEditTimeline *timeline, GdkEventKey *event)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(event != NULL, FALSE);
  
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  if (!state)
    return FALSE;
  
  /* Shift+Space 키 누름 - 손 도구 일시 활성화 */
  if (event->keyval == HAND_TOOL_TEMP_KEY &&
      (event->state & GDK_SHIFT_MASK) == 0) {
    if (state->mode == BLOUEDIT_HAND_TOOL_MODE_OFF) {
      blouedit_timeline_set_hand_tool_mode(timeline, BLOUEDIT_HAND_TOOL_MODE_TEMPORARY);
      return TRUE;
    }
  }
  
  /* 'H' 키 누름 - 손 도구 토글 */
  if (event->keyval == HAND_TOOL_TOGGLE_KEY &&
      (event->state & (GDK_CONTROL_MASK | GDK_SHIFT_MASK)) == 0) {
    if (state->mode == BLOUEDIT_HAND_TOOL_MODE_OFF) {
      blouedit_timeline_set_hand_tool_mode(timeline, BLOUEDIT_HAND_TOOL_MODE_LOCKED);
      return TRUE;
    } else if (state->mode == BLOUEDIT_HAND_TOOL_MODE_LOCKED) {
      blouedit_timeline_set_hand_tool_mode(timeline, BLOUEDIT_HAND_TOOL_MODE_OFF);
      return TRUE;
    }
  }
  
  return FALSE;
}

/* 핸드 툴 키 떼기 처리 함수 */
gboolean 
blouedit_timeline_handle_hand_tool_key_release(BlouEditTimeline *timeline, GdkEventKey *event)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(event != NULL, FALSE);
  
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  if (!state)
    return FALSE;
  
  /* Space 키 뗌 - 손 도구 일시 활성화 해제 */
  if (event->keyval == HAND_TOOL_TEMP_KEY) {
    if (state->mode == BLOUEDIT_HAND_TOOL_MODE_TEMPORARY) {
      /* 패닝 중이었다면 종료 */
      if (state->is_panning) {
        stop_panning(timeline, state, state->last_x, state->last_y);
      }
      
      blouedit_timeline_set_hand_tool_mode(timeline, BLOUEDIT_HAND_TOOL_MODE_OFF);
      return TRUE;
    }
  }
  
  return FALSE;
}

/* 핸드 툴 마우스 버튼 누름 처리 함수 */
gboolean 
blouedit_timeline_handle_hand_tool_button_press(BlouEditTimeline *timeline, GdkEventButton *event)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(event != NULL, FALSE);
  
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  if (!state)
    return FALSE;
  
  /* 임시 모드 또는 잠금 모드일 때만 처리 */
  if (state->mode == BLOUEDIT_HAND_TOOL_MODE_OFF)
    return FALSE;
  
  /* 마우스 왼쪽 버튼일 때만 처리 */
  if (event->button == 1) {
    /* 패닝 시작 */
    start_panning(timeline, state, event->x, event->y);
    return TRUE;
  } else if (event->button == 2) {
    /* 중간 마우스 버튼도 패닝으로 처리 */
    start_panning(timeline, state, event->x, event->y);
    return TRUE;
  }
  
  return FALSE;
}

/* 핸드 툴 마우스 버튼 뗌 처리 함수 */
gboolean 
blouedit_timeline_handle_hand_tool_button_release(BlouEditTimeline *timeline, GdkEventButton *event)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(event != NULL, FALSE);
  
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  if (!state)
    return FALSE;
  
  /* 패닝 중이 아니면 처리하지 않음 */
  if (!state->is_panning)
    return FALSE;
  
  /* 패닝 종료 */
  if (event->button == 1 || event->button == 2) {
    stop_panning(timeline, state, event->x, event->y);
    
    /* 모멘텀 스크롤 시작 */
    if (state->momentum_enabled && fabs(state->momentum_x) > 0.5) {
      if (state->scroll_timeout_id > 0) {
        g_source_remove(state->scroll_timeout_id);
      }
      state->scroll_timeout_id = g_timeout_add(16, momentum_scroll_timeout, timeline);
    }
    
    return TRUE;
  }
  
  return FALSE;
}

/* 핸드 툴 마우스 이동 처리 함수 */
gboolean 
blouedit_timeline_handle_hand_tool_motion(BlouEditTimeline *timeline, GdkEventMotion *event)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(event != NULL, FALSE);
  
  BlouEditHandToolState *state = get_hand_tool_state(timeline);
  if (!state)
    return FALSE;
  
  /* 패닝 중이 아니면 처리하지 않음 */
  if (!state->is_panning)
    return FALSE;
  
  /* 패닝 업데이트 */
  update_panning(timeline, state, event->x, event->y);
  return TRUE;
}

/* 핸드 툴 초기화 함수 */
BlouEditHandToolState* 
blouedit_timeline_init_hand_tool(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  /* 이미 초기화되었는지 확인 */
  BlouEditHandToolState *state = g_object_get_data(G_OBJECT(timeline), HAND_TOOL_KEY);
  if (state)
    return state;
  
  /* 상태 구조체 생성 및 초기화 */
  state = g_new0(BlouEditHandToolState, 1);
  state->mode = BLOUEDIT_HAND_TOOL_MODE_OFF;
  state->is_panning = FALSE;
  state->momentum_enabled = TRUE;
  
  /* 커서 생성 */
  GdkDisplay *display = gtk_widget_get_display(GTK_WIDGET(timeline));
  state->open_hand_cursor = gdk_cursor_new_for_display(display, HAND_CURSOR_OPEN);
  state->closed_hand_cursor = gdk_cursor_new_for_display(display, HAND_CURSOR_CLOSED);
  
  /* 타임라인 객체에 상태 저장 */
  g_object_set_data_full(G_OBJECT(timeline), HAND_TOOL_KEY, state, 
                        (GDestroyNotify)blouedit_timeline_cleanup_hand_tool);
  
  return state;
}

/* 핸드 툴 정리 함수 */
void 
blouedit_timeline_cleanup_hand_tool(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  BlouEditHandToolState *state = g_object_get_data(G_OBJECT(timeline), HAND_TOOL_KEY);
  if (!state)
    return;
  
  /* 타임아웃 정리 */
  if (state->scroll_timeout_id > 0) {
    g_source_remove(state->scroll_timeout_id);
    state->scroll_timeout_id = 0;
  }
  
  /* 커서 해제 */
  if (state->open_hand_cursor) {
    g_object_unref(state->open_hand_cursor);
    state->open_hand_cursor = NULL;
  }
  
  if (state->closed_hand_cursor) {
    g_object_unref(state->closed_hand_cursor);
    state->closed_hand_cursor = NULL;
  }
  
  /* 메모리 해제는 g_object_set_data_full()에 등록된 destroy 함수에서 수행됨 */
} 