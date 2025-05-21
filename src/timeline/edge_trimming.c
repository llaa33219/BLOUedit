#include <gtk/gtk.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include <gst/editing-services/ges-clip.h>
#include "edge_trimming.h"
#include "core/types.h"
#include "core/timeline.h"

/* 타임라인 Edge Trimming 상태 */
static BlouEditEdgeTrimState trim_state = {
  .active = FALSE,
  .mode = BLOUEDIT_EDGE_TRIM_MODE_NORMAL,
  .clip = NULL,
  .edge = BLOUEDIT_EDGE_NONE,
  .original_position = 0,
  .current_position = 0,
  .start_x = 0,
  .snap_enabled = TRUE,
  .precision_level = 0,
  .adjacent_clip = NULL
};

/* Edge Trimming 모드 설정 */
void
blouedit_timeline_set_edge_trim_mode(BlouEditTimeline *timeline, BlouEditEdgeTrimMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  trim_state.mode = mode;
  
  /* 모드 변경 메시지 표시 */
  gchar *mode_name = NULL;
  switch (mode) {
    case BLOUEDIT_EDGE_TRIM_MODE_NORMAL:
      mode_name = "일반 에지 트리밍 모드";
      break;
    case BLOUEDIT_EDGE_TRIM_MODE_PRECISE:
      mode_name = "정밀 에지 트리밍 모드";
      break;
    case BLOUEDIT_EDGE_TRIM_MODE_RIPPLE:
      mode_name = "리플 에지 트리밍 모드";
      break;
    case BLOUEDIT_EDGE_TRIM_MODE_ROLL:
      mode_name = "롤 에지 트리밍 모드";
      break;
  }
  
  blouedit_timeline_show_message(timeline, mode_name);
}

/* Edge Trimming 모드 가져오기 */
BlouEditEdgeTrimMode
blouedit_timeline_get_edge_trim_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_EDGE_TRIM_MODE_NORMAL);
  
  return trim_state.mode;
}

/* 인접 클립 찾기 */
static GESClip*
find_adjacent_clip(BlouEditTimeline *timeline, GESClip *clip, BlouEditClipEdge edge)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  g_return_val_if_fail(GES_IS_CLIP(clip), NULL);
  
  /* 현재 클립의 위치와 길이 가져오기 */
  gint64 clip_start = ges_timeline_element_get_start(GES_TIMELINE_ELEMENT(clip));
  gint64 clip_duration = ges_timeline_element_get_duration(GES_TIMELINE_ELEMENT(clip));
  gint64 clip_end = clip_start + clip_duration;
  
  /* 타임라인의 모든 트랙 검사 */
  GESTrack *target_track = ges_clip_get_track(clip);
  
  if (!target_track) {
    return NULL;
  }
  
  /* 이 트랙의 모든 클립 검사 */
  GList *clips = ges_track_get_clips(target_track);
  GList *tmp;
  GESClip *adjacent = NULL;
  
  for (tmp = clips; tmp; tmp = tmp->next) {
    GESClip *other_clip = GES_CLIP(tmp->data);
    
    /* 자기 자신 무시 */
    if (other_clip == clip) {
      continue;
    }
    
    gint64 other_start = ges_timeline_element_get_start(GES_TIMELINE_ELEMENT(other_clip));
    gint64 other_duration = ges_timeline_element_get_duration(GES_TIMELINE_ELEMENT(other_clip));
    gint64 other_end = other_start + other_duration;
    
    if (edge == BLOUEDIT_EDGE_START && other_end == clip_start) {
      /* 시작 부분에 인접한 클립 찾기 */
      adjacent = other_clip;
      break;
    } else if (edge == BLOUEDIT_EDGE_END && other_start == clip_end) {
      /* 끝 부분에 인접한 클립 찾기 */
      adjacent = other_clip;
      break;
    }
  }
  
  g_list_free(clips);
  return adjacent;
}

/* Edge Trimming 기능 시작 */
gboolean
blouedit_timeline_start_edge_trimming(BlouEditTimeline *timeline, 
                                   GESClip *clip,
                                   BlouEditClipEdge edge,
                                   gdouble x)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(GES_IS_CLIP(clip), FALSE);
  g_return_val_if_fail(edge != BLOUEDIT_EDGE_NONE, FALSE);
  
  /* 이미 활성화된 경우 중단 */
  if (trim_state.active) {
    return FALSE;
  }
  
  /* 트리밍 상태 초기화 */
  trim_state.active = TRUE;
  trim_state.clip = g_object_ref(clip);
  trim_state.edge = edge;
  trim_state.start_x = x;
  trim_state.snap_enabled = (timeline->snap_mode != BLOUEDIT_SNAP_NONE);
  
  /* 원래 위치 저장 */
  gint64 clip_start = ges_timeline_element_get_start(GES_TIMELINE_ELEMENT(clip));
  gint64 clip_duration = ges_timeline_element_get_duration(GES_TIMELINE_ELEMENT(clip));
  
  if (edge == BLOUEDIT_EDGE_START) {
    trim_state.original_position = clip_start;
  } else {
    trim_state.original_position = clip_start + clip_duration;
  }
  
  trim_state.current_position = trim_state.original_position;
  
  /* 롤 트리밍 모드인 경우 인접 클립 찾기 */
  if (trim_state.mode == BLOUEDIT_EDGE_TRIM_MODE_ROLL) {
    trim_state.adjacent_clip = find_adjacent_clip(timeline, clip, edge);
    if (trim_state.adjacent_clip) {
      g_object_ref(trim_state.adjacent_clip);
    }
  } else {
    trim_state.adjacent_clip = NULL;
  }
  
  /* 히스토리 그룹 시작 */
  const gchar *description = (edge == BLOUEDIT_EDGE_START) ? 
                            "클립 시작 지점 트리밍" : "클립 끝 지점 트리밍";
  blouedit_timeline_begin_group_action(timeline, description);
  
  /* UI 업데이트 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  return TRUE;
}

/* Edge Trimming 현재 위치로 업데이트 */
gboolean
blouedit_timeline_update_edge_trimming(BlouEditTimeline *timeline, gdouble x)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 활성화되지 않은 경우 중단 */
  if (!trim_state.active || !trim_state.clip) {
    return FALSE;
  }
  
  /* 위치 변환 */
  gdouble delta_x = x - trim_state.start_x;
  gint64 delta_time = delta_x / timeline->zoom_level;
  
  /* 정밀도 레벨에 따라 조정 */
  if (trim_state.precision_level > 0) {
    /* 프레임 레이트에 따라 프레임 단위로 조정 */
    gint64 frame_duration = GST_SECOND / timeline->framerate;
    
    if (trim_state.precision_level == 1) {
      /* 높은 정밀도: 프레임 단위 */
      delta_time = (delta_time / frame_duration) * frame_duration;
    } else if (trim_state.precision_level == 2) {
      /* 최고 정밀도: 절반 프레임 단위 */
      delta_time = (delta_time / (frame_duration / 2)) * (frame_duration / 2);
    }
  }
  
  /* 새 위치 계산 */
  gint64 new_position = trim_state.original_position + delta_time;
  
  /* 최소값 확인 (0 이하가 되지 않도록) */
  if (new_position < 0) {
    new_position = 0;
  }
  
  /* 스냅 기능 적용 */
  if (trim_state.snap_enabled) {
    new_position = blouedit_timeline_snap_position(timeline, new_position);
  }
  
  /* 위치가 변경되었는지 확인 */
  if (new_position == trim_state.current_position) {
    return TRUE;  /* 변경 없음 */
  }
  
  trim_state.current_position = new_position;
  
  /* 편집 모드에 따라 클립 트리밍 수행 */
  switch (trim_state.mode) {
    case BLOUEDIT_EDGE_TRIM_MODE_NORMAL:
      /* 일반 트리밍 */
      if (trim_state.edge == BLOUEDIT_EDGE_START) {
        blouedit_timeline_trim_clip_start(timeline, trim_state.clip, new_position);
      } else {
        blouedit_timeline_trim_clip_end(timeline, trim_state.clip, new_position);
      }
      break;
      
    case BLOUEDIT_EDGE_TRIM_MODE_PRECISE:
      /* 정밀 트리밍 (일반 트리밍과 동일, 정밀도는 위에서 이미 적용됨) */
      if (trim_state.edge == BLOUEDIT_EDGE_START) {
        blouedit_timeline_trim_clip_start(timeline, trim_state.clip, new_position);
      } else {
        blouedit_timeline_trim_clip_end(timeline, trim_state.clip, new_position);
      }
      break;
      
    case BLOUEDIT_EDGE_TRIM_MODE_RIPPLE:
      /* 리플 트리밍 */
      blouedit_timeline_ripple_trim(timeline, trim_state.clip, trim_state.edge, new_position);
      break;
      
    case BLOUEDIT_EDGE_TRIM_MODE_ROLL:
      /* 롤 트리밍 */
      blouedit_timeline_roll_edit(timeline, trim_state.clip, trim_state.edge, new_position);
      break;
  }
  
  /* UI 업데이트 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  return TRUE;
}

/* Edge Trimming 완료 */
gboolean
blouedit_timeline_finish_edge_trimming(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 활성화되지 않은 경우 중단 */
  if (!trim_state.active || !trim_state.clip) {
    return FALSE;
  }
  
  /* 히스토리 그룹 종료 */
  blouedit_timeline_end_group_action(timeline);
  
  /* 자원 해제 */
  g_object_unref(trim_state.clip);
  trim_state.clip = NULL;
  
  if (trim_state.adjacent_clip) {
    g_object_unref(trim_state.adjacent_clip);
    trim_state.adjacent_clip = NULL;
  }
  
  /* 상태 초기화 */
  trim_state.active = FALSE;
  trim_state.edge = BLOUEDIT_EDGE_NONE;
  
  /* UI 업데이트 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  return TRUE;
}

/* Edge Trimming 취소 */
gboolean
blouedit_timeline_cancel_edge_trimming(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 활성화되지 않은 경우 중단 */
  if (!trim_state.active || !trim_state.clip) {
    return FALSE;
  }
  
  /* 원래 위치로 복원 */
  if (trim_state.edge == BLOUEDIT_EDGE_START) {
    blouedit_timeline_trim_clip_start(timeline, trim_state.clip, trim_state.original_position);
  } else {
    blouedit_timeline_trim_clip_end(timeline, trim_state.clip, trim_state.original_position);
  }
  
  /* 히스토리 그룹 취소 */
  blouedit_timeline_undo(timeline);
  
  /* 자원 해제 */
  g_object_unref(trim_state.clip);
  trim_state.clip = NULL;
  
  if (trim_state.adjacent_clip) {
    g_object_unref(trim_state.adjacent_clip);
    trim_state.adjacent_clip = NULL;
  }
  
  /* 상태 초기화 */
  trim_state.active = FALSE;
  trim_state.edge = BLOUEDIT_EDGE_NONE;
  
  /* UI 업데이트 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  return TRUE;
}

/* Edge Trimming 정밀도 설정 */
void
blouedit_timeline_set_edge_trim_precision(BlouEditTimeline *timeline, gint precision_level)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(precision_level >= 0 && precision_level <= 2);
  
  trim_state.precision_level = precision_level;
  
  /* 정밀도 모드에 대한 메시지 표시 */
  gchar *precision_name = NULL;
  switch (precision_level) {
    case 0:
      precision_name = "일반 정밀도";
      break;
    case 1:
      precision_name = "프레임 단위 정밀도";
      break;
    case 2:
      precision_name = "최고 정밀도";
      break;
  }
  
  blouedit_timeline_show_message(timeline, precision_name);
}

/* Edge Trimming 도구 UI 표시 */
void
blouedit_timeline_draw_edge_trimming_ui(BlouEditTimeline *timeline, 
                                     cairo_t *cr, 
                                     gint width, 
                                     gint height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(cr != NULL);
  
  /* Edge Trimming이 활성화되지 않은 경우 아무것도 그리지 않음 */
  if (!trim_state.active || !trim_state.clip) {
    return;
  }
  
  /* 클립 정보 가져오기 */
  gint64 clip_start = ges_timeline_element_get_start(GES_TIMELINE_ELEMENT(trim_state.clip));
  gint64 clip_duration = ges_timeline_element_get_duration(GES_TIMELINE_ELEMENT(trim_state.clip));
  gint64 clip_end = clip_start + clip_duration;
  
  /* 현재 트리밍 위치 계산 */
  gdouble position_x = blouedit_timeline_get_x_from_position(timeline, trim_state.current_position);
  
  /* 트리밍 가이드라인 그리기 */
  cairo_save(cr);
  
  /* 빨간색 세로 선 그리기 */
  cairo_set_source_rgba(cr, 1.0, 0.0, 0.0, 0.8);
  cairo_set_line_width(cr, 2.0);
  cairo_move_to(cr, position_x, 0);
  cairo_line_to(cr, position_x, height);
  cairo_stroke(cr);
  
  /* 트리밍 정보 표시 */
  gchar *timecode = blouedit_timeline_position_to_timecode(timeline, 
                                                       trim_state.current_position,
                                                       timeline->timecode_format);
  gchar *delta_str = NULL;
  
  gint64 delta = trim_state.current_position - trim_state.original_position;
  if (delta != 0) {
    gchar *delta_timecode = blouedit_timeline_position_to_timecode(timeline, 
                                                                ABS(delta),
                                                                timeline->timecode_format);
    delta_str = g_strdup_printf("%s%s", (delta < 0) ? "-" : "+", delta_timecode);
    g_free(delta_timecode);
  } else {
    delta_str = g_strdup("0");
  }
  
  /* 배경 그리기 */
  cairo_text_extents_t extents;
  gchar *info_text = g_strdup_printf("%s (%s)", timecode, delta_str);
  
  cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size(cr, 12);
  cairo_text_extents(cr, info_text, &extents);
  
  /* 텍스트 위치 계산 (선 오른쪽에 표시, 화면 경계 고려) */
  gdouble text_x = position_x + 5;
  if (text_x + extents.width + 10 > width) {
    text_x = position_x - extents.width - 10;
  }
  
  gdouble text_y = 30;
  
  /* 배경 박스 그리기 */
  cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.7);
  cairo_rectangle(cr, 
                 text_x - 5, 
                 text_y - extents.height - 5, 
                 extents.width + 10, 
                 extents.height + 10);
  cairo_fill(cr);
  
  /* 텍스트 그리기 */
  cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);
  cairo_move_to(cr, text_x, text_y);
  cairo_show_text(cr, info_text);
  
  g_free(info_text);
  g_free(timecode);
  g_free(delta_str);
  
  /* 트림 모드 표시 */
  gchar *mode_text = NULL;
  switch (trim_state.mode) {
    case BLOUEDIT_EDGE_TRIM_MODE_NORMAL:
      mode_text = "일반 트리밍";
      break;
    case BLOUEDIT_EDGE_TRIM_MODE_PRECISE:
      mode_text = "정밀 트리밍";
      break;
    case BLOUEDIT_EDGE_TRIM_MODE_RIPPLE:
      mode_text = "리플 트리밍";
      break;
    case BLOUEDIT_EDGE_TRIM_MODE_ROLL:
      mode_text = "롤 트리밍";
      break;
  }
  
  cairo_text_extents(cr, mode_text, &extents);
  
  /* 배경 박스 그리기 */
  cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.7);
  cairo_rectangle(cr, 
                 width - extents.width - 15, 
                 height - extents.height - 15, 
                 extents.width + 10, 
                 extents.height + 10);
  cairo_fill(cr);
  
  /* 텍스트 그리기 */
  cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);
  cairo_move_to(cr, width - extents.width - 10, height - 10);
  cairo_show_text(cr, mode_text);
  
  cairo_restore(cr);
}

/* 정밀도 컨트롤 변경 핸들러 */
static void
on_precision_changed(GtkComboBox *combo_box, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  gint active = gtk_combo_box_get_active(combo_box);
  
  blouedit_timeline_set_edge_trim_precision(timeline, active);
}

/* 모드 컨트롤 변경 핸들러 */
static void
on_mode_changed(GtkComboBox *combo_box, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  gint active = gtk_combo_box_get_active(combo_box);
  
  blouedit_timeline_set_edge_trim_mode(timeline, (BlouEditEdgeTrimMode)active);
}

/* 스냅 토글 핸들러 */
static void
toggle_snap(GtkToggleButton *togglebutton, gpointer user_data)
{
  gboolean active = gtk_toggle_button_get_active(togglebutton);
  trim_state.snap_enabled = active;
}

/* Edge Trimming 도구 대화상자 표시 */
void
blouedit_timeline_show_edge_trimming_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog;
  GtkWidget *content_area;
  GtkWidget *grid;
  GtkWidget *mode_label, *mode_combo;
  GtkWidget *precision_label, *precision_combo;
  GtkWidget *snap_check;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("에지 트리밍 설정",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                     "_닫기", GTK_RESPONSE_CLOSE,
                                     NULL);
  
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  
  /* 그리드 생성 */
  grid = gtk_grid_new();
  gtk_grid_set_column_spacing(GTK_GRID(grid), 10);
  gtk_grid_set_row_spacing(GTK_GRID(grid), 10);
  gtk_container_add(GTK_CONTAINER(content_area), grid);
  
  /* 모드 설정 */
  mode_label = gtk_label_new("트리밍 모드:");
  gtk_widget_set_halign(mode_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), mode_label, 0, 0, 1, 1);
  
  mode_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "일반 트리밍");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "정밀 트리밍");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "리플 트리밍");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "롤 트리밍");
  gtk_combo_box_set_active(GTK_COMBO_BOX(mode_combo), trim_state.mode);
  g_signal_connect(mode_combo, "changed", G_CALLBACK(on_mode_changed), timeline);
  gtk_grid_attach(GTK_GRID(grid), mode_combo, 1, 0, 1, 1);
  
  /* 정밀도 설정 */
  precision_label = gtk_label_new("트리밍 정밀도:");
  gtk_widget_set_halign(precision_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), precision_label, 0, 1, 1, 1);
  
  precision_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(precision_combo), "일반 정밀도");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(precision_combo), "프레임 단위 정밀도");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(precision_combo), "최고 정밀도");
  gtk_combo_box_set_active(GTK_COMBO_BOX(precision_combo), trim_state.precision_level);
  g_signal_connect(precision_combo, "changed", G_CALLBACK(on_precision_changed), timeline);
  gtk_grid_attach(GTK_GRID(grid), precision_combo, 1, 1, 1, 1);
  
  /* 스냅 체크박스 */
  snap_check = gtk_check_button_new_with_label("스냅 활성화");
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(snap_check), trim_state.snap_enabled);
  g_signal_connect(snap_check, "toggled", G_CALLBACK(toggle_snap), timeline);
  gtk_grid_attach(GTK_GRID(grid), snap_check, 0, 2, 2, 1);
  
  /* 설명 레이블 */
  GtkWidget *desc_label = gtk_label_new(
      "에지 트리밍 도구는 클립의 시작점과 끝점을 정밀하게 조절합니다.\n"
      "정밀도를 높이면 프레임 단위의 트리밍이 가능합니다.");
  gtk_label_set_line_wrap(GTK_LABEL(desc_label), TRUE);
  gtk_widget_set_margin_top(desc_label, 10);
  gtk_grid_attach(GTK_GRID(grid), desc_label, 0, 3, 2, 1);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 실행 */
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
} 