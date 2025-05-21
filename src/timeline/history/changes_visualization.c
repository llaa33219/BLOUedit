#include <gtk/gtk.h>
#include <string.h>
#include "timeline_history.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 변경 시각화 상태 플래그 */
static gboolean changes_visualization_enabled = FALSE;

/* 변경 시각화 활성화 설정 함수 */
void
blouedit_timeline_set_changes_visualization_enabled(BlouEditTimeline *timeline, 
                                                gboolean enabled)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  changes_visualization_enabled = enabled;
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 변경 시각화 활성화 상태 확인 함수 */
gboolean
blouedit_timeline_get_changes_visualization_enabled(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  return changes_visualization_enabled;
}

/* 타임라인 변경 시각화 그리기 함수 - 내부 사용 */
void
blouedit_timeline_draw_changes_visualization(BlouEditTimeline *timeline, 
                                         cairo_t *cr, 
                                         gint width, 
                                         gint height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(cr != NULL);
  
  /* 변경 시각화가 비활성화된 경우 그리지 않음 */
  if (!changes_visualization_enabled) {
    return;
  }
  
  /* 히스토리 항목 가져오기 */
  GList *history_items = blouedit_timeline_history_get_entries(timeline);
  if (history_items == NULL) {
    return;
  }
  
  /* 색상 및 크기 정의 */
  const gdouble mark_height = 5.0;
  
  /* 타임라인 시각적 변경점 표시 */
  for (GList *l = history_items; l != NULL; l = l->next) {
    BlouEditHistoryEntry *entry = (BlouEditHistoryEntry*)l->data;
    
    /* 특정 유형의 히스토리 항목에 대해서만 변경점 표시 */
    if (entry->type == BLOUEDIT_HISTORY_ADD_CLIP ||
        entry->type == BLOUEDIT_HISTORY_REMOVE_CLIP ||
        entry->type == BLOUEDIT_HISTORY_MOVE_CLIP ||
        entry->type == BLOUEDIT_HISTORY_TRIM_CLIP ||
        entry->type == BLOUEDIT_HISTORY_SPLIT_CLIP ||
        entry->type == BLOUEDIT_HISTORY_MERGE_CLIPS) {
      
      /* 변경 지점의 타임라인 위치 계산 (예: 클립의 시작/끝 위치) */
      /* 실제 구현에서는 히스토리 항목의 데이터를 사용하여 정확한 위치 계산 필요 */
      gdouble position_x = g_random_double_range(0, width); /* 임시로 랜덤 위치 사용 */
      
      /* 유형에 따른 색상 선택 */
      switch (entry->type) {
        case BLOUEDIT_HISTORY_ADD_CLIP:
          cairo_set_source_rgba(cr, 0.2, 0.8, 0.2, 0.5); /* 녹색 */
          break;
        case BLOUEDIT_HISTORY_REMOVE_CLIP:
          cairo_set_source_rgba(cr, 0.8, 0.2, 0.2, 0.5); /* 빨간색 */
          break;
        case BLOUEDIT_HISTORY_MOVE_CLIP:
          cairo_set_source_rgba(cr, 0.2, 0.2, 0.8, 0.5); /* 파란색 */
          break;
        case BLOUEDIT_HISTORY_TRIM_CLIP:
          cairo_set_source_rgba(cr, 0.8, 0.8, 0.2, 0.5); /* 노란색 */
          break;
        case BLOUEDIT_HISTORY_SPLIT_CLIP:
          cairo_set_source_rgba(cr, 0.8, 0.2, 0.8, 0.5); /* 보라색 */
          break;
        case BLOUEDIT_HISTORY_MERGE_CLIPS:
          cairo_set_source_rgba(cr, 0.2, 0.8, 0.8, 0.5); /* 청록색 */
          break;
        default:
          cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.5); /* 회색 */
      }
      
      /* 변경점 표시 - 세로선 */
      cairo_set_line_width(cr, 1.0);
      cairo_move_to(cr, position_x, 0);
      cairo_line_to(cr, position_x, height);
      cairo_stroke(cr);
      
      /* 변경점 표시 - 상단 마커 */
      cairo_rectangle(cr, position_x - 2, 0, 4, mark_height);
      cairo_fill(cr);
    }
  }
}

/* 타임라인 변경 시각화 대화상자 */
void
blouedit_timeline_show_changes_visualization(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *switch_widget, *label, *box;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("타임라인 변경 시각화 설정",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_닫기", GTK_RESPONSE_CLOSE,
                                     NULL);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 설명 레이블 */
  label = gtk_label_new("타임라인에 편집 지점을 시각적으로 표시합니다.\n"
                       "각 색상은 다른 유형의 편집을 나타냅니다:\n"
                       "- 녹색: 클립 추가\n"
                       "- 빨간색: 클립 제거\n"
                       "- 파란색: 클립 이동\n"
                       "- 노란색: 클립 트리밍\n"
                       "- 보라색: 클립 분할\n"
                       "- 청록색: 클립 병합");
  gtk_label_set_line_wrap(GTK_LABEL(label), TRUE);
  gtk_container_add(GTK_CONTAINER(content_area), label);
  
  /* 활성화 스위치를 포함한 박스 */
  box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
  gtk_container_add(GTK_CONTAINER(content_area), box);
  
  /* 스위치 레이블 */
  label = gtk_label_new("변경 시각화 활성화:");
  gtk_container_add(GTK_CONTAINER(box), label);
  
  /* 스위치 위젯 */
  switch_widget = gtk_switch_new();
  gtk_switch_set_active(GTK_SWITCH(switch_widget), 
                      blouedit_timeline_get_changes_visualization_enabled(timeline));
  gtk_container_add(GTK_CONTAINER(box), switch_widget);
  
  /* 스위치 변경 핸들러 */
  g_signal_connect(switch_widget, "notify::active", G_CALLBACK(blouedit_timeline_set_changes_visualization_enabled), timeline);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 실행 */
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 정리 */
  gtk_widget_destroy(dialog);
} 