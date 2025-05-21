#include <gtk/gtk.h>
#include <string.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "markers.h"
#include "marker_color.h"

/* 마커 검색 및 필터링 구조체 정의 */
typedef struct {
  BlouEditTimeline *timeline;    /* 타임라인 */
  GSList *filtered_markers;      /* 필터링된 마커 목록 */
  GtkWidget *search_entry;       /* 검색어 입력 위젯 */
  GtkWidget *type_filter_combo;  /* 유형 필터 콤보박스 */
  GtkWidget *result_list;        /* 결과 목록 위젯 */
  GtkListStore *marker_store;    /* 마커 저장소 모델 */
} BlouEditMarkerSearch;

/* 마커 목록 칼럼 정의 */
enum {
  COLUMN_ID,
  COLUMN_TYPE,
  COLUMN_NAME,
  COLUMN_COMMENT,
  COLUMN_POSITION,
  COLUMN_COLOR,
  COLUMN_MARKER_PTR,
  N_COLUMNS
};

/* 마커 유형 문자열 반환 함수 */
static const gchar* 
get_marker_type_name(BlouEditMarkerType type)
{
  switch (type) {
    case BLOUEDIT_MARKER_TYPE_GENERIC:
      return "일반";
    case BLOUEDIT_MARKER_TYPE_CUE:
      return "큐 포인트";
    case BLOUEDIT_MARKER_TYPE_IN:
      return "시작 지점";
    case BLOUEDIT_MARKER_TYPE_OUT:
      return "종료 지점";
    case BLOUEDIT_MARKER_TYPE_CHAPTER:
      return "챕터";
    case BLOUEDIT_MARKER_TYPE_ERROR:
      return "오류";
    case BLOUEDIT_MARKER_TYPE_WARNING:
      return "경고";
    case BLOUEDIT_MARKER_TYPE_COMMENT:
      return "코멘트";
    default:
      return "알 수 없음";
  }
}

/* 타임코드 문자열 반환 함수 */
static gchar*
get_position_as_timecode(BlouEditTimeline *timeline, gint64 position)
{
  BlouEditTimecodeFormat format = blouedit_timeline_get_timecode_format(timeline);
  return blouedit_timeline_position_to_timecode(timeline, position, format);
}

/* 마커 목록 모델 업데이트 함수 */
static void
update_marker_list_model(BlouEditMarkerSearch *search)
{
  g_return_if_fail(search != NULL);
  g_return_if_fail(search->marker_store != NULL);
  
  /* 목록 지우기 */
  gtk_list_store_clear(search->marker_store);
  
  /* 필터링된 마커 목록에서 항목 추가 */
  for (GSList *m = search->filtered_markers; m != NULL; m = m->next) {
    BlouEditTimelineMarker *marker = (BlouEditTimelineMarker *)m->data;
    GtkTreeIter iter;
    gchar *timecode = get_position_as_timecode(search->timeline, marker->position);
    
    gtk_list_store_append(search->marker_store, &iter);
    gtk_list_store_set(search->marker_store, &iter,
                      COLUMN_ID, marker->id,
                      COLUMN_TYPE, get_marker_type_name(marker->type),
                      COLUMN_NAME, marker->name,
                      COLUMN_COMMENT, marker->comment,
                      COLUMN_POSITION, timecode,
                      COLUMN_COLOR, &marker->color,
                      COLUMN_MARKER_PTR, marker,
                      -1);
    
    g_free(timecode);
  }
}

/* 마커 필터링 함수 */
static void
filter_markers(BlouEditMarkerSearch *search, const gchar *search_text, gint type_filter)
{
  g_return_if_fail(search != NULL);
  
  /* 이전 필터링 결과 지우기 */
  g_slist_free(search->filtered_markers);
  search->filtered_markers = NULL;
  
  /* 모든 마커 가져오기 */
  GSList *all_markers = blouedit_timeline_get_markers(search->timeline);
  
  /* 필터링 수행 */
  for (GSList *m = all_markers; m != NULL; m = m->next) {
    BlouEditTimelineMarker *marker = (BlouEditTimelineMarker *)m->data;
    gboolean type_match = (type_filter == -1) || (marker->type == type_filter);
    
    if (!type_match) {
      continue;
    }
    
    if (search_text && *search_text) {
      /* 검색어가 있으면 이름과 코멘트에서 검색 */
      gboolean name_match = marker->name && strstr(marker->name, search_text);
      gboolean comment_match = marker->comment && strstr(marker->comment, search_text);
      
      if (name_match || comment_match) {
        search->filtered_markers = g_slist_append(search->filtered_markers, marker);
      }
    } else {
      /* 검색어가 없으면 모든 해당 유형의 마커 포함 */
      search->filtered_markers = g_slist_append(search->filtered_markers, marker);
    }
  }
  
  /* 목록 업데이트 */
  update_marker_list_model(search);
}

/* 검색어 변경 시 콜백 함수 */
static void
on_search_text_changed(GtkEditable *editable, BlouEditMarkerSearch *search)
{
  const gchar *text = gtk_entry_get_text(GTK_ENTRY(editable));
  gint type_filter = gtk_combo_box_get_active(GTK_COMBO_BOX(search->type_filter_combo)) - 1;
  
  filter_markers(search, text, type_filter);
}

/* 유형 필터 변경 시 콜백 함수 */
static void
on_type_filter_changed(GtkComboBox *widget, BlouEditMarkerSearch *search)
{
  const gchar *text = gtk_entry_get_text(GTK_ENTRY(search->search_entry));
  gint type_filter = gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) - 1;
  
  filter_markers(search, text, type_filter);
}

/* 마커 선택 시 콜백 함수 */
static void
on_marker_selected(GtkTreeSelection *selection, BlouEditMarkerSearch *search)
{
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
    BlouEditTimelineMarker *marker;
    gtk_tree_model_get(model, &iter, COLUMN_MARKER_PTR, &marker, -1);
    
    if (marker) {
      /* 선택한 마커로 이동 */
      blouedit_timeline_set_position(search->timeline, marker->position);
      
      /* 마커 선택 */
      blouedit_timeline_select_marker(search->timeline, marker);
      
      /* 타임라인 다시 그리기 */
      gtk_widget_queue_draw(GTK_WIDGET(search->timeline));
    }
  }
}

/* 색상 셀 렌더링 함수 */
static void
render_color_cell(GtkTreeViewColumn *column,
                 GtkCellRenderer *renderer,
                 GtkTreeModel *model,
                 GtkTreeIter *iter,
                 gpointer user_data)
{
  GdkRGBA *color;
  gtk_tree_model_get(model, iter, COLUMN_COLOR, &color, -1);
  
  if (color) {
    gchar *color_str = gdk_rgba_to_string(color);
    g_object_set(renderer, "background", color_str, NULL);
    g_free(color_str);
  }
}

/* 마커 검색 대화상자 닫기 콜백 */
static void
on_marker_search_dialog_response(GtkDialog *dialog,
                               gint response_id,
                               BlouEditMarkerSearch *search)
{
  if (response_id == GTK_RESPONSE_OK) {
    /* 선택한 마커 처리 - 이미 트리뷰 선택 핸들러에서 처리함 */
  }
  
  /* 검색 데이터 정리 */
  g_slist_free(search->filtered_markers);
  g_free(search);
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(GTK_WIDGET(dialog));
}

/* 마커 검색 대화상자 표시 함수 */
void
blouedit_timeline_show_marker_search(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 검색 데이터 구조체 생성 */
  BlouEditMarkerSearch *search = g_new0(BlouEditMarkerSearch, 1);
  search->timeline = timeline;
  search->filtered_markers = NULL;
  
  /* 대화상자 생성 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "마커 검색",
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    "_취소", GTK_RESPONSE_CANCEL,
    "_확인", GTK_RESPONSE_OK,
    NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 600, 400);
  
  /* 대화상자 콘텐츠 영역 가져오기 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_widget_set_margin_start(content_area, 12);
  gtk_widget_set_margin_end(content_area, 12);
  gtk_widget_set_margin_top(content_area, 12);
  gtk_widget_set_margin_bottom(content_area, 12);
  gtk_box_set_spacing(GTK_BOX(content_area), 6);
  
  /* 필터 영역 컨테이너 */
  GtkWidget *filter_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
  gtk_container_add(GTK_CONTAINER(content_area), filter_box);
  
  /* 검색어 레이블 */
  GtkWidget *search_label = gtk_label_new("검색어:");
  gtk_container_add(GTK_CONTAINER(filter_box), search_label);
  
  /* 검색어 입력 필드 */
  search->search_entry = gtk_entry_new();
  gtk_widget_set_hexpand(search->search_entry, TRUE);
  gtk_container_add(GTK_CONTAINER(filter_box), search->search_entry);
  
  /* 마커 유형 레이블 */
  GtkWidget *type_label = gtk_label_new("마커 유형:");
  gtk_container_add(GTK_CONTAINER(filter_box), type_label);
  
  /* 마커 유형 콤보박스 */
  search->type_filter_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "모든 유형");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "일반");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "큐 포인트");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "시작 지점");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "종료 지점");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "챕터");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "오류");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "경고");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(search->type_filter_combo), "코멘트");
  gtk_combo_box_set_active(GTK_COMBO_BOX(search->type_filter_combo), 0); /* 기본값: 모든 유형 */
  gtk_container_add(GTK_CONTAINER(filter_box), search->type_filter_combo);
  
  /* 목록 스크롤 윈도우 */
  GtkWidget *scrolled_window = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window),
                                GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_widget_set_vexpand(scrolled_window, TRUE);
  gtk_container_add(GTK_CONTAINER(content_area), scrolled_window);
  
  /* 마커 목록 저장소 모델 생성 */
  search->marker_store = gtk_list_store_new(N_COLUMNS,
                                         G_TYPE_UINT,      /* ID */
                                         G_TYPE_STRING,    /* 유형 */
                                         G_TYPE_STRING,    /* 이름 */
                                         G_TYPE_STRING,    /* 코멘트 */
                                         G_TYPE_STRING,    /* 위치 */
                                         GDK_TYPE_RGBA,    /* 색상 */
                                         G_TYPE_POINTER);  /* 마커 포인터 */
  
  /* 트리뷰 생성 */
  search->result_list = gtk_tree_view_new_with_model(GTK_TREE_MODEL(search->marker_store));
  gtk_container_add(GTK_CONTAINER(scrolled_window), search->result_list);
  
  /* 색상 컬럼 */
  GtkCellRenderer *color_renderer = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *color_column = gtk_tree_view_column_new_with_attributes(
    "", color_renderer, NULL);
  gtk_tree_view_column_set_cell_data_func(color_column, color_renderer,
                                        render_color_cell, NULL, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(search->result_list), color_column);
  
  /* ID 컬럼 */
  GtkCellRenderer *id_renderer = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *id_column = gtk_tree_view_column_new_with_attributes(
    "ID", id_renderer, "text", COLUMN_ID, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(search->result_list), id_column);
  
  /* 유형 컬럼 */
  GtkCellRenderer *type_renderer = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *type_column = gtk_tree_view_column_new_with_attributes(
    "유형", type_renderer, "text", COLUMN_TYPE, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(search->result_list), type_column);
  
  /* 이름 컬럼 */
  GtkCellRenderer *name_renderer = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *name_column = gtk_tree_view_column_new_with_attributes(
    "이름", name_renderer, "text", COLUMN_NAME, NULL);
  gtk_tree_view_column_set_expand(name_column, TRUE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(search->result_list), name_column);
  
  /* 코멘트 컬럼 */
  GtkCellRenderer *comment_renderer = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *comment_column = gtk_tree_view_column_new_with_attributes(
    "코멘트", comment_renderer, "text", COLUMN_COMMENT, NULL);
  gtk_tree_view_column_set_expand(comment_column, TRUE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(search->result_list), comment_column);
  
  /* 위치 컬럼 */
  GtkCellRenderer *pos_renderer = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *pos_column = gtk_tree_view_column_new_with_attributes(
    "위치", pos_renderer, "text", COLUMN_POSITION, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(search->result_list), pos_column);
  
  /* 선택 처리 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(search->result_list));
  gtk_tree_selection_set_mode(selection, GTK_SELECTION_SINGLE);
  
  /* 시그널 핸들러 연결 */
  g_signal_connect(search->search_entry, "changed",
                  G_CALLBACK(on_search_text_changed), search);
  g_signal_connect(search->type_filter_combo, "changed",
                  G_CALLBACK(on_type_filter_changed), search);
  g_signal_connect(selection, "changed",
                  G_CALLBACK(on_marker_selected), search);
  g_signal_connect(dialog, "response",
                  G_CALLBACK(on_marker_search_dialog_response), search);
  
  /* 초기 목록 채우기 */
  filter_markers(search, "", -1);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
}

/* 마커 간 이동 함수 - 향상된 버전 */
void
blouedit_timeline_goto_next_marker_by_type(BlouEditTimeline *timeline, BlouEditMarkerType type)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  gint64 current_position = blouedit_timeline_get_position(timeline);
  gint64 best_position = G_MAXINT64;
  BlouEditTimelineMarker *best_marker = NULL;
  
  /* 모든 마커 가져오기 */
  GSList *markers = blouedit_timeline_get_markers(timeline);
  
  /* 현재 위치 이후의 가장 가까운 마커 찾기 */
  for (GSList *m = markers; m != NULL; m = m->next) {
    BlouEditTimelineMarker *marker = (BlouEditTimelineMarker *)m->data;
    
    /* 유형 필터링이 설정되었으면 확인 */
    if (type != -1 && marker->type != type) {
      continue;
    }
    
    /* 현재 위치 이후의 마커만 확인 */
    if (marker->position > current_position && 
        marker->position < best_position) {
      best_position = marker->position;
      best_marker = marker;
    }
  }
  
  /* 적합한 마커를 찾았으면 이동 */
  if (best_marker) {
    blouedit_timeline_set_position(timeline, best_marker->position);
    blouedit_timeline_select_marker(timeline, best_marker);
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
}

void
blouedit_timeline_goto_prev_marker_by_type(BlouEditTimeline *timeline, BlouEditMarkerType type)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  gint64 current_position = blouedit_timeline_get_position(timeline);
  gint64 best_position = -1;
  BlouEditTimelineMarker *best_marker = NULL;
  
  /* 모든 마커 가져오기 */
  GSList *markers = blouedit_timeline_get_markers(timeline);
  
  /* 현재 위치 이전의 가장 가까운 마커 찾기 */
  for (GSList *m = markers; m != NULL; m = m->next) {
    BlouEditTimelineMarker *marker = (BlouEditTimelineMarker *)m->data;
    
    /* 유형 필터링이 설정되었으면 확인 */
    if (type != -1 && marker->type != type) {
      continue;
    }
    
    /* 현재 위치 이전의 마커만 확인 */
    if (marker->position < current_position && 
        marker->position > best_position) {
      best_position = marker->position;
      best_marker = marker;
    }
  }
  
  /* 적합한 마커를 찾았으면 이동 */
  if (best_marker) {
    blouedit_timeline_set_position(timeline, best_marker->position);
    blouedit_timeline_select_marker(timeline, best_marker);
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
} 