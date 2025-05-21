#include <gtk/gtk.h>
#include <gst/gst.h>
#include "timeline_performance.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 프리렌더 세그먼트 구조체 */
typedef struct {
  GstClockTime start;
  GstClockTime end;
  gboolean is_active;
  gboolean is_completed;
  gchar *render_file;
} BlouEditPrerenderSegment;

/* 프리렌더 세그먼트 생성 */
static BlouEditPrerenderSegment*
prerender_segment_new(GstClockTime start, GstClockTime end)
{
  BlouEditPrerenderSegment *segment = g_new0(BlouEditPrerenderSegment, 1);
  
  segment->start = start;
  segment->end = end;
  segment->is_active = FALSE;
  segment->is_completed = FALSE;
  segment->render_file = NULL;
  
  return segment;
}

/* 프리렌더 세그먼트 해제 */
static void
prerender_segment_free(BlouEditPrerenderSegment *segment)
{
  if (segment == NULL) {
    return;
  }
  
  g_free(segment->render_file);
  g_free(segment);
}

/* 세그먼트 위치가 겹치는지 확인 */
static gboolean
segments_overlap(BlouEditPrerenderSegment *a, BlouEditPrerenderSegment *b)
{
  return (a->start < b->end && a->end > b->start);
}

/* 프리렌더 세그먼트 추가 */
void
blouedit_timeline_set_prerender_segment(BlouEditTimeline *timeline, 
                                     GstClockTime start,
                                     GstClockTime end)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(start < end);
  
  /* 새 세그먼트 생성 */
  BlouEditPrerenderSegment *new_segment = prerender_segment_new(start, end);
  
  /* 기존 세그먼트와 겹치는지 확인 */
  GList *existing = NULL;
  for (GList *l = timeline->prerender_segments; l != NULL; l = l->next) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)l->data;
    
    if (segments_overlap(segment, new_segment)) {
      existing = l;
      break;
    }
  }
  
  /* 겹치는 세그먼트가 있으면 해당 세그먼트를 대체 */
  if (existing != NULL) {
    BlouEditPrerenderSegment *old_segment = (BlouEditPrerenderSegment*)existing->data;
    prerender_segment_free(old_segment);
    existing->data = new_segment;
  } else {
    /* 겹치는 세그먼트가 없으면 목록에 추가 */
    timeline->prerender_segments = g_list_append(timeline->prerender_segments, new_segment);
  }
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 프리렌더 세그먼트 제거 */
void
blouedit_timeline_remove_prerender_segment(BlouEditTimeline *timeline, 
                                        GstClockTime start,
                                        GstClockTime end)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 제거할 세그먼트 찾기 */
  GList *to_remove = NULL;
  for (GList *l = timeline->prerender_segments; l != NULL; l = l->next) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)l->data;
    
    if (segment->start == start && segment->end == end) {
      to_remove = l;
      break;
    }
  }
  
  /* 세그먼트 제거 */
  if (to_remove != NULL) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)to_remove->data;
    
    /* 활성 상태면 렌더링 중지 */
    if (segment->is_active) {
      /* 렌더링 중지 코드 (실제 구현에서 추가) */
    }
    
    /* 세그먼트 해제 */
    prerender_segment_free(segment);
    
    /* 목록에서 제거 */
    timeline->prerender_segments = g_list_delete_link(timeline->prerender_segments, to_remove);
    
    /* 타임라인 갱신 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
}

/* 모든 프리렌더 세그먼트 제거 */
void
blouedit_timeline_clear_prerender_segments(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 모든 활성 렌더링 중지 */
  for (GList *l = timeline->prerender_segments; l != NULL; l = l->next) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)l->data;
    
    if (segment->is_active) {
      /* 렌더링 중지 코드 (실제 구현에서 추가) */
    }
    
    /* 세그먼트 해제 */
    prerender_segment_free(segment);
  }
  
  /* 목록 지우기 */
  g_list_free(timeline->prerender_segments);
  timeline->prerender_segments = NULL;
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 시간을 문자열로 변환 (hh:mm:ss.sss 형식) */
static gchar*
format_time(GstClockTime time)
{
  guint hours, minutes, seconds, ms;
  
  hours = (guint)(time / (GST_SECOND * 60 * 60));
  minutes = (guint)((time / (GST_SECOND * 60)) % 60);
  seconds = (guint)((time / GST_SECOND) % 60);
  ms = (guint)((time / (GST_SECOND / 1000)) % 1000);
  
  return g_strdup_printf("%02u:%02u:%02u.%03u", hours, minutes, seconds, ms);
}

/* 세그먼트 렌더링 진행 콜백 (가상 함수) */
static gboolean
simulate_render_progress(gpointer user_data)
{
  GtkWidget *progress_bar = GTK_WIDGET(user_data);
  gdouble fraction = gtk_progress_bar_get_fraction(GTK_PROGRESS_BAR(progress_bar));
  
  /* 진행 상태 업데이트 */
  fraction += 0.01;
  if (fraction > 1.0) {
    fraction = 1.0;
    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), "완료됨");
    return G_SOURCE_REMOVE;  /* 타이머 제거 */
  }
  
  gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar), fraction);
  gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), 
                          g_strdup_printf("%.0f%%", fraction * 100));
  
  return G_SOURCE_CONTINUE;  /* 타이머 계속 */
}

/* 프리렌더 시작 */
void
blouedit_timeline_start_prerender(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 렌더링 대기 중인 세그먼트 확인 */
  gboolean has_pending = FALSE;
  for (GList *l = timeline->prerender_segments; l != NULL; l = l->next) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)l->data;
    
    if (!segment->is_completed && !segment->is_active) {
      has_pending = TRUE;
      break;
    }
  }
  
  if (!has_pending) {
    GtkWidget *message = gtk_message_dialog_new(
      GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
      GTK_MESSAGE_INFO,
      GTK_BUTTONS_OK,
      "렌더링할 세그먼트가 없습니다. 먼저 프리렌더 세그먼트를 설정하세요.");
    
    gtk_dialog_run(GTK_DIALOG(message));
    gtk_widget_destroy(message);
    return;
  }
  
  /* 렌더링 진행 대화상자 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "프리렌더 진행",
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    "_취소", GTK_RESPONSE_CANCEL,
    NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 400, 200);
  
  /* 콘텐츠 영역 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 목록 스크롤 */
  GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                               GTK_POLICY_AUTOMATIC,
                               GTK_POLICY_AUTOMATIC);
  gtk_widget_set_vexpand(scroll, TRUE);
  gtk_container_add(GTK_CONTAINER(content_area), scroll);
  
  /* 목록 컨테이너 */
  GtkWidget *list_box = gtk_list_box_new();
  gtk_container_add(GTK_CONTAINER(scroll), list_box);
  
  /* 세그먼트 목록 추가 */
  for (GList *l = timeline->prerender_segments; l != NULL; l = l->next) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)l->data;
    
    if (segment->is_completed) {
      continue;  /* 이미 완료된 세그먼트는 건너뜀 */
    }
    
    /* 세그먼트 컨테이너 */
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_top(box, 5);
    gtk_widget_set_margin_bottom(box, 5);
    gtk_widget_set_margin_start(box, 5);
    gtk_widget_set_margin_end(box, 5);
    
    /* 세그먼트 정보 레이블 */
    gchar *start_str = format_time(segment->start);
    gchar *end_str = format_time(segment->end);
    gchar *info = g_strdup_printf("세그먼트: %s - %s", start_str, end_str);
    GtkWidget *label = gtk_label_new(info);
    gtk_widget_set_halign(label, GTK_ALIGN_START);
    gtk_container_add(GTK_CONTAINER(box), label);
    g_free(info);
    g_free(start_str);
    g_free(end_str);
    
    /* 진행 바 */
    GtkWidget *progress = gtk_progress_bar_new();
    gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(progress), TRUE);
    if (segment->is_active) {
      gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress), "처리 중...");
    } else {
      gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress), "대기 중...");
    }
    gtk_container_add(GTK_CONTAINER(box), progress);
    
    /* 목록에 추가 */
    gtk_container_add(GTK_CONTAINER(list_box), box);
    
    /* 세그먼트 활성화 및 렌더링 시작 (시뮬레이션) */
    segment->is_active = TRUE;
    g_timeout_add(100, simulate_render_progress, progress);
  }
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_CANCEL) {
    /* 렌더링 취소 처리 (실제 구현에서 추가) */
  }
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
}

/* 프리렌더 중지 */
void
blouedit_timeline_stop_prerender(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 모든 활성 렌더링 중지 */
  for (GList *l = timeline->prerender_segments; l != NULL; l = l->next) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)l->data;
    
    if (segment->is_active) {
      segment->is_active = FALSE;
      /* 렌더링 중지 코드 (실제 구현에서 추가) */
    }
  }
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 인/아웃 포인트 설정 버튼 클릭 핸들러 */
static void
on_set_in_out_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  
  /* 현재 선택 범위 가져오기 */
  GstClockTime in_point = timeline->selection_start;
  GstClockTime out_point = timeline->selection_end;
  
  /* 선택 영역이 있을 경우 해당 영역으로 설정 */
  if (in_point < out_point) {
    GtkEntry *start_entry = GTK_ENTRY(g_object_get_data(G_OBJECT(dialog), "start-entry"));
    GtkEntry *end_entry = GTK_ENTRY(g_object_get_data(G_OBJECT(dialog), "end-entry"));
    
    gchar *start_str = format_time(in_point);
    gchar *end_str = format_time(out_point);
    
    gtk_entry_set_text(start_entry, start_str);
    gtk_entry_set_text(end_entry, end_str);
    
    g_free(start_str);
    g_free(end_str);
  } else {
    /* 선택 영역이 없을 경우 경고 */
    GtkWidget *message = gtk_message_dialog_new(
      GTK_WINDOW(dialog),
      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
      GTK_MESSAGE_WARNING,
      GTK_BUTTONS_OK,
      "타임라인에서 먼저 구간을 선택하세요.");
    
    gtk_dialog_run(GTK_DIALOG(message));
    gtk_widget_destroy(message);
  }
}

/* 프리렌더 세그먼트 추가 핸들러 */
static void
on_add_segment_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  GtkEntry *start_entry = GTK_ENTRY(g_object_get_data(G_OBJECT(dialog), "start-entry"));
  GtkEntry *end_entry = GTK_ENTRY(g_object_get_data(G_OBJECT(dialog), "end-entry"));
  GtkListStore *store = GTK_LIST_STORE(g_object_get_data(G_OBJECT(dialog), "segments-store"));
  
  /* 입력값 확인 */
  const gchar *start_text = gtk_entry_get_text(start_entry);
  const gchar *end_text = gtk_entry_get_text(end_entry);
  
  /* 간단한 검증 (실제 구현에서는 더 정확한 시간 파싱 필요) */
  if (start_text == NULL || start_text[0] == '\0' || 
      end_text == NULL || end_text[0] == '\0') {
    GtkWidget *message = gtk_message_dialog_new(
      GTK_WINDOW(dialog),
      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
      GTK_MESSAGE_ERROR,
      GTK_BUTTONS_OK,
      "시작 시간과 종료 시간을 모두 입력하세요.");
    
    gtk_dialog_run(GTK_DIALOG(message));
    gtk_widget_destroy(message);
    return;
  }
  
  /* 시간 문자열을 GstClockTime으로 변환 (간단한 구현) */
  gint h1, m1, s1, ms1;
  gint h2, m2, s2, ms2;
  if (sscanf(start_text, "%d:%d:%d.%d", &h1, &m1, &s1, &ms1) == 4 &&
      sscanf(end_text, "%d:%d:%d.%d", &h2, &m2, &s2, &ms2) == 4) {
    
    GstClockTime start_time = ((guint64)h1 * 3600 + m1 * 60 + s1) * GST_SECOND + ms1 * GST_MSECOND;
    GstClockTime end_time = ((guint64)h2 * 3600 + m2 * 60 + s2) * GST_SECOND + ms2 * GST_MSECOND;
    
    if (start_time >= end_time) {
      GtkWidget *message = gtk_message_dialog_new(
        GTK_WINDOW(dialog),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        GTK_MESSAGE_ERROR,
        GTK_BUTTONS_OK,
        "종료 시간은 시작 시간보다 나중이어야 합니다.");
      
      gtk_dialog_run(GTK_DIALOG(message));
      gtk_widget_destroy(message);
      return;
    }
    
    /* 세그먼트 추가 */
    blouedit_timeline_set_prerender_segment(timeline, start_time, end_time);
    
    /* 목록에 추가 */
    GtkTreeIter iter;
    gtk_list_store_append(store, &iter);
    gtk_list_store_set(store, &iter,
                     0, start_text,
                     1, end_text,
                     2, "대기 중",
                     -1);
    
    /* 입력 필드 초기화 */
    gtk_entry_set_text(start_entry, "");
    gtk_entry_set_text(end_entry, "");
  } else {
    GtkWidget *message = gtk_message_dialog_new(
      GTK_WINDOW(dialog),
      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
      GTK_MESSAGE_ERROR,
      GTK_BUTTONS_OK,
      "시간 형식이 잘못되었습니다. hh:mm:ss.sss 형식을 사용하세요.");
    
    gtk_dialog_run(GTK_DIALOG(message));
    gtk_widget_destroy(message);
  }
}

/* 세그먼트 선택 변경 핸들러 */
static void
on_segment_selection_changed(GtkTreeSelection *selection, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  GtkWidget *remove_button = GTK_WIDGET(g_object_get_data(G_OBJECT(dialog), "remove-button"));
  
  /* 선택 여부에 따라 제거 버튼 활성화 */
  gtk_widget_set_sensitive(remove_button, gtk_tree_selection_get_selected(selection, NULL, NULL));
}

/* 세그먼트 제거 버튼 클릭 핸들러 */
static void
on_remove_segment_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "segments-view"));
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  /* 선택된 항목 가져오기 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(tree_view);
  if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
    gchar *start_text, *end_text;
    gtk_tree_model_get(model, &iter, 0, &start_text, 1, &end_text, -1);
    
    /* 시간 문자열을 GstClockTime으로 변환 */
    gint h1, m1, s1, ms1;
    gint h2, m2, s2, ms2;
    if (sscanf(start_text, "%d:%d:%d.%d", &h1, &m1, &s1, &ms1) == 4 &&
        sscanf(end_text, "%d:%d:%d.%d", &h2, &m2, &s2, &ms2) == 4) {
      
      GstClockTime start_time = ((guint64)h1 * 3600 + m1 * 60 + s1) * GST_SECOND + ms1 * GST_MSECOND;
      GstClockTime end_time = ((guint64)h2 * 3600 + m2 * 60 + s2) * GST_SECOND + ms2 * GST_MSECOND;
      
      /* 세그먼트 제거 */
      blouedit_timeline_remove_prerender_segment(timeline, start_time, end_time);
      
      /* 목록에서 제거 */
      gtk_list_store_remove(GTK_LIST_STORE(model), &iter);
    }
    
    g_free(start_text);
    g_free(end_text);
  }
}

/* 프리렌더 세그먼트 관리 대화상자 */
void
blouedit_timeline_show_prerender_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 대화상자 생성 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "프리렌더 구간 설정",
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    "_닫기", GTK_RESPONSE_CLOSE,
    NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 500, 400);
  
  /* 콘텐츠 영역 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 입력 영역 프레임 */
  GtkWidget *input_frame = gtk_frame_new("새 프리렌더 구간 추가");
  gtk_container_add(GTK_CONTAINER(content_area), input_frame);
  
  /* 입력 그리드 */
  GtkWidget *input_grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(input_grid), 6);
  gtk_grid_set_column_spacing(GTK_GRID(input_grid), 12);
  gtk_container_set_border_width(GTK_CONTAINER(input_grid), 10);
  gtk_container_add(GTK_CONTAINER(input_frame), input_grid);
  
  /* 시작 시간 레이블 및 입력 필드 */
  GtkWidget *start_label = gtk_label_new("시작 시간:");
  gtk_widget_set_halign(start_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(input_grid), start_label, 0, 0, 1, 1);
  
  GtkWidget *start_entry = gtk_entry_new();
  gtk_entry_set_placeholder_text(GTK_ENTRY(start_entry), "00:00:00.000");
  gtk_grid_attach(GTK_GRID(input_grid), start_entry, 1, 0, 1, 1);
  
  /* 종료 시간 레이블 및 입력 필드 */
  GtkWidget *end_label = gtk_label_new("종료 시간:");
  gtk_widget_set_halign(end_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(input_grid), end_label, 0, 1, 1, 1);
  
  GtkWidget *end_entry = gtk_entry_new();
  gtk_entry_set_placeholder_text(GTK_ENTRY(end_entry), "00:00:00.000");
  gtk_grid_attach(GTK_GRID(input_grid), end_entry, 1, 1, 1, 1);
  
  /* 버튼 컨테이너 */
  GtkWidget *button_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout(GTK_BUTTON_BOX(button_box), GTK_BUTTONBOX_END);
  gtk_box_set_spacing(GTK_BOX(button_box), 6);
  gtk_grid_attach(GTK_GRID(input_grid), button_box, 0, 2, 2, 1);
  
  /* 인/아웃 포인트 설정 버튼 */
  GtkWidget *in_out_button = gtk_button_new_with_label("현재 선택 범위 사용");
  gtk_container_add(GTK_CONTAINER(button_box), in_out_button);
  
  /* 추가 버튼 */
  GtkWidget *add_button = gtk_button_new_with_label("구간 추가");
  gtk_container_add(GTK_CONTAINER(button_box), add_button);
  
  /* 세그먼트 목록 프레임 */
  GtkWidget *list_frame = gtk_frame_new("프리렌더 구간 목록");
  gtk_widget_set_vexpand(list_frame, TRUE);
  gtk_container_add(GTK_CONTAINER(content_area), list_frame);
  
  /* 세그먼트 목록 컨테이너 */
  GtkWidget *list_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
  gtk_container_set_border_width(GTK_CONTAINER(list_box), 10);
  gtk_container_add(GTK_CONTAINER(list_frame), list_box);
  
  /* 스크롤 윈도우 */
  GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                               GTK_POLICY_AUTOMATIC,
                               GTK_POLICY_AUTOMATIC);
  gtk_widget_set_vexpand(scroll, TRUE);
  gtk_container_add(GTK_CONTAINER(list_box), scroll);
  
  /* 세그먼트 목록 모델 */
  GtkListStore *store = gtk_list_store_new(3, G_TYPE_STRING, G_TYPE_STRING, G_TYPE_STRING);
  
  /* 기존 세그먼트 추가 */
  for (GList *l = timeline->prerender_segments; l != NULL; l = l->next) {
    BlouEditPrerenderSegment *segment = (BlouEditPrerenderSegment*)l->data;
    
    gchar *start_str = format_time(segment->start);
    gchar *end_str = format_time(segment->end);
    const gchar *status = segment->is_completed ? "완료됨" : 
                         (segment->is_active ? "처리 중" : "대기 중");
    
    GtkTreeIter iter;
    gtk_list_store_append(store, &iter);
    gtk_list_store_set(store, &iter,
                     0, start_str,
                     1, end_str,
                     2, status,
                     -1);
    
    g_free(start_str);
    g_free(end_str);
  }
  
  /* 세그먼트 목록 트리 뷰 */
  GtkWidget *tree_view = gtk_tree_view_new_with_model(GTK_TREE_MODEL(store));
  gtk_container_add(GTK_CONTAINER(scroll), tree_view);
  
  /* 열 설정 */
  GtkCellRenderer *renderer = gtk_cell_renderer_text_new();
  GtkTreeViewColumn *column = gtk_tree_view_column_new_with_attributes(
    "시작 시간", renderer, "text", 0, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes(
    "종료 시간", renderer, "text", 1, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes(
    "상태", renderer, "text", 2, NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  /* 선택 모드 설정 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(tree_view));
  gtk_tree_selection_set_mode(selection, GTK_SELECTION_SINGLE);
  
  /* 트리 뷰 아래 버튼 박스 */
  GtkWidget *list_button_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout(GTK_BUTTON_BOX(list_button_box), GTK_BUTTONBOX_END);
  gtk_box_set_spacing(GTK_BOX(list_button_box), 6);
  gtk_container_add(GTK_CONTAINER(list_box), list_button_box);
  
  /* 제거 버튼 */
  GtkWidget *remove_button = gtk_button_new_with_label("선택한 구간 제거");
  gtk_widget_set_sensitive(remove_button, FALSE);  /* 초기에는 비활성화 */
  gtk_container_add(GTK_CONTAINER(list_button_box), remove_button);
  
  /* 모두 제거 버튼 */
  GtkWidget *clear_button = gtk_button_new_with_label("모든 구간 제거");
  gtk_container_add(GTK_CONTAINER(list_button_box), clear_button);
  
  /* 하단 버튼 영역 */
  GtkWidget *action_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout(GTK_BUTTON_BOX(action_box), GTK_BUTTONBOX_CENTER);
  gtk_box_set_spacing(GTK_BOX(action_box), 12);
  gtk_container_add(GTK_CONTAINER(content_area), action_box);
  
  /* 프리렌더 시작 버튼 */
  GtkWidget *start_button = gtk_button_new_with_label("프리렌더 시작");
  gtk_container_add(GTK_CONTAINER(action_box), start_button);
  
  /* 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "start-entry", start_entry);
  g_object_set_data(G_OBJECT(dialog), "end-entry", end_entry);
  g_object_set_data(G_OBJECT(dialog), "segments-store", store);
  g_object_set_data(G_OBJECT(dialog), "segments-view", tree_view);
  g_object_set_data(G_OBJECT(dialog), "remove-button", remove_button);
  
  /* 시그널 핸들러 연결 */
  g_signal_connect(in_out_button, "clicked", G_CALLBACK(on_set_in_out_clicked), dialog);
  g_signal_connect(add_button, "clicked", G_CALLBACK(on_add_segment_clicked), dialog);
  g_signal_connect(remove_button, "clicked", G_CALLBACK(on_remove_segment_clicked), dialog);
  g_signal_connect(selection, "changed", G_CALLBACK(on_segment_selection_changed), dialog);
  g_signal_connect(clear_button, "clicked", G_CALLBACK(blouedit_timeline_clear_prerender_segments), timeline);
  g_signal_connect(start_button, "clicked", G_CALLBACK(blouedit_timeline_start_prerender), timeline);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 실행 */
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
} 