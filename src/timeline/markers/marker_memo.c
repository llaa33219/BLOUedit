#include <gtk/gtk.h>
#include <string.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "markers.h"

/* 마커 상세 메모 설정 함수 */
void 
blouedit_timeline_set_marker_detailed_memo (BlouEditTimeline *timeline, 
                                         BlouEditTimelineMarker *marker, 
                                         const gchar *detailed_memo)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(marker != NULL);
  
  /* 기존 메모 해제 */
  g_free(marker->detailed_memo);
  
  /* 새 메모 설정 */
  marker->detailed_memo = g_strdup(detailed_memo);
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  /* 마커 업데이트 시그널 발생 */
  g_signal_emit_by_name(timeline, "marker-updated", marker);
}

/* 마커 상세 메모 가져오기 함수 */
const gchar* 
blouedit_timeline_get_marker_detailed_memo (BlouEditTimelineMarker *marker)
{
  g_return_val_if_fail(marker != NULL, NULL);
  
  return marker->detailed_memo;
}

/* 마커 메모 에디터 창 응답 처리 콜백 */
static void
on_memo_editor_response(GtkDialog *dialog, gint response_id, gpointer user_data)
{
  if (response_id == GTK_RESPONSE_ACCEPT) {
    /* 사용자 데이터에서 필요한 정보 추출 */
    GtkTextBuffer *buffer = g_object_get_data(G_OBJECT(dialog), "buffer");
    BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
    BlouEditTimelineMarker *marker = g_object_get_data(G_OBJECT(dialog), "marker");
    
    GtkTextIter start, end;
    gtk_text_buffer_get_bounds(buffer, &start, &end);
    gchar *text = gtk_text_buffer_get_text(buffer, &start, &end, FALSE);
    
    /* 마커 메모 업데이트 */
    blouedit_timeline_set_marker_detailed_memo(timeline, marker, text);
    
    g_free(text);
  }
  
  /* 대화상자 닫기 */
  gtk_widget_destroy(GTK_WIDGET(dialog));
}

/* 마커 메모 에디터 표시 함수 */
void 
blouedit_timeline_show_marker_memo_editor (BlouEditTimeline *timeline, 
                                        BlouEditTimelineMarker *marker)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(marker != NULL);
  
  GtkWidget *dialog, *content_area, *scrolled_window, *text_view;
  GtkTextBuffer *buffer;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("마커 상세 메모 편집",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_취소", GTK_RESPONSE_CANCEL,
                                     "_저장", GTK_RESPONSE_ACCEPT,
                                     NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 600, 400);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  
  /* 스크롤 윈도우 생성 */
  scrolled_window = gtk_scrolled_window_new(NULL, NULL);
  gtk_widget_set_hexpand(scrolled_window, TRUE);
  gtk_widget_set_vexpand(scrolled_window, TRUE);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window),
                               GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_container_add(GTK_CONTAINER(content_area), scrolled_window);
  
  /* 텍스트 뷰 생성 */
  text_view = gtk_text_view_new();
  gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(text_view), GTK_WRAP_WORD);
  gtk_container_add(GTK_CONTAINER(scrolled_window), text_view);
  
  /* 텍스트 버퍼 가져오기 */
  buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(text_view));
  
  /* 기존 메모 텍스트 설정 */
  if (marker->detailed_memo) {
    gtk_text_buffer_set_text(buffer, marker->detailed_memo, -1);
  }
  
  /* 응답 핸들러 설정 */
  g_signal_connect(dialog, "response", G_CALLBACK(on_memo_editor_response), NULL);
  
  /* 필요한 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "buffer", buffer);
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "marker", marker);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
}

/* 마커 편집기에 메모 버튼 추가하는 함수 */
void
blouedit_timeline_add_memo_button_to_marker_editor(GtkWidget *editor, 
                                                BlouEditTimeline *timeline, 
                                                BlouEditTimelineMarker *marker)
{
  GtkWidget *button_box, *memo_button;
  
  /* 버튼 박스 찾기 */
  button_box = g_object_get_data(G_OBJECT(editor), "button-box");
  if (!button_box) {
    return;
  }
  
  /* 메모 버튼 생성 */
  memo_button = gtk_button_new_with_label("상세 메모 편집");
  gtk_container_add(GTK_CONTAINER(button_box), memo_button);
  
  /* 버튼 클릭 핸들러 */
  g_signal_connect_swapped(memo_button, "clicked",
                         G_CALLBACK(blouedit_timeline_show_marker_memo_editor),
                         timeline);
  
  /* 마커 데이터 설정 */
  g_object_set_data(G_OBJECT(memo_button), "marker", marker);
  
  /* 버튼 표시 */
  gtk_widget_show(memo_button);
} 