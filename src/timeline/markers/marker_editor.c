#include <gtk/gtk.h>
#include <string.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "markers.h"
#include "marker_color.h"
#include "marker_memo.h"

/* 마커 편집기 대화상자 응답 처리 콜백 */
static void
on_marker_editor_response(GtkDialog *dialog, gint response_id, gpointer user_data)
{
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  BlouEditTimelineMarker *marker = g_object_get_data(G_OBJECT(dialog), "marker");
  
  if (response_id == GTK_RESPONSE_OK) {
    /* 위젯에서 값 가져오기 */
    GtkWidget *name_entry = g_object_get_data(G_OBJECT(dialog), "name-entry");
    GtkWidget *comment_entry = g_object_get_data(G_OBJECT(dialog), "comment-entry");
    GtkWidget *type_combo = g_object_get_data(G_OBJECT(dialog), "type-combo");
    
    const gchar *name = gtk_entry_get_text(GTK_ENTRY(name_entry));
    const gchar *comment = gtk_entry_get_text(GTK_ENTRY(comment_entry));
    gint active = gtk_combo_box_get_active(GTK_COMBO_BOX(type_combo));
    
    BlouEditMarkerType type = (BlouEditMarkerType)active;
    
    /* 마커 업데이트 */
    blouedit_timeline_update_marker(timeline, marker, marker->position, type, name, comment);
    
    /* 마커 타입에 맞는 색상 설정 */
    blouedit_timeline_set_marker_color_by_type(timeline, marker);
    
    /* 타임라인 재그리기 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
  
  /* 대화상자 닫기 */
  gtk_widget_destroy(GTK_WIDGET(dialog));
}

/* 마커 타입 콤보박스 생성 함수 */
static GtkWidget*
create_marker_type_combo(BlouEditMarkerType current_type)
{
  GtkWidget *combo = gtk_combo_box_text_new();
  
  /* 마커 타입 옵션 추가 */
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "일반");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "큐 포인트");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "시작 지점");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "종료 지점");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "챕터");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "오류");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "경고");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo), "코멘트");
  
  /* 현재 타입 선택 */
  gtk_combo_box_set_active(GTK_COMBO_BOX(combo), current_type);
  
  return combo;
}

/* 색상 버튼 클릭 콜백 */
static void
on_color_button_clicked(GtkButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(button), "timeline");
  BlouEditTimelineMarker *marker = g_object_get_data(G_OBJECT(button), "marker");
  
  blouedit_timeline_show_marker_color_dialog(timeline, marker);
}

/* 마커 편집기 표시 함수 */
void
blouedit_timeline_show_marker_editor(BlouEditTimeline *timeline, BlouEditTimelineMarker *marker)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(marker != NULL);
  
  GtkWidget *dialog, *content_area, *grid;
  GtkWidget *name_label, *name_entry;
  GtkWidget *comment_label, *comment_entry;
  GtkWidget *type_label, *type_combo;
  GtkWidget *position_label, *position_value;
  GtkWidget *color_button;
  GtkWidget *button_box;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("마커 편집",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                     "_취소", GTK_RESPONSE_CANCEL,
                                     "_확인", GTK_RESPONSE_OK,
                                     NULL);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  
  /* 그리드 생성 */
  grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
  gtk_grid_set_column_spacing(GTK_GRID(grid), 12);
  gtk_container_set_border_width(GTK_CONTAINER(grid), 12);
  gtk_container_add(GTK_CONTAINER(content_area), grid);
  
  /* 위치 표시 */
  gchar *position_text = blouedit_timeline_position_to_timecode(timeline, 
                                                         marker->position, 
                                                         blouedit_timeline_get_timecode_format(timeline));
  
  position_label = gtk_label_new("위치:");
  gtk_widget_set_halign(position_label, GTK_ALIGN_END);
  position_value = gtk_label_new(position_text);
  gtk_widget_set_halign(position_value, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), position_label, 0, 0, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), position_value, 1, 0, 1, 1);
  g_free(position_text);
  
  /* 이름 입력 */
  name_label = gtk_label_new("이름:");
  gtk_widget_set_halign(name_label, GTK_ALIGN_END);
  name_entry = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(name_entry), marker->name ? marker->name : "");
  gtk_grid_attach(GTK_GRID(grid), name_label, 0, 1, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), name_entry, 1, 1, 1, 1);
  
  /* 코멘트 입력 */
  comment_label = gtk_label_new("코멘트:");
  gtk_widget_set_halign(comment_label, GTK_ALIGN_END);
  comment_entry = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(comment_entry), marker->comment ? marker->comment : "");
  gtk_grid_attach(GTK_GRID(grid), comment_label, 0, 2, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), comment_entry, 1, 2, 1, 1);
  
  /* 마커 타입 선택 */
  type_label = gtk_label_new("유형:");
  gtk_widget_set_halign(type_label, GTK_ALIGN_END);
  type_combo = create_marker_type_combo(marker->type);
  gtk_grid_attach(GTK_GRID(grid), type_label, 0, 3, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), type_combo, 1, 3, 1, 1);
  
  /* 색상 버튼 */
  color_button = gtk_button_new_with_label("색상 선택");
  gtk_grid_attach(GTK_GRID(grid), color_button, 1, 4, 1, 1);
  g_object_set_data(G_OBJECT(color_button), "timeline", timeline);
  g_object_set_data(G_OBJECT(color_button), "marker", marker);
  g_signal_connect(color_button, "clicked", G_CALLBACK(on_color_button_clicked), NULL);
  
  /* 버튼 박스 (추가 동작 버튼용) */
  button_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout(GTK_BUTTON_BOX(button_box), GTK_BUTTONBOX_START);
  gtk_grid_attach(GTK_GRID(grid), button_box, 0, 5, 2, 1);
  
  /* 상세 메모 버튼 추가 */
  GtkWidget *memo_button = gtk_button_new_with_label("상세 메모 편집");
  gtk_container_add(GTK_CONTAINER(button_box), memo_button);
  g_object_set_data(G_OBJECT(memo_button), "timeline", timeline);
  g_object_set_data(G_OBJECT(memo_button), "marker", marker);
  g_signal_connect(memo_button, "clicked", 
                 G_CALLBACK(blouedit_timeline_show_marker_memo_editor), 
                 timeline);
  
  /* 필요한 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "marker", marker);
  g_object_set_data(G_OBJECT(dialog), "name-entry", name_entry);
  g_object_set_data(G_OBJECT(dialog), "comment-entry", comment_entry);
  g_object_set_data(G_OBJECT(dialog), "type-combo", type_combo);
  g_object_set_data(G_OBJECT(dialog), "button-box", button_box);
  
  /* 응답 핸들러 설정 */
  g_signal_connect(dialog, "response", G_CALLBACK(on_marker_editor_response), NULL);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
} 