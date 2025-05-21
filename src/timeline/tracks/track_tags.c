#include <gtk/gtk.h>
#include <string.h>
#include "track_tags.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 트랙 태그 설정 함수 */
void 
blouedit_timeline_set_track_tags(BlouEditTimeline *timeline, 
                              BlouEditTimelineTrack *track, 
                              const gchar **tags)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(track != NULL);
  
  /* 기존 태그 해제 */
  if (track->tags != NULL) {
    g_strfreev(track->tags);
    track->tags = NULL;
  }
  
  /* 새 태그 복사 */
  if (tags != NULL) {
    gint count = 0;
    while (tags[count] != NULL) count++;
    
    track->tags = g_new0(gchar*, count + 1);
    for (gint i = 0; i < count; i++) {
      track->tags[i] = g_strdup(tags[i]);
    }
    track->tags[count] = NULL;
  }
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 트랙 태그 가져오기 함수 */
const gchar** 
blouedit_timeline_get_track_tags(BlouEditTimeline *timeline, 
                               BlouEditTimelineTrack *track)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  g_return_val_if_fail(track != NULL, NULL);
  
  return (const gchar**)track->tags;
}

/* 트랙에 태그 추가 함수 */
void 
blouedit_timeline_add_track_tag(BlouEditTimeline *timeline, 
                              BlouEditTimelineTrack *track, 
                              const gchar *tag)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(track != NULL);
  g_return_if_fail(tag != NULL);
  
  /* 이미 해당 태그가 있는지 확인 */
  if (blouedit_timeline_track_has_tag(track, tag)) {
    return;
  }
  
  /* 기존 태그 배열 길이 계산 */
  gint count = 0;
  if (track->tags != NULL) {
    while (track->tags[count] != NULL) count++;
  }
  
  /* 새 태그 배열 생성 */
  gchar **new_tags = g_new0(gchar*, count + 2);
  
  /* 기존 태그 복사 */
  for (gint i = 0; i < count; i++) {
    new_tags[i] = g_strdup(track->tags[i]);
  }
  
  /* 새 태그 추가 */
  new_tags[count] = g_strdup(tag);
  new_tags[count + 1] = NULL;
  
  /* 기존 태그 배열 해제 */
  if (track->tags != NULL) {
    g_strfreev(track->tags);
  }
  
  /* 새 태그 배열 설정 */
  track->tags = new_tags;
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 트랙에서 태그 제거 함수 */
void 
blouedit_timeline_remove_track_tag(BlouEditTimeline *timeline, 
                                 BlouEditTimelineTrack *track, 
                                 const gchar *tag)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(track != NULL);
  g_return_if_fail(tag != NULL);
  
  /* 태그가 없으면 아무것도 하지 않음 */
  if (track->tags == NULL) {
    return;
  }
  
  /* 태그 배열 길이 계산 */
  gint count = 0;
  while (track->tags[count] != NULL) count++;
  
  /* 제거할 태그 위치 찾기 */
  gint tag_index = -1;
  for (gint i = 0; i < count; i++) {
    if (g_strcmp0(track->tags[i], tag) == 0) {
      tag_index = i;
      break;
    }
  }
  
  /* 태그가 없으면 아무것도 하지 않음 */
  if (tag_index == -1) {
    return;
  }
  
  /* 새 태그 배열 생성 (1개 줄임) */
  gchar **new_tags = g_new0(gchar*, count);
  
  /* 태그 위치 이전까지 복사 */
  for (gint i = 0; i < tag_index; i++) {
    new_tags[i] = g_strdup(track->tags[i]);
  }
  
  /* 태그 위치 이후부터 복사 */
  for (gint i = tag_index + 1; i < count; i++) {
    new_tags[i - 1] = g_strdup(track->tags[i]);
  }
  
  /* 기존 태그 배열 해제 */
  g_strfreev(track->tags);
  
  /* 새 태그 배열 설정 */
  track->tags = new_tags;
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 트랙이 특정 태그를 가지고 있는지 확인하는 함수 */
gboolean 
blouedit_timeline_track_has_tag(BlouEditTimelineTrack *track, const gchar *tag)
{
  g_return_val_if_fail(track != NULL, FALSE);
  g_return_val_if_fail(tag != NULL, FALSE);
  
  if (track->tags == NULL) {
    return FALSE;
  }
  
  for (gint i = 0; track->tags[i] != NULL; i++) {
    if (g_strcmp0(track->tags[i], tag) == 0) {
      return TRUE;
    }
  }
  
  return FALSE;
}

/* 태그로 트랙 필터링 함수 */
void 
blouedit_timeline_filter_tracks_by_tag(BlouEditTimeline *timeline, const gchar *tag)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 필터링 상태 설정 */
  if (timeline->track_filter_tag != NULL) {
    g_free(timeline->track_filter_tag);
    timeline->track_filter_tag = NULL;
  }
  
  if (tag != NULL && tag[0] != '\0') {
    timeline->track_filter_tag = g_strdup(tag);
    timeline->track_filtering_active = TRUE;
  } else {
    timeline->track_filtering_active = FALSE;
  }
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 트랙 필터 초기화 함수 */
void 
blouedit_timeline_clear_track_filter(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 필터링 상태 초기화 */
  if (timeline->track_filter_tag != NULL) {
    g_free(timeline->track_filter_tag);
    timeline->track_filter_tag = NULL;
  }
  
  timeline->track_filtering_active = FALSE;
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 태그 칩 생성 함수 - 내부 사용 */
static GtkWidget*
create_tag_chip(const gchar *tag, gpointer user_data)
{
  GtkWidget *box, *label, *button;
  GtkStyleContext *context;
  
  /* 박스 생성 */
  box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 3);
  gtk_widget_set_margin_start(box, 2);
  gtk_widget_set_margin_end(box, 2);
  gtk_widget_set_margin_top(box, 2);
  gtk_widget_set_margin_bottom(box, 2);
  
  /* 스타일 설정 */
  context = gtk_widget_get_style_context(box);
  gtk_style_context_add_class(context, "tag-chip");
  
  /* 레이블 생성 */
  label = gtk_label_new(tag);
  gtk_widget_set_margin_start(label, 5);
  gtk_container_add(GTK_CONTAINER(box), label);
  
  /* 버튼 생성 */
  button = gtk_button_new_from_icon_name("window-close-symbolic", GTK_ICON_SIZE_MENU);
  gtk_button_set_relief(GTK_BUTTON(button), GTK_RELIEF_NONE);
  gtk_container_add(GTK_CONTAINER(box), button);
  
  /* 버튼 데이터 설정 */
  g_object_set_data_full(G_OBJECT(button), "tag", g_strdup(tag), g_free);
  g_object_set_data(G_OBJECT(button), "user-data", user_data);
  
  /* 박스 표시 */
  gtk_widget_show_all(box);
  
  return box;
}

/* 태그 추가 버튼 클릭 핸들러 */
static void
on_add_tag_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  BlouEditTimelineTrack *track = g_object_get_data(G_OBJECT(dialog), "track");
  GtkWidget *entry = g_object_get_data(G_OBJECT(dialog), "tag-entry");
  GtkWidget *flow_box = g_object_get_data(G_OBJECT(dialog), "tag-flow");
  
  /* 태그 가져오기 */
  const gchar *tag = gtk_entry_get_text(GTK_ENTRY(entry));
  
  /* 태그가 비어있지 않고 중복이 아니면 추가 */
  if (tag != NULL && tag[0] != '\0' && !blouedit_timeline_track_has_tag(track, tag)) {
    /* 태그 추가 */
    blouedit_timeline_add_track_tag(timeline, track, tag);
    
    /* 태그 칩 생성 */
    GtkWidget *chip = create_tag_chip(tag, dialog);
    gtk_container_add(GTK_CONTAINER(flow_box), chip);
    
    /* 입력 필드 지우기 */
    gtk_entry_set_text(GTK_ENTRY(entry), "");
  }
}

/* 태그 제거 버튼 클릭 핸들러 */
static void
on_remove_tag_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = g_object_get_data(G_OBJECT(button), "user-data");
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  BlouEditTimelineTrack *track = g_object_get_data(G_OBJECT(dialog), "track");
  const gchar *tag = g_object_get_data(G_OBJECT(button), "tag");
  
  /* 태그 제거 */
  blouedit_timeline_remove_track_tag(timeline, track, tag);
  
  /* 칩 위젯 제거 */
  GtkWidget *chip = gtk_widget_get_parent(GTK_WIDGET(button));
  gtk_widget_destroy(chip);
}

/* 트랙 태그 관리 대화상자 표시 함수 */
void 
blouedit_timeline_show_track_tags_dialog(BlouEditTimeline *timeline, 
                                      BlouEditTimelineTrack *track)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(track != NULL);
  
  GtkWidget *dialog, *content_area, *box, *label, *flow_box;
  GtkWidget *entry_box, *entry, *add_button;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("트랙 태그 관리",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_취소", GTK_RESPONSE_CANCEL,
                                     "_확인", GTK_RESPONSE_ACCEPT,
                                     NULL);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 안내 레이블 */
  label = gtk_label_new("트랙 태그를 관리합니다. 태그를 사용하여 트랙을 구성하고 필터링할 수 있습니다.");
  gtk_label_set_line_wrap(GTK_LABEL(label), TRUE);
  gtk_container_add(GTK_CONTAINER(content_area), label);
  
  /* 태그 입력 박스 */
  entry_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  gtk_container_add(GTK_CONTAINER(content_area), entry_box);
  
  /* 입력 필드 */
  entry = gtk_entry_new();
  gtk_entry_set_placeholder_text(GTK_ENTRY(entry), "새 태그 입력");
  gtk_widget_set_hexpand(entry, TRUE);
  gtk_container_add(GTK_CONTAINER(entry_box), entry);
  
  /* 추가 버튼 */
  add_button = gtk_button_new_with_label("추가");
  gtk_container_add(GTK_CONTAINER(entry_box), add_button);
  
  /* 흐름 상자 (태그 표시용) */
  box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_container_add(GTK_CONTAINER(content_area), box);
  
  /* 레이블 */
  label = gtk_label_new("현재 태그:");
  gtk_widget_set_halign(label, GTK_ALIGN_START);
  gtk_container_add(GTK_CONTAINER(box), label);
  
  /* 흐름 상자 */
  flow_box = gtk_flow_box_new();
  gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(flow_box), GTK_SELECTION_NONE);
  gtk_flow_box_set_max_children_per_line(GTK_FLOW_BOX(flow_box), 5);
  gtk_widget_set_hexpand(flow_box, TRUE);
  gtk_widget_set_vexpand(flow_box, TRUE);
  gtk_container_add(GTK_CONTAINER(box), flow_box);
  
  /* 기존 태그 표시 */
  const gchar **tags = blouedit_timeline_get_track_tags(timeline, track);
  if (tags != NULL) {
    for (gint i = 0; tags[i] != NULL; i++) {
      GtkWidget *chip = create_tag_chip(tags[i], dialog);
      gtk_container_add(GTK_CONTAINER(flow_box), chip);
    }
  }
  
  /* 데이터 설정 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "track", track);
  g_object_set_data(G_OBJECT(dialog), "tag-entry", entry);
  g_object_set_data(G_OBJECT(dialog), "tag-flow", flow_box);
  
  /* 시그널 연결 */
  g_signal_connect(add_button, "clicked", G_CALLBACK(on_add_tag_clicked), dialog);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    /* 태그 변경이 이미 적용됨 */
  }
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
} 