#include <gtk/gtk.h>
#include <string.h>
#include <json-glib/json-glib.h>
#include "track_tags.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 기본 미디어 구성 규칙 */
static const gchar *DEFAULT_ORGANIZATION_RULES = 
"{\n"
"  \"rules\": [\n"
"    {\n"
"      \"media_type\": \"video\",\n"
"      \"track_name\": \"비디오 트랙\",\n"
"      \"tags\": [\"비디오\", \"주요\"],\n"
"      \"color\": \"rgba(51,153,255,0.7)\"\n"
"    },\n"
"    {\n"
"      \"media_type\": \"audio\",\n"
"      \"track_name\": \"오디오 트랙\",\n"
"      \"tags\": [\"오디오\", \"주요\"],\n"
"      \"color\": \"rgba(51,204,51,0.7)\"\n"
"    },\n"
"    {\n"
"      \"media_type\": \"image\",\n"
"      \"track_name\": \"이미지 트랙\",\n"
"      \"tags\": [\"이미지\"],\n"
"      \"color\": \"rgba(255,153,51,0.7)\"\n"
"    },\n"
"    {\n"
"      \"media_type\": \"text\",\n"
"      \"track_name\": \"텍스트 트랙\",\n"
"      \"tags\": [\"텍스트\"],\n"
"      \"color\": \"rgba(204,51,204,0.7)\"\n"
"    },\n"
"    {\n"
"      \"media_type\": \"effect\",\n"
"      \"track_name\": \"효과 트랙\",\n"
"      \"tags\": [\"효과\"],\n"
"      \"color\": \"rgba(204,204,51,0.7)\"\n"
"    },\n"
"    {\n"
"      \"media_type\": \"transition\",\n"
"      \"track_name\": \"전환 트랙\",\n"
"      \"tags\": [\"전환\"],\n"
"      \"color\": \"rgba(255,51,51,0.7)\"\n"
"    }\n"
"  ]\n"
"}";

/* 미디어 타입에서 트랙 타입으로 변환 */
static GESTrackType
get_track_type_from_media_type(const gchar *media_type)
{
  if (g_strcmp0(media_type, "video") == 0) {
    return GES_TRACK_TYPE_VIDEO;
  }
  else if (g_strcmp0(media_type, "audio") == 0) {
    return GES_TRACK_TYPE_AUDIO;
  }
  else if (g_strcmp0(media_type, "image") == 0) {
    return GES_TRACK_TYPE_VIDEO;  /* 이미지는 비디오 트랙에 표시됨 */
  }
  else if (g_strcmp0(media_type, "text") == 0) {
    return GES_TRACK_TYPE_VIDEO;  /* 텍스트는 비디오 트랙에 표시됨 */
  }
  else if (g_strcmp0(media_type, "effect") == 0) {
    return GES_TRACK_TYPE_VIDEO;  /* 효과는 비디오 트랙에 표시됨 */
  }
  else if (g_strcmp0(media_type, "transition") == 0) {
    return GES_TRACK_TYPE_VIDEO;  /* 전환은 비디오 트랙에 표시됨 */
  }
  
  return GES_TRACK_TYPE_UNKNOWN;
}

/* 미디어 구성 규칙 가져오기 */
const gchar* 
blouedit_timeline_get_media_track_organization_rules(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  if (timeline->media_organization_rules == NULL) {
    return DEFAULT_ORGANIZATION_RULES;
  }
  
  return timeline->media_organization_rules;
}

/* 미디어 구성 규칙 설정 */
void 
blouedit_timeline_set_media_track_organization_rules(BlouEditTimeline *timeline, 
                                                  const gchar *rules_json)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 기존 규칙 해제 */
  if (timeline->media_organization_rules != NULL) {
    g_free(timeline->media_organization_rules);
    timeline->media_organization_rules = NULL;
  }
  
  /* 새 규칙 설정 */
  if (rules_json != NULL) {
    timeline->media_organization_rules = g_strdup(rules_json);
  }
}

/* 미디어 타입별 트랙 자동 구성 */
void 
blouedit_timeline_auto_organize_tracks_by_media(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  const gchar *rules_json = blouedit_timeline_get_media_track_organization_rules(timeline);
  
  /* JSON 파싱 */
  JsonParser *parser = json_parser_new();
  GError *error = NULL;
  json_parser_load_from_data(parser, rules_json, -1, &error);
  
  if (error) {
    g_warning("미디어 구성 규칙 파싱 오류: %s", error->message);
    g_error_free(error);
    g_object_unref(parser);
    return;
  }
  
  /* 루트 객체 가져오기 */
  JsonNode *root = json_parser_get_root(parser);
  JsonObject *root_obj = json_node_get_object(root);
  
  /* 규칙 배열 가져오기 */
  JsonArray *rules_array = json_object_get_array_member(root_obj, "rules");
  if (rules_array == NULL) {
    g_warning("미디어 구성 규칙에서 'rules' 배열을 찾을 수 없습니다");
    g_object_unref(parser);
    return;
  }
  
  /* 현재 트랙 제거 (사용자 확인 필요) */
  GList *tracks_to_remove = g_list_copy(timeline->tracks);
  for (GList *l = tracks_to_remove; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack*)l->data;
    blouedit_timeline_remove_track(timeline, track);
  }
  g_list_free(tracks_to_remove);
  
  /* 규칙에 따라 트랙 생성 */
  guint rules_len = json_array_get_length(rules_array);
  for (guint i = 0; i < rules_len; i++) {
    JsonObject *rule_obj = json_array_get_object_element(rules_array, i);
    
    /* 미디어 타입 */
    const gchar *media_type = json_object_get_string_member(rule_obj, "media_type");
    GESTrackType track_type = get_track_type_from_media_type(media_type);
    
    /* 트랙 이름 */
    const gchar *track_name = json_object_get_string_member(rule_obj, "track_name");
    
    /* 새 트랙 생성 */
    BlouEditTimelineTrack *new_track = blouedit_timeline_add_track(timeline, track_type, track_name);
    
    /* 색상 설정 */
    if (json_object_has_member(rule_obj, "color")) {
      const gchar *color_str = json_object_get_string_member(rule_obj, "color");
      GdkRGBA color;
      
      if (gdk_rgba_parse(&color, color_str)) {
        new_track->color = color;
      }
    }
    
    /* 태그 설정 */
    if (json_object_has_member(rule_obj, "tags")) {
      JsonArray *tags_array = json_object_get_array_member(rule_obj, "tags");
      guint tags_len = json_array_get_length(tags_array);
      
      for (guint j = 0; j < tags_len; j++) {
        const gchar *tag = json_array_get_string_element(tags_array, j);
        blouedit_timeline_add_track_tag(timeline, new_track, tag);
      }
    }
  }
  
  /* 메모리 해제 */
  g_object_unref(parser);
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 미디어 트랙 구성 대화상자 */
void 
blouedit_timeline_show_media_track_organization_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *scroll, *text_view;
  GtkTextBuffer *buffer;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("미디어 타입별 트랙 자동 구성",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_취소", GTK_RESPONSE_CANCEL,
                                     "_적용", GTK_RESPONSE_ACCEPT,
                                     NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 600, 500);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 스크롤 윈도우 생성 */
  scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_widget_set_hexpand(scroll, TRUE);
  gtk_widget_set_vexpand(scroll, TRUE);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                               GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_container_add(GTK_CONTAINER(content_area), scroll);
  
  /* 텍스트 뷰 생성 */
  text_view = gtk_text_view_new();
  gtk_text_view_set_monospace(GTK_TEXT_VIEW(text_view), TRUE);
  gtk_container_add(GTK_CONTAINER(scroll), text_view);
  
  /* 텍스트 버퍼 가져오기 */
  buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(text_view));
  
  /* 기존 JSON 설정 */
  const gchar *rules_json = blouedit_timeline_get_media_track_organization_rules(timeline);
  gtk_text_buffer_set_text(buffer, rules_json, -1);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    /* 텍스트 가져오기 */
    GtkTextIter start, end;
    gtk_text_buffer_get_bounds(buffer, &start, &end);
    gchar *text = gtk_text_buffer_get_text(buffer, &start, &end, FALSE);
    
    /* JSON 유효성 검사 */
    JsonParser *parser = json_parser_new();
    GError *error = NULL;
    
    if (json_parser_load_from_data(parser, text, -1, &error)) {
      /* 규칙 설정 */
      blouedit_timeline_set_media_track_organization_rules(timeline, text);
      
      /* 자동 구성 적용 */
      blouedit_timeline_auto_organize_tracks_by_media(timeline);
    } else {
      /* 오류 표시 */
      GtkWidget *error_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                                    GTK_DIALOG_MODAL,
                                                    GTK_MESSAGE_ERROR,
                                                    GTK_BUTTONS_OK,
                                                    "JSON 파싱 오류: %s",
                                                    error->message);
      gtk_dialog_run(GTK_DIALOG(error_dialog));
      gtk_widget_destroy(error_dialog);
      g_error_free(error);
    }
    
    g_object_unref(parser);
    g_free(text);
  }
  
  /* 대화상자 닫기 */
  gtk_widget_destroy(dialog);
} 