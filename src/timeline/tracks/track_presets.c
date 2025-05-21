#include <gtk/gtk.h>
#include <string.h>
#include <json-glib/json-glib.h>
#include "track_tags.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 프리셋 저장 경로 */
#define BLOUEDIT_TRACK_PRESETS_DIR "~/.config/blouedit/track_presets"

/* 트랙 프리셋 구조체 - 내부 사용 */
typedef struct {
  gchar *name;
  gchar *description;
  GList *tracks;  /* BlouEditTrackInfo 객체 목록 */
} BlouEditTrackPreset;

/* 트랙 정보 구조체 - 내부 사용 */
typedef struct {
  GESTrackType type;
  gchar *name;
  gchar **tags;
  GdkRGBA color;
} BlouEditTrackInfo;

/* 트랙 정보 생성 함수 - 내부 사용 */
static BlouEditTrackInfo*
track_info_new(BlouEditTimelineTrack *track)
{
  BlouEditTrackInfo *info = g_new0(BlouEditTrackInfo, 1);
  
  info->type = track->type;
  info->name = g_strdup(track->name);
  
  /* 태그 복사 */
  if (track->tags != NULL) {
    gint count = 0;
    while (track->tags[count] != NULL) count++;
    
    info->tags = g_new0(gchar*, count + 1);
    for (gint i = 0; i < count; i++) {
      info->tags[i] = g_strdup(track->tags[i]);
    }
    info->tags[count] = NULL;
  }
  
  /* 색상 복사 */
  info->color = track->color;
  
  return info;
}

/* 트랙 정보 해제 함수 - 내부 사용 */
static void
track_info_free(BlouEditTrackInfo *info)
{
  if (info == NULL) {
    return;
  }
  
  g_free(info->name);
  
  if (info->tags != NULL) {
    g_strfreev(info->tags);
  }
  
  g_free(info);
}

/* 트랙 프리셋 해제 함수 - 내부 사용 */
static void
track_preset_free(BlouEditTrackPreset *preset)
{
  if (preset == NULL) {
    return;
  }
  
  g_free(preset->name);
  g_free(preset->description);
  
  /* 트랙 정보 목록 해제 */
  g_list_free_full(preset->tracks, (GDestroyNotify)track_info_free);
  
  g_free(preset);
}

/* 프리셋 디렉토리 확인 및 생성 함수 - 내부 사용 */
static gchar*
ensure_presets_dir(void)
{
  gchar *dir_path = g_strdup_printf("%s", BLOUEDIT_TRACK_PRESETS_DIR);
  gchar *expanded_path = NULL;
  
  /* 경로의 ~ 확장 */
  if (dir_path[0] == '~') {
    const gchar *home = g_get_home_dir();
    expanded_path = g_strdup_printf("%s%s", home, dir_path + 1);
    g_free(dir_path);
    dir_path = expanded_path;
  }
  
  /* 디렉토리 존재 확인 */
  if (!g_file_test(dir_path, G_FILE_TEST_IS_DIR)) {
    /* 디렉토리 생성 */
    g_mkdir_with_parents(dir_path, 0755);
  }
  
  return dir_path;
}

/* 트랙 프리셋 저장 함수 */
void 
blouedit_timeline_save_track_preset(BlouEditTimeline *timeline, const gchar *preset_name)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(preset_name != NULL && preset_name[0] != '\0');
  
  /* 프리셋 디렉토리 확인 */
  gchar *dir_path = ensure_presets_dir();
  gchar *file_path = g_strdup_printf("%s/%s.json", dir_path, preset_name);
  
  /* JSON 생성 */
  JsonBuilder *builder = json_builder_new();
  
  /* 루트 객체 시작 */
  json_builder_begin_object(builder);
  
  /* 프리셋 이름 */
  json_builder_set_member_name(builder, "name");
  json_builder_add_string_value(builder, preset_name);
  
  /* 프리셋 생성 시간 */
  json_builder_set_member_name(builder, "created");
  GDateTime *now = g_date_time_new_now_local();
  gchar *date_str = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
  json_builder_add_string_value(builder, date_str);
  g_free(date_str);
  g_date_time_unref(now);
  
  /* 트랙 배열 시작 */
  json_builder_set_member_name(builder, "tracks");
  json_builder_begin_array(builder);
  
  /* 모든 트랙 정보 저장 */
  for (GList *l = timeline->tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack*)l->data;
    
    /* 트랙 객체 시작 */
    json_builder_begin_object(builder);
    
    /* 트랙 유형 */
    json_builder_set_member_name(builder, "type");
    json_builder_add_int_value(builder, track->type);
    
    /* 트랙 이름 */
    json_builder_set_member_name(builder, "name");
    json_builder_add_string_value(builder, track->name ? track->name : "");
    
    /* 트랙 색상 */
    json_builder_set_member_name(builder, "color");
    gchar *color_str = g_strdup_printf("rgba(%d,%d,%d,%f)",
                                     (int)(track->color.red * 255),
                                     (int)(track->color.green * 255),
                                     (int)(track->color.blue * 255),
                                     track->color.alpha);
    json_builder_add_string_value(builder, color_str);
    g_free(color_str);
    
    /* 트랙 태그 배열 */
    json_builder_set_member_name(builder, "tags");
    json_builder_begin_array(builder);
    
    if (track->tags != NULL) {
      for (gint i = 0; track->tags[i] != NULL; i++) {
        json_builder_add_string_value(builder, track->tags[i]);
      }
    }
    
    /* 태그 배열 종료 */
    json_builder_end_array(builder);
    
    /* 트랙 객체 종료 */
    json_builder_end_object(builder);
  }
  
  /* 트랙 배열 종료 */
  json_builder_end_array(builder);
  
  /* 루트 객체 종료 */
  json_builder_end_object(builder);
  
  /* JSON 생성 */
  JsonGenerator *generator = json_generator_new();
  JsonNode *root = json_builder_get_root(builder);
  json_generator_set_root(generator, root);
  json_generator_set_pretty(generator, TRUE);
  
  /* 파일에 저장 */
  GError *error = NULL;
  json_generator_to_file(generator, file_path, &error);
  
  if (error) {
    g_warning("프리셋 저장 오류: %s", error->message);
    g_error_free(error);
  }
  
  /* 메모리 해제 */
  json_node_free(root);
  g_object_unref(generator);
  g_object_unref(builder);
  g_free(file_path);
  g_free(dir_path);
}

/* 트랙 프리셋 적용 함수 */
void 
blouedit_timeline_apply_track_preset(BlouEditTimeline *timeline, const gchar *preset_name)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(preset_name != NULL && preset_name[0] != '\0');
  
  /* 프리셋 파일 경로 */
  gchar *dir_path = ensure_presets_dir();
  gchar *file_path = g_strdup_printf("%s/%s.json", dir_path, preset_name);
  
  /* 파일 존재 확인 */
  if (!g_file_test(file_path, G_FILE_TEST_EXISTS)) {
    g_warning("프리셋 파일이 존재하지 않습니다: %s", file_path);
    g_free(file_path);
    g_free(dir_path);
    return;
  }
  
  /* JSON 파싱 */
  JsonParser *parser = json_parser_new();
  GError *error = NULL;
  json_parser_load_from_file(parser, file_path, &error);
  
  if (error) {
    g_warning("프리셋 파일 로드 오류: %s", error->message);
    g_error_free(error);
    g_object_unref(parser);
    g_free(file_path);
    g_free(dir_path);
    return;
  }
  
  /* 루트 객체 가져오기 */
  JsonNode *root = json_parser_get_root(parser);
  JsonObject *root_obj = json_node_get_object(root);
  
  /* 트랙 배열 가져오기 */
  JsonArray *tracks_array = json_object_get_array_member(root_obj, "tracks");
  
  /* 현재 트랙 모두 제거 (사용자 확인 필요) */
  GList *tracks_to_remove = g_list_copy(timeline->tracks);
  for (GList *l = tracks_to_remove; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack*)l->data;
    blouedit_timeline_remove_track(timeline, track);
  }
  g_list_free(tracks_to_remove);
  
  /* 프리셋의 트랙 추가 */
  guint tracks_len = json_array_get_length(tracks_array);
  for (guint i = 0; i < tracks_len; i++) {
    JsonObject *track_obj = json_array_get_object_element(tracks_array, i);
    
    /* 트랙 유형 */
    GESTrackType track_type = json_object_get_int_member(track_obj, "type");
    
    /* 트랙 이름 */
    const gchar *track_name = json_object_get_string_member(track_obj, "name");
    
    /* 새 트랙 생성 */
    BlouEditTimelineTrack *new_track = blouedit_timeline_add_track(timeline, track_type, track_name);
    
    /* 색상 설정 */
    if (json_object_has_member(track_obj, "color")) {
      const gchar *color_str = json_object_get_string_member(track_obj, "color");
      GdkRGBA color;
      
      if (gdk_rgba_parse(&color, color_str)) {
        new_track->color = color;
      }
    }
    
    /* 태그 설정 */
    if (json_object_has_member(track_obj, "tags")) {
      JsonArray *tags_array = json_object_get_array_member(track_obj, "tags");
      guint tags_len = json_array_get_length(tags_array);
      
      for (guint j = 0; j < tags_len; j++) {
        const gchar *tag = json_array_get_string_element(tags_array, j);
        blouedit_timeline_add_track_tag(timeline, new_track, tag);
      }
    }
  }
  
  /* 메모리 해제 */
  g_object_unref(parser);
  g_free(file_path);
  g_free(dir_path);
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 트랙 프리셋 삭제 함수 */
void 
blouedit_timeline_delete_track_preset(const gchar *preset_name)
{
  g_return_if_fail(preset_name != NULL && preset_name[0] != '\0');
  
  /* 프리셋 파일 경로 */
  gchar *dir_path = ensure_presets_dir();
  gchar *file_path = g_strdup_printf("%s/%s.json", dir_path, preset_name);
  
  /* 파일 존재 확인 및 삭제 */
  if (g_file_test(file_path, G_FILE_TEST_EXISTS)) {
    if (g_unlink(file_path) != 0) {
      g_warning("프리셋 파일 삭제 실패: %s", file_path);
    }
  }
  
  /* 메모리 해제 */
  g_free(file_path);
  g_free(dir_path);
}

/* 트랙 프리셋 목록 가져오기 함수 */
GList* 
blouedit_timeline_get_track_presets(void)
{
  GList *presets = NULL;
  
  /* 프리셋 디렉토리 */
  gchar *dir_path = ensure_presets_dir();
  
  /* 디렉토리 열기 */
  GDir *dir = g_dir_open(dir_path, 0, NULL);
  if (dir == NULL) {
    g_free(dir_path);
    return NULL;
  }
  
  /* 디렉토리 내용 읽기 */
  const gchar *filename;
  while ((filename = g_dir_read_name(dir)) != NULL) {
    /* .json 파일만 처리 */
    if (g_str_has_suffix(filename, ".json")) {
      /* 확장자 제거 */
      gchar *preset_name = g_strndup(filename, strlen(filename) - 5);
      presets = g_list_append(presets, preset_name);
    }
  }
  
  /* 정리 */
  g_dir_close(dir);
  g_free(dir_path);
  
  return presets;
}

/* 프리셋 삭제 버튼 클릭 핸들러 */
static void
on_delete_preset_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "tree-view"));
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  /* 선택된 항목 가져오기 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(tree_view);
  if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
    gchar *preset_name;
    gtk_tree_model_get(model, &iter, 0, &preset_name, -1);
    
    /* 확인 대화상자 */
    GtkWidget *confirm_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                                    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                    GTK_MESSAGE_QUESTION,
                                                    GTK_BUTTONS_YES_NO,
                                                    "정말로 '%s' 프리셋을 삭제하시겠습니까?",
                                                    preset_name);
    
    gint response = gtk_dialog_run(GTK_DIALOG(confirm_dialog));
    
    if (response == GTK_RESPONSE_YES) {
      /* 프리셋 삭제 */
      blouedit_timeline_delete_track_preset(preset_name);
      
      /* 목록에서 제거 */
      gtk_list_store_remove(GTK_LIST_STORE(model), &iter);
    }
    
    gtk_widget_destroy(confirm_dialog);
    g_free(preset_name);
  }
}

/* 트랙 프리셋 관리 대화상자 표시 함수 */
void 
blouedit_timeline_show_track_presets_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *box, *scroll, *tree_view;
  GtkWidget *button_box, *apply_button, *save_button, *delete_button;
  GtkListStore *store;
  GtkTreeIter iter;
  GtkTreeViewColumn *column;
  GtkCellRenderer *renderer;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("트랙 프리셋 관리",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_닫기", GTK_RESPONSE_CLOSE,
                                     NULL);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 박스 생성 */
  box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  gtk_container_add(GTK_CONTAINER(content_area), box);
  
  /* 스크롤 윈도우 */
  scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_shadow_type(GTK_SCROLLED_WINDOW(scroll), GTK_SHADOW_ETCHED_IN);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll), 
                               GTK_POLICY_AUTOMATIC, 
                               GTK_POLICY_AUTOMATIC);
  gtk_widget_set_vexpand(scroll, TRUE);
  gtk_container_add(GTK_CONTAINER(box), scroll);
  
  /* 목록 저장소 생성 */
  store = gtk_list_store_new(2, G_TYPE_STRING, G_TYPE_STRING);
  
  /* 프리셋 목록 가져오기 */
  GList *presets = blouedit_timeline_get_track_presets();
  for (GList *l = presets; l != NULL; l = l->next) {
    gchar *preset_name = (gchar*)l->data;
    
    /* 프리셋 정보 로드 */
    gchar *dir_path = ensure_presets_dir();
    gchar *file_path = g_strdup_printf("%s/%s.json", dir_path, preset_name);
    
    /* 생성 시간 가져오기 */
    gchar *created_time = g_strdup("알 수 없음");
    
    JsonParser *parser = json_parser_new();
    GError *error = NULL;
    
    if (json_parser_load_from_file(parser, file_path, &error)) {
      JsonNode *root = json_parser_get_root(parser);
      JsonObject *root_obj = json_node_get_object(root);
      
      if (json_object_has_member(root_obj, "created")) {
        g_free(created_time);
        created_time = g_strdup(json_object_get_string_member(root_obj, "created"));
      }
    }
    
    if (error) {
      g_error_free(error);
    }
    
    g_object_unref(parser);
    g_free(file_path);
    g_free(dir_path);
    
    /* 목록에 추가 */
    gtk_list_store_append(store, &iter);
    gtk_list_store_set(store, &iter, 
                     0, preset_name,
                     1, created_time,
                     -1);
    
    g_free(created_time);
  }
  
  /* 목록 메모리 해제 */
  g_list_free_full(presets, g_free);
  
  /* 트리 뷰 생성 */
  tree_view = gtk_tree_view_new_with_model(GTK_TREE_MODEL(store));
  g_object_unref(store);
  
  /* 이름 열 */
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("프리셋 이름", 
                                                 renderer, 
                                                 "text", 0, 
                                                 NULL);
  gtk_tree_view_column_set_expand(column, TRUE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  /* 생성일 열 */
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("생성일", 
                                                 renderer, 
                                                 "text", 1, 
                                                 NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  /* 트리 뷰 설정 */
  gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(tree_view), TRUE);
  gtk_container_add(GTK_CONTAINER(scroll), tree_view);
  
  /* 버튼 상자 */
  button_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout(GTK_BUTTON_BOX(button_box), GTK_BUTTONBOX_END);
  gtk_box_set_spacing(GTK_BOX(button_box), 5);
  gtk_container_add(GTK_CONTAINER(box), button_box);
  
  /* 적용 버튼 */
  apply_button = gtk_button_new_with_label("적용");
  gtk_container_add(GTK_CONTAINER(button_box), apply_button);
  
  /* 저장 버튼 */
  save_button = gtk_button_new_with_label("현재 구성 저장");
  gtk_container_add(GTK_CONTAINER(button_box), save_button);
  
  /* 삭제 버튼 */
  delete_button = gtk_button_new_with_label("삭제");
  gtk_container_add(GTK_CONTAINER(button_box), delete_button);
  
  /* 데이터 설정 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "tree-view", tree_view);
  
  /* 시그널 연결 */
  g_signal_connect(delete_button, "clicked", G_CALLBACK(on_delete_preset_clicked), dialog);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 실행 */
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 정리 */
  gtk_widget_destroy(dialog);
} 