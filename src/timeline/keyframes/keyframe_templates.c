#include <gtk/gtk.h>
#include <string.h>
#include <json-glib/json-glib.h>
#include "keyframe_templates.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 템플릿 저장 경로 */
#define BLOUEDIT_KEYFRAME_TEMPLATES_DIR "~/.config/blouedit/keyframe_templates"

/* 키프레임 템플릿 디렉토리 확인 및 생성 함수 */
static gchar*
ensure_templates_dir(void)
{
  gchar *dir_path = g_strdup_printf("%s", BLOUEDIT_KEYFRAME_TEMPLATES_DIR);
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

/* 파일명에 유효하지 않은 문자 제거 */
static gchar*
sanitize_filename(const gchar *name)
{
  gchar *result = g_strdup(name);
  
  /* 파일명에 유효하지 않은 문자를 '_'로 대체 */
  for (gchar *p = result; *p; p++) {
    if (*p == '/' || *p == '\\' || *p == ':' || *p == '*' || 
        *p == '?' || *p == '"' || *p == '<' || *p == '>' || 
        *p == '|') {
      *p = '_';
    }
  }
  
  return result;
}

/* 키프레임 템플릿 저장 함수 */
void
blouedit_timeline_save_keyframe_template(BlouEditTimeline *timeline, 
                                     const gchar *template_name,
                                     GList *keyframes)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(template_name != NULL && template_name[0] != '\0');
  
  /* 템플릿 디렉토리 확인 */
  gchar *dir_path = ensure_templates_dir();
  
  /* 파일명에 유효하지 않은 문자 제거 */
  gchar *safe_name = sanitize_filename(template_name);
  
  /* 템플릿 파일 경로 */
  gchar *file_path = g_strdup_printf("%s/%s.json", dir_path, safe_name);
  
  /* JSON 생성 */
  JsonBuilder *builder = json_builder_new();
  
  /* 루트 객체 시작 */
  json_builder_begin_object(builder);
  
  /* 템플릿 이름 */
  json_builder_set_member_name(builder, "name");
  json_builder_add_string_value(builder, template_name);
  
  /* 템플릿 생성 시간 */
  json_builder_set_member_name(builder, "created");
  GDateTime *now = g_date_time_new_now_local();
  gchar *date_str = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
  json_builder_add_string_value(builder, date_str);
  g_free(date_str);
  g_date_time_unref(now);
  
  /* 키프레임 배열 시작 */
  json_builder_set_member_name(builder, "keyframes");
  json_builder_begin_array(builder);
  
  /* 모든 키프레임 정보 저장 */
  for (GList *l = keyframes; l != NULL; l = l->next) {
    GESKeyFrameInfo *kf = (GESKeyFrameInfo*)l->data;
    
    /* 키프레임 객체 시작 */
    json_builder_begin_object(builder);
    
    /* 키프레임 시간 */
    json_builder_set_member_name(builder, "time");
    json_builder_add_int_value(builder, kf->timestamp);
    
    /* 키프레임 값 */
    json_builder_set_member_name(builder, "value");
    if (G_VALUE_HOLDS_DOUBLE(&kf->value)) {
      json_builder_add_double_value(builder, g_value_get_double(&kf->value));
    } else if (G_VALUE_HOLDS_INT(&kf->value)) {
      json_builder_add_int_value(builder, g_value_get_int(&kf->value));
    } else if (G_VALUE_HOLDS_BOOLEAN(&kf->value)) {
      json_builder_add_boolean_value(builder, g_value_get_boolean(&kf->value));
    } else if (G_VALUE_HOLDS_STRING(&kf->value)) {
      json_builder_add_string_value(builder, g_value_get_string(&kf->value));
    } else {
      /* 지원하지 않는 타입은 널로 저장 */
      json_builder_add_null_value(builder);
    }
    
    /* 키프레임 보간 유형 */
    json_builder_set_member_name(builder, "interpolation");
    json_builder_add_int_value(builder, kf->interpolation);
    
    /* 키프레임 객체 종료 */
    json_builder_end_object(builder);
  }
  
  /* 키프레임 배열 종료 */
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
    g_warning("키프레임 템플릿 저장 오류: %s", error->message);
    g_error_free(error);
  }
  
  /* 메모리 해제 */
  json_node_free(root);
  g_object_unref(generator);
  g_object_unref(builder);
  g_free(file_path);
  g_free(safe_name);
  g_free(dir_path);
}

/* 키프레임 템플릿 로드 함수 */
GList*
blouedit_timeline_load_keyframe_template(BlouEditTimeline *timeline, 
                                     const gchar *template_name)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  g_return_val_if_fail(template_name != NULL && template_name[0] != '\0', NULL);
  
  GList *keyframes = NULL;
  
  /* 템플릿 디렉토리 */
  gchar *dir_path = ensure_templates_dir();
  
  /* 파일명에 유효하지 않은 문자 제거 */
  gchar *safe_name = sanitize_filename(template_name);
  
  /* 템플릿 파일 경로 */
  gchar *file_path = g_strdup_printf("%s/%s.json", dir_path, safe_name);
  
  /* 파일 존재 확인 */
  if (!g_file_test(file_path, G_FILE_TEST_EXISTS)) {
    g_warning("템플릿 파일이 존재하지 않습니다: %s", file_path);
    g_free(file_path);
    g_free(safe_name);
    g_free(dir_path);
    return NULL;
  }
  
  /* JSON 파싱 */
  JsonParser *parser = json_parser_new();
  GError *error = NULL;
  json_parser_load_from_file(parser, file_path, &error);
  
  if (error) {
    g_warning("템플릿 파일 로드 오류: %s", error->message);
    g_error_free(error);
    g_object_unref(parser);
    g_free(file_path);
    g_free(safe_name);
    g_free(dir_path);
    return NULL;
  }
  
  /* 루트 객체 가져오기 */
  JsonNode *root = json_parser_get_root(parser);
  JsonObject *root_obj = json_node_get_object(root);
  
  /* 키프레임 배열 가져오기 */
  JsonArray *keyframes_array = json_object_get_array_member(root_obj, "keyframes");
  
  /* 키프레임 파싱 */
  guint keyframes_len = json_array_get_length(keyframes_array);
  for (guint i = 0; i < keyframes_len; i++) {
    JsonObject *kf_obj = json_array_get_object_element(keyframes_array, i);
    
    /* 키프레임 정보 생성 */
    GESKeyFrameInfo *kf = g_new0(GESKeyFrameInfo, 1);
    
    /* 시간 설정 */
    kf->timestamp = json_object_get_int_member(kf_obj, "time");
    
    /* 보간 유형 설정 */
    kf->interpolation = json_object_get_int_member(kf_obj, "interpolation");
    
    /* 값 설정 (임시로 실수 값만 지원) */
    JsonNode *value_node = json_object_get_member(kf_obj, "value");
    
    if (json_node_get_node_type(value_node) == JSON_NODE_VALUE) {
      switch (json_node_get_value_type(value_node)) {
        case G_TYPE_DOUBLE:
          g_value_init(&kf->value, G_TYPE_DOUBLE);
          g_value_set_double(&kf->value, json_node_get_double(value_node));
          break;
          
        case G_TYPE_INT64:
          g_value_init(&kf->value, G_TYPE_INT);
          g_value_set_int(&kf->value, (gint)json_node_get_int(value_node));
          break;
          
        case G_TYPE_BOOLEAN:
          g_value_init(&kf->value, G_TYPE_BOOLEAN);
          g_value_set_boolean(&kf->value, json_node_get_boolean(value_node));
          break;
          
        case G_TYPE_STRING:
          g_value_init(&kf->value, G_TYPE_STRING);
          g_value_set_string(&kf->value, json_node_get_string(value_node));
          break;
          
        default:
          g_value_init(&kf->value, G_TYPE_DOUBLE);
          g_value_set_double(&kf->value, 0.0);
      }
    } else {
      g_value_init(&kf->value, G_TYPE_DOUBLE);
      g_value_set_double(&kf->value, 0.0);
    }
    
    /* 리스트에 추가 */
    keyframes = g_list_append(keyframes, kf);
  }
  
  /* 메모리 해제 */
  g_object_unref(parser);
  g_free(file_path);
  g_free(safe_name);
  g_free(dir_path);
  
  return keyframes;
}

/* 키프레임 템플릿 삭제 함수 */
void
blouedit_timeline_delete_keyframe_template(const gchar *template_name)
{
  g_return_if_fail(template_name != NULL && template_name[0] != '\0');
  
  /* 템플릿 디렉토리 */
  gchar *dir_path = ensure_templates_dir();
  
  /* 파일명에 유효하지 않은 문자 제거 */
  gchar *safe_name = sanitize_filename(template_name);
  
  /* 템플릿 파일 경로 */
  gchar *file_path = g_strdup_printf("%s/%s.json", dir_path, safe_name);
  
  /* 파일 삭제 */
  if (g_file_test(file_path, G_FILE_TEST_EXISTS)) {
    if (g_unlink(file_path) != 0) {
      g_warning("템플릿 파일 삭제 실패: %s", file_path);
    }
  }
  
  /* 메모리 해제 */
  g_free(file_path);
  g_free(safe_name);
  g_free(dir_path);
}

/* 키프레임 템플릿 목록 가져오기 함수 */
GList*
blouedit_timeline_get_keyframe_templates(void)
{
  GList *templates = NULL;
  
  /* 템플릿 디렉토리 */
  gchar *dir_path = ensure_templates_dir();
  
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
      gchar *template_name = g_strndup(filename, strlen(filename) - 5);
      
      /* JSON 파싱하여 실제 템플릿 이름 가져오기 */
      gchar *file_path = g_strdup_printf("%s/%s", dir_path, filename);
      JsonParser *parser = json_parser_new();
      GError *error = NULL;
      
      if (json_parser_load_from_file(parser, file_path, &error)) {
        JsonNode *root = json_parser_get_root(parser);
        JsonObject *root_obj = json_node_get_object(root);
        
        if (json_object_has_member(root_obj, "name")) {
          const gchar *display_name = json_object_get_string_member(root_obj, "name");
          
          /* 생성 시간 확인 */
          const gchar *created = NULL;
          if (json_object_has_member(root_obj, "created")) {
            created = json_object_get_string_member(root_obj, "created");
          }
          
          /* 템플릿 정보 구성 */
          gchar *template_info = g_strdup_printf("%s|%s|%s", 
                                            template_name, 
                                            display_name,
                                            created ? created : "");
          
          templates = g_list_append(templates, template_info);
        }
      }
      
      if (error) {
        g_error_free(error);
      }
      
      g_object_unref(parser);
      g_free(file_path);
      g_free(template_name);
    }
  }
  
  /* 정리 */
  g_dir_close(dir);
  g_free(dir_path);
  
  return templates;
}

/* 키프레임 템플릿 적용 함수 */
void
blouedit_timeline_apply_keyframe_template(BlouEditTimeline *timeline, 
                                      const gchar *template_name,
                                      GESClip *clip,
                                      const gchar *property_name)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(template_name != NULL && template_name[0] != '\0');
  g_return_if_fail(GES_IS_CLIP(clip));
  g_return_if_fail(property_name != NULL && property_name[0] != '\0');
  
  /* 템플릿 키프레임 목록 로드 */
  GList *keyframes = blouedit_timeline_load_keyframe_template(timeline, template_name);
  if (keyframes == NULL) {
    g_warning("키프레임 템플릿을 로드할 수 없습니다: %s", template_name);
    return;
  }
  
  /* 기존 키프레임 제거 */
  GESTrackElement *element = ges_clip_find_track_element(clip, NULL, GES_TYPE_TRACK_ELEMENT);
  if (element != NULL) {
    ges_track_element_remove_all_control_bindings(element);
  }
  
  /* 새 키프레임 설정 */
  for (GList *l = keyframes; l != NULL; l = l->next) {
    GESKeyFrameInfo *kf = (GESKeyFrameInfo*)l->data;
    
    /* 키프레임 추가 (여기서는 단순화된 예시) */
    ges_track_element_set_control_source_for_property(element, 
                                                   NULL, 
                                                   property_name, 
                                                   "direct");
    
    ges_timeline_element_set_child_property_by_pspec(GES_TIMELINE_ELEMENT(element), 
                                                  NULL, 
                                                  property_name, 
                                                  &kf->value);
  }
  
  /* 메모리 해제 */
  g_list_free_full(keyframes, g_free);
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 템플릿 삭제 버튼 클릭 핸들러 */
static void
on_delete_template_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "tree-view"));
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  /* 선택된 항목 가져오기 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(tree_view);
  if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
    gchar *template_file_name;
    gtk_tree_model_get(model, &iter, 0, &template_file_name, -1);
    
    /* 확인 대화상자 */
    GtkWidget *confirm_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                                    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                    GTK_MESSAGE_QUESTION,
                                                    GTK_BUTTONS_YES_NO,
                                                    "정말로 키프레임 템플릿을 삭제하시겠습니까?");
    
    gint response = gtk_dialog_run(GTK_DIALOG(confirm_dialog));
    
    if (response == GTK_RESPONSE_YES) {
      /* 템플릿 삭제 */
      blouedit_timeline_delete_keyframe_template(template_file_name);
      
      /* 목록에서 제거 */
      gtk_list_store_remove(GTK_LIST_STORE(model), &iter);
    }
    
    gtk_widget_destroy(confirm_dialog);
    g_free(template_file_name);
  }
}

/* 템플릿 적용 버튼 클릭 핸들러 */
static void
on_apply_template_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "tree-view"));
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  GESClip *clip = g_object_get_data(G_OBJECT(dialog), "clip");
  const gchar *property_name = g_object_get_data(G_OBJECT(dialog), "property");
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  /* 선택된 항목 가져오기 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(tree_view);
  if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
    gchar *template_file_name;
    gtk_tree_model_get(model, &iter, 0, &template_file_name, -1);
    
    /* 템플릿 적용 */
    blouedit_timeline_apply_keyframe_template(timeline, template_file_name, clip, property_name);
    
    /* 완료 메시지 */
    GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                             GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                             GTK_MESSAGE_INFO,
                                             GTK_BUTTONS_OK,
                                             "키프레임 템플릿이 적용되었습니다.");
    
    gtk_dialog_run(GTK_DIALOG(message));
    gtk_widget_destroy(message);
    
    /* 다이얼로그 닫기 */
    gtk_dialog_response(GTK_DIALOG(dialog), GTK_RESPONSE_CLOSE);
    
    g_free(template_file_name);
  }
}

/* 새 템플릿 저장 버튼 클릭 핸들러 */
static void
on_save_new_template_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  GESClip *clip = g_object_get_data(G_OBJECT(dialog), "clip");
  const gchar *property_name = g_object_get_data(G_OBJECT(dialog), "property");
  
  /* 이름 입력 대화상자 */
  GtkWidget *name_dialog = gtk_dialog_new_with_buttons("키프레임 템플릿 이름 입력",
                                                    GTK_WINDOW(dialog),
                                                    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                    "_취소", GTK_RESPONSE_CANCEL,
                                                    "_저장", GTK_RESPONSE_ACCEPT,
                                                    NULL);
  
  /* 콘텐츠 영역 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(name_dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 레이블 */
  GtkWidget *label = gtk_label_new("템플릿 이름을 입력하세요:");
  gtk_container_add(GTK_CONTAINER(content_area), label);
  
  /* 입력 필드 */
  GtkWidget *entry = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(entry), "새 키프레임 템플릿");
  gtk_container_add(GTK_CONTAINER(content_area), entry);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(name_dialog);
  
  /* 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(name_dialog)) == GTK_RESPONSE_ACCEPT) {
    const gchar *template_name = gtk_entry_get_text(GTK_ENTRY(entry));
    
    if (template_name != NULL && template_name[0] != '\0') {
      /* 현재 클립의 키프레임 가져오기 */
      GList *keyframes = NULL; /* 실제 구현에서는 클립에서 키프레임 목록 추출 */
      
      /* 템플릿 저장 */
      blouedit_timeline_save_keyframe_template(timeline, template_name, keyframes);
      
      /* 완료 메시지 */
      GtkWidget *message = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                               GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                               GTK_MESSAGE_INFO,
                                               GTK_BUTTONS_OK,
                                               "키프레임 템플릿이 저장되었습니다.");
      
      gtk_dialog_run(GTK_DIALOG(message));
      gtk_widget_destroy(message);
      
      /* 목록 새로고침 */
      GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "tree-view"));
      GtkListStore *store = GTK_LIST_STORE(gtk_tree_view_get_model(tree_view));
      gtk_list_store_clear(store);
      
      GList *templates = blouedit_timeline_get_keyframe_templates();
      for (GList *l = templates; l != NULL; l = l->next) {
        gchar *template_info = (gchar*)l->data;
        gchar **parts = g_strsplit(template_info, "|", 3);
        
        GtkTreeIter iter;
        gtk_list_store_append(store, &iter);
        gtk_list_store_set(store, &iter, 
                         0, parts[0], 
                         1, parts[1], 
                         2, parts[2] ? parts[2] : "",
                         -1);
        
        g_strfreev(parts);
        g_free(template_info);
      }
      g_list_free(templates);
    }
  }
  
  gtk_widget_destroy(name_dialog);
}

/* 키프레임 템플릿 관리 대화상자 */
void
blouedit_timeline_show_keyframe_templates_dialog(BlouEditTimeline *timeline,
                                             GESClip *clip,
                                             const gchar *property_name)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *box, *scroll, *tree_view;
  GtkWidget *button_box, *apply_button, *save_button, *delete_button;
  GtkListStore *store;
  GtkTreeIter iter;
  GtkCellRenderer *renderer;
  GtkTreeViewColumn *column;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("키프레임 템플릿 관리",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_닫기", GTK_RESPONSE_CLOSE,
                                     NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 500, 400);
  
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
  
  /* 모델 생성 */
  store = gtk_list_store_new(3, G_TYPE_STRING, G_TYPE_STRING, G_TYPE_STRING);
  
  /* 템플릿 목록 가져오기 */
  GList *templates = blouedit_timeline_get_keyframe_templates();
  for (GList *l = templates; l != NULL; l = l->next) {
    gchar *template_info = (gchar*)l->data;
    gchar **parts = g_strsplit(template_info, "|", 3);
    
    gtk_list_store_append(store, &iter);
    gtk_list_store_set(store, &iter, 
                     0, parts[0], 
                     1, parts[1], 
                     2, parts[2] ? parts[2] : "",
                     -1);
    
    g_strfreev(parts);
    g_free(template_info);
  }
  g_list_free(templates);
  
  /* 트리 뷰 생성 */
  tree_view = gtk_tree_view_new_with_model(GTK_TREE_MODEL(store));
  g_object_unref(store);
  
  /* 선택 모드 설정 */
  gtk_tree_selection_set_mode(gtk_tree_view_get_selection(GTK_TREE_VIEW(tree_view)),
                            GTK_SELECTION_SINGLE);
  
  /* 열 추가 (파일명은 숨김 - 내부 사용) */
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("FileName", 
                                                 renderer, 
                                                 "text", 0, 
                                                 NULL);
  gtk_tree_view_column_set_visible(column, FALSE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  /* 이름 열 */
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("템플릿 이름", 
                                                 renderer, 
                                                 "text", 1, 
                                                 NULL);
  gtk_tree_view_column_set_expand(column, TRUE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  /* 생성일 열 */
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("생성일", 
                                                 renderer, 
                                                 "text", 2, 
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
  save_button = gtk_button_new_with_label("새 템플릿 저장");
  gtk_container_add(GTK_CONTAINER(button_box), save_button);
  
  /* 삭제 버튼 */
  delete_button = gtk_button_new_with_label("삭제");
  gtk_container_add(GTK_CONTAINER(button_box), delete_button);
  
  /* 데이터 설정 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "tree-view", tree_view);
  g_object_set_data(G_OBJECT(dialog), "clip", clip);
  g_object_set_data_full(G_OBJECT(dialog), "property", g_strdup(property_name), g_free);
  
  /* 시그널 연결 */
  g_signal_connect(delete_button, "clicked", G_CALLBACK(on_delete_template_clicked), dialog);
  g_signal_connect(apply_button, "clicked", G_CALLBACK(on_apply_template_clicked), dialog);
  g_signal_connect(save_button, "clicked", G_CALLBACK(on_save_new_template_clicked), dialog);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 실행 */
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 정리 */
  gtk_widget_destroy(dialog);
} 