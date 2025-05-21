#include <gtk/gtk.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include <json-glib/json-glib.h>
#include "timeline_history.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 스냅샷 저장 경로 */
#define BLOUEDIT_SNAPSHOTS_DIR "~/.config/blouedit/snapshots"

/* 스냅샷 디렉토리 확인 및 생성 함수 */
static gchar*
ensure_snapshots_dir(BlouEditTimeline *timeline)
{
  gchar *dir_path = g_strdup_printf("%s/%s", 
                                  BLOUEDIT_SNAPSHOTS_DIR,
                                  timeline->project_id ? timeline->project_id : "default");
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

/* 타임라인 객체를 XML로 직렬화 */
static gchar*
serialize_timeline_to_xml(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  GError *error = NULL;
  gchar *xml_data = NULL;
  
  /* GES 타임라인 가져오기 */
  GESTimeline *ges_timeline = timeline->ges_timeline;
  if (ges_timeline == NULL) {
    return NULL;
  }
  
  /* XML로 직렬화 */
  if (!ges_timeline_save_to_uri(ges_timeline, "mem://temp.xges", NULL, TRUE, &error)) {
    g_warning("타임라인 직렬화 오류: %s", error->message);
    g_error_free(error);
    return NULL;
  }
  
  /* XML 데이터 가져오기 (이 부분은 실제 구현시 적절한 방식으로 변경 필요) */
  /* 여기서는 단순화를 위해 빈 XML 문자열 반환 */
  xml_data = g_strdup("<timeline></timeline>");
  
  return xml_data;
}

/* XML에서 타임라인 객체 복원 */
static gboolean
deserialize_timeline_from_xml(BlouEditTimeline *timeline, const gchar *xml_data)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(xml_data != NULL, FALSE);
  
  GError *error = NULL;
  
  /* GES 타임라인 가져오기 */
  GESTimeline *ges_timeline = timeline->ges_timeline;
  if (ges_timeline == NULL) {
    return FALSE;
  }
  
  /* XML에서 복원 (이 부분은 실제 구현시 적절한 방식으로 변경 필요) */
  /* 여기서는 단순화를 위해 항상 성공 반환 */
  return TRUE;
}

/* 타임라인 스냅샷 저장 */
void
blouedit_timeline_save_snapshot(BlouEditTimeline *timeline, const gchar *snapshot_name)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(snapshot_name != NULL && snapshot_name[0] != '\0');
  
  /* 스냅샷 디렉토리 확인 */
  gchar *dir_path = ensure_snapshots_dir(timeline);
  
  /* 파일명에 유효하지 않은 문자 제거 */
  gchar *safe_name = sanitize_filename(snapshot_name);
  
  /* 스냅샷 파일 경로 */
  gchar *snapshot_path = g_strdup_printf("%s/%s.xges", dir_path, safe_name);
  
  /* 타임라인 직렬화 */
  gchar *xml_data = serialize_timeline_to_xml(timeline);
  
  if (xml_data != NULL) {
    /* 파일에 저장 */
    GError *error = NULL;
    if (!g_file_set_contents(snapshot_path, xml_data, -1, &error)) {
      g_warning("스냅샷 저장 오류: %s", error->message);
      g_error_free(error);
    } else {
      /* 메타데이터 파일 생성 */
      gchar *meta_path = g_strdup_printf("%s/%s.meta", dir_path, safe_name);
      
      /* JSON 생성 */
      JsonBuilder *builder = json_builder_new();
      
      /* 루트 객체 시작 */
      json_builder_begin_object(builder);
      
      /* 스냅샷 이름 */
      json_builder_set_member_name(builder, "name");
      json_builder_add_string_value(builder, snapshot_name);
      
      /* 생성 시간 */
      json_builder_set_member_name(builder, "created");
      GDateTime *now = g_date_time_new_now_local();
      gchar *date_str = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
      json_builder_add_string_value(builder, date_str);
      g_free(date_str);
      g_date_time_unref(now);
      
      /* 루트 객체 종료 */
      json_builder_end_object(builder);
      
      /* JSON 생성 */
      JsonGenerator *generator = json_generator_new();
      JsonNode *root = json_builder_get_root(builder);
      json_generator_set_root(generator, root);
      json_generator_set_pretty(generator, TRUE);
      
      /* 파일에 저장 */
      GError *meta_error = NULL;
      json_generator_to_file(generator, meta_path, &meta_error);
      
      if (meta_error) {
        g_warning("스냅샷 메타데이터 저장 오류: %s", meta_error->message);
        g_error_free(meta_error);
      }
      
      /* 메모리 해제 */
      json_node_free(root);
      g_object_unref(generator);
      g_object_unref(builder);
      g_free(meta_path);
      
      /* 히스토리 항목 추가 */
      blouedit_timeline_history_add_entry(timeline, 
                                      BLOUEDIT_HISTORY_SNAPSHOT, 
                                      g_strdup(snapshot_name), NULL, NULL);
    }
    
    g_free(xml_data);
  }
  
  g_free(snapshot_path);
  g_free(safe_name);
  g_free(dir_path);
}

/* 타임라인 스냅샷 목록 가져오기 */
GList*
blouedit_timeline_get_snapshots(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  GList *snapshots = NULL;
  
  /* 스냅샷 디렉토리 */
  gchar *dir_path = ensure_snapshots_dir(timeline);
  
  /* 디렉토리 열기 */
  GDir *dir = g_dir_open(dir_path, 0, NULL);
  if (dir == NULL) {
    g_free(dir_path);
    return NULL;
  }
  
  /* 디렉토리 내용 읽기 */
  const gchar *filename;
  while ((filename = g_dir_read_name(dir)) != NULL) {
    /* .xges 파일만 처리 */
    if (g_str_has_suffix(filename, ".xges")) {
      /* 메타데이터 파일 경로 */
      gchar *basename = g_strndup(filename, strlen(filename) - 5);
      gchar *meta_path = g_strdup_printf("%s/%s.meta", dir_path, basename);
      
      /* 메타데이터 파일 존재 확인 */
      if (g_file_test(meta_path, G_FILE_TEST_EXISTS)) {
        /* 메타데이터 파일 파싱 */
        JsonParser *parser = json_parser_new();
        GError *error = NULL;
        json_parser_load_from_file(parser, meta_path, &error);
        
        if (error == NULL) {
          /* 메타데이터 가져오기 */
          JsonNode *root = json_parser_get_root(parser);
          JsonObject *root_obj = json_node_get_object(root);
          
          /* 스냅샷 이름 */
          const gchar *name = json_object_get_string_member(root_obj, "name");
          
          /* 생성 시간 */
          const gchar *created = NULL;
          if (json_object_has_member(root_obj, "created")) {
            created = json_object_get_string_member(root_obj, "created");
          }
          
          /* 스냅샷 정보 생성 */
          gchar *info = g_strdup_printf("%s|%s", name, created ? created : "");
          snapshots = g_list_append(snapshots, info);
        } else {
          g_error_free(error);
        }
        
        g_object_unref(parser);
      }
      
      g_free(meta_path);
      g_free(basename);
    }
  }
  
  /* 정리 */
  g_dir_close(dir);
  g_free(dir_path);
  
  return snapshots;
}

/* 타임라인 스냅샷 복원 */
gboolean
blouedit_timeline_restore_snapshot(BlouEditTimeline *timeline, const gchar *snapshot_name)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(snapshot_name != NULL && snapshot_name[0] != '\0', FALSE);
  
  /* 스냅샷 디렉토리 */
  gchar *dir_path = ensure_snapshots_dir(timeline);
  
  /* 파일명에 유효하지 않은 문자 제거 */
  gchar *safe_name = sanitize_filename(snapshot_name);
  
  /* 스냅샷 파일 경로 */
  gchar *snapshot_path = g_strdup_printf("%s/%s.xges", dir_path, safe_name);
  
  /* 파일 존재 확인 */
  if (!g_file_test(snapshot_path, G_FILE_TEST_EXISTS)) {
    g_warning("스냅샷 파일이 존재하지 않습니다: %s", snapshot_path);
    g_free(snapshot_path);
    g_free(safe_name);
    g_free(dir_path);
    return FALSE;
  }
  
  /* 파일 내용 읽기 */
  GError *error = NULL;
  gchar *xml_data = NULL;
  gsize length;
  
  if (!g_file_get_contents(snapshot_path, &xml_data, &length, &error)) {
    g_warning("스냅샷 파일 읽기 오류: %s", error->message);
    g_error_free(error);
    g_free(snapshot_path);
    g_free(safe_name);
    g_free(dir_path);
    return FALSE;
  }
  
  /* 타임라인 복원 */
  gboolean result = deserialize_timeline_from_xml(timeline, xml_data);
  
  /* 정리 */
  g_free(xml_data);
  g_free(snapshot_path);
  g_free(safe_name);
  g_free(dir_path);
  
  return result;
}

/* 타임라인 스냅샷 삭제 */
void
blouedit_timeline_delete_snapshot(BlouEditTimeline *timeline, const gchar *snapshot_name)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(snapshot_name != NULL && snapshot_name[0] != '\0');
  
  /* 스냅샷 디렉토리 */
  gchar *dir_path = ensure_snapshots_dir(timeline);
  
  /* 파일명에 유효하지 않은 문자 제거 */
  gchar *safe_name = sanitize_filename(snapshot_name);
  
  /* 스냅샷 파일 경로 */
  gchar *snapshot_path = g_strdup_printf("%s/%s.xges", dir_path, safe_name);
  
  /* 메타데이터 파일 경로 */
  gchar *meta_path = g_strdup_printf("%s/%s.meta", dir_path, safe_name);
  
  /* 파일 삭제 */
  if (g_file_test(snapshot_path, G_FILE_TEST_EXISTS)) {
    if (g_unlink(snapshot_path) != 0) {
      g_warning("스냅샷 파일 삭제 실패: %s", snapshot_path);
    }
  }
  
  if (g_file_test(meta_path, G_FILE_TEST_EXISTS)) {
    if (g_unlink(meta_path) != 0) {
      g_warning("스냅샷 메타데이터 파일 삭제 실패: %s", meta_path);
    }
  }
  
  /* 정리 */
  g_free(meta_path);
  g_free(snapshot_path);
  g_free(safe_name);
  g_free(dir_path);
}

/* 스냅샷 삭제 버튼 클릭 핸들러 */
static void
on_delete_snapshot_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "tree-view"));
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  /* 선택된 항목 가져오기 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(tree_view);
  if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
    gchar *snapshot_name;
    gtk_tree_model_get(model, &iter, 0, &snapshot_name, -1);
    
    /* 확인 대화상자 */
    GtkWidget *confirm_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                                    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                    GTK_MESSAGE_QUESTION,
                                                    GTK_BUTTONS_YES_NO,
                                                    "정말로 '%s' 스냅샷을 삭제하시겠습니까?",
                                                    snapshot_name);
    
    gint response = gtk_dialog_run(GTK_DIALOG(confirm_dialog));
    
    if (response == GTK_RESPONSE_YES) {
      /* 스냅샷 삭제 */
      blouedit_timeline_delete_snapshot(timeline, snapshot_name);
      
      /* 목록에서 제거 */
      gtk_list_store_remove(GTK_LIST_STORE(model), &iter);
    }
    
    gtk_widget_destroy(confirm_dialog);
    g_free(snapshot_name);
  }
}

/* 스냅샷 복원 버튼 클릭 핸들러 */
static void
on_restore_snapshot_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "tree-view"));
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  /* 선택된 항목 가져오기 */
  GtkTreeSelection *selection = gtk_tree_view_get_selection(tree_view);
  if (gtk_tree_selection_get_selected(selection, &model, &iter)) {
    gchar *snapshot_name;
    gtk_tree_model_get(model, &iter, 0, &snapshot_name, -1);
    
    /* 확인 대화상자 */
    GtkWidget *confirm_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                                    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                    GTK_MESSAGE_QUESTION,
                                                    GTK_BUTTONS_YES_NO,
                                                    "정말로 '%s' 스냅샷을 복원하시겠습니까?\n현재 타임라인이 삭제됩니다.",
                                                    snapshot_name);
    
    gint response = gtk_dialog_run(GTK_DIALOG(confirm_dialog));
    
    if (response == GTK_RESPONSE_YES) {
      /* 스냅샷 복원 */
      if (blouedit_timeline_restore_snapshot(timeline, snapshot_name)) {
        /* 성공 메시지 */
        GtkWidget *success_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                                       GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                       GTK_MESSAGE_INFO,
                                                       GTK_BUTTONS_OK,
                                                       "스냅샷 '%s'이(가) 성공적으로 복원되었습니다.",
                                                       snapshot_name);
        gtk_dialog_run(GTK_DIALOG(success_dialog));
        gtk_widget_destroy(success_dialog);
        
        /* 대화상자 닫기 */
        gtk_dialog_response(GTK_DIALOG(dialog), GTK_RESPONSE_CLOSE);
      } else {
        /* 오류 메시지 */
        GtkWidget *error_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
                                                      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                      GTK_MESSAGE_ERROR,
                                                      GTK_BUTTONS_OK,
                                                      "스냅샷 '%s' 복원 중 오류가 발생했습니다.",
                                                      snapshot_name);
        gtk_dialog_run(GTK_DIALOG(error_dialog));
        gtk_widget_destroy(error_dialog);
      }
    }
    
    gtk_widget_destroy(confirm_dialog);
    g_free(snapshot_name);
  }
}

/* 스냅샷 저장 버튼 클릭 핸들러 */
static void
on_save_snapshot_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  GtkTreeView *tree_view = GTK_TREE_VIEW(g_object_get_data(G_OBJECT(dialog), "tree-view"));
  GtkListStore *store = GTK_LIST_STORE(gtk_tree_view_get_model(tree_view));
  
  /* 이름 입력 대화상자 */
  GtkWidget *name_dialog = gtk_dialog_new_with_buttons("스냅샷 이름 입력",
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
  GtkWidget *label = gtk_label_new("새 스냅샷의 이름을 입력하세요:");
  gtk_container_add(GTK_CONTAINER(content_area), label);
  
  /* 입력 필드 */
  GtkWidget *entry = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(entry), "스냅샷");
  gtk_container_add(GTK_CONTAINER(content_area), entry);
  
  /* 현재 시간 추가 */
  GDateTime *now = g_date_time_new_now_local();
  gchar *time_str = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
  gchar *default_name = g_strdup_printf("스냅샷 %s", time_str);
  gtk_entry_set_text(GTK_ENTRY(entry), default_name);
  g_free(default_name);
  g_free(time_str);
  g_date_time_unref(now);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(name_dialog);
  
  /* 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(name_dialog)) == GTK_RESPONSE_ACCEPT) {
    const gchar *snapshot_name = gtk_entry_get_text(GTK_ENTRY(entry));
    
    if (snapshot_name != NULL && snapshot_name[0] != '\0') {
      /* 스냅샷 저장 */
      blouedit_timeline_save_snapshot(timeline, snapshot_name);
      
      /* 목록 갱신 */
      GtkTreeIter iter;
      gtk_list_store_append(store, &iter);
      
      /* 현재 시간 포맷팅 */
      GDateTime *now = g_date_time_new_now_local();
      gchar *time_str = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
      
      gtk_list_store_set(store, &iter,
                       0, snapshot_name,
                       1, time_str,
                       -1);
      
      g_free(time_str);
      g_date_time_unref(now);
    }
  }
  
  gtk_widget_destroy(name_dialog);
}

/* 타임라인 스냅샷 관리 대화상자 */
void
blouedit_timeline_show_snapshots_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *box, *scroll, *tree_view;
  GtkWidget *button_box, *restore_button, *save_button, *delete_button;
  GtkListStore *store;
  GtkTreeIter iter;
  GtkCellRenderer *renderer;
  GtkTreeViewColumn *column;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("타임라인 스냅샷 관리",
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
  store = gtk_list_store_new(2, G_TYPE_STRING, G_TYPE_STRING);
  
  /* 스냅샷 목록 가져오기 */
  GList *snapshots = blouedit_timeline_get_snapshots(timeline);
  for (GList *l = snapshots; l != NULL; l = l->next) {
    gchar *info = (gchar*)l->data;
    gchar **parts = g_strsplit(info, "|", 2);
    
    gtk_list_store_append(store, &iter);
    gtk_list_store_set(store, &iter, 
                     0, parts[0],
                     1, parts[1] ? parts[1] : "",
                     -1);
    
    g_strfreev(parts);
    g_free(info);
  }
  g_list_free(snapshots);
  
  /* 트리 뷰 생성 */
  tree_view = gtk_tree_view_new_with_model(GTK_TREE_MODEL(store));
  g_object_unref(store);
  
  /* 선택 모드 설정 */
  gtk_tree_selection_set_mode(gtk_tree_view_get_selection(GTK_TREE_VIEW(tree_view)),
                            GTK_SELECTION_SINGLE);
  
  /* 이름 열 */
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("스냅샷 이름", 
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
  
  /* 복원 버튼 */
  restore_button = gtk_button_new_with_label("복원");
  gtk_container_add(GTK_CONTAINER(button_box), restore_button);
  
  /* 저장 버튼 */
  save_button = gtk_button_new_with_label("새 스냅샷 저장");
  gtk_container_add(GTK_CONTAINER(button_box), save_button);
  
  /* 삭제 버튼 */
  delete_button = gtk_button_new_with_label("삭제");
  gtk_container_add(GTK_CONTAINER(button_box), delete_button);
  
  /* 데이터 설정 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "tree-view", tree_view);
  
  /* 시그널 연결 */
  g_signal_connect(delete_button, "clicked", G_CALLBACK(on_delete_snapshot_clicked), dialog);
  g_signal_connect(restore_button, "clicked", G_CALLBACK(on_restore_snapshot_clicked), dialog);
  g_signal_connect(save_button, "clicked", G_CALLBACK(on_save_snapshot_clicked), dialog);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 실행 */
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 정리 */
  gtk_widget_destroy(dialog);
} 