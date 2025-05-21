#include <gtk/gtk.h>
#include <string.h>
#include "timeline_history.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 자동 저장 경로 */
#define BLOUEDIT_AUTO_SAVE_DIR "~/.config/blouedit/auto_save"

/* 자동 저장 콜백 함수 ID */
static guint auto_save_callback_id = 0;

/* 자동 저장 디렉토리 확인 및 생성 함수 */
static gchar*
ensure_auto_save_dir(BlouEditTimeline *timeline)
{
  gchar *dir_path = g_strdup_printf("%s/%s", 
                                  BLOUEDIT_AUTO_SAVE_DIR,
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

/* 타임라인 자동 저장 함수 - 내부 사용 */
static void
auto_save_timeline(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 자동 저장 디렉토리 */
  gchar *dir_path = ensure_auto_save_dir(timeline);
  
  /* 자동 저장 파일 경로 */
  gchar *auto_save_path = g_strdup_printf("%s/auto_save.xges", dir_path);
  
  /* 타임라인 직렬화 (timeline_snapshot.c에서 가져온 함수 호출) */
  extern gchar* serialize_timeline_to_xml(BlouEditTimeline *timeline);
  gchar *xml_data = serialize_timeline_to_xml(timeline);
  
  if (xml_data != NULL) {
    /* 파일에 저장 */
    GError *error = NULL;
    if (!g_file_set_contents(auto_save_path, xml_data, -1, &error)) {
      g_warning("자동 저장 오류: %s", error->message);
      g_error_free(error);
    } else {
      /* 메타데이터 파일 생성 */
      gchar *meta_path = g_strdup_printf("%s/auto_save.meta", dir_path);
      
      /* JSON 형식으로 메타데이터 저장 */
      GString *meta_data = g_string_new("{\n");
      
      /* 타임스탬프 추가 */
      GDateTime *now = g_date_time_new_now_local();
      gchar *date_str = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
      g_string_append_printf(meta_data, "  \"timestamp\": \"%s\"\n}", date_str);
      g_free(date_str);
      g_date_time_unref(now);
      
      /* 메타데이터 파일 저장 */
      if (!g_file_set_contents(meta_path, meta_data->str, -1, &error)) {
        g_warning("자동 저장 메타데이터 오류: %s", error->message);
        g_error_free(error);
      }
      
      g_string_free(meta_data, TRUE);
      g_free(meta_path);
    }
    
    g_free(xml_data);
  }
  
  g_free(auto_save_path);
  g_free(dir_path);
}

/* 자동 저장 콜백 함수 */
static gboolean
auto_save_callback(gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  
  /* 타임라인이 유효한지 확인 */
  if (!BLOUEDIT_IS_TIMELINE(timeline)) {
    auto_save_callback_id = 0;
    return G_SOURCE_REMOVE;
  }
  
  /* 자동 저장 실행 */
  auto_save_timeline(timeline);
  
  return G_SOURCE_CONTINUE;
}

/* 자동 저장 설정 함수 */
void
blouedit_timeline_setup_auto_save(BlouEditTimeline *timeline, guint interval_seconds)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(interval_seconds > 0);
  
  /* 기존 자동 저장 해제 */
  if (auto_save_callback_id > 0) {
    g_source_remove(auto_save_callback_id);
    auto_save_callback_id = 0;
  }
  
  /* 자동 저장 간격 설정 */
  timeline->auto_save_interval = interval_seconds;
  
  /* 자동 저장 활성화 */
  auto_save_callback_id = g_timeout_add_seconds(interval_seconds, 
                                             auto_save_callback, 
                                             timeline);
}

/* 자동 저장 복원 함수 */
gboolean
blouedit_timeline_restore_auto_save(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 자동 저장 디렉토리 */
  gchar *dir_path = ensure_auto_save_dir(timeline);
  
  /* 자동 저장 파일 경로 */
  gchar *auto_save_path = g_strdup_printf("%s/auto_save.xges", dir_path);
  
  /* 메타데이터 파일 경로 */
  gchar *meta_path = g_strdup_printf("%s/auto_save.meta", dir_path);
  
  /* 자동 저장 파일 존재 확인 */
  if (!g_file_test(auto_save_path, G_FILE_TEST_EXISTS)) {
    g_free(auto_save_path);
    g_free(meta_path);
    g_free(dir_path);
    return FALSE;
  }
  
  /* 메타데이터 파일 읽기 */
  GDateTime *save_time = NULL;
  if (g_file_test(meta_path, G_FILE_TEST_EXISTS)) {
    GError *error = NULL;
    gchar *meta_content = NULL;
    
    if (g_file_get_contents(meta_path, &meta_content, NULL, &error)) {
      /* 간단한 타임스탬프 파싱 (실제 구현에서는 JSON 파서 사용 권장) */
      gchar *timestamp_start = strstr(meta_content, "\"timestamp\": \"");
      if (timestamp_start) {
        timestamp_start += 14; /* "timestamp": " 길이 */
        gchar *timestamp_end = strchr(timestamp_start, '"');
        if (timestamp_end) {
          *timestamp_end = '\0';
          save_time = g_date_time_new_from_iso8601(timestamp_start, NULL);
        }
      }
      
      g_free(meta_content);
    } else {
      g_warning("자동 저장 메타데이터 읽기 오류: %s", error->message);
      g_error_free(error);
    }
  }
  
  /* 파일 내용 읽기 */
  GError *error = NULL;
  gchar *xml_data = NULL;
  
  if (!g_file_get_contents(auto_save_path, &xml_data, NULL, &error)) {
    g_warning("자동 저장 파일 읽기 오류: %s", error->message);
    g_error_free(error);
    if (save_time) g_date_time_unref(save_time);
    g_free(auto_save_path);
    g_free(meta_path);
    g_free(dir_path);
    return FALSE;
  }
  
  /* 복원 전 확인 대화상자 */
  GtkWidget *parent = gtk_widget_get_toplevel(GTK_WIDGET(timeline));
  GtkWidget *dialog;
  
  if (save_time) {
    gchar *time_str = g_date_time_format(save_time, "%Y-%m-%d %H:%M:%S");
    dialog = gtk_message_dialog_new(GTK_WINDOW(parent),
                                  GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                  GTK_MESSAGE_QUESTION,
                                  GTK_BUTTONS_YES_NO,
                                  "자동 저장된 프로젝트가 있습니다 (%s).\n\n복원하시겠습니까?",
                                  time_str);
    g_free(time_str);
  } else {
    dialog = gtk_message_dialog_new(GTK_WINDOW(parent),
                                  GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                  GTK_MESSAGE_QUESTION,
                                  GTK_BUTTONS_YES_NO,
                                  "자동 저장된 프로젝트가 있습니다.\n\n복원하시겠습니까?");
  }
  
  gint response = gtk_dialog_run(GTK_DIALOG(dialog));
  gtk_widget_destroy(dialog);
  
  if (response != GTK_RESPONSE_YES) {
    /* 사용자가 복원을 취소함 */
    if (save_time) g_date_time_unref(save_time);
    g_free(xml_data);
    g_free(auto_save_path);
    g_free(meta_path);
    g_free(dir_path);
    return FALSE;
  }
  
  /* 타임라인 복원 (timeline_snapshot.c에서 가져온 함수 호출) */
  extern gboolean deserialize_timeline_from_xml(BlouEditTimeline *timeline, const gchar *xml_data);
  gboolean result = deserialize_timeline_from_xml(timeline, xml_data);
  
  /* 정리 */
  if (save_time) g_date_time_unref(save_time);
  g_free(xml_data);
  g_free(auto_save_path);
  g_free(meta_path);
  g_free(dir_path);
  
  return result;
}

/* 자동 저장 파일 지우기 */
void
blouedit_timeline_clear_auto_saves(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 자동 저장 디렉토리 */
  gchar *dir_path = ensure_auto_save_dir(timeline);
  
  /* 자동 저장 파일 경로 */
  gchar *auto_save_path = g_strdup_printf("%s/auto_save.xges", dir_path);
  
  /* 메타데이터 파일 경로 */
  gchar *meta_path = g_strdup_printf("%s/auto_save.meta", dir_path);
  
  /* 파일 삭제 */
  if (g_file_test(auto_save_path, G_FILE_TEST_EXISTS)) {
    g_unlink(auto_save_path);
  }
  
  if (g_file_test(meta_path, G_FILE_TEST_EXISTS)) {
    g_unlink(meta_path);
  }
  
  /* 정리 */
  g_free(auto_save_path);
  g_free(meta_path);
  g_free(dir_path);
} 