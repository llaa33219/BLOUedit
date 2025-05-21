#include <gtk/gtk.h>
#include <string.h>
#include "timeline_performance.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 렌더링 캐시 설정 구조체 */
typedef struct {
  gboolean enabled;
  gchar *cache_dir;
  guint max_cache_size_mb;
  gboolean auto_cache_during_playback;
} BlouEditRenderingCacheSettings;

/* 캐시 초기화 함수 */
void
blouedit_timeline_init_rendering_cache(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 이미 초기화되었는지 확인 */
  if (timeline->rendering_cache_settings != NULL) {
    return;
  }
  
  /* 설정 구조체 생성 */
  BlouEditRenderingCacheSettings *settings = g_new0(BlouEditRenderingCacheSettings, 1);
  
  /* 기본 값 설정 */
  settings->enabled = FALSE;
  settings->cache_dir = g_build_filename(g_get_user_cache_dir(), "blouedit", "render_cache", NULL);
  settings->max_cache_size_mb = 1024;  /* 기본 1GB */
  settings->auto_cache_during_playback = TRUE;
  
  /* 캐시 디렉토리 생성 */
  g_mkdir_with_parents(settings->cache_dir, 0755);
  
  /* 타임라인에 설정 저장 */
  timeline->rendering_cache_settings = settings;
}

/* 캐시 활성화 설정 함수 */
void
blouedit_timeline_set_rendering_cache_enabled(BlouEditTimeline *timeline, gboolean enabled)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->rendering_cache_settings == NULL) {
    blouedit_timeline_init_rendering_cache(timeline);
  }
  
  BlouEditRenderingCacheSettings *settings = 
    (BlouEditRenderingCacheSettings *)timeline->rendering_cache_settings;
  
  /* 상태가 변경된 경우에만 처리 */
  if (settings->enabled != enabled) {
    settings->enabled = enabled;
    
    /* 캐시 비활성화 시 메모리 캐시 해제 */
    if (!enabled) {
      /* 메모리 캐시 해제 코드 (실제 구현에서 추가) */
    }
    
    /* 상태 변경 시그널 발생 (필요시) */
    g_signal_emit_by_name(timeline, "rendering-cache-state-changed", enabled);
  }
}

/* 캐시 활성화 상태 확인 함수 */
gboolean
blouedit_timeline_get_rendering_cache_enabled(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->rendering_cache_settings == NULL) {
    blouedit_timeline_init_rendering_cache(timeline);
  }
  
  BlouEditRenderingCacheSettings *settings = 
    (BlouEditRenderingCacheSettings *)timeline->rendering_cache_settings;
  
  return settings->enabled;
}

/* 캐시 지우기 함수 */
void
blouedit_timeline_clear_rendering_cache(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->rendering_cache_settings == NULL) {
    blouedit_timeline_init_rendering_cache(timeline);
    return;
  }
  
  BlouEditRenderingCacheSettings *settings = 
    (BlouEditRenderingCacheSettings *)timeline->rendering_cache_settings;
  
  /* 캐시 디렉토리 확인 */
  if (settings->cache_dir == NULL || !g_file_test(settings->cache_dir, G_FILE_TEST_IS_DIR)) {
    return;
  }
  
  /* 캐시 디렉토리 내 모든 파일 삭제 - 실제 구현에서는 더 안전한 방법 사용 */
  GDir *dir = g_dir_open(settings->cache_dir, 0, NULL);
  if (dir != NULL) {
    const gchar *filename;
    while ((filename = g_dir_read_name(dir)) != NULL) {
      gchar *file_path = g_build_filename(settings->cache_dir, filename, NULL);
      if (g_file_test(file_path, G_FILE_TEST_IS_REGULAR)) {
        g_unlink(file_path);
      }
      g_free(file_path);
    }
    g_dir_close(dir);
  }
  
  /* 메모리 캐시 지우기 (실제 구현에서 추가) */
  
  /* 캐시 지움 완료 시그널 발생 (필요시) */
  g_signal_emit_by_name(timeline, "rendering-cache-cleared");
}

/* 캐시 설정 변경 버튼 핸들러 */
static void
on_cache_dir_button_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  BlouEditRenderingCacheSettings *settings = 
    (BlouEditRenderingCacheSettings *)timeline->rendering_cache_settings;
  
  /* 폴더 선택 대화상자 생성 */
  GtkWidget *folder_dialog = gtk_file_chooser_dialog_new(
    "캐시 디렉토리 선택",
    GTK_WINDOW(dialog),
    GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
    "_취소", GTK_RESPONSE_CANCEL,
    "_선택", GTK_RESPONSE_ACCEPT,
    NULL);
  
  /* 현재 폴더 설정 */
  if (settings->cache_dir != NULL) {
    gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(folder_dialog), settings->cache_dir);
  }
  
  /* 대화상자 실행 */
  if (gtk_dialog_run(GTK_DIALOG(folder_dialog)) == GTK_RESPONSE_ACCEPT) {
    /* 새 캐시 디렉토리 설정 */
    gchar *new_dir = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(folder_dialog));
    
    /* 기존 디렉토리 해제 */
    g_free(settings->cache_dir);
    
    /* 새 디렉토리 설정 */
    settings->cache_dir = new_dir;
    
    /* 업데이트된 경로 표시 */
    GtkEntry *entry = GTK_ENTRY(g_object_get_data(G_OBJECT(dialog), "cache-dir-entry"));
    gtk_entry_set_text(entry, new_dir);
    
    /* 디렉토리 생성 */
    g_mkdir_with_parents(settings->cache_dir, 0755);
  }
  
  gtk_widget_destroy(folder_dialog);
}

/* 캐시 지우기 버튼 핸들러 */
static void
on_clear_cache_button_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  
  /* 확인 대화상자 */
  GtkWidget *confirm_dialog = gtk_message_dialog_new(
    GTK_WINDOW(dialog),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    GTK_MESSAGE_QUESTION,
    GTK_BUTTONS_YES_NO,
    "정말로 모든 렌더링 캐시를 지우시겠습니까?");
  
  /* 대화상자 실행 */
  if (gtk_dialog_run(GTK_DIALOG(confirm_dialog)) == GTK_RESPONSE_YES) {
    /* 캐시 지우기 */
    blouedit_timeline_clear_rendering_cache(timeline);
    
    /* 성공 메시지 */
    GtkWidget *info_dialog = gtk_message_dialog_new(
      GTK_WINDOW(dialog),
      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
      GTK_MESSAGE_INFO,
      GTK_BUTTONS_OK,
      "렌더링 캐시가 성공적으로 지워졌습니다.");
    
    gtk_dialog_run(GTK_DIALOG(info_dialog));
    gtk_widget_destroy(info_dialog);
  }
  
  gtk_widget_destroy(confirm_dialog);
}

/* 캐시 설정 대화상자 */
void
blouedit_timeline_configure_cache_settings(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->rendering_cache_settings == NULL) {
    blouedit_timeline_init_rendering_cache(timeline);
  }
  
  BlouEditRenderingCacheSettings *settings = 
    (BlouEditRenderingCacheSettings *)timeline->rendering_cache_settings;
  
  /* 대화상자 생성 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "타임라인 렌더링 캐시 설정",
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    "_취소", GTK_RESPONSE_CANCEL,
    "_저장", GTK_RESPONSE_ACCEPT,
    NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 500, 300);
  
  /* 콘텐츠 영역 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 활성화 체크박스 */
  GtkWidget *enable_check = gtk_check_button_new_with_label("렌더링 캐시 활성화");
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(enable_check), settings->enabled);
  gtk_container_add(GTK_CONTAINER(content_area), enable_check);
  
  /* 그리드 레이아웃 */
  GtkWidget *grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
  gtk_grid_set_column_spacing(GTK_GRID(grid), 12);
  gtk_container_add(GTK_CONTAINER(content_area), grid);
  
  /* 캐시 디렉토리 설정 */
  GtkWidget *dir_label = gtk_label_new("캐시 디렉토리:");
  gtk_widget_set_halign(dir_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), dir_label, 0, 0, 1, 1);
  
  GtkWidget *dir_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  gtk_grid_attach(GTK_GRID(grid), dir_box, 1, 0, 1, 1);
  
  GtkWidget *dir_entry = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(dir_entry), settings->cache_dir ? settings->cache_dir : "");
  gtk_widget_set_hexpand(dir_entry, TRUE);
  gtk_container_add(GTK_CONTAINER(dir_box), dir_entry);
  
  GtkWidget *dir_button = gtk_button_new_with_label("찾아보기...");
  gtk_container_add(GTK_CONTAINER(dir_box), dir_button);
  
  /* 최대 캐시 크기 설정 */
  GtkWidget *size_label = gtk_label_new("최대 캐시 크기 (MB):");
  gtk_widget_set_halign(size_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), size_label, 0, 1, 1, 1);
  
  GtkWidget *size_spin = gtk_spin_button_new_with_range(100, 10000, 100);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(size_spin), settings->max_cache_size_mb);
  gtk_grid_attach(GTK_GRID(grid), size_spin, 1, 1, 1, 1);
  
  /* 자동 캐싱 설정 */
  GtkWidget *auto_check = gtk_check_button_new_with_label("재생 중 자동 캐싱 활성화");
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(auto_check), settings->auto_cache_during_playback);
  gtk_grid_attach(GTK_GRID(grid), auto_check, 0, 2, 2, 1);
  
  /* 캐시 지우기 버튼 */
  GtkWidget *clear_button = gtk_button_new_with_label("모든 캐시 지우기");
  gtk_grid_attach(GTK_GRID(grid), clear_button, 0, 3, 2, 1);
  
  /* 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "cache-dir-entry", dir_entry);
  
  /* 시그널 핸들러 연결 */
  g_signal_connect(dir_button, "clicked", G_CALLBACK(on_cache_dir_button_clicked), dialog);
  g_signal_connect(clear_button, "clicked", G_CALLBACK(on_clear_cache_button_clicked), dialog);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    /* 설정 업데이트 */
    settings->enabled = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(enable_check));
    settings->max_cache_size_mb = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(size_spin));
    settings->auto_cache_during_playback = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(auto_check));
    
    /* 캐시 디렉토리는 이미 on_cache_dir_button_clicked에서 업데이트됨 */
    
    /* 디렉토리가 존재하는지 확인하고 필요시 생성 */
    if (settings->cache_dir != NULL) {
      g_mkdir_with_parents(settings->cache_dir, 0755);
    }
  }
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
} 