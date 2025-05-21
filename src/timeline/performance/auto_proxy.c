#include <gtk/gtk.h>
#include <string.h>
#include <gst/gst.h>
#include "timeline_performance.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 프록시 생성 설정 구조체 */
typedef struct {
  gboolean enabled;
  gint width;
  gint height;
  gchar *format;
  gchar *proxy_dir;
} BlouEditProxySettings;

/* 자동 프록시 활성화 설정 함수 */
void
blouedit_timeline_enable_auto_proxy(BlouEditTimeline *timeline, gboolean enabled)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->proxy_settings == NULL) {
    BlouEditProxySettings *settings = g_new0(BlouEditProxySettings, 1);
    
    settings->enabled = FALSE;
    settings->width = 1280;   /* 기본 프록시 해상도 */
    settings->height = 720;
    settings->format = g_strdup("mp4");
    settings->proxy_dir = g_build_filename(g_get_user_cache_dir(), "blouedit", "proxies", NULL);
    
    /* 디렉토리 생성 */
    g_mkdir_with_parents(settings->proxy_dir, 0755);
    
    timeline->proxy_settings = settings;
  }
  
  BlouEditProxySettings *settings = (BlouEditProxySettings *)timeline->proxy_settings;
  
  /* 현재 설정과 같으면 무시 */
  if (settings->enabled == enabled) {
    return;
  }
  
  /* 설정 변경 */
  settings->enabled = enabled;
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  /* 자동 프록시 변경 시그널 발생 (필요시) */
  g_signal_emit_by_name(timeline, "auto-proxy-changed", enabled);
}

/* 자동 프록시 활성화 상태 가져오기 */
gboolean
blouedit_timeline_get_auto_proxy_enabled(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->proxy_settings == NULL) {
    blouedit_timeline_enable_auto_proxy(timeline, FALSE);
  }
  
  BlouEditProxySettings *settings = (BlouEditProxySettings *)timeline->proxy_settings;
  
  return settings->enabled;
}

/* 프록시 해상도 설정 함수 */
void
blouedit_timeline_set_proxy_resolution(BlouEditTimeline *timeline, gint width, gint height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(width > 0 && height > 0);
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->proxy_settings == NULL) {
    blouedit_timeline_enable_auto_proxy(timeline, FALSE);
  }
  
  BlouEditProxySettings *settings = (BlouEditProxySettings *)timeline->proxy_settings;
  
  /* 설정 변경 */
  settings->width = width;
  settings->height = height;
}

/* 프록시 해상도 가져오기 함수 */
void
blouedit_timeline_get_proxy_resolution(BlouEditTimeline *timeline, gint *width, gint *height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(width != NULL && height != NULL);
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->proxy_settings == NULL) {
    blouedit_timeline_enable_auto_proxy(timeline, FALSE);
  }
  
  BlouEditProxySettings *settings = (BlouEditProxySettings *)timeline->proxy_settings;
  
  /* 해상도 반환 */
  *width = settings->width;
  *height = settings->height;
}

/* 클립에 대한 프록시 생성 함수 */
void
blouedit_timeline_generate_proxy_for_clip(BlouEditTimeline *timeline, GESClip *clip)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(GES_IS_CLIP(clip));
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->proxy_settings == NULL) {
    blouedit_timeline_enable_auto_proxy(timeline, FALSE);
  }
  
  BlouEditProxySettings *settings = (BlouEditProxySettings *)timeline->proxy_settings;
  
  /* 프록시가 필요한지 확인 */
  gboolean needs_proxy = FALSE;
  
  /* 클립 속성 확인 */
  if (GES_IS_URI_CLIP(clip)) {
    const gchar *uri = ges_uri_clip_get_uri(GES_URI_CLIP(clip));
    
    /* 실제 구현에서는 해상도, 비트레이트 등 확인하여 판단 */
    needs_proxy = TRUE;
    
    if (needs_proxy) {
      /* 프록시 생성 대화상자 */
      GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "프록시 생성",
        GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "_취소", GTK_RESPONSE_CANCEL,
        NULL);
      
      /* 대화상자 크기 설정 */
      gtk_window_set_default_size(GTK_WINDOW(dialog), 400, 150);
      
      /* 콘텐츠 영역 */
      GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
      gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
      gtk_box_set_spacing(GTK_BOX(content_area), 10);
      
      /* 정보 레이블 */
      gchar *filename = g_path_get_basename(uri);
      gchar *message = g_strdup_printf("'%s' 파일의 프록시를 생성하는 중...", filename);
      GtkWidget *label = gtk_label_new(message);
      gtk_container_add(GTK_CONTAINER(content_area), label);
      g_free(message);
      g_free(filename);
      
      /* 진행 바 */
      GtkWidget *progress = gtk_progress_bar_new();
      gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(progress), TRUE);
      gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress), "0%");
      gtk_container_add(GTK_CONTAINER(content_area), progress);
      
      /* 대화상자 표시 */
      gtk_widget_show_all(dialog);
      
      /* 프록시 생성 시뮬레이션 */
      guint timeout_id = g_timeout_add(100, (GSourceFunc)simulate_proxy_progress, progress);
      
      /* 대화상자 응답 처리 */
      if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_CANCEL) {
        /* 취소 처리 - 타임아웃 제거 */
        g_source_remove(timeout_id);
      }
      
      /* 대화상자 파괴 */
      gtk_widget_destroy(dialog);
    }
  }
}

/* 프록시 생성 진행 시뮬레이션 */
static gboolean
simulate_proxy_progress(gpointer user_data)
{
  GtkWidget *progress_bar = GTK_WIDGET(user_data);
  gdouble fraction = gtk_progress_bar_get_fraction(GTK_PROGRESS_BAR(progress_bar));
  
  /* 진행 상태 업데이트 */
  fraction += 0.01;
  if (fraction > 1.0) {
    fraction = 1.0;
    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), "완료됨");
    gtk_dialog_response(GTK_DIALOG(gtk_widget_get_toplevel(progress_bar)), GTK_RESPONSE_ACCEPT);
    return G_SOURCE_REMOVE;  /* 타이머 제거 */
  }
  
  gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar), fraction);
  gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), 
                          g_strdup_printf("%.0f%%", fraction * 100));
  
  return G_SOURCE_CONTINUE;  /* 타이머 계속 */
}

/* 모든 클립에 대한 프록시 생성 함수 */
void
blouedit_timeline_generate_proxies_for_all_clips(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 모든 클립에 대해 프록시 생성 */
  GList *clips = ges_timeline_get_clips(timeline->ges_timeline);
  
  if (clips == NULL) {
    /* 클립이 없음 알림 */
    GtkWidget *message = gtk_message_dialog_new(
      GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
      GTK_MESSAGE_INFO,
      GTK_BUTTONS_OK,
      "프록시를 생성할 클립이 없습니다.");
    
    gtk_dialog_run(GTK_DIALOG(message));
    gtk_widget_destroy(message);
    return;
  }
  
  /* 확인 대화상자 */
  gint clip_count = g_list_length(clips);
  GtkWidget *confirm = gtk_message_dialog_new(
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    GTK_MESSAGE_QUESTION,
    GTK_BUTTONS_YES_NO,
    "%d개 클립에 대한 프록시를 모두 생성하시겠습니까?", clip_count);
  
  if (gtk_dialog_run(GTK_DIALOG(confirm)) == GTK_RESPONSE_YES) {
    /* 프록시 생성 대화상자 */
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
      "프록시 일괄 생성",
      GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
      GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
      "_취소", GTK_RESPONSE_CANCEL,
      NULL);
    
    /* 대화상자 크기 설정 */
    gtk_window_set_default_size(GTK_WINDOW(dialog), 500, 300);
    
    /* 콘텐츠 영역 */
    GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
    gtk_box_set_spacing(GTK_BOX(content_area), 10);
    
    /* 정보 레이블 */
    gchar *message = g_strdup_printf("%d개 클립에 대한 프록시를 생성하는 중...", clip_count);
    GtkWidget *label = gtk_label_new(message);
    gtk_container_add(GTK_CONTAINER(content_area), label);
    g_free(message);
    
    /* 스크롤 윈도우 */
    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                 GTK_POLICY_AUTOMATIC,
                                 GTK_POLICY_AUTOMATIC);
    gtk_widget_set_vexpand(scroll, TRUE);
    gtk_container_add(GTK_CONTAINER(content_area), scroll);
    
    /* 목록 */
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(scroll), box);
    
    /* 각 클립에 대한 진행 상황 표시 */
    for (GList *l = clips; l != NULL; l = l->next) {
      GESClip *clip = GES_CLIP(l->data);
      
      if (GES_IS_URI_CLIP(clip)) {
        const gchar *uri = ges_uri_clip_get_uri(GES_URI_CLIP(clip));
        gchar *filename = g_path_get_basename(uri);
        
        /* 개별 클립 컨테이너 */
        GtkWidget *clip_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 3);
        gtk_widget_set_margin_top(clip_box, 5);
        gtk_widget_set_margin_bottom(clip_box, 5);
        gtk_container_add(GTK_CONTAINER(box), clip_box);
        
        /* 클립 이름 */
        GtkWidget *clip_label = gtk_label_new(filename);
        gtk_widget_set_halign(clip_label, GTK_ALIGN_START);
        gtk_container_add(GTK_CONTAINER(clip_box), clip_label);
        
        /* 진행 바 */
        GtkWidget *progress = gtk_progress_bar_new();
        gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(progress), TRUE);
        gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress), "대기 중...");
        gtk_container_add(GTK_CONTAINER(clip_box), progress);
        
        /* 진행 시뮬레이션 시작 */
        g_timeout_add(500 + g_random_int_range(0, 2000), (GSourceFunc)simulate_proxy_progress, progress);
        
        g_free(filename);
      }
    }
    
    /* 전체 진행 상황 */
    GtkWidget *total_label = gtk_label_new("전체 진행 상황:");
    gtk_widget_set_halign(total_label, GTK_ALIGN_START);
    gtk_widget_set_margin_top(total_label, 10);
    gtk_container_add(GTK_CONTAINER(content_area), total_label);
    
    GtkWidget *total_progress = gtk_progress_bar_new();
    gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(total_progress), TRUE);
    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(total_progress), "0%");
    gtk_container_add(GTK_CONTAINER(content_area), total_progress);
    
    /* 전체 진행 시뮬레이션 시작 */
    g_timeout_add(300, (GSourceFunc)simulate_proxy_progress, total_progress);
    
    /* 대화상자 표시 */
    gtk_widget_show_all(dialog);
    
    /* 대화상자 응답 처리 */
    gtk_dialog_run(GTK_DIALOG(dialog));
    
    /* 대화상자 파괴 */
    gtk_widget_destroy(dialog);
  }
  
  gtk_widget_destroy(confirm);
  g_list_free(clips);
}

/* 프록시 해상도 변경 버튼 핸들러 */
static void
on_resolution_button_clicked(GtkButton *button, gpointer user_data)
{
  GtkWidget *dialog = GTK_WIDGET(user_data);
  GtkWidget *preset_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(dialog), "preset-combo"));
  GtkWidget *width_spin = GTK_WIDGET(g_object_get_data(G_OBJECT(dialog), "width-spin"));
  GtkWidget *height_spin = GTK_WIDGET(g_object_get_data(G_OBJECT(dialog), "height-spin"));
  
  /* 선택된 프리셋 */
  gint active = gtk_combo_box_get_active(GTK_COMBO_BOX(preset_combo));
  
  /* 프리셋에 따른 해상도 설정 */
  switch (active) {
    case 0:  /* 사용자 정의 - 아무것도 안 함 */
      break;
    case 1:  /* 1080p */
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin), 1920);
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin), 1080);
      break;
    case 2:  /* 720p */
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin), 1280);
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin), 720);
      break;
    case 3:  /* 480p */
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin), 854);
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin), 480);
      break;
    case 4:  /* 360p */
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin), 640);
      gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin), 360);
      break;
    default:
      break;
  }
}

/* 프록시 설정 대화상자 */
void
blouedit_timeline_show_proxy_settings_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 설정이 초기화되지 않았으면 초기화 */
  if (timeline->proxy_settings == NULL) {
    blouedit_timeline_enable_auto_proxy(timeline, FALSE);
  }
  
  BlouEditProxySettings *settings = (BlouEditProxySettings *)timeline->proxy_settings;
  
  /* 대화상자 생성 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "프록시 설정",
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    "_취소", GTK_RESPONSE_CANCEL,
    "_저장", GTK_RESPONSE_ACCEPT,
    NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 450, 350);
  
  /* 콘텐츠 영역 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 안내 레이블 */
  GtkWidget *header_label = gtk_label_new(
    "프록시는 고해상도 영상의 편집 성능을 향상시키기 위해 저해상도 버전을 사용하는 기술입니다.");
  gtk_label_set_line_wrap(GTK_LABEL(header_label), TRUE);
  gtk_widget_set_halign(header_label, GTK_ALIGN_START);
  gtk_container_add(GTK_CONTAINER(content_area), header_label);
  
  /* 활성화 체크박스 */
  GtkWidget *enable_check = gtk_check_button_new_with_label("자동 프록시 생성 활성화");
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(enable_check), settings->enabled);
  gtk_container_add(GTK_CONTAINER(content_area), enable_check);
  
  /* 설정 그리드 */
  GtkWidget *grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
  gtk_grid_set_column_spacing(GTK_GRID(grid), 12);
  gtk_container_add(GTK_CONTAINER(content_area), grid);
  
  /* 해상도 프리셋 레이블 */
  GtkWidget *preset_label = gtk_label_new("해상도 프리셋:");
  gtk_widget_set_halign(preset_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), preset_label, 0, 0, 1, 1);
  
  /* 해상도 프리셋 콤보박스 */
  GtkWidget *preset_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(preset_combo), "사용자 정의");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(preset_combo), "1080p (1920x1080)");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(preset_combo), "720p (1280x720)");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(preset_combo), "480p (854x480)");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(preset_combo), "360p (640x360)");
  gtk_combo_box_set_active(GTK_COMBO_BOX(preset_combo), 0);
  gtk_grid_attach(GTK_GRID(grid), preset_combo, 1, 0, 1, 1);
  
  /* 해상도 적용 버튼 */
  GtkWidget *preset_button = gtk_button_new_with_label("적용");
  gtk_grid_attach(GTK_GRID(grid), preset_button, 2, 0, 1, 1);
  
  /* 너비 레이블 */
  GtkWidget *width_label = gtk_label_new("너비:");
  gtk_widget_set_halign(width_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), width_label, 0, 1, 1, 1);
  
  /* 너비 스핀 버튼 */
  GtkWidget *width_spin = gtk_spin_button_new_with_range(320, 3840, 16);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin), settings->width);
  gtk_grid_attach(GTK_GRID(grid), width_spin, 1, 1, 2, 1);
  
  /* 높이 레이블 */
  GtkWidget *height_label = gtk_label_new("높이:");
  gtk_widget_set_halign(height_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), height_label, 0, 2, 1, 1);
  
  /* 높이 스핀 버튼 */
  GtkWidget *height_spin = gtk_spin_button_new_with_range(240, 2160, 16);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin), settings->height);
  gtk_grid_attach(GTK_GRID(grid), height_spin, 1, 2, 2, 1);
  
  /* 포맷 레이블 */
  GtkWidget *format_label = gtk_label_new("프록시 형식:");
  gtk_widget_set_halign(format_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), format_label, 0, 3, 1, 1);
  
  /* 포맷 콤보박스 */
  GtkWidget *format_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(format_combo), "MP4 (H.264)");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(format_combo), "WebM (VP9)");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(format_combo), "MKV (H.265)");
  gtk_combo_box_set_active(GTK_COMBO_BOX(format_combo), 0);
  gtk_grid_attach(GTK_GRID(grid), format_combo, 1, 3, 2, 1);
  
  /* 프록시 저장 위치 */
  GtkWidget *dir_label = gtk_label_new("저장 위치:");
  gtk_widget_set_halign(dir_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), dir_label, 0, 4, 1, 1);
  
  /* 저장 위치 표시 */
  GtkWidget *dir_entry = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(dir_entry), settings->proxy_dir);
  gtk_entry_set_editable(GTK_ENTRY(dir_entry), FALSE);
  gtk_grid_attach(GTK_GRID(grid), dir_entry, 1, 4, 2, 1);
  
  /* 작업 버튼 영역 */
  GtkWidget *action_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
  gtk_widget_set_margin_top(action_box, 10);
  gtk_container_add(GTK_CONTAINER(content_area), action_box);
  
  /* 모든 클립 프록시 생성 버튼 */
  GtkWidget *generate_all_button = gtk_button_new_with_label("모든 클립 프록시 생성");
  gtk_container_add(GTK_CONTAINER(action_box), generate_all_button);
  
  /* 프록시 캐시 지우기 버튼 */
  GtkWidget *clear_button = gtk_button_new_with_label("프록시 캐시 비우기");
  gtk_container_add(GTK_CONTAINER(action_box), clear_button);
  
  /* 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "preset-combo", preset_combo);
  g_object_set_data(G_OBJECT(dialog), "width-spin", width_spin);
  g_object_set_data(G_OBJECT(dialog), "height-spin", height_spin);
  
  /* 시그널 핸들러 연결 */
  g_signal_connect(preset_button, "clicked", G_CALLBACK(on_resolution_button_clicked), dialog);
  g_signal_connect(generate_all_button, "clicked", G_CALLBACK(blouedit_timeline_generate_proxies_for_all_clips), timeline);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    /* 설정 저장 */
    settings->enabled = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(enable_check));
    settings->width = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(width_spin));
    settings->height = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(height_spin));
    
    /* 포맷 설정 */
    gint format_index = gtk_combo_box_get_active(GTK_COMBO_BOX(format_combo));
    g_free(settings->format);
    
    switch (format_index) {
      case 1:
        settings->format = g_strdup("webm");
        break;
      case 2:
        settings->format = g_strdup("mkv");
        break;
      case 0:
      default:
        settings->format = g_strdup("mp4");
        break;
    }
  }
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
}