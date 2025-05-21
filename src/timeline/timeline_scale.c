#include <gtk/gtk.h>
#include <string.h>
#include <math.h>
#include "timeline_scale.h"
#include "core/types.h"
#include "core/timeline.h"

/* 타임라인 구조체에 필요한 필드를 추가했다고 가정합니다.
   필요한 필드:
   - BlouEditTimelineScaleMode scale_mode
   - guint64 custom_scale_interval
*/

/* 타임라인 눈금 모드 설정 함수 */
void 
blouedit_timeline_set_scale_mode(BlouEditTimeline *timeline, 
                              BlouEditTimelineScaleMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  timeline->scale_mode = mode;
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 타임라인 눈금 모드 가져오기 함수 */
BlouEditTimelineScaleMode 
blouedit_timeline_get_scale_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_TIMELINE_SCALE_SECONDS);
  
  return timeline->scale_mode;
}

/* 타임라인 사용자 지정 눈금 간격 설정 함수 */
void 
blouedit_timeline_set_custom_scale_interval(BlouEditTimeline *timeline, 
                                        guint64 interval_ms)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(interval_ms > 0);
  
  timeline->custom_scale_interval = interval_ms;
  
  /* 모드가 사용자 지정이 아니라면 사용자 지정으로 변경 */
  if (timeline->scale_mode != BLOUEDIT_TIMELINE_SCALE_CUSTOM) {
    timeline->scale_mode = BLOUEDIT_TIMELINE_SCALE_CUSTOM;
  }
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 타임라인 사용자 지정 눈금 간격 가져오기 함수 */
guint64 
blouedit_timeline_get_custom_scale_interval(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), 1000); /* 기본값 1초 */
  
  return timeline->custom_scale_interval;
}

/* 타임라인 눈금에 사용할 적절한 간격 계산 */
static guint64
calculate_scale_interval(BlouEditTimeline *timeline, gdouble zoom_level)
{
  guint64 interval_ms;
  
  /* 현재 모드에 따라 기본 간격 설정 */
  switch (timeline->scale_mode) {
    case BLOUEDIT_TIMELINE_SCALE_SECONDS:
      interval_ms = 1000; /* 1초 */
      break;
    case BLOUEDIT_TIMELINE_SCALE_MINUTES:
      interval_ms = 60000; /* 1분 */
      break;
    case BLOUEDIT_TIMELINE_SCALE_HOURS:
      interval_ms = 3600000; /* 1시간 */
      break;
    case BLOUEDIT_TIMELINE_SCALE_FRAMES:
      /* 프레임레이트를 고려하여 계산 (기본값 30fps 가정) */
      interval_ms = (guint64)(1000.0 / (timeline->framerate > 0 ? timeline->framerate : 30.0));
      break;
    case BLOUEDIT_TIMELINE_SCALE_CUSTOM:
      interval_ms = timeline->custom_scale_interval;
      break;
    default:
      interval_ms = 1000; /* 기본값 1초 */
  }
  
  /* 줌 레벨에 따라 간격 조정 */
  if (zoom_level > 2.0) {
    /* 줌인 상태에서는 더 작은 간격으로 표시 */
    interval_ms /= 2;
    
    /* 더 많이 줌인된 경우 더 작은 간격으로 */
    if (zoom_level > 4.0) {
      interval_ms /= 2;
    }
  } else if (zoom_level < 0.5) {
    /* 줌아웃 상태에서는 더 큰 간격으로 표시 */
    interval_ms *= 2;
    
    /* 더 많이 줌아웃된 경우 더 큰 간격으로 */
    if (zoom_level < 0.25) {
      interval_ms *= 2;
    }
  }
  
  return interval_ms;
}

/* 타임코드 포맷팅 함수 - 내부 사용 */
static gchar* 
format_timecode_for_scale(BlouEditTimeline *timeline, gint64 position_ms)
{
  switch (timeline->scale_mode) {
    case BLOUEDIT_TIMELINE_SCALE_SECONDS:
      return g_strdup_printf("%.1fs", position_ms / 1000.0);
    
    case BLOUEDIT_TIMELINE_SCALE_MINUTES: {
      gint minutes = position_ms / 60000;
      gint seconds = (position_ms % 60000) / 1000;
      return g_strdup_printf("%d:%02d", minutes, seconds);
    }
    
    case BLOUEDIT_TIMELINE_SCALE_HOURS: {
      gint hours = position_ms / 3600000;
      gint minutes = (position_ms % 3600000) / 60000;
      return g_strdup_printf("%d:%02d", hours, minutes);
    }
    
    case BLOUEDIT_TIMELINE_SCALE_FRAMES: {
      /* 프레임레이트를 고려하여 계산 (기본값 30fps 가정) */
      gdouble fps = timeline->framerate > 0 ? timeline->framerate : 30.0;
      gint64 frames = (gint64)(position_ms * fps / 1000.0);
      return g_strdup_printf("f%lld", frames);
    }
    
    case BLOUEDIT_TIMELINE_SCALE_CUSTOM:
    default:
      /* 사용자 지정 모드나 기본값은 밀리초를 초로 표시 */
      return g_strdup_printf("%.1fs", position_ms / 1000.0);
  }
}

/* 타임라인 눈금 그리기 함수 */
void 
blouedit_timeline_draw_scale(BlouEditTimeline *timeline, 
                         cairo_t *cr, 
                         gint width, 
                         gint ruler_height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(cr != NULL);
  
  /* 폰트 설정 */
  cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  cairo_set_font_size(cr, 10);
  
  /* 눈금 간격 계산 */
  guint64 interval_ms = calculate_scale_interval(timeline, timeline->zoom_level);
  gdouble pixels_per_ms = timeline->zoom_level / 10.0; /* 밀리초당 픽셀 수 (예: zoom_level 1.0에서 10픽셀/100ms) */
  
  /* 타임라인 시작 위치 (오프셋) 계산 */
  gint timeline_start_x = timeline->timeline_start_x;
  gdouble pixel_interval = interval_ms * pixels_per_ms;
  
  /* 픽셀 간격이 너무 작으면 조정 (최소 20픽셀) */
  if (pixel_interval < 20) {
    interval_ms = (guint64)(20.0 / pixels_per_ms);
    pixel_interval = interval_ms * pixels_per_ms;
  }
  
  /* 표시 영역의 첫 눈금 위치 계산 */
  gint64 start_position = timeline->horizontal_scroll_position;
  gint64 first_mark_position = (start_position / interval_ms) * interval_ms;
  if (first_mark_position < start_position) {
    first_mark_position += interval_ms;
  }
  
  /* 눈금선 및 라벨 그리기 */
  cairo_set_line_width(cr, 1.0);
  
  for (gint64 position_ms = first_mark_position; 
       position_ms < start_position + (width / pixels_per_ms); 
       position_ms += interval_ms) {
    
    /* 픽셀 위치 계산 */
    gdouble x = timeline_start_x + ((position_ms - start_position) * pixels_per_ms);
    
    /* 경계 확인 */
    if (x < timeline_start_x || x > width) {
      continue;
    }
    
    /* 눈금선 그리기 */
    cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.8);
    cairo_move_to(cr, x, 0);
    cairo_line_to(cr, x, ruler_height);
    cairo_stroke(cr);
    
    /* 라벨 그리기 */
    gchar *label = format_timecode_for_scale(timeline, position_ms);
    cairo_text_extents_t extents;
    cairo_text_extents(cr, label, &extents);
    
    cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 1.0);
    cairo_move_to(cr, x - (extents.width / 2), ruler_height - 5);
    cairo_show_text(cr, label);
    
    g_free(label);
  }
}

/* 타임라인 스케일 설정 대화상자 응답 처리 */
static void
on_scale_settings_response(GtkDialog *dialog, gint response_id, gpointer user_data)
{
  if (response_id == GTK_RESPONSE_ACCEPT) {
    BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
    GtkComboBox *mode_combo = GTK_COMBO_BOX(g_object_get_data(G_OBJECT(dialog), "mode-combo"));
    GtkSpinButton *custom_interval = GTK_SPIN_BUTTON(g_object_get_data(G_OBJECT(dialog), "custom-interval"));
    
    /* 모드 가져오기 */
    gint active = gtk_combo_box_get_active(mode_combo);
    BlouEditTimelineScaleMode mode = (BlouEditTimelineScaleMode)active;
    
    /* 모드 설정 */
    blouedit_timeline_set_scale_mode(timeline, mode);
    
    /* 사용자 지정 모드인 경우 간격 설정 */
    if (mode == BLOUEDIT_TIMELINE_SCALE_CUSTOM) {
      guint64 interval_ms = (guint64)gtk_spin_button_get_value(custom_interval);
      blouedit_timeline_set_custom_scale_interval(timeline, interval_ms);
    }
  }
  
  /* 대화상자 닫기 */
  gtk_widget_destroy(GTK_WIDGET(dialog));
}

/* 콤보박스 변경 핸들러 */
static void
on_mode_combo_changed(GtkComboBox *combo, gpointer user_data)
{
  GtkWidget *custom_box = GTK_WIDGET(user_data);
  gint active = gtk_combo_box_get_active(combo);
  
  /* 사용자 지정 모드인 경우 사용자 지정 영역 활성화 */
  if (active == BLOUEDIT_TIMELINE_SCALE_CUSTOM) {
    gtk_widget_set_sensitive(custom_box, TRUE);
  } else {
    gtk_widget_set_sensitive(custom_box, FALSE);
  }
}

/* 타임라인 눈금 설정 대화상자 표시 함수 */
void 
blouedit_timeline_show_scale_settings_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *mode_label, *mode_combo;
  GtkWidget *custom_box, *custom_label, *custom_interval;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("타임라인 눈금 설정",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_취소", GTK_RESPONSE_CANCEL,
                                     "_적용", GTK_RESPONSE_ACCEPT,
                                     NULL);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 모드 선택 UI */
  GtkWidget *mode_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
  gtk_container_add(GTK_CONTAINER(content_area), mode_box);
  
  mode_label = gtk_label_new("눈금 모드:");
  gtk_container_add(GTK_CONTAINER(mode_box), mode_label);
  
  mode_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "초 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "분 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "시간 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "프레임 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "사용자 지정");
  gtk_combo_box_set_active(GTK_COMBO_BOX(mode_combo), (gint)timeline->scale_mode);
  gtk_container_add(GTK_CONTAINER(mode_box), mode_combo);
  gtk_widget_set_hexpand(mode_combo, TRUE);
  
  /* 사용자 지정 간격 설정 UI */
  custom_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
  gtk_container_add(GTK_CONTAINER(content_area), custom_box);
  
  custom_label = gtk_label_new("사용자 지정 간격 (ms):");
  gtk_container_add(GTK_CONTAINER(custom_box), custom_label);
  
  custom_interval = gtk_spin_button_new_with_range(1, 100000, 100);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(custom_interval), (gdouble)timeline->custom_scale_interval);
  gtk_container_add(GTK_CONTAINER(custom_box), custom_interval);
  
  /* 현재 모드가 사용자 지정이 아니면 사용자 지정 영역 비활성화 */
  if (timeline->scale_mode != BLOUEDIT_TIMELINE_SCALE_CUSTOM) {
    gtk_widget_set_sensitive(custom_box, FALSE);
  }
  
  /* 모드 변경 시 사용자 지정 영역 활성화/비활성화 */
  g_signal_connect(mode_combo, "changed", G_CALLBACK(on_mode_combo_changed), custom_box);
  
  /* 응답 핸들러 설정 */
  g_signal_connect(dialog, "response", G_CALLBACK(on_scale_settings_response), timeline);
  
  /* 필요한 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "mode-combo", mode_combo);
  g_object_set_data(G_OBJECT(dialog), "custom-interval", custom_interval);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
} 