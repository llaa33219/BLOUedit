#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "timeline_scale.h"

/* 타임라인 눈금 모드 설정 함수 */
void 
blouedit_timeline_set_scale_mode(BlouEditTimeline *timeline, BlouEditTimelineScaleMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 구조체에 모드 저장 */
  timeline->scale_mode = mode;
  
  /* 타임라인 다시 그리기 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 타임라인 눈금 모드 가져오기 함수 */
BlouEditTimelineScaleMode 
blouedit_timeline_get_scale_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_SCALE_MODE_SECONDS);
  
  return timeline->scale_mode;
}

/* 타임라인 사용자 지정 눈금 간격 설정 함수 */
void 
blouedit_timeline_set_custom_scale_interval(BlouEditTimeline *timeline, gint64 interval)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(interval > 0);
  
  /* 구조체에 간격 저장 */
  timeline->custom_scale_interval = interval;
  
  /* 모드를 사용자 지정으로 설정 */
  timeline->scale_mode = BLOUEDIT_SCALE_MODE_CUSTOM;
  
  /* 타임라인 다시 그리기 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 타임라인 사용자 지정 눈금 간격 가져오기 함수 */
gint64 
blouedit_timeline_get_custom_scale_interval(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), 1000); /* 기본값 1초 */
  
  return timeline->custom_scale_interval;
}

/* 모드에 따른 눈금 간격 계산 함수 */
gint64 
blouedit_timeline_calculate_scale_interval(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), 1000);
  
  gdouble framerate = blouedit_timeline_get_framerate(timeline);
  
  switch (timeline->scale_mode) {
    case BLOUEDIT_SCALE_MODE_FRAMES:
      return (gint64)(GST_SECOND / framerate); /* 1 프레임 간격 */
      
    case BLOUEDIT_SCALE_MODE_SECONDS:
      return GST_SECOND; /* 1초 간격 */
      
    case BLOUEDIT_SCALE_MODE_MINUTES:
      return GST_SECOND * 60; /* 1분 간격 */
      
    case BLOUEDIT_SCALE_MODE_HOURS:
      return GST_SECOND * 60 * 60; /* 1시간 간격 */
      
    case BLOUEDIT_SCALE_MODE_CUSTOM:
      return timeline->custom_scale_interval; /* 사용자 지정 간격 */
      
    default:
      return GST_SECOND; /* 기본값 1초 */
  }
}

/* 눈금 설정 대화상자에서 사용자 정의 간격 입력 활성화/비활성화 */
static void
on_scale_mode_changed(GtkComboBox *combo, GtkWidget *custom_box)
{
  gint active = gtk_combo_box_get_active(combo);
  
  /* 사용자 정의 모드인 경우에만 사용자 정의 간격 입력란 활성화 */
  gtk_widget_set_sensitive(custom_box, active == BLOUEDIT_SCALE_MODE_CUSTOM);
}

/* 눈금 설정 대화상자 응답 처리 콜백 */
static void
on_scale_settings_response(GtkDialog *dialog, gint response_id, gpointer user_data)
{
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(dialog), "timeline");
  
  if (response_id == GTK_RESPONSE_OK) {
    GtkWidget *mode_combo = g_object_get_data(G_OBJECT(dialog), "mode-combo");
    GtkWidget *custom_spin = g_object_get_data(G_OBJECT(dialog), "custom-spin");
    GtkWidget *unit_combo = g_object_get_data(G_OBJECT(dialog), "unit-combo");
    
    gint mode = gtk_combo_box_get_active(GTK_COMBO_BOX(mode_combo));
    gint64 custom_value = gtk_spin_button_get_value(GTK_SPIN_BUTTON(custom_spin));
    gint unit = gtk_combo_box_get_active(GTK_COMBO_BOX(unit_combo));
    
    /* 값 적용 */
    blouedit_timeline_set_scale_mode(timeline, (BlouEditTimelineScaleMode)mode);
    
    /* 사용자 정의 간격을 선택한 경우, 단위에 따라 값 변환 */
    if (mode == BLOUEDIT_SCALE_MODE_CUSTOM) {
      gint64 interval = 0;
      
      switch (unit) {
        case 0: /* 프레임 */
          interval = (gint64)(GST_SECOND / blouedit_timeline_get_framerate(timeline) * custom_value);
          break;
          
        case 1: /* 밀리초 */
          interval = GST_MSECOND * custom_value;
          break;
          
        case 2: /* 초 */
          interval = GST_SECOND * custom_value;
          break;
          
        case 3: /* 분 */
          interval = GST_SECOND * 60 * custom_value;
          break;
          
        default:
          interval = GST_SECOND * custom_value;
          break;
      }
      
      blouedit_timeline_set_custom_scale_interval(timeline, interval);
    }
  }
  
  /* 대화상자 닫기 */
  gtk_widget_destroy(GTK_WIDGET(dialog));
}

/* 타임라인 눈금 설정 대화상자 표시 함수 */
void 
blouedit_timeline_show_scale_settings(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *grid;
  GtkWidget *mode_label, *mode_combo;
  GtkWidget *custom_box, *custom_label, *custom_spin, *unit_combo;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("타임라인 눈금 설정",
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
  
  /* 모드 선택 */
  mode_label = gtk_label_new("눈금 모드:");
  gtk_widget_set_halign(mode_label, GTK_ALIGN_END);
  
  mode_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "초 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "분 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "시간 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "프레임 단위");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "사용자 지정");
  
  /* 현재 모드 선택 */
  gtk_combo_box_set_active(GTK_COMBO_BOX(mode_combo), blouedit_timeline_get_scale_mode(timeline));
  
  gtk_grid_attach(GTK_GRID(grid), mode_label, 0, 0, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), mode_combo, 1, 0, 1, 1);
  
  /* 사용자 지정 간격 입력 */
  custom_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
  
  custom_label = gtk_label_new("간격:");
  gtk_widget_set_halign(custom_label, GTK_ALIGN_END);
  
  custom_spin = gtk_spin_button_new_with_range(1, 1000, 1);
  gtk_spin_button_set_digits(GTK_SPIN_BUTTON(custom_spin), 0);
  
  unit_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(unit_combo), "프레임");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(unit_combo), "밀리초");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(unit_combo), "초");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(unit_combo), "분");
  
  /* 기본 단위는 초로 설정 */
  gtk_combo_box_set_active(GTK_COMBO_BOX(unit_combo), 2);
  
  /* 현재 사용자 지정 간격 설정 */
  gint64 interval = blouedit_timeline_get_custom_scale_interval(timeline);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(custom_spin), interval / GST_SECOND);
  
  /* 사용자 지정 박스에 위젯 추가 */
  gtk_box_pack_start(GTK_BOX(custom_box), custom_spin, FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(custom_box), unit_combo, FALSE, FALSE, 0);
  
  gtk_grid_attach(GTK_GRID(grid), custom_label, 0, 1, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), custom_box, 1, 1, 1, 1);
  
  /* 현재 모드에 따라 사용자 지정 입력란 활성화/비활성화 */
  gtk_widget_set_sensitive(
    custom_box, 
    blouedit_timeline_get_scale_mode(timeline) == BLOUEDIT_SCALE_MODE_CUSTOM
  );
  
  /* 모드 변경 시 사용자 지정 입력란 활성화/비활성화 */
  g_signal_connect(mode_combo, "changed", G_CALLBACK(on_scale_mode_changed), custom_box);
  
  /* 응답 핸들러 설정 */
  g_signal_connect(dialog, "response", G_CALLBACK(on_scale_settings_response), NULL);
  
  /* 필요한 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "mode-combo", mode_combo);
  g_object_set_data(G_OBJECT(dialog), "custom-spin", custom_spin);
  g_object_set_data(G_OBJECT(dialog), "unit-combo", unit_combo);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
} 