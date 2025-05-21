#include <gtk/gtk.h>
#include <string.h>
#include "timeline_performance.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 성능 모드 설정 함수 */
void
blouedit_timeline_set_performance_mode(BlouEditTimeline *timeline, BlouEditPerformanceMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 현재 모드와 같은 모드로 설정하려는 경우 무시 */
  if (timeline->performance_mode == mode) {
    return;
  }
  
  /* 모드 설정 */
  timeline->performance_mode = mode;
  
  /* 모드별 최적화 설정 적용 */
  switch (mode) {
    case BLOUEDIT_PERFORMANCE_MODE_QUALITY:
      /* 고품질 재생 모드 설정 */
      timeline->playback_quality = 1.0;  /* 최대 품질 */
      timeline->use_hardware_decoding = TRUE;
      break;
      
    case BLOUEDIT_PERFORMANCE_MODE_RESPONSIVE:
      /* 반응성 우선 모드 설정 */
      timeline->playback_quality = 0.5;  /* 낮은 품질 */
      timeline->use_hardware_decoding = TRUE;
      break;
      
    case BLOUEDIT_PERFORMANCE_MODE_BALANCED:
    default:
      /* 균형 모드 설정 */
      timeline->playback_quality = 0.75;  /* 중간 품질 */
      timeline->use_hardware_decoding = TRUE;
      break;
  }
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  /* 성능 모드 변경 시그널 발생 (필요시) */
  g_signal_emit_by_name(timeline, "performance-mode-changed", mode);
}

/* 현재 성능 모드 가져오기 함수 */
BlouEditPerformanceMode
blouedit_timeline_get_performance_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_PERFORMANCE_MODE_BALANCED);
  
  return timeline->performance_mode;
}

/* 성능 모드별 설명 문자열 */
static const gchar*
get_performance_mode_description(BlouEditPerformanceMode mode)
{
  switch (mode) {
    case BLOUEDIT_PERFORMANCE_MODE_QUALITY:
      return "최고 품질의 재생을 위한 설정입니다. 고해상도 영상과 효과를 처리할 때 가장 좋지만\n"
             "시스템 요구 사항이 높아 편집 반응성이 떨어질 수 있습니다.";
      
    case BLOUEDIT_PERFORMANCE_MODE_RESPONSIVE:
      return "편집 작업의 반응성을 최우선으로 하는 설정입니다. 재생 품질이 다소 떨어질 수 있지만\n"
             "편집 작업이 더 빠르고 원활하게 진행됩니다.";
      
    case BLOUEDIT_PERFORMANCE_MODE_BALANCED:
    default:
      return "재생 품질과 편집 반응성 사이의 균형을 맞춘 설정입니다.\n"
             "대부분의 프로젝트에 적합한 기본 설정입니다.";
  }
}

/* 성능 모드 선택 변경 핸들러 */
static void
on_performance_mode_changed(GtkToggleButton *button, gpointer user_data)
{
  if (!gtk_toggle_button_get_active(button)) {
    return;  /* 비활성화되는 버튼은 무시 */
  }
  
  GtkWidget *dialog = GTK_WIDGET(user_data);
  GtkLabel *description_label = GTK_LABEL(g_object_get_data(G_OBJECT(dialog), "description-label"));
  
  /* 선택된 모드 식별 */
  BlouEditPerformanceMode mode;
  const gchar *mode_name = gtk_buildable_get_name(GTK_BUILDABLE(button));
  
  if (g_strcmp0(mode_name, "quality-radio") == 0) {
    mode = BLOUEDIT_PERFORMANCE_MODE_QUALITY;
  } 
  else if (g_strcmp0(mode_name, "responsive-radio") == 0) {
    mode = BLOUEDIT_PERFORMANCE_MODE_RESPONSIVE;
  } 
  else {
    mode = BLOUEDIT_PERFORMANCE_MODE_BALANCED;
  }
  
  /* 설명 업데이트 */
  gtk_label_set_text(description_label, get_performance_mode_description(mode));
  
  /* 선택된 모드 저장 */
  g_object_set_data(G_OBJECT(dialog), "selected-mode", GINT_TO_POINTER(mode));
}

/* 성능 설정 대화상자 */
void
blouedit_timeline_show_performance_settings_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 현재 모드 */
  BlouEditPerformanceMode current_mode = blouedit_timeline_get_performance_mode(timeline);
  
  /* 대화상자 생성 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "성능 모드 설정",
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    "_취소", GTK_RESPONSE_CANCEL,
    "_적용", GTK_RESPONSE_ACCEPT,
    NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 450, 350);
  
  /* 콘텐츠 영역 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 안내 레이블 */
  GtkWidget *header_label = gtk_label_new("성능 모드는 재생 품질과 편집 반응성 사이의 균형을 조절합니다.");
  gtk_widget_set_halign(header_label, GTK_ALIGN_START);
  gtk_container_add(GTK_CONTAINER(content_area), header_label);
  
  /* 옵션 프레임 */
  GtkWidget *frame = gtk_frame_new("모드 선택");
  gtk_container_add(GTK_CONTAINER(content_area), frame);
  
  /* 옵션 컨테이너 */
  GtkWidget *options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_container_set_border_width(GTK_CONTAINER(options_box), 10);
  gtk_container_add(GTK_CONTAINER(frame), options_box);
  
  /* 라디오 버튼 그룹 */
  GSList *group = NULL;
  
  /* 품질 우선 모드 */
  GtkWidget *quality_radio = gtk_radio_button_new_with_label(group, "재생 품질 우선");
  gtk_buildable_set_name(GTK_BUILDABLE(quality_radio), "quality-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(quality_radio));
  gtk_container_add(GTK_CONTAINER(options_box), quality_radio);
  
  /* 균형 모드 */
  GtkWidget *balanced_radio = gtk_radio_button_new_with_label(group, "균형 모드");
  gtk_buildable_set_name(GTK_BUILDABLE(balanced_radio), "balanced-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(balanced_radio));
  gtk_container_add(GTK_CONTAINER(options_box), balanced_radio);
  
  /* 반응성 우선 모드 */
  GtkWidget *responsive_radio = gtk_radio_button_new_with_label(group, "편집 반응성 우선");
  gtk_buildable_set_name(GTK_BUILDABLE(responsive_radio), "responsive-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(responsive_radio));
  gtk_container_add(GTK_CONTAINER(options_box), responsive_radio);
  
  /* 현재 모드 선택 */
  switch (current_mode) {
    case BLOUEDIT_PERFORMANCE_MODE_QUALITY:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(quality_radio), TRUE);
      break;
    case BLOUEDIT_PERFORMANCE_MODE_RESPONSIVE:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(responsive_radio), TRUE);
      break;
    case BLOUEDIT_PERFORMANCE_MODE_BALANCED:
    default:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(balanced_radio), TRUE);
      break;
  }
  
  /* 설명 프레임 */
  GtkWidget *desc_frame = gtk_frame_new("설명");
  gtk_container_add(GTK_CONTAINER(content_area), desc_frame);
  
  /* 설명 레이블 */
  GtkWidget *description_label = gtk_label_new(get_performance_mode_description(current_mode));
  gtk_label_set_line_wrap(GTK_LABEL(description_label), TRUE);
  gtk_widget_set_margin_start(description_label, 10);
  gtk_widget_set_margin_end(description_label, 10);
  gtk_widget_set_margin_top(description_label, 10);
  gtk_widget_set_margin_bottom(description_label, 10);
  gtk_container_add(GTK_CONTAINER(desc_frame), description_label);
  
  /* 팁 레이블 */
  GtkWidget *tip_label = gtk_label_new("팁: 편집 중에는 '편집 반응성' 모드를, 최종 미리보기 시에는 '재생 품질' 모드를 사용하세요.");
  gtk_label_set_line_wrap(GTK_LABEL(tip_label), TRUE);
  gtk_widget_set_margin_top(tip_label, 10);
  gtk_container_add(GTK_CONTAINER(content_area), tip_label);
  
  /* 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "description-label", description_label);
  g_object_set_data(G_OBJECT(dialog), "selected-mode", GINT_TO_POINTER(current_mode));
  
  /* 시그널 핸들러 연결 */
  g_signal_connect(quality_radio, "toggled", G_CALLBACK(on_performance_mode_changed), dialog);
  g_signal_connect(balanced_radio, "toggled", G_CALLBACK(on_performance_mode_changed), dialog);
  g_signal_connect(responsive_radio, "toggled", G_CALLBACK(on_performance_mode_changed), dialog);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    /* 선택된 모드 적용 */
    BlouEditPerformanceMode selected_mode = 
      (BlouEditPerformanceMode)GPOINTER_TO_INT(g_object_get_data(G_OBJECT(dialog), "selected-mode"));
    
    blouedit_timeline_set_performance_mode(timeline, selected_mode);
  }
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
} 