#include <gtk/gtk.h>
#include <string.h>
#include "timeline_performance.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 하드웨어 가속 설정 함수 */
void
blouedit_timeline_set_hw_acceleration(BlouEditTimeline *timeline, BlouEditHwAccelType accel_type)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 현재 설정과 같으면 무시 */
  if (timeline->hw_acceleration_type == accel_type) {
    return;
  }
  
  /* 설정 변경 */
  timeline->hw_acceleration_type = accel_type;
  
  /* 하드웨어 가속 활성화 상태 업데이트 */
  timeline->use_hardware_acceleration = (accel_type != BLOUEDIT_HW_ACCEL_DISABLED);
  
  /* 설정에 따른 처리 */
  switch (accel_type) {
    case BLOUEDIT_HW_ACCEL_INTEL:
      /* Intel 하드웨어 가속 활성화 코드 */
      timeline->hw_accel_vendor = "intel";
      break;
      
    case BLOUEDIT_HW_ACCEL_NVIDIA:
      /* NVIDIA 하드웨어 가속 활성화 코드 */
      timeline->hw_accel_vendor = "nvidia";
      break;
      
    case BLOUEDIT_HW_ACCEL_AMD:
      /* AMD 하드웨어 가속 활성화 코드 */
      timeline->hw_accel_vendor = "amd";
      break;
      
    case BLOUEDIT_HW_ACCEL_AUTO:
      /* 자동 감지 코드 */
      timeline->hw_accel_vendor = "auto";
      /* 실제 구현에서는 시스템 GPU 감지 후 적절한 가속 설정 */
      break;
      
    case BLOUEDIT_HW_ACCEL_DISABLED:
    default:
      /* 하드웨어 가속 비활성화 */
      timeline->hw_accel_vendor = NULL;
      break;
  }
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  /* 하드웨어 가속 변경 시그널 발생 (필요시) */
  g_signal_emit_by_name(timeline, "hw-acceleration-changed", accel_type);
}

/* 현재 하드웨어 가속 설정 가져오기 */
BlouEditHwAccelType
blouedit_timeline_get_hw_acceleration(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_HW_ACCEL_DISABLED);
  
  return timeline->hw_acceleration_type;
}

/* 특정 하드웨어 가속이 사용 가능한지 확인 (간단한 구현) */
gboolean
blouedit_timeline_is_hw_acceleration_available(BlouEditTimeline *timeline, 
                                           BlouEditHwAccelType accel_type)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 실제 구현에서는 시스템에서 해당 하드웨어 가속이 가능한지 확인 */
  switch (accel_type) {
    case BLOUEDIT_HW_ACCEL_AUTO:
      /* 자동 감지는 항상 사용 가능 */
      return TRUE;
      
    case BLOUEDIT_HW_ACCEL_INTEL:
      /* Intel GPU 확인 코드 */
      return TRUE;  /* 임시로 TRUE 반환 */
      
    case BLOUEDIT_HW_ACCEL_NVIDIA:
      /* NVIDIA GPU 확인 코드 */
      return TRUE;  /* 임시로 TRUE 반환 */
      
    case BLOUEDIT_HW_ACCEL_AMD:
      /* AMD GPU 확인 코드 */
      return TRUE;  /* 임시로 TRUE 반환 */
      
    case BLOUEDIT_HW_ACCEL_DISABLED:
      /* 비활성화는 항상 가능 */
      return TRUE;
      
    default:
      return FALSE;
  }
}

/* 하드웨어 가속 유형별 설명 */
static const gchar*
get_hw_accel_description(BlouEditHwAccelType accel_type)
{
  switch (accel_type) {
    case BLOUEDIT_HW_ACCEL_AUTO:
      return "시스템에서 사용 가능한 가장 적합한 하드웨어 가속을 자동으로 감지하여 사용합니다.\n"
             "대부분의 경우 이 옵션이 권장됩니다.";
      
    case BLOUEDIT_HW_ACCEL_INTEL:
      return "Intel 그래픽 칩셋을 사용한 하드웨어 가속을 활성화합니다.\n"
             "Intel HD Graphics, Iris, UHD Graphics 등의 그래픽 카드에 최적화됩니다.";
      
    case BLOUEDIT_HW_ACCEL_NVIDIA:
      return "NVIDIA GPU를 사용한 하드웨어 가속을 활성화합니다.\n"
             "GeForce, Quadro 시리즈 그래픽 카드에 최적화됩니다.";
      
    case BLOUEDIT_HW_ACCEL_AMD:
      return "AMD GPU를 사용한 하드웨어 가속을 활성화합니다.\n"
             "Radeon 시리즈 그래픽 카드에 최적화됩니다.";
      
    case BLOUEDIT_HW_ACCEL_DISABLED:
    default:
      return "모든 하드웨어 가속을 비활성화하고 CPU만 사용합니다.\n"
             "호환성 문제가 있거나 그래픽 드라이버 문제가 있을 때 이 옵션을 선택하세요.";
  }
}

/* 하드웨어 가속 선택 변경 핸들러 */
static void
on_hw_accel_changed(GtkToggleButton *button, gpointer user_data)
{
  if (!gtk_toggle_button_get_active(button)) {
    return;  /* 비활성화되는 버튼은 무시 */
  }
  
  GtkWidget *dialog = GTK_WIDGET(user_data);
  GtkLabel *description_label = GTK_LABEL(g_object_get_data(G_OBJECT(dialog), "description-label"));
  
  /* 선택된 모드 식별 */
  BlouEditHwAccelType accel_type;
  const gchar *mode_name = gtk_buildable_get_name(GTK_BUILDABLE(button));
  
  if (g_strcmp0(mode_name, "intel-radio") == 0) {
    accel_type = BLOUEDIT_HW_ACCEL_INTEL;
  } 
  else if (g_strcmp0(mode_name, "nvidia-radio") == 0) {
    accel_type = BLOUEDIT_HW_ACCEL_NVIDIA;
  } 
  else if (g_strcmp0(mode_name, "amd-radio") == 0) {
    accel_type = BLOUEDIT_HW_ACCEL_AMD;
  } 
  else if (g_strcmp0(mode_name, "disabled-radio") == 0) {
    accel_type = BLOUEDIT_HW_ACCEL_DISABLED;
  } 
  else {
    accel_type = BLOUEDIT_HW_ACCEL_AUTO;
  }
  
  /* 설명 업데이트 */
  gtk_label_set_text(description_label, get_hw_accel_description(accel_type));
  
  /* 선택된 모드 저장 */
  g_object_set_data(G_OBJECT(dialog), "selected-accel", GINT_TO_POINTER(accel_type));
}

/* 하드웨어 가속 설정 대화상자 */
void
blouedit_timeline_show_hw_acceleration_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 현재 설정 */
  BlouEditHwAccelType current_accel = blouedit_timeline_get_hw_acceleration(timeline);
  
  /* 대화상자 생성 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "하드웨어 가속 설정",
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
  GtkWidget *header_label = gtk_label_new(
    "하드웨어 가속은 그래픽 카드의 전용 기능을 사용하여 비디오 디코딩 및 인코딩 속도를 향상시킵니다.");
  gtk_label_set_line_wrap(GTK_LABEL(header_label), TRUE);
  gtk_widget_set_halign(header_label, GTK_ALIGN_START);
  gtk_container_add(GTK_CONTAINER(content_area), header_label);
  
  /* 옵션 프레임 */
  GtkWidget *frame = gtk_frame_new("가속 유형 선택");
  gtk_container_add(GTK_CONTAINER(content_area), frame);
  
  /* 옵션 컨테이너 */
  GtkWidget *options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_container_set_border_width(GTK_CONTAINER(options_box), 10);
  gtk_container_add(GTK_CONTAINER(frame), options_box);
  
  /* 라디오 버튼 그룹 */
  GSList *group = NULL;
  
  /* 자동 감지 */
  GtkWidget *auto_radio = gtk_radio_button_new_with_label(group, "자동 감지");
  gtk_buildable_set_name(GTK_BUILDABLE(auto_radio), "auto-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(auto_radio));
  gtk_widget_set_tooltip_text(auto_radio, "시스템에서 사용 가능한 최적의 하드웨어 가속을 자동으로 선택합니다");
  gtk_container_add(GTK_CONTAINER(options_box), auto_radio);
  
  /* Intel */
  GtkWidget *intel_radio = gtk_radio_button_new_with_label(group, "Intel 그래픽스");
  gtk_buildable_set_name(GTK_BUILDABLE(intel_radio), "intel-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(intel_radio));
  gtk_widget_set_tooltip_text(intel_radio, "Intel 그래픽 칩셋을 사용한 하드웨어 가속");
  gtk_container_add(GTK_CONTAINER(options_box), intel_radio);
  
  /* NVIDIA */
  GtkWidget *nvidia_radio = gtk_radio_button_new_with_label(group, "NVIDIA 그래픽스");
  gtk_buildable_set_name(GTK_BUILDABLE(nvidia_radio), "nvidia-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(nvidia_radio));
  gtk_widget_set_tooltip_text(nvidia_radio, "NVIDIA GPU를 사용한 하드웨어 가속");
  gtk_container_add(GTK_CONTAINER(options_box), nvidia_radio);
  
  /* AMD */
  GtkWidget *amd_radio = gtk_radio_button_new_with_label(group, "AMD 그래픽스");
  gtk_buildable_set_name(GTK_BUILDABLE(amd_radio), "amd-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(amd_radio));
  gtk_widget_set_tooltip_text(amd_radio, "AMD GPU를 사용한 하드웨어 가속");
  gtk_container_add(GTK_CONTAINER(options_box), amd_radio);
  
  /* 비활성화 */
  GtkWidget *disabled_radio = gtk_radio_button_new_with_label(group, "하드웨어 가속 비활성화");
  gtk_buildable_set_name(GTK_BUILDABLE(disabled_radio), "disabled-radio");
  group = gtk_radio_button_get_group(GTK_RADIO_BUTTON(disabled_radio));
  gtk_widget_set_tooltip_text(disabled_radio, "CPU만 사용하여 처리");
  gtk_container_add(GTK_CONTAINER(options_box), disabled_radio);
  
  /* 현재 설정 선택 */
  switch (current_accel) {
    case BLOUEDIT_HW_ACCEL_AUTO:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(auto_radio), TRUE);
      break;
    case BLOUEDIT_HW_ACCEL_INTEL:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(intel_radio), TRUE);
      break;
    case BLOUEDIT_HW_ACCEL_NVIDIA:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(nvidia_radio), TRUE);
      break;
    case BLOUEDIT_HW_ACCEL_AMD:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(amd_radio), TRUE);
      break;
    case BLOUEDIT_HW_ACCEL_DISABLED:
    default:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(disabled_radio), TRUE);
      break;
  }
  
  /* 각 옵션의 가용성 설정 - 실제 구현에서 활성화 */
  /*
  gtk_widget_set_sensitive(intel_radio, 
                         blouedit_timeline_is_hw_acceleration_available(timeline, BLOUEDIT_HW_ACCEL_INTEL));
  gtk_widget_set_sensitive(nvidia_radio, 
                         blouedit_timeline_is_hw_acceleration_available(timeline, BLOUEDIT_HW_ACCEL_NVIDIA));
  gtk_widget_set_sensitive(amd_radio, 
                         blouedit_timeline_is_hw_acceleration_available(timeline, BLOUEDIT_HW_ACCEL_AMD));
  */
  
  /* 설명 프레임 */
  GtkWidget *desc_frame = gtk_frame_new("설명");
  gtk_container_add(GTK_CONTAINER(content_area), desc_frame);
  
  /* 설명 레이블 */
  GtkWidget *description_label = gtk_label_new(get_hw_accel_description(current_accel));
  gtk_label_set_line_wrap(GTK_LABEL(description_label), TRUE);
  gtk_widget_set_margin_start(description_label, 10);
  gtk_widget_set_margin_end(description_label, 10);
  gtk_widget_set_margin_top(description_label, 10);
  gtk_widget_set_margin_bottom(description_label, 10);
  gtk_container_add(GTK_CONTAINER(desc_frame), description_label);
  
  /* 경고 레이블 */
  GtkWidget *warning_label = gtk_label_new(
    "참고: 하드웨어 가속 설정을 변경하면 편집기를 다시 시작해야 완전히 적용됩니다.");
  gtk_label_set_line_wrap(GTK_LABEL(warning_label), TRUE);
  gtk_widget_set_margin_top(warning_label, 10);
  gtk_container_add(GTK_CONTAINER(content_area), warning_label);
  
  /* 데이터 연결 */
  g_object_set_data(G_OBJECT(dialog), "timeline", timeline);
  g_object_set_data(G_OBJECT(dialog), "description-label", description_label);
  g_object_set_data(G_OBJECT(dialog), "selected-accel", GINT_TO_POINTER(current_accel));
  
  /* 시그널 핸들러 연결 */
  g_signal_connect(auto_radio, "toggled", G_CALLBACK(on_hw_accel_changed), dialog);
  g_signal_connect(intel_radio, "toggled", G_CALLBACK(on_hw_accel_changed), dialog);
  g_signal_connect(nvidia_radio, "toggled", G_CALLBACK(on_hw_accel_changed), dialog);
  g_signal_connect(amd_radio, "toggled", G_CALLBACK(on_hw_accel_changed), dialog);
  g_signal_connect(disabled_radio, "toggled", G_CALLBACK(on_hw_accel_changed), dialog);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    /* 선택된 가속 방식 적용 */
    BlouEditHwAccelType selected_accel = 
      (BlouEditHwAccelType)GPOINTER_TO_INT(g_object_get_data(G_OBJECT(dialog), "selected-accel"));
    
    blouedit_timeline_set_hw_acceleration(timeline, selected_accel);
  }
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
} 