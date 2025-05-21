#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "timeline_view_mode.h"

struct _BlouEditTimelineViewData {
  BlouEditTimelineViewMode mode;
  BlouEditTimelineVisualizationFlags visualization_flags;
  gint thumbnail_width;
  gint thumbnail_height;
  gint waveform_resolution;
  GdkRGBA waveform_color;
};

/* 타임라인 구조체에 뷰 데이터를 저장하는 키 */
static const gchar *VIEW_DATA_KEY = "blouedit-timeline-view-data";

/* 타임라인에서 뷰 데이터 가져오기 */
static BlouEditTimelineViewData*
get_view_data(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  /* 타임라인에서 뷰 데이터 가져오기 */
  BlouEditTimelineViewData *view_data = g_object_get_data(G_OBJECT(timeline), VIEW_DATA_KEY);
  
  if (!view_data) {
    /* 첫 접근 시 데이터 초기화 */
    view_data = g_new0(BlouEditTimelineViewData, 1);
    view_data->mode = BLOUEDIT_TIMELINE_VIEW_STANDARD;
    view_data->visualization_flags = BLOUEDIT_TIMELINE_SHOW_THUMBNAILS | 
                                    BLOUEDIT_TIMELINE_SHOW_WAVEFORMS | 
                                    BLOUEDIT_TIMELINE_SHOW_LABELS;
    view_data->thumbnail_width = 80;
    view_data->thumbnail_height = 45;
    view_data->waveform_resolution = 2;
    
    /* 기본 파형 색상: 녹색 */
    view_data->waveform_color.red = 0.0;
    view_data->waveform_color.green = 0.8;
    view_data->waveform_color.blue = 0.0;
    view_data->waveform_color.alpha = 1.0;
    
    /* 타임라인에 데이터 저장 */
    g_object_set_data_full(G_OBJECT(timeline), VIEW_DATA_KEY, view_data, g_free);
  }
  
  return view_data;
}

/* 타임라인 시각화 모드 설정 함수 */
void 
blouedit_timeline_set_view_mode(BlouEditTimeline *timeline, BlouEditTimelineViewMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  
  if (view_data->mode != mode) {
    view_data->mode = mode;
    
    /* 모드에 따라 기본 시각화 플래그 설정 */
    switch (mode) {
      case BLOUEDIT_TIMELINE_VIEW_STANDARD:
        view_data->visualization_flags = BLOUEDIT_TIMELINE_SHOW_THUMBNAILS | 
                                        BLOUEDIT_TIMELINE_SHOW_WAVEFORMS | 
                                        BLOUEDIT_TIMELINE_SHOW_LABELS;
        break;
      
      case BLOUEDIT_TIMELINE_VIEW_COMPACT:
        view_data->visualization_flags = BLOUEDIT_TIMELINE_SHOW_LABELS;
        break;
      
      case BLOUEDIT_TIMELINE_VIEW_ICONIC:
        view_data->visualization_flags = BLOUEDIT_TIMELINE_SHOW_THUMBNAILS | 
                                        BLOUEDIT_TIMELINE_SHOW_LABELS |
                                        BLOUEDIT_TIMELINE_SHOW_EFFECTS;
        /* 아이코닉 모드에서는 썸네일 크기 증가 */
        view_data->thumbnail_width = 120;
        view_data->thumbnail_height = 68;
        break;
      
      case BLOUEDIT_TIMELINE_VIEW_WAVEFORM:
        view_data->visualization_flags = BLOUEDIT_TIMELINE_SHOW_WAVEFORMS | 
                                        BLOUEDIT_TIMELINE_SHOW_LABELS;
        break;
      
      case BLOUEDIT_TIMELINE_VIEW_MINIMAL:
        view_data->visualization_flags = BLOUEDIT_TIMELINE_SHOW_NONE;
        break;
      
      default:
        break;
    }
    
    /* 타임라인 다시 그리기 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
}

/* 타임라인 시각화 모드 가져오기 함수 */
BlouEditTimelineViewMode 
blouedit_timeline_get_view_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_TIMELINE_VIEW_STANDARD);
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  return view_data->mode;
}

/* 시각화 플래그 설정 함수 */
void 
blouedit_timeline_set_visualization_flags(BlouEditTimeline *timeline, 
                                        BlouEditTimelineVisualizationFlags flags)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  
  if (view_data->visualization_flags != flags) {
    view_data->visualization_flags = flags;
    
    /* 타임라인 다시 그리기 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
}

/* 시각화 플래그 가져오기 함수 */
BlouEditTimelineVisualizationFlags 
blouedit_timeline_get_visualization_flags(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_TIMELINE_SHOW_NONE);
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  return view_data->visualization_flags;
}

/* 썸네일 크기 설정 함수 */
void 
blouedit_timeline_set_thumbnail_size(BlouEditTimeline *timeline, gint width, gint height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(width > 0 && height > 0);
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  
  if (view_data->thumbnail_width != width || view_data->thumbnail_height != height) {
    view_data->thumbnail_width = width;
    view_data->thumbnail_height = height;
    
    /* 썸네일 캐시 지우기 필요 - 구현 필요 */
    
    /* 타임라인 다시 그리기 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
}

/* 썸네일 크기 가져오기 함수 */
void 
blouedit_timeline_get_thumbnail_size(BlouEditTimeline *timeline, gint *width, gint *height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  
  if (width)
    *width = view_data->thumbnail_width;
  
  if (height)
    *height = view_data->thumbnail_height;
}

/* 파형 해상도 설정 함수 */
void 
blouedit_timeline_set_waveform_resolution(BlouEditTimeline *timeline, gint resolution)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(resolution > 0);
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  
  if (view_data->waveform_resolution != resolution) {
    view_data->waveform_resolution = resolution;
    
    /* 파형 캐시 지우기 필요 - 구현 필요 */
    
    /* 타임라인 다시 그리기 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
}

/* 파형 해상도 가져오기 함수 */
gint 
blouedit_timeline_get_waveform_resolution(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), 2);
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  return view_data->waveform_resolution;
}

/* 파형 색상 설정 함수 */
void 
blouedit_timeline_set_waveform_color(BlouEditTimeline *timeline, const GdkRGBA *color)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(color != NULL);
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  
  view_data->waveform_color = *color;
  
  /* 타임라인 다시 그리기 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 파형 색상 가져오기 함수 */
void 
blouedit_timeline_get_waveform_color(BlouEditTimeline *timeline, GdkRGBA *color)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(color != NULL);
  
  BlouEditTimelineViewData *view_data = get_view_data(timeline);
  *color = view_data->waveform_color;
}

/* 모드 이름 문자열 반환 함수 */
static const gchar*
get_view_mode_name(BlouEditTimelineViewMode mode)
{
  switch (mode) {
    case BLOUEDIT_TIMELINE_VIEW_STANDARD:
      return "표준";
    case BLOUEDIT_TIMELINE_VIEW_COMPACT:
      return "압축";
    case BLOUEDIT_TIMELINE_VIEW_ICONIC:
      return "아이코닉";
    case BLOUEDIT_TIMELINE_VIEW_WAVEFORM:
      return "파형";
    case BLOUEDIT_TIMELINE_VIEW_MINIMAL:
      return "최소화";
    default:
      return "알 수 없음";
  }
}

/* 체크박스 토글 시 콜백 함수 */
static void
on_visualization_toggle(GtkToggleButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  BlouEditTimelineVisualizationFlags flag = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button), "flag"));
  BlouEditTimelineVisualizationFlags current_flags = blouedit_timeline_get_visualization_flags(timeline);
  
  if (gtk_toggle_button_get_active(button))
    current_flags |= flag;
  else
    current_flags &= ~flag;
  
  blouedit_timeline_set_visualization_flags(timeline, current_flags);
}

/* 모드 라디오 버튼 토글 시 콜백 함수 */
static void
on_mode_toggle(GtkToggleButton *button, gpointer user_data)
{
  if (gtk_toggle_button_get_active(button)) {
    BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
    BlouEditTimelineViewMode mode = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button), "mode"));
    blouedit_timeline_set_view_mode(timeline, mode);
  }
}

/* 크기 조절 스핀 버튼 값 변경 시 콜백 함수 */
static void
on_thumbnail_size_changed(GtkSpinButton *spin_button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  gint width, height;
  
  /* 현재 썸네일 크기 가져오기 */
  blouedit_timeline_get_thumbnail_size(timeline, &width, &height);
  
  /* 아스펙트 비율 유지 비율 (16:9) */
  gdouble aspect_ratio = 16.0 / 9.0;
  
  /* 어떤 스핀 버튼이 변경되었는지 확인 */
  const gchar *name = gtk_buildable_get_name(GTK_BUILDABLE(spin_button));
  
  if (g_strcmp0(name, "width_spin") == 0) {
    width = gtk_spin_button_get_value_as_int(spin_button);
    height = (gint)(width / aspect_ratio);
  } else {
    height = gtk_spin_button_get_value_as_int(spin_button);
    width = (gint)(height * aspect_ratio);
  }
  
  /* 새 크기 설정 */
  blouedit_timeline_set_thumbnail_size(timeline, width, height);
  
  /* 다른 스핀 버튼 업데이트 */
  if (g_strcmp0(name, "width_spin") == 0) {
    GtkSpinButton *height_spin = GTK_SPIN_BUTTON(g_object_get_data(G_OBJECT(timeline), "height_spin"));
    if (height_spin)
      gtk_spin_button_set_value(height_spin, height);
  } else {
    GtkSpinButton *width_spin = GTK_SPIN_BUTTON(g_object_get_data(G_OBJECT(timeline), "width_spin"));
    if (width_spin)
      gtk_spin_button_set_value(width_spin, width);
  }
}

/* 파형 해상도 변경 시 콜백 함수 */
static void
on_waveform_resolution_changed(GtkSpinButton *spin_button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  gint resolution = gtk_spin_button_get_value_as_int(spin_button);
  
  blouedit_timeline_set_waveform_resolution(timeline, resolution);
}

/* 파형 색상 변경 시 콜백 함수 */
static void
on_waveform_color_set(GtkColorButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  GdkRGBA color;
  
  gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(button), &color);
  blouedit_timeline_set_waveform_color(timeline, &color);
}

/* 설정 대화상자 응답 콜백 함수 */
static void
on_view_settings_response(GtkDialog *dialog, gint response_id, gpointer user_data)
{
  /* 대화상자 닫기 */
  gtk_widget_destroy(GTK_WIDGET(dialog));
}

/* 시각화 설정 대화상자 표시 함수 */
void
blouedit_timeline_show_view_settings_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 현재 설정 가져오기 */
  BlouEditTimelineViewMode current_mode = blouedit_timeline_get_view_mode(timeline);
  BlouEditTimelineVisualizationFlags current_flags = blouedit_timeline_get_visualization_flags(timeline);
  gint thumb_width, thumb_height;
  blouedit_timeline_get_thumbnail_size(timeline, &thumb_width, &thumb_height);
  gint waveform_resolution = blouedit_timeline_get_waveform_resolution(timeline);
  GdkRGBA waveform_color;
  blouedit_timeline_get_waveform_color(timeline, &waveform_color);
  
  /* 대화상자 생성 */
  GtkWidget *dialog = gtk_dialog_new_with_buttons(
    "타임라인 보기 설정",
    GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
    GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
    "_확인", GTK_RESPONSE_OK,
    NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 500, 400);
  
  /* 대화상자 콘텐츠 영역 가져오기 */
  GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_widget_set_margin_start(content_area, 12);
  gtk_widget_set_margin_end(content_area, 12);
  gtk_widget_set_margin_top(content_area, 12);
  gtk_widget_set_margin_bottom(content_area, 12);
  gtk_box_set_spacing(GTK_BOX(content_area), 6);
  
  /* 노트북 생성 */
  GtkWidget *notebook = gtk_notebook_new();
  gtk_container_add(GTK_CONTAINER(content_area), notebook);
  
  /* 모드 페이지 */
  GtkWidget *mode_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
  gtk_widget_set_margin_start(mode_page, 12);
  gtk_widget_set_margin_end(mode_page, 12);
  gtk_widget_set_margin_top(mode_page, 12);
  gtk_widget_set_margin_bottom(mode_page, 12);
  gtk_notebook_append_page(GTK_NOTEBOOK(notebook), mode_page, gtk_label_new("보기 모드"));
  
  /* 모드 라디오 버튼 그룹 */
  GtkWidget *mode_frame = gtk_frame_new("타임라인 보기 모드");
  gtk_container_add(GTK_CONTAINER(mode_page), mode_frame);
  
  GtkWidget *mode_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
  gtk_widget_set_margin_start(mode_box, 12);
  gtk_widget_set_margin_end(mode_box, 12);
  gtk_widget_set_margin_top(mode_box, 12);
  gtk_widget_set_margin_bottom(mode_box, 12);
  gtk_container_add(GTK_CONTAINER(mode_frame), mode_box);
  
  /* 모드 라디오 버튼 생성 */
  GtkWidget *radio_standard = gtk_radio_button_new_with_label(NULL, "표준 모드");
  g_object_set_data(G_OBJECT(radio_standard), "mode", GINT_TO_POINTER(BLOUEDIT_TIMELINE_VIEW_STANDARD));
  gtk_container_add(GTK_CONTAINER(mode_box), radio_standard);
  
  GtkWidget *radio_compact = gtk_radio_button_new_with_label_from_widget(
    GTK_RADIO_BUTTON(radio_standard), "압축 모드");
  g_object_set_data(G_OBJECT(radio_compact), "mode", GINT_TO_POINTER(BLOUEDIT_TIMELINE_VIEW_COMPACT));
  gtk_container_add(GTK_CONTAINER(mode_box), radio_compact);
  
  GtkWidget *radio_iconic = gtk_radio_button_new_with_label_from_widget(
    GTK_RADIO_BUTTON(radio_standard), "아이코닉 모드 (큰 썸네일)");
  g_object_set_data(G_OBJECT(radio_iconic), "mode", GINT_TO_POINTER(BLOUEDIT_TIMELINE_VIEW_ICONIC));
  gtk_container_add(GTK_CONTAINER(mode_box), radio_iconic);
  
  GtkWidget *radio_waveform = gtk_radio_button_new_with_label_from_widget(
    GTK_RADIO_BUTTON(radio_standard), "파형 모드 (오디오 파형 강조)");
  g_object_set_data(G_OBJECT(radio_waveform), "mode", GINT_TO_POINTER(BLOUEDIT_TIMELINE_VIEW_WAVEFORM));
  gtk_container_add(GTK_CONTAINER(mode_box), radio_waveform);
  
  GtkWidget *radio_minimal = gtk_radio_button_new_with_label_from_widget(
    GTK_RADIO_BUTTON(radio_standard), "최소화 모드 (기본 시각화만)");
  g_object_set_data(G_OBJECT(radio_minimal), "mode", GINT_TO_POINTER(BLOUEDIT_TIMELINE_VIEW_MINIMAL));
  gtk_container_add(GTK_CONTAINER(mode_box), radio_minimal);
  
  /* 현재 모드에 맞게 라디오 버튼 설정 */
  switch (current_mode) {
    case BLOUEDIT_TIMELINE_VIEW_STANDARD:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radio_standard), TRUE);
      break;
    case BLOUEDIT_TIMELINE_VIEW_COMPACT:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radio_compact), TRUE);
      break;
    case BLOUEDIT_TIMELINE_VIEW_ICONIC:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radio_iconic), TRUE);
      break;
    case BLOUEDIT_TIMELINE_VIEW_WAVEFORM:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radio_waveform), TRUE);
      break;
    case BLOUEDIT_TIMELINE_VIEW_MINIMAL:
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radio_minimal), TRUE);
      break;
  }
  
  /* 모드 설명 레이블 */
  GtkWidget *mode_desc_label = gtk_label_new(
    "아이코닉 모드는 클립 썸네일을 크게 표시하여 시각적 편집에 적합합니다.\n"
    "파형 모드는 오디오 파형을 강조하여 오디오 편집에 유용합니다.\n"
    "압축 모드는 여러 트랙을 한 번에 보기에 좋습니다.\n"
    "최소화 모드는 성능이 낮은 시스템에서 유용합니다.");
  gtk_container_add(GTK_CONTAINER(mode_page), mode_desc_label);
  
  /* 시각화 요소 페이지 */
  GtkWidget *vis_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
  gtk_widget_set_margin_start(vis_page, 12);
  gtk_widget_set_margin_end(vis_page, 12);
  gtk_widget_set_margin_top(vis_page, 12);
  gtk_widget_set_margin_bottom(vis_page, 12);
  gtk_notebook_append_page(GTK_NOTEBOOK(notebook), vis_page, gtk_label_new("표시 요소"));
  
  /* 시각화 요소 프레임 */
  GtkWidget *vis_frame = gtk_frame_new("표시할 요소 선택");
  gtk_container_add(GTK_CONTAINER(vis_page), vis_frame);
  
  GtkWidget *vis_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
  gtk_widget_set_margin_start(vis_box, 12);
  gtk_widget_set_margin_end(vis_box, 12);
  gtk_widget_set_margin_top(vis_box, 12);
  gtk_widget_set_margin_bottom(vis_box, 12);
  gtk_container_add(GTK_CONTAINER(vis_frame), vis_box);
  
  /* 시각화 요소 체크박스 */
  GtkWidget *check_thumbnails = gtk_check_button_new_with_label("비디오 클립 썸네일 표시");
  g_object_set_data(G_OBJECT(check_thumbnails), "flag", 
                   GINT_TO_POINTER(BLOUEDIT_TIMELINE_SHOW_THUMBNAILS));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check_thumbnails), 
                              (current_flags & BLOUEDIT_TIMELINE_SHOW_THUMBNAILS) != 0);
  gtk_container_add(GTK_CONTAINER(vis_box), check_thumbnails);
  
  GtkWidget *check_waveforms = gtk_check_button_new_with_label("오디오 클립 파형 표시");
  g_object_set_data(G_OBJECT(check_waveforms), "flag", 
                   GINT_TO_POINTER(BLOUEDIT_TIMELINE_SHOW_WAVEFORMS));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check_waveforms), 
                              (current_flags & BLOUEDIT_TIMELINE_SHOW_WAVEFORMS) != 0);
  gtk_container_add(GTK_CONTAINER(vis_box), check_waveforms);
  
  GtkWidget *check_labels = gtk_check_button_new_with_label("클립 레이블 표시");
  g_object_set_data(G_OBJECT(check_labels), "flag", 
                   GINT_TO_POINTER(BLOUEDIT_TIMELINE_SHOW_LABELS));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check_labels), 
                              (current_flags & BLOUEDIT_TIMELINE_SHOW_LABELS) != 0);
  gtk_container_add(GTK_CONTAINER(vis_box), check_labels);
  
  GtkWidget *check_effects = gtk_check_button_new_with_label("효과 아이콘 표시");
  g_object_set_data(G_OBJECT(check_effects), "flag", 
                   GINT_TO_POINTER(BLOUEDIT_TIMELINE_SHOW_EFFECTS));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check_effects), 
                              (current_flags & BLOUEDIT_TIMELINE_SHOW_EFFECTS) != 0);
  gtk_container_add(GTK_CONTAINER(vis_box), check_effects);
  
  GtkWidget *check_keyframes = gtk_check_button_new_with_label("키프레임 표시");
  g_object_set_data(G_OBJECT(check_keyframes), "flag", 
                   GINT_TO_POINTER(BLOUEDIT_TIMELINE_SHOW_KEYFRAMES));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check_keyframes), 
                              (current_flags & BLOUEDIT_TIMELINE_SHOW_KEYFRAMES) != 0);
  gtk_container_add(GTK_CONTAINER(vis_box), check_keyframes);
  
  GtkWidget *check_in_out = gtk_check_button_new_with_label("클립 시작/끝점 표시");
  g_object_set_data(G_OBJECT(check_in_out), "flag", 
                   GINT_TO_POINTER(BLOUEDIT_TIMELINE_SHOW_IN_OUT_POINTS));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check_in_out), 
                              (current_flags & BLOUEDIT_TIMELINE_SHOW_IN_OUT_POINTS) != 0);
  gtk_container_add(GTK_CONTAINER(vis_box), check_in_out);
  
  GtkWidget *check_durations = gtk_check_button_new_with_label("클립 지속 시간 표시");
  g_object_set_data(G_OBJECT(check_durations), "flag", 
                   GINT_TO_POINTER(BLOUEDIT_TIMELINE_SHOW_DURATIONS));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(check_durations), 
                              (current_flags & BLOUEDIT_TIMELINE_SHOW_DURATIONS) != 0);
  gtk_container_add(GTK_CONTAINER(vis_box), check_durations);
  
  /* 고급 설정 페이지 */
  GtkWidget *adv_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
  gtk_widget_set_margin_start(adv_page, 12);
  gtk_widget_set_margin_end(adv_page, 12);
  gtk_widget_set_margin_top(adv_page, 12);
  gtk_widget_set_margin_bottom(adv_page, 12);
  gtk_notebook_append_page(GTK_NOTEBOOK(notebook), adv_page, gtk_label_new("고급 설정"));
  
  /* 썸네일 설정 프레임 */
  GtkWidget *thumb_frame = gtk_frame_new("썸네일 설정");
  gtk_container_add(GTK_CONTAINER(adv_page), thumb_frame);
  
  GtkWidget *thumb_grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(thumb_grid), 6);
  gtk_grid_set_column_spacing(GTK_GRID(thumb_grid), 12);
  gtk_widget_set_margin_start(thumb_grid, 12);
  gtk_widget_set_margin_end(thumb_grid, 12);
  gtk_widget_set_margin_top(thumb_grid, 12);
  gtk_widget_set_margin_bottom(thumb_grid, 12);
  gtk_container_add(GTK_CONTAINER(thumb_frame), thumb_grid);
  
  /* 썸네일 너비 설정 */
  GtkWidget *width_label = gtk_label_new("썸네일 너비:");
  gtk_grid_attach(GTK_GRID(thumb_grid), width_label, 0, 0, 1, 1);
  
  GtkWidget *width_spin = gtk_spin_button_new_with_range(20, 300, 1);
  gtk_widget_set_hexpand(width_spin, TRUE);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin), thumb_width);
  gtk_buildable_set_name(GTK_BUILDABLE(width_spin), "width_spin");
  gtk_grid_attach(GTK_GRID(thumb_grid), width_spin, 1, 0, 1, 1);
  
  /* 썸네일 높이 설정 */
  GtkWidget *height_label = gtk_label_new("썸네일 높이:");
  gtk_grid_attach(GTK_GRID(thumb_grid), height_label, 0, 1, 1, 1);
  
  GtkWidget *height_spin = gtk_spin_button_new_with_range(20, 300, 1);
  gtk_widget_set_hexpand(height_spin, TRUE);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin), thumb_height);
  gtk_buildable_set_name(GTK_BUILDABLE(height_spin), "height_spin");
  gtk_grid_attach(GTK_GRID(thumb_grid), height_spin, 1, 1, 1, 1);
  
  /* 스핀 버튼 참조 저장 (상호 업데이트용) */
  g_object_set_data(G_OBJECT(timeline), "width_spin", width_spin);
  g_object_set_data(G_OBJECT(timeline), "height_spin", height_spin);
  
  /* 파형 설정 프레임 */
  GtkWidget *wave_frame = gtk_frame_new("파형 설정");
  gtk_container_add(GTK_CONTAINER(adv_page), wave_frame);
  
  GtkWidget *wave_grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(wave_grid), 6);
  gtk_grid_set_column_spacing(GTK_GRID(wave_grid), 12);
  gtk_widget_set_margin_start(wave_grid, 12);
  gtk_widget_set_margin_end(wave_grid, 12);
  gtk_widget_set_margin_top(wave_grid, 12);
  gtk_widget_set_margin_bottom(wave_grid, 12);
  gtk_container_add(GTK_CONTAINER(wave_frame), wave_grid);
  
  /* 파형 해상도 설정 */
  GtkWidget *res_label = gtk_label_new("파형 해상도:");
  gtk_grid_attach(GTK_GRID(wave_grid), res_label, 0, 0, 1, 1);
  
  GtkWidget *res_spin = gtk_spin_button_new_with_range(1, 10, 1);
  gtk_widget_set_hexpand(res_spin, TRUE);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(res_spin), waveform_resolution);
  gtk_grid_attach(GTK_GRID(wave_grid), res_spin, 1, 0, 1, 1);
  
  /* 파형 색상 설정 */
  GtkWidget *color_label = gtk_label_new("파형 색상:");
  gtk_grid_attach(GTK_GRID(wave_grid), color_label, 0, 1, 1, 1);
  
  GtkWidget *color_button = gtk_color_button_new_with_rgba(&waveform_color);
  gtk_widget_set_hexpand(color_button, TRUE);
  gtk_grid_attach(GTK_GRID(wave_grid), color_button, 1, 1, 1, 1);
  
  /* 시그널 연결 */
  g_signal_connect(radio_standard, "toggled", G_CALLBACK(on_mode_toggle), timeline);
  g_signal_connect(radio_compact, "toggled", G_CALLBACK(on_mode_toggle), timeline);
  g_signal_connect(radio_iconic, "toggled", G_CALLBACK(on_mode_toggle), timeline);
  g_signal_connect(radio_waveform, "toggled", G_CALLBACK(on_mode_toggle), timeline);
  g_signal_connect(radio_minimal, "toggled", G_CALLBACK(on_mode_toggle), timeline);
  
  g_signal_connect(check_thumbnails, "toggled", G_CALLBACK(on_visualization_toggle), timeline);
  g_signal_connect(check_waveforms, "toggled", G_CALLBACK(on_visualization_toggle), timeline);
  g_signal_connect(check_labels, "toggled", G_CALLBACK(on_visualization_toggle), timeline);
  g_signal_connect(check_effects, "toggled", G_CALLBACK(on_visualization_toggle), timeline);
  g_signal_connect(check_keyframes, "toggled", G_CALLBACK(on_visualization_toggle), timeline);
  g_signal_connect(check_in_out, "toggled", G_CALLBACK(on_visualization_toggle), timeline);
  g_signal_connect(check_durations, "toggled", G_CALLBACK(on_visualization_toggle), timeline);
  
  g_signal_connect(width_spin, "value-changed", G_CALLBACK(on_thumbnail_size_changed), timeline);
  g_signal_connect(height_spin, "value-changed", G_CALLBACK(on_thumbnail_size_changed), timeline);
  g_signal_connect(res_spin, "value-changed", G_CALLBACK(on_waveform_resolution_changed), timeline);
  g_signal_connect(color_button, "color-set", G_CALLBACK(on_waveform_color_set), timeline);
  
  g_signal_connect(dialog, "response", G_CALLBACK(on_view_settings_response), timeline);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
} 