#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "media_type_visualizer.h"

/* 미디어 유형별 기본 색상 */
static const GdkRGBA default_media_colors[] = {
  { 0.2, 0.6, 1.0, 1.0 },  /* BLOUEDIT_FILTER_VIDEO - 파란색 */
  { 0.3, 0.8, 0.3, 1.0 },  /* BLOUEDIT_FILTER_AUDIO - 녹색 */
  { 1.0, 0.8, 0.2, 1.0 },  /* BLOUEDIT_FILTER_IMAGE - 노란색 */
  { 0.8, 0.4, 1.0, 1.0 },  /* BLOUEDIT_FILTER_TEXT - 보라색 */
  { 1.0, 0.5, 0.3, 1.0 },  /* BLOUEDIT_FILTER_EFFECT - 주황색 */
  { 0.5, 0.9, 0.9, 1.0 }   /* BLOUEDIT_FILTER_TRANSITION - 청록색 */
};

/* 미디어 유형별 아이콘 이름 */
static const gchar *media_type_icons[] = {
  "video-x-generic",     /* BLOUEDIT_FILTER_VIDEO */
  "audio-x-generic",     /* BLOUEDIT_FILTER_AUDIO */
  "image-x-generic",     /* BLOUEDIT_FILTER_IMAGE */
  "text-x-generic",      /* BLOUEDIT_FILTER_TEXT */
  "applications-graphics", /* BLOUEDIT_FILTER_EFFECT */
  "view-refresh"         /* BLOUEDIT_FILTER_TRANSITION */
};

/* 비트 플래그에서 인덱스 구하기 */
static int 
get_index_from_media_type(BlouEditMediaFilterType type)
{
  switch (type) {
    case BLOUEDIT_FILTER_VIDEO:
      return 0;
    case BLOUEDIT_FILTER_AUDIO:
      return 1;
    case BLOUEDIT_FILTER_IMAGE:
      return 2;
    case BLOUEDIT_FILTER_TEXT:
      return 3;
    case BLOUEDIT_FILTER_EFFECT:
      return 4;
    case BLOUEDIT_FILTER_TRANSITION:
      return 5;
    default:
      return 0;
  }
}

/* 미디어 유형별 시각적 구분 모드 설정 함수 */
void 
blouedit_timeline_set_media_visual_mode(BlouEditTimeline *timeline, BlouEditMediaVisualMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  timeline->media_visual_mode = mode;
  
  /* 타임라인 다시 그리기 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 미디어 유형별 시각적 구분 모드 가져오기 함수 */
BlouEditMediaVisualMode 
blouedit_timeline_get_media_visual_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_MEDIA_VISUAL_MODE_NONE);
  
  return timeline->media_visual_mode;
}

/* 특정 미디어 유형의 색상 설정 함수 */
void 
blouedit_timeline_set_media_type_color(BlouEditTimeline *timeline, BlouEditMediaFilterType type, const GdkRGBA *color)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(color != NULL);
  
  int index = get_index_from_media_type(type);
  timeline->media_type_colors[index] = *color;
  
  /* 타임라인 다시 그리기 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 특정 미디어 유형의 색상 가져오기 함수 */
void 
blouedit_timeline_get_media_type_color(BlouEditTimeline *timeline, BlouEditMediaFilterType type, GdkRGBA *color)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(color != NULL);
  
  int index = get_index_from_media_type(type);
  
  /* timeline->media_type_colors가 초기화되지 않았으면 기본 색상 반환 */
  if (!timeline->media_type_colors_initialized) {
    *color = default_media_colors[index];
  } else {
    *color = timeline->media_type_colors[index];
  }
}

/* 클립에서 미디어 유형 가져오기 함수 */
BlouEditMediaFilterType 
blouedit_timeline_get_clip_media_type(GESClip *clip)
{
  g_return_val_if_fail(GES_IS_CLIP(clip), BLOUEDIT_FILTER_VIDEO);
  
  GESTrackType track_types = ges_clip_get_supported_formats(clip);
  
  if (track_types & GES_TRACK_TYPE_VIDEO) {
    if (GES_IS_TRANSITION_CLIP(clip)) {
      return BLOUEDIT_FILTER_TRANSITION;
    } else if (GES_IS_EFFECT_CLIP(clip)) {
      return BLOUEDIT_FILTER_EFFECT;
    } else if (GES_IS_IMAGE_CLIP(clip) || GES_IS_STILL_CLIP(clip)) {
      return BLOUEDIT_FILTER_IMAGE;
    } else if (GES_IS_TEXT_OVERLAY_CLIP(clip) || GES_IS_TITLE_CLIP(clip)) {
      return BLOUEDIT_FILTER_TEXT;
    } else {
      return BLOUEDIT_FILTER_VIDEO;
    }
  } else if (track_types & GES_TRACK_TYPE_AUDIO) {
    return BLOUEDIT_FILTER_AUDIO;
  }
  
  return BLOUEDIT_FILTER_VIDEO; /* 기본값 */
}

/* 미디어 유형에 따라 클립 색상 가져오기 */
void 
blouedit_timeline_get_color_for_clip(BlouEditTimeline *timeline, GESClip *clip, GdkRGBA *color)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(GES_IS_CLIP(clip));
  g_return_if_fail(color != NULL);
  
  BlouEditMediaFilterType type = blouedit_timeline_get_clip_media_type(clip);
  blouedit_timeline_get_media_type_color(timeline, type, color);
}

/* 미디어 유형에 따라 아이콘 이름 가져오기 */
const gchar* 
blouedit_timeline_get_icon_for_media_type(BlouEditMediaFilterType type)
{
  int index = get_index_from_media_type(type);
  return media_type_icons[index];
}

/* 색상 버튼 클릭 콜백 */
static void
on_color_button_clicked(GtkColorButton *button, gpointer user_data)
{
  /* 사용자 데이터에서 정보 추출 */
  BlouEditTimeline *timeline = g_object_get_data(G_OBJECT(button), "timeline");
  BlouEditMediaFilterType type = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button), "media-type"));
  
  /* 색상 가져오기 */
  GdkRGBA color;
  gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(button), &color);
  
  /* 새 색상 설정 */
  blouedit_timeline_set_media_type_color(timeline, type, &color);
}

/* 미디어 유형 시각화 설정 대화상자 표시 함수 */
void 
blouedit_timeline_show_media_visual_settings(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *grid;
  GtkWidget *mode_label, *mode_combo;
  GtkWidget *colors_frame, *colors_grid;
  GtkWidget *video_label, *video_color;
  GtkWidget *audio_label, *audio_color;
  GtkWidget *image_label, *image_color;
  GtkWidget *text_label, *text_color;
  GtkWidget *effect_label, *effect_color;
  GtkWidget *transition_label, *transition_color;
  GtkWidget *reset_button;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("미디어 유형별 시각화 설정",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                     "_취소", GTK_RESPONSE_CANCEL,
                                     "_확인", GTK_RESPONSE_OK,
                                     NULL);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  
  /* 그리드 생성 */
  grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(grid), 12);
  gtk_grid_set_column_spacing(GTK_GRID(grid), 12);
  gtk_container_set_border_width(GTK_CONTAINER(grid), 12);
  gtk_container_add(GTK_CONTAINER(content_area), grid);
  
  /* 모드 선택 */
  mode_label = gtk_label_new("시각화 모드:");
  gtk_widget_set_halign(mode_label, GTK_ALIGN_START);
  
  mode_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "표시 안함");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "색상으로 구분");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "아이콘으로 구분");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "색상과 아이콘 모두 사용");
  
  /* 현재 모드 선택 */
  gtk_combo_box_set_active(GTK_COMBO_BOX(mode_combo), 
                         blouedit_timeline_get_media_visual_mode(timeline));
  
  gtk_grid_attach(GTK_GRID(grid), mode_label, 0, 0, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), mode_combo, 1, 0, 1, 1);
  
  /* 색상 설정 프레임 */
  colors_frame = gtk_frame_new("미디어 유형별 색상");
  gtk_grid_attach(GTK_GRID(grid), colors_frame, 0, 1, 2, 1);
  
  /* 색상 선택 그리드 */
  colors_grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(colors_grid), 6);
  gtk_grid_set_column_spacing(GTK_GRID(colors_grid), 12);
  gtk_container_set_border_width(GTK_CONTAINER(colors_grid), 12);
  gtk_container_add(GTK_CONTAINER(colors_frame), colors_grid);
  
  /* 비디오 색상 */
  video_label = gtk_label_new("비디오:");
  gtk_widget_set_halign(video_label, GTK_ALIGN_START);
  
  GdkRGBA video_rgba;
  blouedit_timeline_get_media_type_color(timeline, BLOUEDIT_FILTER_VIDEO, &video_rgba);
  
  video_color = gtk_color_button_new_with_rgba(&video_rgba);
  g_object_set_data(G_OBJECT(video_color), "timeline", timeline);
  g_object_set_data(G_OBJECT(video_color), "media-type", 
                  GINT_TO_POINTER(BLOUEDIT_FILTER_VIDEO));
  g_signal_connect(video_color, "color-set", 
                 G_CALLBACK(on_color_button_clicked), NULL);
  
  gtk_grid_attach(GTK_GRID(colors_grid), video_label, 0, 0, 1, 1);
  gtk_grid_attach(GTK_GRID(colors_grid), video_color, 1, 0, 1, 1);
  
  /* 오디오 색상 */
  audio_label = gtk_label_new("오디오:");
  gtk_widget_set_halign(audio_label, GTK_ALIGN_START);
  
  GdkRGBA audio_rgba;
  blouedit_timeline_get_media_type_color(timeline, BLOUEDIT_FILTER_AUDIO, &audio_rgba);
  
  audio_color = gtk_color_button_new_with_rgba(&audio_rgba);
  g_object_set_data(G_OBJECT(audio_color), "timeline", timeline);
  g_object_set_data(G_OBJECT(audio_color), "media-type", 
                  GINT_TO_POINTER(BLOUEDIT_FILTER_AUDIO));
  g_signal_connect(audio_color, "color-set", 
                 G_CALLBACK(on_color_button_clicked), NULL);
  
  gtk_grid_attach(GTK_GRID(colors_grid), audio_label, 0, 1, 1, 1);
  gtk_grid_attach(GTK_GRID(colors_grid), audio_color, 1, 1, 1, 1);
  
  /* 이미지 색상 */
  image_label = gtk_label_new("이미지:");
  gtk_widget_set_halign(image_label, GTK_ALIGN_START);
  
  GdkRGBA image_rgba;
  blouedit_timeline_get_media_type_color(timeline, BLOUEDIT_FILTER_IMAGE, &image_rgba);
  
  image_color = gtk_color_button_new_with_rgba(&image_rgba);
  g_object_set_data(G_OBJECT(image_color), "timeline", timeline);
  g_object_set_data(G_OBJECT(image_color), "media-type", 
                  GINT_TO_POINTER(BLOUEDIT_FILTER_IMAGE));
  g_signal_connect(image_color, "color-set", 
                 G_CALLBACK(on_color_button_clicked), NULL);
  
  gtk_grid_attach(GTK_GRID(colors_grid), image_label, 0, 2, 1, 1);
  gtk_grid_attach(GTK_GRID(colors_grid), image_color, 1, 2, 1, 1);
  
  /* 텍스트 색상 */
  text_label = gtk_label_new("텍스트:");
  gtk_widget_set_halign(text_label, GTK_ALIGN_START);
  
  GdkRGBA text_rgba;
  blouedit_timeline_get_media_type_color(timeline, BLOUEDIT_FILTER_TEXT, &text_rgba);
  
  text_color = gtk_color_button_new_with_rgba(&text_rgba);
  g_object_set_data(G_OBJECT(text_color), "timeline", timeline);
  g_object_set_data(G_OBJECT(text_color), "media-type", 
                  GINT_TO_POINTER(BLOUEDIT_FILTER_TEXT));
  g_signal_connect(text_color, "color-set", 
                 G_CALLBACK(on_color_button_clicked), NULL);
  
  gtk_grid_attach(GTK_GRID(colors_grid), text_label, 0, 3, 1, 1);
  gtk_grid_attach(GTK_GRID(colors_grid), text_color, 1, 3, 1, 1);
  
  /* 이펙트 색상 */
  effect_label = gtk_label_new("이펙트:");
  gtk_widget_set_halign(effect_label, GTK_ALIGN_START);
  
  GdkRGBA effect_rgba;
  blouedit_timeline_get_media_type_color(timeline, BLOUEDIT_FILTER_EFFECT, &effect_rgba);
  
  effect_color = gtk_color_button_new_with_rgba(&effect_rgba);
  g_object_set_data(G_OBJECT(effect_color), "timeline", timeline);
  g_object_set_data(G_OBJECT(effect_color), "media-type", 
                  GINT_TO_POINTER(BLOUEDIT_FILTER_EFFECT));
  g_signal_connect(effect_color, "color-set", 
                 G_CALLBACK(on_color_button_clicked), NULL);
  
  gtk_grid_attach(GTK_GRID(colors_grid), effect_label, 0, 4, 1, 1);
  gtk_grid_attach(GTK_GRID(colors_grid), effect_color, 1, 4, 1, 1);
  
  /* 트랜지션 색상 */
  transition_label = gtk_label_new("트랜지션:");
  gtk_widget_set_halign(transition_label, GTK_ALIGN_START);
  
  GdkRGBA transition_rgba;
  blouedit_timeline_get_media_type_color(timeline, BLOUEDIT_FILTER_TRANSITION, &transition_rgba);
  
  transition_color = gtk_color_button_new_with_rgba(&transition_rgba);
  g_object_set_data(G_OBJECT(transition_color), "timeline", timeline);
  g_object_set_data(G_OBJECT(transition_color), "media-type", 
                  GINT_TO_POINTER(BLOUEDIT_FILTER_TRANSITION));
  g_signal_connect(transition_color, "color-set", 
                 G_CALLBACK(on_color_button_clicked), NULL);
  
  gtk_grid_attach(GTK_GRID(colors_grid), transition_label, 0, 5, 1, 1);
  gtk_grid_attach(GTK_GRID(colors_grid), transition_color, 1, 5, 1, 1);
  
  /* 기본값으로 초기화 버튼 */
  reset_button = gtk_button_new_with_label("기본 색상으로 초기화");
  gtk_grid_attach(GTK_GRID(grid), reset_button, 0, 2, 2, 1);
  
  /* 대화상자 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_OK) {
    /* 모드 설정 적용 */
    BlouEditMediaVisualMode mode = gtk_combo_box_get_active(GTK_COMBO_BOX(mode_combo));
    blouedit_timeline_set_media_visual_mode(timeline, mode);
    
    /* media_type_colors_initialized 플래그 설정 */
    timeline->media_type_colors_initialized = TRUE;
  }
  
  /* 대화상자 닫기 */
  gtk_widget_destroy(dialog);
} 