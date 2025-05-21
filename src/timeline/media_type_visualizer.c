#include <gtk/gtk.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-clip.h>
#include "media_type_visualizer.h"
#include "core/types.h"
#include "core/timeline.h"

/* 기본 미디어 유형 색상 */
static const GdkRGBA DEFAULT_MEDIA_COLORS[] = {
  { 0.2, 0.6, 1.0, 0.7 },    /* 비디오 - 푸른색 */
  { 0.2, 0.8, 0.2, 0.7 },    /* 오디오 - 녹색 */
  { 0.9, 0.6, 0.3, 0.7 },    /* 이미지 - 주황색 */
  { 0.8, 0.3, 0.8, 0.7 },    /* 텍스트 - 보라색 */
  { 0.7, 0.7, 0.2, 0.7 },    /* 효과 - 노란색 */
  { 0.9, 0.3, 0.3, 0.7 }     /* 전환 - 빨간색 */
};

/* 미디어 유형별 아이콘 */
static const gchar* MEDIA_TYPE_ICONS[] = {
  "video-x-generic",         /* 비디오 */
  "audio-x-generic",         /* 오디오 */
  "image-x-generic",         /* 이미지 */
  "text-x-generic",          /* 텍스트 */
  "applications-graphics",   /* 효과 */
  "view-refresh"             /* 전환 */
};

/* 미디어 유형에서 배열 인덱스로 변환 (비트 플래그에서 실제 인덱스로) */
static int 
get_index_from_media_type(BlouEditMediaFilterType type)
{
  switch (type) {
    case BLOUEDIT_FILTER_VIDEO:      return 0;
    case BLOUEDIT_FILTER_AUDIO:      return 1;
    case BLOUEDIT_FILTER_IMAGE:      return 2;
    case BLOUEDIT_FILTER_TEXT:       return 3;
    case BLOUEDIT_FILTER_EFFECT:     return 4;
    case BLOUEDIT_FILTER_TRANSITION: return 5;
    default:                       return 0; /* 기본값으로 비디오 색상 사용 */
  }
}

/* 타임라인에 기본 미디어 유형 색상 초기화 함수 */
void 
blouedit_timeline_initialize_media_type_colors(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 이미 초기화되었는지 확인 */
  if (timeline->media_type_colors_initialized) {
    return;
  }
  
  /* 기본 색상 복사 */
  for (int i = 0; i < 6; i++) {
    timeline->media_type_colors[i] = DEFAULT_MEDIA_COLORS[i];
  }
  
  /* 초기화 완료 표시 */
  timeline->media_type_colors_initialized = TRUE;
}

/* 미디어 시각화 모드 설정 함수 */
void 
blouedit_timeline_set_media_visual_mode(BlouEditTimeline *timeline, 
                                     BlouEditMediaVisualMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 모드 변경 */
  timeline->media_visual_mode = mode;
  
  /* 색상이 초기화되지 않았다면 초기화 */
  if (!timeline->media_type_colors_initialized) {
    blouedit_timeline_initialize_media_type_colors(timeline);
  }
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 미디어 시각화 모드 가져오기 함수 */
BlouEditMediaVisualMode 
blouedit_timeline_get_media_visual_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_MEDIA_VISUAL_MODE_NONE);
  
  return timeline->media_visual_mode;
}

/* 미디어 유형별 색상 설정 함수 */
void 
blouedit_timeline_set_media_type_color(BlouEditTimeline *timeline, 
                                    BlouEditMediaFilterType type, 
                                    const GdkRGBA *color)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(color != NULL);
  
  /* 색상이 초기화되지 않았다면 초기화 */
  if (!timeline->media_type_colors_initialized) {
    blouedit_timeline_initialize_media_type_colors(timeline);
  }
  
  /* 인덱스 계산 및 색상 설정 */
  int index = get_index_from_media_type(type);
  timeline->media_type_colors[index] = *color;
  
  /* 타임라인 갱신 요청 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 미디어 유형별 색상 가져오기 함수 */
void 
blouedit_timeline_get_media_type_color(BlouEditTimeline *timeline, 
                                    BlouEditMediaFilterType type, 
                                    GdkRGBA *color)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(color != NULL);
  
  /* 색상이 초기화되지 않았다면 초기화 */
  if (!timeline->media_type_colors_initialized) {
    blouedit_timeline_initialize_media_type_colors(timeline);
  }
  
  /* 인덱스 계산 및 색상 반환 */
  int index = get_index_from_media_type(type);
  *color = timeline->media_type_colors[index];
}

/* 클립의 미디어 유형 식별 함수 */
BlouEditMediaFilterType 
blouedit_timeline_get_clip_media_type(GESClip *clip)
{
  g_return_val_if_fail(GES_IS_CLIP(clip), BLOUEDIT_FILTER_VIDEO);
  
  GESTrackType track_types = ges_clip_get_supported_formats(clip);
  
  if (track_types & GES_TRACK_TYPE_VIDEO) {
    if (GES_IS_TRANSITION_CLIP(clip)) {
      return BLOUEDIT_FILTER_TRANSITION;
    }
    
    if (GES_IS_EFFECT_CLIP(clip)) {
      return BLOUEDIT_FILTER_EFFECT;
    }
    
    if (GES_IS_TEXT_OVERLAY_CLIP(clip) || 
        GES_IS_TITLE_CLIP(clip)) {
      return BLOUEDIT_FILTER_TEXT;
    }
    
    if (GES_IS_IMAGE_CLIP(clip) || 
        GES_IS_STILL_CLIP(clip)) {
      return BLOUEDIT_FILTER_IMAGE;
    }
    
    return BLOUEDIT_FILTER_VIDEO;
  }
  
  if (track_types & GES_TRACK_TYPE_AUDIO) {
    return BLOUEDIT_FILTER_AUDIO;
  }
  
  return BLOUEDIT_FILTER_VIDEO;  /* 기본값 */
}

/* 클립의 미디어 유형에 해당하는 색상 가져오기 함수 */
void 
blouedit_timeline_get_color_for_clip(BlouEditTimeline *timeline, 
                                  GESClip *clip, 
                                  GdkRGBA *color)
{
  BlouEditMediaFilterType type = blouedit_timeline_get_clip_media_type(clip);
  blouedit_timeline_get_media_type_color(timeline, type, color);
}

/* 미디어 유형별 아이콘 이름 가져오기 함수 */
const gchar* 
blouedit_timeline_get_icon_for_media_type(BlouEditMediaFilterType type)
{
  int index = get_index_from_media_type(type);
  return MEDIA_TYPE_ICONS[index];
}

/* 색상 버튼 클릭 핸들러 */
static void
on_color_button_clicked(GtkColorButton *button, gpointer user_data)
{
  BlouEditMediaFilterType *type_ptr = (BlouEditMediaFilterType*)g_object_get_data(G_OBJECT(button), "media-type");
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(g_object_get_data(G_OBJECT(button), "timeline"));
  
  if (type_ptr && timeline) {
    GdkRGBA color;
    gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(button), &color);
    blouedit_timeline_set_media_type_color(timeline, *type_ptr, &color);
  }
}

/* 모드 콤보박스 변경 핸들러 */
static void
on_visual_mode_changed(GtkComboBox *combo, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  gint active = gtk_combo_box_get_active(combo);
  
  blouedit_timeline_set_media_visual_mode(timeline, (BlouEditMediaVisualMode)active);
}

/* 미디어 시각화 설정 대화상자 표시 함수 */
void 
blouedit_timeline_show_media_visual_settings(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *grid;
  GtkWidget *mode_label, *mode_combo;
  GtkWidget *video_label, *video_color, *audio_label, *audio_color;
  GtkWidget *image_label, *image_color, *text_label, *text_color;
  GtkWidget *effect_label, *effect_color, *trans_label, *trans_color;
  
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 색상이 초기화되지 않았다면 초기화 */
  if (!timeline->media_type_colors_initialized) {
    blouedit_timeline_initialize_media_type_colors(timeline);
  }
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("미디어 유형별 시각화 설정",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_취소", GTK_RESPONSE_CANCEL,
                                     "_적용", GTK_RESPONSE_ACCEPT,
                                     NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 400, 350);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 그리드 생성 */
  grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(grid), 10);
  gtk_grid_set_column_spacing(GTK_GRID(grid), 10);
  gtk_container_add(GTK_CONTAINER(content_area), grid);
  
  /* 시각화 모드 선택 UI */
  mode_label = gtk_label_new("시각화 모드:");
  gtk_widget_set_halign(mode_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), mode_label, 0, 0, 1, 1);
  
  mode_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "구분 없음");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "색상으로 구분");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "아이콘으로 구분");
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(mode_combo), "색상과 아이콘 모두 사용");
  gtk_combo_box_set_active(GTK_COMBO_BOX(mode_combo), (gint)timeline->media_visual_mode);
  gtk_grid_attach(GTK_GRID(grid), mode_combo, 1, 0, 2, 1);
  
  /* 비디오 색상 설정 */
  video_label = gtk_label_new("비디오 클립:");
  gtk_widget_set_halign(video_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), video_label, 0, 1, 1, 1);
  
  video_color = gtk_color_button_new_with_rgba(&timeline->media_type_colors[0]);
  gtk_grid_attach(GTK_GRID(grid), video_color, 1, 1, 2, 1);
  
  /* 오디오 색상 설정 */
  audio_label = gtk_label_new("오디오 클립:");
  gtk_widget_set_halign(audio_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), audio_label, 0, 2, 1, 1);
  
  audio_color = gtk_color_button_new_with_rgba(&timeline->media_type_colors[1]);
  gtk_grid_attach(GTK_GRID(grid), audio_color, 1, 2, 2, 1);
  
  /* 이미지 색상 설정 */
  image_label = gtk_label_new("이미지 클립:");
  gtk_widget_set_halign(image_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), image_label, 0, 3, 1, 1);
  
  image_color = gtk_color_button_new_with_rgba(&timeline->media_type_colors[2]);
  gtk_grid_attach(GTK_GRID(grid), image_color, 1, 3, 2, 1);
  
  /* 텍스트 색상 설정 */
  text_label = gtk_label_new("텍스트 클립:");
  gtk_widget_set_halign(text_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), text_label, 0, 4, 1, 1);
  
  text_color = gtk_color_button_new_with_rgba(&timeline->media_type_colors[3]);
  gtk_grid_attach(GTK_GRID(grid), text_color, 1, 4, 2, 1);
  
  /* 효과 색상 설정 */
  effect_label = gtk_label_new("효과 클립:");
  gtk_widget_set_halign(effect_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), effect_label, 0, 5, 1, 1);
  
  effect_color = gtk_color_button_new_with_rgba(&timeline->media_type_colors[4]);
  gtk_grid_attach(GTK_GRID(grid), effect_color, 1, 5, 2, 1);
  
  /* 전환 색상 설정 */
  trans_label = gtk_label_new("전환 클립:");
  gtk_widget_set_halign(trans_label, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), trans_label, 0, 6, 1, 1);
  
  trans_color = gtk_color_button_new_with_rgba(&timeline->media_type_colors[5]);
  gtk_grid_attach(GTK_GRID(grid), trans_color, 1, 6, 2, 1);
  
  /* 색상 버튼 클릭 핸들러 설정 및 데이터 연결 */
  BlouEditMediaFilterType video_type = BLOUEDIT_FILTER_VIDEO;
  g_object_set_data(G_OBJECT(video_color), "media-type", &video_type);
  g_object_set_data(G_OBJECT(video_color), "timeline", timeline);
  g_signal_connect(video_color, "color-set", G_CALLBACK(on_color_button_clicked), NULL);
  
  BlouEditMediaFilterType audio_type = BLOUEDIT_FILTER_AUDIO;
  g_object_set_data(G_OBJECT(audio_color), "media-type", &audio_type);
  g_object_set_data(G_OBJECT(audio_color), "timeline", timeline);
  g_signal_connect(audio_color, "color-set", G_CALLBACK(on_color_button_clicked), NULL);
  
  BlouEditMediaFilterType image_type = BLOUEDIT_FILTER_IMAGE;
  g_object_set_data(G_OBJECT(image_color), "media-type", &image_type);
  g_object_set_data(G_OBJECT(image_color), "timeline", timeline);
  g_signal_connect(image_color, "color-set", G_CALLBACK(on_color_button_clicked), NULL);
  
  BlouEditMediaFilterType text_type = BLOUEDIT_FILTER_TEXT;
  g_object_set_data(G_OBJECT(text_color), "media-type", &text_type);
  g_object_set_data(G_OBJECT(text_color), "timeline", timeline);
  g_signal_connect(text_color, "color-set", G_CALLBACK(on_color_button_clicked), NULL);
  
  BlouEditMediaFilterType effect_type = BLOUEDIT_FILTER_EFFECT;
  g_object_set_data(G_OBJECT(effect_color), "media-type", &effect_type);
  g_object_set_data(G_OBJECT(effect_color), "timeline", timeline);
  g_signal_connect(effect_color, "color-set", G_CALLBACK(on_color_button_clicked), NULL);
  
  BlouEditMediaFilterType trans_type = BLOUEDIT_FILTER_TRANSITION;
  g_object_set_data(G_OBJECT(trans_color), "media-type", &trans_type);
  g_object_set_data(G_OBJECT(trans_color), "timeline", timeline);
  g_signal_connect(trans_color, "color-set", G_CALLBACK(on_color_button_clicked), NULL);
  
  /* 모드 콤보박스 변경 핸들러 설정 */
  g_signal_connect(mode_combo, "changed", G_CALLBACK(on_visual_mode_changed), timeline);
  
  /* 대화상자 응답 핸들러 */
  g_signal_connect(dialog, "response", G_CALLBACK(gtk_widget_destroy), NULL);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
} 