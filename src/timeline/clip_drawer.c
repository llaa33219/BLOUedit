#include <gtk/gtk.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include <gst/editing-services/ges-clip.h>
#include "clip_drawer.h"
#include "core/types.h"
#include "core/timeline.h"
#include "media_type_visualizer.h"

/* 클립 그리기 함수 */
void 
blouedit_timeline_draw_clip (BlouEditTimeline *timeline, 
                          cairo_t *cr, 
                          GESClip *clip, 
                          int track_y, 
                          int track_height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(GES_IS_CLIP(clip));
  
  /* 클립 정보 가져오기 */
  guint64 start = ges_timeline_element_get_start(GES_TIMELINE_ELEMENT(clip));
  guint64 duration = ges_timeline_element_get_duration(GES_TIMELINE_ELEMENT(clip));
  const gchar *name = ges_timeline_element_get_name(GES_TIMELINE_ELEMENT(clip));
  if (!name) name = "Unnamed clip";
  
  /* 화면 좌표 계산 */
  double zoom = timeline->zoom_level;
  int x = timeline->timeline_start_x + (start * zoom / GST_SECOND);
  int w = duration * zoom / GST_SECOND;
  
  /* 미디어 유형 및 색상 가져오기 */
  BlouEditMediaFilterType media_type = blouedit_timeline_get_clip_media_type(clip);
  GdkRGBA color;
  
  /* 미디어 시각화 모드에 따라 처리 */
  if (timeline->media_visual_mode != BLOUEDIT_MEDIA_VISUAL_MODE_NONE) {
    /* 색상 가져오기 */
    blouedit_timeline_get_color_for_clip(timeline, clip, &color);
  } else {
    /* 기본 색상 설정 */
    color.red = 0.4;
    color.green = 0.6;
    color.blue = 0.9;
    color.alpha = 0.7;
  }
  
  /* 클립 배경 그리기 */
  cairo_set_source_rgba(cr, color.red, color.green, color.blue, color.alpha);
  cairo_rectangle(cr, x, track_y, w, track_height);
  cairo_fill(cr);
  
  /* 클립 테두리 그리기 */
  if (blouedit_timeline_is_clip_selected(timeline, clip)) {
    /* 선택된 클립은 더 굵고 밝은 테두리 */
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.8);
    cairo_set_line_width(cr, 2.0);
  } else {
    /* 일반 클립은 얇은 테두리 */
    cairo_set_source_rgba(cr, 0.2, 0.2, 0.2, 0.5);
    cairo_set_line_width(cr, 1.0);
  }
  cairo_rectangle(cr, x, track_y, w, track_height);
  cairo_stroke(cr);
  
  /* 아이콘 그리기 (미디어 시각화 모드가 적용된 경우) */
  if (timeline->media_visual_mode == BLOUEDIT_MEDIA_VISUAL_MODE_ICON || 
      timeline->media_visual_mode == BLOUEDIT_MEDIA_VISUAL_MODE_BOTH) {
    blouedit_timeline_draw_clip_icon(timeline, cr, media_type, x + 5, track_y + 5, 16);
  }
  
  /* 텍스트 라벨 그리기 */
  if (w > 50) { /* 클립이 충분히 넓은 경우에만 이름 표시 */
    blouedit_timeline_draw_clip_label(timeline, cr, clip, x + 25, track_y, w - 30, track_height);
  }
}

/* 트랙의 모든 클립 그리기 함수 */
void 
blouedit_timeline_draw_track_clips (BlouEditTimeline *timeline, 
                                 cairo_t *cr, 
                                 BlouEditTimelineTrack *track, 
                                 int track_y, 
                                 int track_height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(track != NULL);
  g_return_if_fail(GES_IS_TRACK(track->ges_track));
  
  GList *clip_elements, *tmp;
  GESTimeline *ges_timeline = timeline->ges_timeline;
  
  /* 타임라인의 모든 요소(클립) 가져오기 */
  clip_elements = ges_timeline_get_tracks_elements(ges_timeline);
  
  /* 클립 정렬하기 (시작 시간 기준) */
  clip_elements = g_list_sort(clip_elements, (GCompareFunc)blouedit_timeline_compare_clips_by_start);
  
  /* 각 클립 그리기 */
  for (tmp = clip_elements; tmp; tmp = tmp->next) {
    GESTrackElement *track_element = GES_TRACK_ELEMENT(tmp->data);
    GESTrack *element_track = ges_track_element_get_track(track_element);
    
    /* 현재 트랙에 속한 클립만 그리기 */
    if (element_track == track->ges_track) {
      GESClip *clip = ges_track_element_get_parent(track_element);
      if (clip) {
        /* 현재 필터 설정에 맞는 미디어 타입인지 확인 */
        BlouEditMediaFilterType media_type = blouedit_timeline_get_clip_media_type(clip);
        if ((timeline->media_filter & media_type) || timeline->media_filter == BLOUEDIT_FILTER_ALL) {
          /* 클립 그리기 */
          blouedit_timeline_draw_clip(timeline, cr, clip, track_y, track_height);
        }
      }
    }
  }
  
  /* 리스트 해제 */
  g_list_free(clip_elements);
}

/* 클립 간 시작 시간 기준 정렬을 위한 비교 함수 */
int 
blouedit_timeline_compare_clips_by_start (GESTimelineElement *a, GESTimelineElement *b)
{
  guint64 start_a = ges_timeline_element_get_start(a);
  guint64 start_b = ges_timeline_element_get_start(b);
  
  if (start_a < start_b)
    return -1;
  else if (start_a > start_b)
    return 1;
  else
    return 0;
}

/* 클립 텍스트 라벨 그리기 함수 */
void 
blouedit_timeline_draw_clip_label (BlouEditTimeline *timeline, 
                                cairo_t *cr, 
                                GESClip *clip, 
                                int x, int y, 
                                int width, int height)
{
  const gchar *name = ges_timeline_element_get_name(GES_TIMELINE_ELEMENT(clip));
  if (!name) name = "Unnamed clip";
  
  /* 텍스트 설정 */
  cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  cairo_set_font_size(cr, 12);
  cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.8);
  
  /* 텍스트 크기 측정 */
  cairo_text_extents_t extents;
  cairo_text_extents(cr, name, &extents);
  
  /* 텍스트가 너무 길면 자르기 */
  char *display_name = g_strdup(name);
  if (extents.width > width) {
    /* 텍스트가 너무 길면 줄임표로 표시 */
    int max_chars = strlen(name) * (width / extents.width);
    if (max_chars < 3) max_chars = 3;
    if (max_chars < strlen(name)) {
      display_name[max_chars-3] = '.';
      display_name[max_chars-2] = '.';
      display_name[max_chars-1] = '.';
      display_name[max_chars] = '\0';
    }
  }
  
  /* 텍스트 그리기 */
  cairo_move_to(cr, x, y + (height / 2) + (extents.height / 2));
  cairo_show_text(cr, display_name);
  g_free(display_name);
}

/* 클립 아이콘 그리기 함수 */
void 
blouedit_timeline_draw_clip_icon (BlouEditTimeline *timeline, 
                               cairo_t *cr, 
                               BlouEditMediaFilterType type, 
                               int x, int y, 
                               int size)
{
  /* 아이콘 그리기 - 여기서는 간단한 형태로 그림 */
  /* GTK 아이콘 테마를 사용할 수도 있음 */
  
  cairo_save(cr);
  
  /* 아이콘 배경 */
  cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.9);
  cairo_arc(cr, x + size/2, y + size/2, size/2, 0, 2 * G_PI);
  cairo_fill(cr);
  
  /* 아이콘 형태 (미디어 유형별로 다르게) */
  cairo_set_source_rgba(cr, 0.2, 0.2, 0.2, 0.8);
  cairo_set_line_width(cr, 1.5);
  
  switch (type) {
    case BLOUEDIT_FILTER_VIDEO:
      /* 비디오 아이콘 (사각형 + 삼각형) */
      cairo_rectangle(cr, x + size*0.25, y + size*0.25, size*0.5, size*0.5);
      cairo_stroke(cr);
      cairo_move_to(cr, x + size*0.35, y + size*0.35);
      cairo_line_to(cr, x + size*0.65, y + size*0.5);
      cairo_line_to(cr, x + size*0.35, y + size*0.65);
      cairo_close_path(cr);
      cairo_fill(cr);
      break;
      
    case BLOUEDIT_FILTER_AUDIO:
      /* 오디오 아이콘 (파형) */
      cairo_move_to(cr, x + size*0.25, y + size*0.5);
      cairo_line_to(cr, x + size*0.35, y + size*0.25);
      cairo_line_to(cr, x + size*0.45, y + size*0.5);
      cairo_line_to(cr, x + size*0.55, y + size*0.75);
      cairo_line_to(cr, x + size*0.65, y + size*0.5);
      cairo_line_to(cr, x + size*0.75, y + size*0.5);
      cairo_stroke(cr);
      break;
      
    case BLOUEDIT_FILTER_IMAGE:
      /* 이미지 아이콘 (사각형 + 원) */
      cairo_rectangle(cr, x + size*0.25, y + size*0.25, size*0.5, size*0.5);
      cairo_stroke(cr);
      cairo_arc(cr, x + size*0.4, y + size*0.4, size*0.1, 0, 2 * G_PI);
      cairo_fill(cr);
      break;
      
    case BLOUEDIT_FILTER_TEXT:
      /* 텍스트 아이콘 (가로선) */
      cairo_move_to(cr, x + size*0.25, y + size*0.4);
      cairo_line_to(cr, x + size*0.75, y + size*0.4);
      cairo_stroke(cr);
      cairo_move_to(cr, x + size*0.25, y + size*0.6);
      cairo_line_to(cr, x + size*0.65, y + size*0.6);
      cairo_stroke(cr);
      break;
      
    case BLOUEDIT_FILTER_EFFECT:
      /* 효과 아이콘 (별 모양) */
      for (int i = 0; i < 5; i++) {
        double angle = i * 2 * G_PI / 5 - G_PI / 2;
        double next_angle = (i + 2) % 5 * 2 * G_PI / 5 - G_PI / 2;
        
        cairo_move_to(cr, 
                     x + size/2 + size/4 * cos(angle),
                     y + size/2 + size/4 * sin(angle));
        cairo_line_to(cr, 
                     x + size/2 + size/4 * cos(next_angle),
                     y + size/2 + size/4 * sin(next_angle));
      }
      cairo_close_path(cr);
      cairo_stroke(cr);
      break;
      
    case BLOUEDIT_FILTER_TRANSITION:
      /* 전환 아이콘 (두 개의 겹치는 사각형) */
      cairo_rectangle(cr, x + size*0.2, y + size*0.3, size*0.4, size*0.4);
      cairo_stroke(cr);
      cairo_rectangle(cr, x + size*0.4, y + size*0.4, size*0.4, size*0.4);
      cairo_stroke(cr);
      break;
      
    default:
      /* 기본 아이콘 (원) */
      cairo_arc(cr, x + size/2, y + size/2, size/4, 0, 2 * G_PI);
      cairo_stroke(cr);
  }
  
  cairo_restore(cr);
} 