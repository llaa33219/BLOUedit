#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include "timeline.h"
#include "core/types.h"
#include "tracks/tracks.h"
#include "clip_drawer.h"

/**
 * blouedit_timeline_show_message:
 * @timeline: 타임라인 객체
 * @message: 표시할 메시지
 *
 * 사용자에게 간단한 메시지를 표시합니다.
 * 현재는 로그에만 출력합니다.
 */
void
blouedit_timeline_show_message (BlouEditTimeline *timeline, const gchar *message)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (message != NULL);
  
  // 로그에 메시지 출력
  g_message ("%s", message);
}

/**
 * on_track_type_color_changed:
 * @radio: 트랙 유형 라디오 버튼
 * @color_button: 색상 선택 버튼
 *
 * 트랙 유형이 변경되었을 때 색상 기본값을 업데이트합니다.
 */
static void
on_track_type_color_changed (GtkToggleButton *radio, GtkWidget *color_button)
{
  if (gtk_toggle_button_get_active (radio)) {
    // 비디오 트랙 기본 색상
    GdkRGBA color;
    gdk_rgba_parse (&color, "#CC5588");
    gtk_color_chooser_set_rgba (GTK_COLOR_CHOOSER (color_button), &color);
  } else {
    // 오디오 트랙 기본 색상
    GdkRGBA color;
    gdk_rgba_parse (&color, "#5588CC");
    gtk_color_chooser_set_rgba (GTK_COLOR_CHOOSER (color_button), &color);
  }
}

/**
 * blouedit_timeline_draw_tracks:
 * @timeline: 타임라인 객체
 * @cr: Cairo 컨텍스트
 * @width: 그리기 영역 너비
 * @height: 그리기 영역 높이
 *
 * 타임라인의 모든 트랙을 그립니다.
 * 무제한 트랙 지원을 위해 스크롤 뷰를 사용하도록 개선되었습니다.
 */
void
blouedit_timeline_draw_tracks (BlouEditTimeline *timeline, cairo_t *cr, int width, int height)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));

  GSList *l;
  int y_pos = timeline->ruler_height;
  int track_x = timeline->timeline_start_x;
  int track_width = width - track_x;
  
  // 트랙 헤더 배경 그리기
  cairo_set_source_rgb (cr, 0.2, 0.2, 0.2);
  cairo_rectangle (cr, 0, timeline->ruler_height, timeline->timeline_start_x, height - timeline->ruler_height);
  cairo_fill (cr);
  
  // 트랙 헤더와 타임라인 영역 구분선
  cairo_set_source_rgb (cr, 0.3, 0.3, 0.3);
  cairo_set_line_width (cr, 1.0);
  cairo_move_to (cr, timeline->timeline_start_x, timeline->ruler_height);
  cairo_line_to (cr, timeline->timeline_start_x, height);
  cairo_stroke (cr);
  
  // 각 트랙 그리기
  for (l = timeline->tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)l->data;
    int track_height = track->folded ? track->folded_height : track->height;
    
    // 현재 표시 영역에 보이는 트랙만 그리기 (성능 최적화)
    if ((y_pos + track_height) >= timeline->ruler_height && y_pos <= height) {
      // 트랙 헤더 배경
      cairo_set_source_rgba (cr, track->color.red, track->color.green, track->color.blue, 0.3);
      cairo_rectangle (cr, 0, y_pos, timeline->timeline_start_x, track_height);
      cairo_fill (cr);
      
      // 트랙 헤더 텍스트
      cairo_set_source_rgb (cr, 0.9, 0.9, 0.9);
      cairo_select_font_face (cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
      cairo_set_font_size (cr, 12);
      
      cairo_text_extents_t extents;
      cairo_text_extents (cr, track->name, &extents);
      
      // 텍스트가 너무 길 경우 자르기
      char *display_name = g_strdup (track->name);
      if (extents.width > timeline->timeline_start_x - 20) {
        // 텍스트가 너무 길면 줄임표로 표시
        int max_chars = strlen(track->name) * ((timeline->timeline_start_x - 20) / extents.width);
        if (max_chars < 3) max_chars = 3;
        if (max_chars < strlen(track->name)) {
          display_name[max_chars-3] = '.';
          display_name[max_chars-2] = '.';
          display_name[max_chars-1] = '.';
          display_name[max_chars] = '\0';
        }
      }
      
      cairo_move_to (cr, 10, y_pos + (track_height / 2) + (extents.height / 2));
      cairo_show_text (cr, display_name);
      g_free (display_name);
      
      // 타임라인 트랙 배경
      GESTrack *ges_track = track->ges_track;
      GESTrackType track_type = ges_track_get_track_type (ges_track);
      
      if (track_type == GES_TRACK_TYPE_VIDEO) {
        // 비디오 트랙 배경색
        cairo_set_source_rgba (cr, track->color.red * 0.5, track->color.green * 0.5, track->color.blue * 0.5, 0.2);
      } else if (track_type == GES_TRACK_TYPE_AUDIO) {
        // 오디오 트랙 배경색
        cairo_set_source_rgba (cr, track->color.red * 0.5, track->color.green * 0.5, track->color.blue * 0.5, 0.2);
      } else {
        // 기타 트랙 배경색
        cairo_set_source_rgba (cr, 0.3, 0.3, 0.3, 0.2);
      }
      
      cairo_rectangle (cr, timeline->timeline_start_x, y_pos, track_width, track_height);
      cairo_fill (cr);
      
      // 트랙 구분선
      cairo_set_source_rgba (cr, 0.3, 0.3, 0.3, 0.8);
      cairo_set_line_width (cr, 1.0);
      cairo_move_to (cr, 0, y_pos);
      cairo_line_to (cr, width, y_pos);
      cairo_stroke (cr);
      
      // 만약 이 트랙이 선택된 트랙이라면 강조 표시
      if (timeline->selected_track == track) {
        cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 0.2);
        cairo_rectangle (cr, 0, y_pos, width, track_height);
        cairo_stroke (cr);
      }
      
      // 트랙의 모든 클립 그리기
      blouedit_timeline_draw_track_clips(timeline, cr, track, y_pos, track_height);
    }
    
    // 다음 트랙 위치 계산
    y_pos += track_height + timeline->track_spacing;
  }
  
  // 트랙이 없거나 트랙 전체 높이가 타임라인보다 작은 경우 빈 공간 채우기
  if (y_pos < height) {
    cairo_set_source_rgb (cr, 0.15, 0.15, 0.15);
    cairo_rectangle (cr, 0, y_pos, width, height - y_pos);
    cairo_fill (cr);
  }
}

/**
 * blouedit_timeline_create_scrolled_view:
 * @timeline: 타임라인 객체
 *
 * 무제한 트랙을 위한 스크롤 가능한 타임라인 뷰를 생성합니다.
 * 이 함수는 타임라인 생성 중에 호출되어야 합니다.
 *
 * Returns: 새로 생성된 스크롤 창 위젯
 */
GtkWidget *
blouedit_timeline_create_scrolled_view (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  // 스크롤 창 생성
  GtkWidget *scrolled_window = gtk_scrolled_window_new (NULL, NULL);
  gtk_scrolled_window_set_policy (GTK_SCROLLED_WINDOW (scrolled_window),
                                 GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  
  // 그리기 영역 생성 및 스크롤 창에 추가
  GtkWidget *drawing_area = gtk_drawing_area_new ();
  gtk_container_add (GTK_CONTAINER (scrolled_window), drawing_area);
  
  // 그리기 영역 이벤트 연결
  g_signal_connect (drawing_area, "draw", G_CALLBACK (blouedit_timeline_draw_tracks), timeline);
  gtk_widget_add_events (drawing_area,
                        GDK_BUTTON_PRESS_MASK |
                        GDK_BUTTON_RELEASE_MASK |
                        GDK_BUTTON_MOTION_MASK |
                        GDK_POINTER_MOTION_MASK |
                        GDK_SCROLL_MASK);
  
  // 마우스 이벤트 연결 (필요시 추가)
  
  // 스크롤 창 반환
  return scrolled_window;
} 