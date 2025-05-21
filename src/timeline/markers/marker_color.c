#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"
#include "markers.h"

/* 마커 타입별 기본 색상 정의 */
static const GdkRGBA marker_type_colors[] = {
  { 1.0, 1.0, 1.0, 1.0 },  /* BLOUEDIT_MARKER_TYPE_GENERIC - 흰색 */
  { 0.0, 0.8, 0.0, 1.0 },  /* BLOUEDIT_MARKER_TYPE_CUE - 녹색 */
  { 0.0, 0.8, 0.8, 1.0 },  /* BLOUEDIT_MARKER_TYPE_IN - 청록색 */
  { 0.8, 0.0, 0.8, 1.0 },  /* BLOUEDIT_MARKER_TYPE_OUT - 보라색 */
  { 0.8, 0.8, 0.0, 1.0 },  /* BLOUEDIT_MARKER_TYPE_CHAPTER - 노란색 */
  { 1.0, 0.0, 0.0, 1.0 },  /* BLOUEDIT_MARKER_TYPE_ERROR - 빨간색 */
  { 1.0, 0.6, 0.0, 1.0 },  /* BLOUEDIT_MARKER_TYPE_WARNING - 주황색 */
  { 0.4, 0.8, 1.0, 1.0 }   /* BLOUEDIT_MARKER_TYPE_COMMENT - 하늘색 */
};

/* 마커 타입별 기본 색상 반환 함수 */
void blouedit_marker_get_default_color_for_type(BlouEditMarkerType type, GdkRGBA *color)
{
  g_return_if_fail(color != NULL);
  
  if (type >= 0 && type < G_N_ELEMENTS(marker_type_colors)) {
    *color = marker_type_colors[type];
  } else {
    /* 기본값은 흰색 */
    color->red = 1.0;
    color->green = 1.0;
    color->blue = 1.0;
    color->alpha = 1.0;
  }
}

/* 마커 타입에 따라 자동으로 색상 설정하는 함수 */
void blouedit_timeline_set_marker_color_by_type(BlouEditTimeline *timeline, 
                                              BlouEditTimelineMarker *marker)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(marker != NULL);
  
  GdkRGBA color;
  blouedit_marker_get_default_color_for_type(marker->type, &color);
  
  /* 마커에 색상 설정 */
  blouedit_timeline_set_marker_color(timeline, marker, &color);
}

/* 타임라인의 모든 마커를 타입에 따라 색상 자동 설정 */
void blouedit_timeline_recolor_all_markers_by_type(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 타임라인에 있는 모든 마커 가져오기 */
  GSList *markers = blouedit_timeline_get_markers(timeline);
  
  /* 각 마커에 타입별 색상 적용 */
  for (GSList *m = markers; m != NULL; m = m->next) {
    BlouEditTimelineMarker *marker = (BlouEditTimelineMarker *)m->data;
    blouedit_timeline_set_marker_color_by_type(timeline, marker);
  }
  
  /* 타임라인 다시 그리기 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 사용자 지정 색상 선택 대화상자 표시 */
void blouedit_timeline_show_marker_color_dialog(BlouEditTimeline *timeline, 
                                              BlouEditTimelineMarker *marker)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(marker != NULL);
  
  /* 색상 선택 대화상자 생성 */
  GtkWidget *dialog = gtk_color_chooser_dialog_new("마커 색상 선택", 
                                                 GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))));
  
  /* 현재 마커 색상 설정 */
  gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(dialog), &marker->color);
  
  /* 대화상자 응답 처리 */
  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_OK) {
    GdkRGBA color;
    gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(dialog), &color);
    
    /* 선택한 색상으로 마커 색상 설정 */
    blouedit_timeline_set_marker_color(timeline, marker, &color);
    
    /* 타임라인 다시 그리기 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
  }
  
  /* 대화상자 파괴 */
  gtk_widget_destroy(dialog);
} 