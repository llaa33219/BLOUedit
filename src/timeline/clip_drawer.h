#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include <gst/editing-services/ges-clip.h>
#include "core/types.h"
#include "core/timeline.h"
#include "media_type_visualizer.h"

G_BEGIN_DECLS

/* 클립 그리기 함수 */
void blouedit_timeline_draw_clip (BlouEditTimeline *timeline, 
                               cairo_t *cr, 
                               GESClip *clip, 
                               int track_y, 
                               int track_height);

/* 트랙의 모든 클립 그리기 함수 */
void blouedit_timeline_draw_track_clips (BlouEditTimeline *timeline, 
                                      cairo_t *cr, 
                                      BlouEditTimelineTrack *track, 
                                      int track_y, 
                                      int track_height);

/* 클립 텍스트 라벨 그리기 함수 */
void blouedit_timeline_draw_clip_label (BlouEditTimeline *timeline, 
                                     cairo_t *cr, 
                                     GESClip *clip, 
                                     int x, int y, 
                                     int width, int height);

/* 클립 아이콘 그리기 함수 */
void blouedit_timeline_draw_clip_icon (BlouEditTimeline *timeline, 
                                    cairo_t *cr, 
                                    BlouEditMediaFilterType type, 
                                    int x, int y, 
                                    int size);

G_END_DECLS 