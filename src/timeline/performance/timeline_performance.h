#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 타임라인 렌더링 캐시 관련 함수 */
void         blouedit_timeline_init_rendering_cache(BlouEditTimeline *timeline);
void         blouedit_timeline_clear_rendering_cache(BlouEditTimeline *timeline);
void         blouedit_timeline_set_rendering_cache_enabled(BlouEditTimeline *timeline, gboolean enabled);
gboolean     blouedit_timeline_get_rendering_cache_enabled(BlouEditTimeline *timeline);
void         blouedit_timeline_configure_cache_settings(BlouEditTimeline *timeline);

/* 성능 모드 관련 함수 */
typedef enum {
  BLOUEDIT_PERFORMANCE_MODE_QUALITY,     /* 재생 품질 우선 */
  BLOUEDIT_PERFORMANCE_MODE_RESPONSIVE,  /* 편집 반응성 우선 */
  BLOUEDIT_PERFORMANCE_MODE_BALANCED     /* 균형 모드 */
} BlouEditPerformanceMode;

void                    blouedit_timeline_set_performance_mode(BlouEditTimeline *timeline, BlouEditPerformanceMode mode);
BlouEditPerformanceMode blouedit_timeline_get_performance_mode(BlouEditTimeline *timeline);
void                    blouedit_timeline_show_performance_settings_dialog(BlouEditTimeline *timeline);

/* 프리렌더 구간 설정 관련 함수 */
void         blouedit_timeline_set_prerender_segment(BlouEditTimeline *timeline, GstClockTime start, GstClockTime end);
void         blouedit_timeline_remove_prerender_segment(BlouEditTimeline *timeline, GstClockTime start, GstClockTime end);
void         blouedit_timeline_clear_prerender_segments(BlouEditTimeline *timeline);
void         blouedit_timeline_start_prerender(BlouEditTimeline *timeline);
void         blouedit_timeline_stop_prerender(BlouEditTimeline *timeline);
void         blouedit_timeline_show_prerender_dialog(BlouEditTimeline *timeline);

/* 하드웨어 가속 관련 함수 */
typedef enum {
  BLOUEDIT_HW_ACCEL_DISABLED,            /* 하드웨어 가속 비활성화 */
  BLOUEDIT_HW_ACCEL_AUTO,                /* 자동 감지 */
  BLOUEDIT_HW_ACCEL_INTEL,               /* Intel HW 가속 */
  BLOUEDIT_HW_ACCEL_NVIDIA,              /* NVIDIA HW 가속 */
  BLOUEDIT_HW_ACCEL_AMD                  /* AMD HW 가속 */
} BlouEditHwAccelType;

void                blouedit_timeline_set_hw_acceleration(BlouEditTimeline *timeline, BlouEditHwAccelType accel_type);
BlouEditHwAccelType blouedit_timeline_get_hw_acceleration(BlouEditTimeline *timeline);
gboolean            blouedit_timeline_is_hw_acceleration_available(BlouEditTimeline *timeline, BlouEditHwAccelType accel_type);
void                blouedit_timeline_show_hw_acceleration_dialog(BlouEditTimeline *timeline);

/* 자동 프록시 생성 관련 함수 */
void         blouedit_timeline_enable_auto_proxy(BlouEditTimeline *timeline, gboolean enabled);
gboolean     blouedit_timeline_get_auto_proxy_enabled(BlouEditTimeline *timeline);
void         blouedit_timeline_set_proxy_resolution(BlouEditTimeline *timeline, gint width, gint height);
void         blouedit_timeline_get_proxy_resolution(BlouEditTimeline *timeline, gint *width, gint *height);
void         blouedit_timeline_generate_proxy_for_clip(BlouEditTimeline *timeline, GESClip *clip);
void         blouedit_timeline_generate_proxies_for_all_clips(BlouEditTimeline *timeline);
void         blouedit_timeline_show_proxy_settings_dialog(BlouEditTimeline *timeline);

G_END_DECLS 