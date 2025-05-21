#pragma once

#include <gtk/gtk.h>
#include "core/types.h"
#include "core/timeline.h"

G_BEGIN_DECLS

/* Edge Trimming 모드 정의 */
typedef enum {
  BLOUEDIT_EDGE_TRIM_MODE_NORMAL,    /* 기본 에지 트리밍 */
  BLOUEDIT_EDGE_TRIM_MODE_PRECISE,   /* 프레임 단위 정밀 트리밍 */
  BLOUEDIT_EDGE_TRIM_MODE_RIPPLE,    /* 리플 트리밍 (연결된 클립 이동) */
  BLOUEDIT_EDGE_TRIM_MODE_ROLL       /* 롤 트리밍 (인접 클립 조정) */
} BlouEditEdgeTrimMode;

/* Edge Trimming 상태 구조체 */
typedef struct _BlouEditEdgeTrimState BlouEditEdgeTrimState;
struct _BlouEditEdgeTrimState {
  gboolean active;                /* 활성화 여부 */
  BlouEditEdgeTrimMode mode;      /* 트리밍 모드 */
  GESClip *clip;                  /* 대상 클립 */
  BlouEditClipEdge edge;          /* 트리밍할 에지 (시작/끝) */
  gint64 original_position;       /* 원래 에지 위치 */
  gint64 current_position;        /* 현재 에지 위치 */
  gdouble start_x;                /* 트리밍 시작 X 좌표 */
  gboolean snap_enabled;          /* 스냅 기능 활성화 여부 */
  gint precision_level;           /* 정밀도 레벨 (0: 일반, 1: 높음, 2: 최고) */
  GESClip *adjacent_clip;         /* 인접한 클립 (롤 트리밍에 사용) */
};

/**
 * Edge Trimming 모드 설정
 * 
 * @param timeline 타임라인
 * @param mode 설정할 에지 트리밍 모드
 */
void blouedit_timeline_set_edge_trim_mode(BlouEditTimeline *timeline, BlouEditEdgeTrimMode mode);

/**
 * Edge Trimming 모드 가져오기
 * 
 * @param timeline 타임라인
 * @return 현재 에지 트리밍 모드
 */
BlouEditEdgeTrimMode blouedit_timeline_get_edge_trim_mode(BlouEditTimeline *timeline);

/**
 * Edge Trimming 기능 시작 
 * 
 * @param timeline 타임라인
 * @param clip 트리밍할 클립
 * @param edge 트리밍할 에지 (시작/끝)
 * @param x 마우스 시작 X 좌표
 * @return 성공 여부
 */
gboolean blouedit_timeline_start_edge_trimming(BlouEditTimeline *timeline, 
                                            GESClip *clip,
                                            BlouEditClipEdge edge,
                                            gdouble x);

/**
 * Edge Trimming 현재 위치로 업데이트
 * 
 * @param timeline 타임라인
 * @param x 현재 마우스 X 좌표
 * @return 성공 여부
 */
gboolean blouedit_timeline_update_edge_trimming(BlouEditTimeline *timeline, gdouble x);

/**
 * Edge Trimming 완료
 * 
 * @param timeline 타임라인
 * @return 성공 여부
 */
gboolean blouedit_timeline_finish_edge_trimming(BlouEditTimeline *timeline);

/**
 * Edge Trimming 취소 (원래 위치로 복원)
 * 
 * @param timeline 타임라인
 * @return 성공 여부
 */
gboolean blouedit_timeline_cancel_edge_trimming(BlouEditTimeline *timeline);

/**
 * Edge Trimming 정밀도 설정
 * 
 * @param timeline 타임라인
 * @param precision_level 정밀도 레벨 (0: 일반, 1: 높음, 2: 최고)
 */
void blouedit_timeline_set_edge_trim_precision(BlouEditTimeline *timeline, gint precision_level);

/**
 * Edge Trimming 도구 UI 표시
 * 
 * @param timeline 타임라인
 * @param cr Cairo 컨텍스트
 * @param width 영역 너비
 * @param height 영역 높이
 */
void blouedit_timeline_draw_edge_trimming_ui(BlouEditTimeline *timeline, 
                                          cairo_t *cr, 
                                          gint width, 
                                          gint height);

/**
 * Edge Trimming 도구 대화상자 표시
 * 
 * @param timeline 타임라인
 */
void blouedit_timeline_show_edge_trimming_dialog(BlouEditTimeline *timeline);

G_END_DECLS 