#pragma once

#include <gtk/gtk.h>
#include "core/types.h"
#include "core/timeline.h"

G_BEGIN_DECLS

/**
 * 멀티캠 관련 정의
 */

/* 멀티캠 모드 정의 */
typedef enum {
  BLOUEDIT_MULTICAM_MODE_DISABLED,     /* 멀티캠 기능 비활성화 */
  BLOUEDIT_MULTICAM_MODE_SOURCE_VIEW,  /* 소스 뷰 모드 (모든 카메라 앵글 보기) */
  BLOUEDIT_MULTICAM_MODE_EDIT          /* 편집 모드 (앵글 전환 편집) */
} BlouEditMulticamMode;

/* 멀티캠 소스 구조체 */
typedef struct _BlouEditMulticamSource BlouEditMulticamSource;
struct _BlouEditMulticamSource {
  guint id;                    /* 소스 ID */
  gchar *name;                 /* 소스 이름 */
  GESClip *source_clip;        /* 원본 소스 클립 */
  GdkRGBA color;               /* 소스 색상 (시각적 구분용) */
  gboolean active;             /* 현재 활성화 여부 */
  gboolean synced;             /* 동기화 여부 */
  gint64 sync_offset;          /* 동기화 오프셋 */
};

/* 멀티캠 전환 구조체 */
typedef struct _BlouEditMulticamSwitch BlouEditMulticamSwitch;
struct _BlouEditMulticamSwitch {
  guint id;                    /* 전환 ID */
  guint source_id;             /* 소스 ID */
  gint64 position;             /* 타임라인 상의 위치 */
  gint64 duration;             /* 전환 지속 시간 (0이면 즉시 전환) */
  gchar *transition_type;      /* 전환 유형 (즉시, 크로스페이드 등) */
};

/* 멀티캠 그룹 구조체 */
typedef struct _BlouEditMulticamGroup BlouEditMulticamGroup;
struct _BlouEditMulticamGroup {
  guint id;                    /* 그룹 ID */
  gchar *name;                 /* 그룹 이름 */
  GSList *sources;             /* 소스 목록 (BlouEditMulticamSource) */
  GSList *switches;            /* 전환 목록 (BlouEditMulticamSwitch) */
  guint next_source_id;        /* 다음 소스 ID */
  guint next_switch_id;        /* 다음 전환 ID */
  guint active_source_id;      /* 현재 활성화된 소스 ID */
  BlouEditTimelineTrack *output_track; /* 출력 트랙 */
};

/**
 * 멀티캠 모드 설정 함수
 */
void blouedit_timeline_set_multicam_mode(BlouEditTimeline *timeline, BlouEditMulticamMode mode);

/**
 * 멀티캠 모드 가져오기 함수
 */
BlouEditMulticamMode blouedit_timeline_get_multicam_mode(BlouEditTimeline *timeline);

/**
 * 멀티캠 그룹 생성 함수
 */
BlouEditMulticamGroup* blouedit_timeline_create_multicam_group(BlouEditTimeline *timeline, const gchar *name);

/**
 * 멀티캠 그룹 제거 함수
 */
void blouedit_timeline_remove_multicam_group(BlouEditTimeline *timeline, BlouEditMulticamGroup *group);

/**
 * 멀티캠 소스 추가 함수
 */
BlouEditMulticamSource* blouedit_multicam_group_add_source(BlouEditMulticamGroup *group, 
                                                        GESClip *clip, 
                                                        const gchar *name);

/**
 * 멀티캠 소스 제거 함수
 */
void blouedit_multicam_group_remove_source(BlouEditMulticamGroup *group, guint source_id);

/**
 * 멀티캠 전환 추가 함수
 */
BlouEditMulticamSwitch* blouedit_multicam_group_add_switch(BlouEditMulticamGroup *group, 
                                                        guint source_id, 
                                                        gint64 position,
                                                        gint64 duration,
                                                        const gchar *transition_type);

/**
 * 멀티캠 전환 제거 함수
 */
void blouedit_multicam_group_remove_switch(BlouEditMulticamGroup *group, guint switch_id);

/**
 * 멀티캠 소스 동기화 설정 함수
 */
void blouedit_multicam_source_set_sync_offset(BlouEditMulticamSource *source, gint64 offset);

/**
 * 멀티캠 편집 UI 표시 함수
 */
void blouedit_timeline_show_multicam_editor(BlouEditTimeline *timeline);

/**
 * 멀티캠 소스 뷰 드로잉 함수
 */
void blouedit_timeline_draw_multicam_source_view(BlouEditTimeline *timeline, cairo_t *cr, int width, int height);

/**
 * 멀티캠 미리 컴파일 함수 (렌더링 준비)
 */
void blouedit_multicam_group_compile(BlouEditTimeline *timeline, BlouEditMulticamGroup *group);

G_END_DECLS 