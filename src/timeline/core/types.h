#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>

G_BEGIN_DECLS

/* 타임코드 형식 정의 */
typedef enum {
  BLOUEDIT_TIMECODE_FORMAT_FRAMES,      /* 프레임 (예: 1234) */
  BLOUEDIT_TIMECODE_FORMAT_SECONDS,     /* 초 (예: 123.45) */
  BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_FF, /* 시:분:초:프레임 (예: 00:12:34:56) */
  BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_MS  /* 시:분:초.밀리초 (예: 00:12:34.567) */
} BlouEditTimecodeFormat;

/* 마커 타입 정의 */
typedef enum {
  BLOUEDIT_MARKER_TYPE_GENERIC,     /* 일반 마커 */
  BLOUEDIT_MARKER_TYPE_CUE,         /* 큐 포인트 */
  BLOUEDIT_MARKER_TYPE_IN,          /* 시작 지점 */
  BLOUEDIT_MARKER_TYPE_OUT,         /* 종료 지점 */
  BLOUEDIT_MARKER_TYPE_CHAPTER,     /* 챕터 지점 */
  BLOUEDIT_MARKER_TYPE_ERROR,       /* 오류 지점 */
  BLOUEDIT_MARKER_TYPE_WARNING,     /* 경고 지점 */
  BLOUEDIT_MARKER_TYPE_COMMENT      /* 코멘트 지점 */
} BlouEditMarkerType;

/* 마커 구조체 정의 */
typedef struct _BlouEditTimelineMarker BlouEditTimelineMarker;

struct _BlouEditTimelineMarker
{
  gint64 position;                /* 마커 위치 (타임라인 단위) */
  BlouEditMarkerType type;        /* 마커 유형 */
  gchar *name;                    /* 마커 이름 */
  gchar *comment;                 /* 마커 설명/코멘트 */
  gchar *detailed_memo;           /* 마커에 연결된 상세 메모 */
  GdkRGBA color;                  /* 마커 색상 */
  guint id;                       /* 마커 고유 ID */
};

/* 트랙 구조체 추가 */
typedef struct _BlouEditTimelineTrack BlouEditTimelineTrack;

struct _BlouEditTimelineTrack
{
  GESTrack *ges_track;        /* 실제 GES 트랙 */
  gchar *name;                /* 트랙 이름 */
  gboolean folded;            /* 접힘 상태 */
  gint height;                /* 기본 트랙 높이 */
  gint folded_height;         /* 접힌 상태의 높이 */
  gboolean visible;           /* 표시 여부 */
  GdkRGBA color;              /* 트랙 색상 */
  gboolean height_resizing;   /* 트랙 높이 조절 중인지 여부 */
};

/* 키프레임 보간 유형 정의 */
typedef enum {
  BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR,     /* 선형 보간 */
  BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER,     /* 베지어 곡선 보간 */
  BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT,   /* 상수 보간 (계단식) */
  BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN,    /* 감속 (천천히 끝나는) */
  BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT,   /* 가속 (천천히 시작하는) */
  BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT /* 가속 후 감속 */
} BlouEditKeyframeInterpolation;

/* 키프레임 구조체 정의 */
typedef struct _BlouEditKeyframe BlouEditKeyframe;

struct _BlouEditKeyframe
{
  gint64 position;                      /* 키프레임 위치 (타임라인 단위) */
  BlouEditKeyframeInterpolation interpolation; /* 보간 유형 */
  gdouble value;                        /* 키프레임 값 */
  gdouble handle_left_x;                /* 왼쪽 베지어 핸들 X 좌표 (상대적) */
  gdouble handle_left_y;                /* 왼쪽 베지어 핸들 Y 좌표 (상대적) */
  gdouble handle_right_x;               /* 오른쪽 베지어 핸들 X 좌표 (상대적) */
  gdouble handle_right_y;               /* 오른쪽 베지어 핸들 Y 좌표 (상대적) */
  guint id;                             /* 키프레임 고유 ID */
};

/* 애니메이션 가능한 속성 타입 정의 */
typedef enum {
  BLOUEDIT_PROPERTY_TYPE_DOUBLE,   /* 실수 값 */
  BLOUEDIT_PROPERTY_TYPE_INT,      /* 정수 값 */
  BLOUEDIT_PROPERTY_TYPE_BOOLEAN,  /* 불리언 값 */
  BLOUEDIT_PROPERTY_TYPE_COLOR,    /* 색상 값 */
  BLOUEDIT_PROPERTY_TYPE_ENUM,     /* 열거형 값 */
  BLOUEDIT_PROPERTY_TYPE_POSITION, /* 위치 값 (x, y) */
  BLOUEDIT_PROPERTY_TYPE_SCALE,    /* 크기 값 (width, height) */
  BLOUEDIT_PROPERTY_TYPE_ROTATION  /* 회전 값 (각도) */
} BlouEditPropertyType;

/* 애니메이션 속성 구조체 정의 */
typedef struct _BlouEditAnimatableProperty BlouEditAnimatableProperty;

struct _BlouEditAnimatableProperty
{
  gchar *name;                      /* 속성 이름 */
  gchar *display_name;              /* 화면에 표시할 속성 이름 */
  BlouEditPropertyType type;        /* 속성 유형 */
  GObject *object;                  /* 속성을 가진 객체 */
  gchar *property_name;             /* GObject 속성 이름 */
  GSList *keyframes;                /* 키프레임 목록 */
  guint id;                         /* 속성 고유 ID */
  gdouble min_value;                /* 최소값 */
  gdouble max_value;                /* 최대값 */
  gdouble default_value;            /* 기본값 */
  gboolean visible;                 /* 타임라인에 표시 여부 */
  gboolean expanded;                /* 확장 표시 여부 */
};

/* Scrubbing modes */
typedef enum {
  BLOUEDIT_SCRUB_MODE_NORMAL,     /* Standard scrubbing */
  BLOUEDIT_SCRUB_MODE_PRECISE,    /* Frame-by-frame precision */
  BLOUEDIT_SCRUB_MODE_SHUTTLE     /* Variable speed scrubbing */
} BlouEditScrubMode;

/* Snap modes */
typedef enum {
  BLOUEDIT_SNAP_NONE       = 0,
  BLOUEDIT_SNAP_TO_GRID    = 1 << 0,
  BLOUEDIT_SNAP_TO_MARKERS = 1 << 1,
  BLOUEDIT_SNAP_TO_CLIPS   = 1 << 2,
  BLOUEDIT_SNAP_ALL        = (BLOUEDIT_SNAP_TO_GRID | BLOUEDIT_SNAP_TO_MARKERS | BLOUEDIT_SNAP_TO_CLIPS)
} BlouEditSnapMode;

/* Clip edit modes */
typedef enum {
  BLOUEDIT_EDIT_MODE_NORMAL,      /* Standard edit - overwrite existing content */
  BLOUEDIT_EDIT_MODE_RIPPLE,      /* Ripple edit - shift subsequent clips */
  BLOUEDIT_EDIT_MODE_ROLL,        /* Roll edit - adjust neighboring clips */
  BLOUEDIT_EDIT_MODE_SLIP,        /* Slip edit - change clip in/out points without changing duration */
  BLOUEDIT_EDIT_MODE_SLIDE        /* Slide edit - move clip between neighbors without changing their durations */
} BlouEditEditMode;

/* Clip edge selection for trimming */
typedef enum {
  BLOUEDIT_EDGE_NONE,
  BLOUEDIT_EDGE_START,
  BLOUEDIT_EDGE_END
} BlouEditClipEdge;

/* Timeline autoscroll modes */
typedef enum {
  BLOUEDIT_AUTOSCROLL_NONE,    /* No auto-scrolling */
  BLOUEDIT_AUTOSCROLL_PAGE,    /* Page scrolling when playhead reaches edge */
  BLOUEDIT_AUTOSCROLL_SMOOTH,  /* Smooth scrolling to follow playhead */
  BLOUEDIT_AUTOSCROLL_SCROLL   /* Keep playhead centered during playback */
} BlouEditAutoscrollMode;

/* Media filter types for timeline */
typedef enum {
  BLOUEDIT_FILTER_NONE       = 0,        /* Show all media types */
  BLOUEDIT_FILTER_VIDEO      = 1 << 0,   /* Show video clips only */
  BLOUEDIT_FILTER_AUDIO      = 1 << 1,   /* Show audio clips only */
  BLOUEDIT_FILTER_IMAGE      = 1 << 2,   /* Show image clips only */
  BLOUEDIT_FILTER_TEXT       = 1 << 3,   /* Show text clips only */
  BLOUEDIT_FILTER_EFFECT     = 1 << 4,   /* Show effect clips only */
  BLOUEDIT_FILTER_TRANSITION = 1 << 5,   /* Show transitions only */
  BLOUEDIT_FILTER_ALL        = (BLOUEDIT_FILTER_VIDEO | BLOUEDIT_FILTER_AUDIO | 
                              BLOUEDIT_FILTER_IMAGE | BLOUEDIT_FILTER_TEXT | 
                              BLOUEDIT_FILTER_EFFECT | BLOUEDIT_FILTER_TRANSITION)
} BlouEditMediaFilterType;

/* Timeline history operation types */
typedef enum {
  BLOUEDIT_HISTORY_NONE,          /* No action */
  BLOUEDIT_HISTORY_ADD_CLIP,      /* Add clip to timeline */
  BLOUEDIT_HISTORY_REMOVE_CLIP,   /* Remove clip from timeline */
  BLOUEDIT_HISTORY_MOVE_CLIP,     /* Move clip in timeline */
  BLOUEDIT_HISTORY_TRIM_CLIP,     /* Trim clip start/end */
  BLOUEDIT_HISTORY_SPLIT_CLIP,    /* Split clip at position */
  BLOUEDIT_HISTORY_JOIN_CLIPS,    /* Join adjacent clips */
  BLOUEDIT_HISTORY_RIPPLE_EDIT,   /* Ripple edit operation */
  BLOUEDIT_HISTORY_ROLL_EDIT,     /* Roll edit operation */
  BLOUEDIT_HISTORY_SLIP_CLIP,     /* Slip clip in/out points */
  BLOUEDIT_HISTORY_SLIDE_CLIP,    /* Slide clip between neighbors */
  BLOUEDIT_HISTORY_CHANGE_SPEED,  /* Change clip speed */
  BLOUEDIT_HISTORY_GROUP_CLIPS,   /* Group clips together */
  BLOUEDIT_HISTORY_UNGROUP_CLIPS, /* Ungroup clips */
  BLOUEDIT_HISTORY_LOCK_CLIPS,    /* Lock clips */
  BLOUEDIT_HISTORY_UNLOCK_CLIPS,  /* Unlock clips */
  BLOUEDIT_HISTORY_ADD_EFFECT,    /* Add effect to clip */
  BLOUEDIT_HISTORY_REMOVE_EFFECT, /* Remove effect from clip */
  BLOUEDIT_HISTORY_ADD_MARKER,    /* Add timeline marker */
  BLOUEDIT_HISTORY_REMOVE_MARKER, /* Remove timeline marker */
  BLOUEDIT_HISTORY_MOVE_MARKER,   /* Move timeline marker */
  BLOUEDIT_HISTORY_ADD_KEYFRAME,    /* Add keyframe to property */
  BLOUEDIT_HISTORY_REMOVE_KEYFRAME, /* Remove keyframe from property */
  BLOUEDIT_HISTORY_MOVE_KEYFRAME,   /* Move keyframe in time */
  BLOUEDIT_HISTORY_EDIT_KEYFRAME,   /* Edit keyframe value or interpolation */
  BLOUEDIT_HISTORY_ADD_TRACK,       /* Add track to timeline */
  BLOUEDIT_HISTORY_REMOVE_TRACK     /* Remove track from timeline */
} BlouEditHistoryActionType;

/* Timeline history operation metadata structure */
typedef struct _BlouEditHistoryAction BlouEditHistoryAction;

struct _BlouEditHistoryAction
{
  BlouEditHistoryActionType type;    /* Type of action */
  GESTimelineElement *element;       /* Element affected */
  gint64 time_stamp;                 /* When the action occurred */
  gchar *description;                /* Human-readable description */
  GValue before_value;               /* State before action (type depends on action) */
  GValue after_value;                /* State after action (type depends on action) */
};

/* Timeline group structure */
typedef struct _BlouEditTimelineGroup BlouEditTimelineGroup;

G_END_DECLS 