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
  BLOUEDIT_HISTORY_EDIT_KEYFRAME    /* Edit keyframe value or interpolation */
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

#define BLOUEDIT_TYPE_TIMELINE (blouedit_timeline_get_type())

G_DECLARE_FINAL_TYPE (BlouEditTimeline, blouedit_timeline, BLOUEDIT, TIMELINE, GtkWidget)

BlouEditTimeline *blouedit_timeline_new (void);

void blouedit_timeline_add_file (BlouEditTimeline *timeline, GFile *file);
void blouedit_timeline_add_clip (BlouEditTimeline *timeline, const char *uri, GESTrack *track, gint64 start_time, gint64 duration);
void blouedit_timeline_split_clip (BlouEditTimeline *timeline, gint64 position);
void blouedit_timeline_delete_clip (BlouEditTimeline *timeline, GESClip *clip);
void blouedit_timeline_add_transition (BlouEditTimeline *timeline, gint64 position, const char *transition_type);

/* Timeline control */
void blouedit_timeline_set_position (BlouEditTimeline *timeline, gint64 position);
gint64 blouedit_timeline_get_position (BlouEditTimeline *timeline);
gint64 blouedit_timeline_get_duration (BlouEditTimeline *timeline);

/* Timecode functions */
gchar *blouedit_timeline_position_to_timecode (BlouEditTimeline *timeline, gint64 position, BlouEditTimecodeFormat format);
gint64 blouedit_timeline_timecode_to_position (BlouEditTimeline *timeline, const gchar *timecode, BlouEditTimecodeFormat format);
void blouedit_timeline_set_timecode_format (BlouEditTimeline *timeline, BlouEditTimecodeFormat format);
BlouEditTimecodeFormat blouedit_timeline_get_timecode_format (BlouEditTimeline *timeline);
void blouedit_timeline_set_framerate (BlouEditTimeline *timeline, gdouble framerate);
gdouble blouedit_timeline_get_framerate (BlouEditTimeline *timeline);
void blouedit_timeline_goto_timecode (BlouEditTimeline *timeline, const gchar *timecode);
void blouedit_timeline_show_timecode (BlouEditTimeline *timeline, gboolean show);
gboolean blouedit_timeline_get_show_timecode (BlouEditTimeline *timeline);

/* Timeline marker functions */
BlouEditTimelineMarker *blouedit_timeline_add_marker (BlouEditTimeline *timeline, gint64 position, BlouEditMarkerType type, const gchar *name);
void blouedit_timeline_remove_marker (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker);
void blouedit_timeline_remove_marker_at_position (BlouEditTimeline *timeline, gint64 position, gint64 tolerance);
void blouedit_timeline_remove_all_markers (BlouEditTimeline *timeline);
void blouedit_timeline_update_marker (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker, gint64 position, BlouEditMarkerType type, const gchar *name);
BlouEditTimelineMarker *blouedit_timeline_get_marker_at_position (BlouEditTimeline *timeline, gint64 position, gint64 tolerance);
GSList *blouedit_timeline_get_markers (BlouEditTimeline *timeline);
GSList *blouedit_timeline_get_markers_in_range (BlouEditTimeline *timeline, gint64 start, gint64 end);
void blouedit_timeline_goto_next_marker (BlouEditTimeline *timeline);
void blouedit_timeline_goto_prev_marker (BlouEditTimeline *timeline);
void blouedit_timeline_set_marker_color (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker, const GdkRGBA *color);
void blouedit_timeline_set_marker_comment (BlouEditTimeline *timeline, BlouEditTimelineMarker *marker, const gchar *comment);
gboolean blouedit_timeline_export_markers (BlouEditTimeline *timeline, GFile *file);
gboolean blouedit_timeline_import_markers (BlouEditTimeline *timeline, GFile *file);

/* Keyframe functions */
BlouEditAnimatableProperty *blouedit_timeline_register_property (BlouEditTimeline *timeline, GObject *object, 
                                                              const gchar *property_name, const gchar *display_name,
                                                              BlouEditPropertyType type,
                                                              gdouble min_value, gdouble max_value, gdouble default_value);
void blouedit_timeline_unregister_property (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property);
GSList *blouedit_timeline_get_properties (BlouEditTimeline *timeline);
BlouEditAnimatableProperty *blouedit_timeline_get_property_by_id (BlouEditTimeline *timeline, guint id);
BlouEditAnimatableProperty *blouedit_timeline_get_property_by_name (BlouEditTimeline *timeline, GObject *object, const gchar *property_name);

BlouEditKeyframe *blouedit_timeline_add_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, 
                                               gint64 position, gdouble value,
                                               BlouEditKeyframeInterpolation interpolation);
void blouedit_timeline_remove_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, BlouEditKeyframe *keyframe);
void blouedit_timeline_remove_keyframe_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position, gint64 tolerance);
void blouedit_timeline_remove_all_keyframes (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property);
void blouedit_timeline_update_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, 
                                     BlouEditKeyframe *keyframe, gint64 position, gdouble value,
                                     BlouEditKeyframeInterpolation interpolation);
void blouedit_timeline_update_keyframe_handles (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property,
                                             BlouEditKeyframe *keyframe,
                                             gdouble handle_left_x, gdouble handle_left_y,
                                             gdouble handle_right_x, gdouble handle_right_y);
BlouEditKeyframe *blouedit_timeline_get_keyframe_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position, gint64 tolerance);
GSList *blouedit_timeline_get_keyframes (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property);
GSList *blouedit_timeline_get_keyframes_in_range (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 start, gint64 end);
gdouble blouedit_timeline_evaluate_property_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position);
void blouedit_timeline_show_keyframe_editor (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, BlouEditKeyframe *keyframe);
gboolean blouedit_timeline_apply_keyframes (BlouEditTimeline *timeline);

/* Track handling functions */
BlouEditTimelineTrack *blouedit_timeline_add_track (BlouEditTimeline *timeline, GESTrackType track_type, const gchar *name);
void blouedit_timeline_remove_track (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
GSList *blouedit_timeline_get_tracks (BlouEditTimeline *timeline);
BlouEditTimelineTrack *blouedit_timeline_get_track_at_y (BlouEditTimeline *timeline, gint y);

/* Track folding functions */
void blouedit_timeline_toggle_track_folded (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
void blouedit_timeline_set_track_folded (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gboolean folded);
gboolean blouedit_timeline_get_track_folded (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
void blouedit_timeline_fold_all_tracks (BlouEditTimeline *timeline);
void blouedit_timeline_unfold_all_tracks (BlouEditTimeline *timeline);

/* Track visibility functions */
void blouedit_timeline_set_track_visible (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gboolean visible);
gboolean blouedit_timeline_get_track_visible (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);

/* Track color functions */
void blouedit_timeline_set_track_color (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, const GdkRGBA *color);
void blouedit_timeline_get_track_color (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, GdkRGBA *color);

/* Track height functions */
void blouedit_timeline_set_track_height (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint height);
gint blouedit_timeline_get_track_height (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
void blouedit_timeline_start_track_resize (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint y);
void blouedit_timeline_resize_track_to (BlouEditTimeline *timeline, gint y);
void blouedit_timeline_end_track_resize (BlouEditTimeline *timeline);
void blouedit_timeline_reset_track_height (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
void blouedit_timeline_set_default_track_height (BlouEditTimeline *timeline, gint height);
gint blouedit_timeline_get_default_track_height (BlouEditTimeline *timeline);

/* Track reordering functions */
void blouedit_timeline_start_track_reorder (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint y);
void blouedit_timeline_reorder_track_to (BlouEditTimeline *timeline, gint y);
void blouedit_timeline_end_track_reorder (BlouEditTimeline *timeline);
void blouedit_timeline_move_track_up (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
void blouedit_timeline_move_track_down (BlouEditTimeline *timeline, BlouEditTimelineTrack *track);

/* Timeline scrubbing functions */
void blouedit_timeline_set_scrub_mode (BlouEditTimeline *timeline, BlouEditScrubMode mode);
BlouEditScrubMode blouedit_timeline_get_scrub_mode (BlouEditTimeline *timeline);
void blouedit_timeline_start_scrubbing (BlouEditTimeline *timeline, double x);
void blouedit_timeline_scrub_to (BlouEditTimeline *timeline, double x);
void blouedit_timeline_end_scrubbing (BlouEditTimeline *timeline);
void blouedit_timeline_set_scrub_sensitivity (BlouEditTimeline *timeline, gdouble sensitivity);
gdouble blouedit_timeline_get_scrub_sensitivity (BlouEditTimeline *timeline);
void blouedit_timeline_set_playhead_position_from_x (BlouEditTimeline *timeline, double x);
double blouedit_timeline_get_x_from_position (BlouEditTimeline *timeline, gint64 position);

/* Timeline zoom functions */
void blouedit_timeline_zoom_in (BlouEditTimeline *timeline);
void blouedit_timeline_zoom_out (BlouEditTimeline *timeline);
void blouedit_timeline_set_zoom_level (BlouEditTimeline *timeline, gdouble zoom_level);
gdouble blouedit_timeline_get_zoom_level (BlouEditTimeline *timeline);
void blouedit_timeline_zoom_fit (BlouEditTimeline *timeline);

/* Snap functions */
void blouedit_timeline_set_snap_mode (BlouEditTimeline *timeline, BlouEditSnapMode mode);
void blouedit_timeline_set_snap_distance (BlouEditTimeline *timeline, guint distance);
guint blouedit_timeline_get_snap_distance (BlouEditTimeline *timeline);
gint64 blouedit_timeline_snap_position (BlouEditTimeline *timeline, gint64 position);
gboolean blouedit_timeline_toggle_snap (BlouEditTimeline *timeline);

/* Edit mode functions */
void blouedit_timeline_set_edit_mode (BlouEditTimeline *timeline, BlouEditEditMode mode);
BlouEditEditMode blouedit_timeline_get_edit_mode (BlouEditTimeline *timeline);
gboolean blouedit_timeline_toggle_ripple_mode (BlouEditTimeline *timeline);

/* Clip handling functions */
GESClip *blouedit_timeline_get_clip_at_position (BlouEditTimeline *timeline, gint64 position, BlouEditTimelineTrack *track);
void blouedit_timeline_select_clip (BlouEditTimeline *timeline, GESClip *clip, gboolean clear_selection);
void blouedit_timeline_select_clips_in_range (BlouEditTimeline *timeline, gint64 start, gint64 end, BlouEditTimelineTrack *track);
void blouedit_timeline_unselect_clip (BlouEditTimeline *timeline, GESClip *clip);
void blouedit_timeline_clear_selection (BlouEditTimeline *timeline);
void blouedit_timeline_select_all_clips (BlouEditTimeline *timeline);
gboolean blouedit_timeline_is_clip_selected (BlouEditTimeline *timeline, GESClip *clip);

/* Clip editing functions */
void blouedit_timeline_trim_clip (BlouEditTimeline *timeline, GESClip *clip, BlouEditClipEdge edge, gint64 position);
void blouedit_timeline_trim_clip_start (BlouEditTimeline *timeline, GESClip *clip, gint64 position);
void blouedit_timeline_trim_clip_end (BlouEditTimeline *timeline, GESClip *clip, gint64 position);
void blouedit_timeline_ripple_trim (BlouEditTimeline *timeline, GESClip *clip, BlouEditClipEdge edge, gint64 position);
void blouedit_timeline_roll_edit (BlouEditTimeline *timeline, GESClip *clip, BlouEditClipEdge edge, gint64 position);
void blouedit_timeline_slip_clip (BlouEditTimeline *timeline, GESClip *clip, gint64 offset);
void blouedit_timeline_slide_clip (BlouEditTimeline *timeline, GESClip *clip, gint64 position);
BlouEditClipEdge blouedit_timeline_get_clip_edge_at_position (BlouEditTimeline *timeline, GESClip *clip, double x, double tolerance);

/* Clip grouping functions */
void blouedit_timeline_group_selected_clips (BlouEditTimeline *timeline);
void blouedit_timeline_ungroup_clips (BlouEditTimeline *timeline, GESGroup *group);
void blouedit_timeline_lock_selected_clips (BlouEditTimeline *timeline, gboolean lock);
gboolean blouedit_timeline_is_clip_locked (BlouEditTimeline *timeline, GESClip *clip);

/* Event handling functions */
gboolean blouedit_timeline_handle_key_press (BlouEditTimeline *timeline, GdkEventKey *event);
gboolean blouedit_timeline_handle_scroll (BlouEditTimeline *timeline, GdkEventScroll *event);
gboolean blouedit_timeline_handle_motion (BlouEditTimeline *timeline, GdkEventMotion *event);
gboolean blouedit_timeline_handle_button_press (BlouEditTimeline *timeline, GdkEventButton *event);
gboolean blouedit_timeline_handle_button_release (BlouEditTimeline *timeline, GdkEventButton *event);

/* Timeline filtering functions */
void blouedit_timeline_set_media_filter (BlouEditTimeline *timeline, BlouEditMediaFilterType filter_type);
BlouEditMediaFilterType blouedit_timeline_get_media_filter (BlouEditTimeline *timeline);
gboolean blouedit_timeline_is_clip_visible (BlouEditTimeline *timeline, GESClip *clip);
void blouedit_timeline_toggle_filter (BlouEditTimeline *timeline, BlouEditMediaFilterType filter_type);

/* Timeline history functions */
void blouedit_timeline_record_action (BlouEditTimeline *timeline, BlouEditHistoryActionType type, 
                                     GESTimelineElement *element, const gchar *description,
                                     const GValue *before_value, const GValue *after_value);
gboolean blouedit_timeline_undo (BlouEditTimeline *timeline);
gboolean blouedit_timeline_redo (BlouEditTimeline *timeline);
void blouedit_timeline_clear_history (BlouEditTimeline *timeline);
void blouedit_timeline_begin_group_action (BlouEditTimeline *timeline, const gchar *description);
void blouedit_timeline_end_group_action (BlouEditTimeline *timeline);
void blouedit_timeline_set_max_history_size (BlouEditTimeline *timeline, gint max_size);
gint blouedit_timeline_get_max_history_size (BlouEditTimeline *timeline);
GSList* blouedit_timeline_get_history_actions (BlouEditTimeline *timeline, gint limit);
void blouedit_timeline_show_history_dialog (BlouEditTimeline *timeline);
gboolean blouedit_timeline_can_undo (BlouEditTimeline *timeline);
gboolean blouedit_timeline_can_redo (BlouEditTimeline *timeline);

/* Timeline autoscroll functions */
void blouedit_timeline_set_autoscroll_mode (BlouEditTimeline *timeline, BlouEditAutoscrollMode mode);
BlouEditAutoscrollMode blouedit_timeline_get_autoscroll_mode (BlouEditTimeline *timeline);
void blouedit_timeline_handle_autoscroll (BlouEditTimeline *timeline);

/* Multi-timeline functions */
typedef struct _BlouEditTimelineGroup BlouEditTimelineGroup;

BlouEditTimelineGroup *blouedit_timeline_group_new (void);
void blouedit_timeline_group_free (BlouEditTimelineGroup *group);

void blouedit_timeline_group_set_active (BlouEditTimelineGroup *group, BlouEditTimeline *timeline);
void blouedit_timeline_group_add (BlouEditTimelineGroup *group, BlouEditTimeline *timeline, const gchar *name);
void blouedit_timeline_group_remove (BlouEditTimelineGroup *group, BlouEditTimeline *timeline);
GSList *blouedit_timeline_group_get_timelines (BlouEditTimelineGroup *group);
BlouEditTimeline *blouedit_timeline_group_get_active (BlouEditTimelineGroup *group);
void blouedit_timeline_group_set_name (BlouEditTimelineGroup *group, BlouEditTimeline *timeline, const gchar *name);
gboolean blouedit_timeline_group_copy_clip (BlouEditTimelineGroup *group, BlouEditTimeline *src, BlouEditTimeline *dest, GESClip *clip);
void blouedit_timeline_group_sync_position (BlouEditTimelineGroup *group, BlouEditTimeline *src);

/* Serialize/Deserialize functions */
gboolean blouedit_timeline_save_to_file (BlouEditTimeline *timeline, GFile *file);
gboolean blouedit_timeline_load_from_file (BlouEditTimeline *timeline, GFile *file);

G_END_DECLS 