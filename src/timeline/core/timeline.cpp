#include "timeline.h"
#include "types.h"
#include <glib/gi18n.h>

/* BlouEditTimeline 구조체 정의 */
struct _BlouEditTimeline
{
  GtkWidget parent_instance;
  
  GESTimeline *ges_timeline;
  GstElement *pipeline;
  
  gdouble zoom_level;           /* Current zoom level (1.0 is default) */
  gdouble min_zoom_level;       /* Minimum zoom level (zoomed out) */
  gdouble max_zoom_level;       /* Maximum zoom level (zoomed in) */
  gdouble zoom_step;            /* How much to change per zoom action */
  
  /* Snap properties */
  BlouEditSnapMode snap_mode;   /* Current snap mode */
  guint snap_distance;          /* Snap distance in pixels */
  guint grid_interval;          /* Grid interval in timeline units (e.g. frames or ms) */
  
  /* Scrubbing properties */
  gboolean is_scrubbing;        /* Whether we are currently scrubbing */
  BlouEditScrubMode scrub_mode; /* Current scrubbing mode */
  gdouble scrub_start_x;        /* X position where scrubbing started */
  gint64 scrub_start_position;  /* Timeline position when scrubbing started */
  gdouble scrub_sensitivity;    /* Scrubbing sensitivity */
  
  /* Timeline layout properties */
  gint ruler_height;            /* Height of the ruler at the top */
  gint timeline_start_x;        /* Horizontal offset where timeline starts (for labels) */
  gint playhead_x;              /* X position of the playhead */
  
  /* Track properties */
  GSList *tracks;               /* List of BlouEditTimelineTrack */
  gint default_track_height;    /* Default height for tracks */
  gint folded_track_height;     /* Height for folded tracks */
  gint track_header_width;      /* Width of track header area */
  gint track_spacing;           /* Spacing between tracks */
  BlouEditTimelineTrack *selected_track; /* Currently selected track */
  
  /* Track resizing properties */
  gboolean is_resizing_track;        /* Whether a track is being resized */
  BlouEditTimelineTrack *resizing_track; /* Track currently being resized */
  gint resize_start_y;               /* Y position where resize started */
  gint resize_start_height;          /* Original height of track before resize */
  gint min_track_height;             /* Minimum track height allowed */
  gint max_track_height;             /* Maximum track height allowed */
  
  /* Track reordering properties */
  gboolean is_reordering_track;      /* Whether a track is being reordered */
  BlouEditTimelineTrack *reordering_track; /* Track currently being reordered */
  gint reorder_start_y;              /* Y position where reorder started */
  gint reorder_original_index;       /* Original index of the track being reordered */
  gint reorder_current_index;        /* Current index during reordering */
  
  /* Marker properties */
  GSList *markers;                   /* List of BlouEditTimelineMarker */
  guint next_marker_id;              /* Next unique marker ID to assign */
  BlouEditTimelineMarker *selected_marker; /* Currently selected marker */
  gboolean show_markers;             /* Whether to show markers */
  gint marker_height;                /* Height of marker display in ruler */
  
  /* Clip editing properties */
  BlouEditEditMode edit_mode;        /* Current edit mode (normal, ripple, etc.) */
  GSList *selected_clips;            /* List of selected GESClip objects */
  gboolean is_trimming;              /* Whether we are currently trimming a clip */
  GESClip *trimming_clip;            /* Clip currently being trimmed */
  BlouEditClipEdge trimming_edge;    /* Which edge is being trimmed */
  gboolean is_moving_clip;           /* Whether we are currently moving a clip */
  GESClip *moving_clip;              /* Clip currently being moved */
  gint64 moving_start_position;      /* Original position of clip being moved */
  gdouble moving_start_x;            /* X position where move started */
  
  /* Keyframe properties */
  GSList *animatable_properties;     /* List of BlouEditAnimatableProperty */
  guint next_property_id;            /* Next unique property ID to assign */
  guint next_keyframe_id;            /* Next unique keyframe ID to assign */
  BlouEditAnimatableProperty *selected_property; /* Currently selected property */
  BlouEditKeyframe *selected_keyframe; /* Currently selected keyframe */
  gboolean show_keyframes;           /* Whether to show keyframes */
  gboolean is_moving_keyframe;       /* Whether we are currently moving a keyframe */
  BlouEditKeyframe *moving_keyframe; /* Keyframe currently being moved */
  gint64 moving_keyframe_start_position; /* Original position of keyframe being moved */
  gdouble moving_keyframe_start_value; /* Original value of keyframe being moved */
  gdouble moving_keyframe_start_x;   /* X position where keyframe move started */
  gdouble moving_keyframe_start_y;   /* Y position where keyframe move started */
  gboolean is_editing_keyframe_handle; /* Whether we are editing a keyframe's bezier handle */
  BlouEditKeyframe *handle_keyframe; /* Keyframe whose handle is being edited */
  gboolean is_editing_left_handle;   /* Whether editing left or right handle */
  gdouble handle_start_x;            /* Original X position of handle being edited */
  gdouble handle_start_y;            /* Original Y position of handle being edited */
  GtkWidget *keyframe_editor;        /* Keyframe editor widget */
  gint keyframe_area_height;         /* Height of keyframe display area */
  gboolean show_keyframe_values;     /* Whether to show keyframe values */
  
  /* Filtering properties */
  BlouEditMediaFilterType media_filter; /* Current media filter */
  
  /* History properties */
  GSList *history;                   /* List of history actions */
  gint history_position;             /* Current position in history */
  gint max_history_size;             /* Maximum number of history items to keep */
  gboolean group_actions;            /* Whether to group actions together */
  GSList *current_group;             /* Current group of actions */
  gchar *current_group_description;  /* Description of current group */
  
  /* Timecode properties */
  BlouEditTimecodeFormat timecode_format; /* Current timecode format */
  gdouble framerate;                 /* Current framerate */
  gboolean show_timecode;            /* Whether to show timecode display */
  GtkWidget *timecode_entry;         /* Timecode entry widget */
  
  /* Autoscroll properties */
  BlouEditAutoscrollMode autoscroll_mode; /* Current autoscroll mode */
  
  /* Player connection */
  GstElement *player;                /* Connected player element */
  gulong player_position_handler;    /* Handler ID for player position updates */
  
  /* Multi-timeline properties */
  BlouEditTimelineGroup *timeline_group; /* Group this timeline belongs to */
  gchar *timeline_name;              /* Name of this timeline in the group */
  
  /* Clip position properties */
  BlouEditClipPositionMode clip_position_mode; /* Current clip position mode */
  gboolean use_frames_for_position;  /* Whether to use frames for position */
  gint clip_position_font_size;      /* Font size for clip position */
  GdkRGBA clip_position_color;        /* Color for clip position */
  
  /* Playback range properties */
  gboolean use_playback_range;         /* Whether to use playback range */
  gint64 in_point;                    /* In point position */
  gint64 out_point;                   /* Out point position */
  gboolean show_range_markers;          /* Whether to show range markers */
};

G_DEFINE_TYPE (BlouEditTimeline, blouedit_timeline, GTK_TYPE_WIDGET)

/* 메모리 해제 도우미 함수들 */
static void
track_free (BlouEditTimelineTrack *track)
{
  if (track) {
    g_free (track->name);
    g_free (track);
  }
}

static void
marker_free (BlouEditTimelineMarker *marker)
{
  if (marker) {
    g_free (marker->name);
    g_free (marker->comment);
    g_free (marker);
  }
}

static void
keyframe_free (BlouEditKeyframe *keyframe)
{
  if (keyframe) {
    g_free (keyframe);
  }
}

static void
animatable_property_free (BlouEditAnimatableProperty *property)
{
  if (property) {
    g_free (property->name);
    g_free (property->display_name);
    g_free (property->property_name);
    
    g_slist_free_full (property->keyframes, (GDestroyNotify) keyframe_free);
    g_free (property);
  }
}

static void
history_action_free (BlouEditHistoryAction *action)
{
  if (action) {
    g_free (action->description);
    g_value_unset (&action->before_value);
    g_value_unset (&action->after_value);
    g_free (action);
  }
}

/* GObject 기본 메서드 구현 */
static void
blouedit_timeline_dispose (GObject *object)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (object);
  
  /* 각종 리소스 해제 */
  g_slist_free_full (timeline->tracks, (GDestroyNotify) track_free);
  timeline->tracks = NULL;
  
  g_slist_free_full (timeline->markers, (GDestroyNotify) marker_free);
  timeline->markers = NULL;
  
  g_slist_free_full (timeline->animatable_properties, (GDestroyNotify) animatable_property_free);
  timeline->animatable_properties = NULL;
  
  g_slist_free_full (timeline->history, (GDestroyNotify) history_action_free);
  timeline->history = NULL;
  
  g_slist_free_full (timeline->current_group, (GDestroyNotify) history_action_free);
  timeline->current_group = NULL;
  
  g_free (timeline->current_group_description);
  timeline->current_group_description = NULL;
  
  g_free (timeline->timeline_name);
  timeline->timeline_name = NULL;
  
  /* GES 타임라인 해제 */
  if (timeline->ges_timeline) {
    g_object_unref (timeline->ges_timeline);
    timeline->ges_timeline = NULL;
  }
  
  /* 부모 클래스의 dispose 호출 */
  G_OBJECT_CLASS (blouedit_timeline_parent_class)->dispose (object);
}

static void
blouedit_timeline_class_init (BlouEditTimelineClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  
  /* GObject 메서드 오버라이드 */
  object_class->dispose = blouedit_timeline_dispose;
}

static void
blouedit_timeline_init (BlouEditTimeline *timeline)
{
  /* 기본값 설정 */
  timeline->ges_timeline = ges_timeline_new ();
  
  /* 줌 설정 */
  timeline->zoom_level = 1.0;
  timeline->min_zoom_level = 0.01;
  timeline->max_zoom_level = 100.0;
  timeline->zoom_step = 0.25;
  
  /* 스냅 설정 */
  timeline->snap_mode = BLOUEDIT_SNAP_ALL;
  timeline->snap_distance = 10;
  timeline->grid_interval = 100; /* ms */
  
  /* 스크러빙 설정 */
  timeline->is_scrubbing = FALSE;
  timeline->scrub_mode = BLOUEDIT_SCRUB_MODE_NORMAL;
  timeline->scrub_sensitivity = 1.0;
  
  /* 타임라인 레이아웃 설정 */
  timeline->ruler_height = 30;
  timeline->timeline_start_x = 100;
  timeline->playhead_x = 0;
  
  /* 트랙 설정 */
  timeline->tracks = NULL;
  timeline->default_track_height = 80;
  timeline->folded_track_height = 20;
  timeline->track_header_width = 100;
  timeline->track_spacing = 2;
  timeline->selected_track = NULL;
  
  /* 트랙 크기 조절 설정 */
  timeline->is_resizing_track = FALSE;
  timeline->resizing_track = NULL;
  timeline->min_track_height = 20;
  timeline->max_track_height = 200;
  
  /* 트랙 순서 변경 설정 */
  timeline->is_reordering_track = FALSE;
  timeline->reordering_track = NULL;
  
  /* 마커 설정 */
  timeline->markers = NULL;
  timeline->next_marker_id = 1;
  timeline->selected_marker = NULL;
  timeline->show_markers = TRUE;
  timeline->marker_height = 15;
  
  /* 클립 편집 설정 */
  timeline->edit_mode = BLOUEDIT_EDIT_MODE_NORMAL;
  timeline->selected_clips = NULL;
  timeline->is_trimming = FALSE;
  timeline->trimming_clip = NULL;
  timeline->trimming_edge = BLOUEDIT_EDGE_NONE;
  timeline->is_moving_clip = FALSE;
  timeline->moving_clip = NULL;
  
  /* 키프레임 설정 */
  timeline->animatable_properties = NULL;
  timeline->next_property_id = 1;
  timeline->next_keyframe_id = 1;
  timeline->selected_property = NULL;
  timeline->selected_keyframe = NULL;
  timeline->show_keyframes = TRUE;
  timeline->is_moving_keyframe = FALSE;
  timeline->moving_keyframe = NULL;
  timeline->is_editing_keyframe_handle = FALSE;
  timeline->handle_keyframe = NULL;
  timeline->keyframe_area_height = 50;
  timeline->show_keyframe_values = TRUE;
  
  /* 필터링 설정 */
  timeline->media_filter = BLOUEDIT_FILTER_ALL;
  
  /* 히스토리 설정 */
  timeline->history = NULL;
  timeline->history_position = -1;
  timeline->max_history_size = 100;
  timeline->group_actions = FALSE;
  timeline->current_group = NULL;
  timeline->current_group_description = NULL;
  
  /* 타임코드 설정 */
  timeline->timecode_format = BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_FF;
  timeline->framerate = 30.0;
  timeline->show_timecode = TRUE;
  timeline->timecode_entry = NULL;
  
  /* 오토스크롤 설정 */
  timeline->autoscroll_mode = BLOUEDIT_AUTOSCROLL_PAGE;
  
  /* 플레이어 연결 설정 */
  timeline->player = NULL;
  timeline->player_position_handler = 0;
  
  /* 멀티 타임라인 설정 */
  timeline->timeline_group = NULL;
  timeline->timeline_name = NULL;
  
  /* Clip position defaults */
  blouedit_timeline_init_clip_position_defaults (timeline);
  
  /* Playback range defaults */
  blouedit_timeline_init_playback_range_defaults (timeline);
  
  /* GtkWidget 설정 */
  gtk_widget_set_can_focus (GTK_WIDGET (timeline), TRUE);
  
  /* 이벤트 마스크 설정 */
  gtk_widget_add_events (GTK_WIDGET (timeline),
                         GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK |
                         GDK_POINTER_MOTION_MASK | GDK_POINTER_MOTION_HINT_MASK |
                         GDK_KEY_PRESS_MASK | GDK_KEY_RELEASE_MASK |
                         GDK_FOCUS_CHANGE_MASK | GDK_SCROLL_MASK);
}

/* 타임라인 기본 함수 구현 */
BlouEditTimeline *
blouedit_timeline_new (void)
{
  BlouEditTimeline *timeline = g_object_new (BLOUEDIT_TYPE_TIMELINE, NULL);
  
  // 타임라인이 생성된 후 기본 트랙 설정
  if (timeline) {
    // 기본 트랙 생성 (비디오 트랙 1개, 오디오 트랙 1개)
    // blouedit_timeline_create_default_tracks (timeline);
    // 다른 모듈에서 구현될 함수이므로 여기서는 주석 처리
  }
  
  return timeline;
}

/* 줌 관련 함수 구현 */
void
blouedit_timeline_zoom_in (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  gdouble new_zoom = timeline->zoom_level * (1.0 + timeline->zoom_step);
  blouedit_timeline_set_zoom_level (timeline, new_zoom);
}

void
blouedit_timeline_zoom_out (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  gdouble new_zoom = timeline->zoom_level / (1.0 + timeline->zoom_step);
  blouedit_timeline_set_zoom_level (timeline, new_zoom);
}

void
blouedit_timeline_set_zoom_level (BlouEditTimeline *timeline, gdouble zoom_level)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 줌 레벨 제한 */
  zoom_level = CLAMP (zoom_level, timeline->min_zoom_level, timeline->max_zoom_level);
  
  /* 변경이 없으면 아무것도 하지 않음 */
  if (ABS (timeline->zoom_level - zoom_level) < 0.001)
    return;
  
  timeline->zoom_level = zoom_level;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

gdouble
blouedit_timeline_get_zoom_level (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 1.0);
  
  return timeline->zoom_level;
}

void
blouedit_timeline_zoom_fit (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 타임라인 길이 확인 */
  gint64 duration = blouedit_timeline_get_duration (timeline);
  if (duration <= 0)
    return;
  
  /* 위젯 크기 확인 */
  GtkAllocation allocation;
  gtk_widget_get_allocation (GTK_WIDGET (timeline), &allocation);
  
  /* 타임라인 영역의 너비 확인 */
  gint timeline_width = allocation.width - timeline->timeline_start_x;
  if (timeline_width <= 0)
    return;
  
  /* 줌 레벨 계산 */
  gdouble zoom_level = (gdouble) timeline_width / (gdouble) duration;
  
  /* 줌 레벨 설정 */
  blouedit_timeline_set_zoom_level (timeline, zoom_level);
}

/* 위치 및 길이 관련 함수 구현 */
void 
blouedit_timeline_set_position (BlouEditTimeline *timeline, gint64 position)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 위치 설정 및 UI 업데이트 */
}

gint64 
blouedit_timeline_get_position (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  /* 현재 위치 반환 */
  return 0;
}

gint64 
blouedit_timeline_get_duration (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  g_return_val_if_fail (timeline->ges_timeline != NULL, 0);
  
  /* GES 타임라인의 길이 반환 */
  return ges_timeline_get_duration (timeline->ges_timeline);
}

/* 스냅 관련 함수 구현 */
void
blouedit_timeline_set_snap_mode (BlouEditTimeline *timeline, BlouEditSnapMode mode)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  timeline->snap_mode = mode;
}

BlouEditSnapMode
blouedit_timeline_get_snap_mode (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), BLOUEDIT_SNAP_NONE);
  
  return timeline->snap_mode;
}

void
blouedit_timeline_set_snap_distance (BlouEditTimeline *timeline, guint distance)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  timeline->snap_distance = distance;
}

guint
blouedit_timeline_get_snap_distance (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 10);
  
  return timeline->snap_distance;
}

gboolean
blouedit_timeline_toggle_snap (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  if (timeline->snap_mode != BLOUEDIT_SNAP_NONE) {
    /* 스냅 비활성화 */
    blouedit_timeline_set_snap_mode (timeline, BLOUEDIT_SNAP_NONE);
    return FALSE;
  } else {
    /* 스냅 활성화 (모든 스냅) */
    blouedit_timeline_set_snap_mode (timeline, BLOUEDIT_SNAP_ALL);
    return TRUE;
  }
}

gint64
blouedit_timeline_snap_position (BlouEditTimeline *timeline, gint64 position)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), position);
  
  /* 스냅이 비활성화되어 있으면 원래 위치 반환 */
  if (timeline->snap_mode == BLOUEDIT_SNAP_NONE)
    return position;
  
  /* 스냅 알고리즘 구현 */
  /* 이 함수는 단순히 메인 프레임워크를 보여주기 위한 것이므로 세부 구현은 생략 */
  
  return position;
}

/* 끝으로 추가할 수 있는 다른 줌 관련 함수들 */

/**
 * blouedit_timeline_set_clip_position_mode:
 * @timeline: 타임라인 객체
 * @mode: 클립 위치 표시 모드
 *
 * 타임라인에서 클립 위치를 표시하는 방식을 설정합니다.
 */
void
blouedit_timeline_set_clip_position_mode (BlouEditTimeline *timeline, BlouEditClipPositionMode mode)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 모드가 변경되지 않았으면 아무 작업도 하지 않음 */
  if (timeline->clip_position_mode == mode)
    return;
  
  timeline->clip_position_mode = mode;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_clip_position_mode:
 * @timeline: 타임라인 객체
 *
 * 현재 클립 위치 표시 모드를 반환합니다.
 *
 * Returns: 현재 클립 위치 표시 모드
 */
BlouEditClipPositionMode
blouedit_timeline_get_clip_position_mode (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), BLOUEDIT_CLIP_POSITION_HIDDEN);
  
  return timeline->clip_position_mode;
}

/**
 * blouedit_timeline_set_use_frames_for_position:
 * @timeline: 타임라인 객체
 * @use_frames: 프레임 번호 사용 여부
 *
 * 클립 위치를 프레임 번호로 표시할지, 타임코드로 표시할지 설정합니다.
 */
void
blouedit_timeline_set_use_frames_for_position (BlouEditTimeline *timeline, gboolean use_frames)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 설정이 변경되지 않았으면 아무 작업도 하지 않음 */
  if (timeline->use_frames_for_position == use_frames)
    return;
  
  timeline->use_frames_for_position = use_frames;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_use_frames_for_position:
 * @timeline: 타임라인 객체
 *
 * 클립 위치 표시에 프레임 번호를 사용하는지 여부를 반환합니다.
 *
 * Returns: 프레임 번호 사용 시 TRUE, 타임코드 사용 시 FALSE
 */
gboolean
blouedit_timeline_get_use_frames_for_position (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  return timeline->use_frames_for_position;
}

/**
 * blouedit_timeline_set_clip_position_color:
 * @timeline: 타임라인 객체
 * @color: 표시 색상
 *
 * 클립 위치 텍스트의 색상을 설정합니다.
 */
void
blouedit_timeline_set_clip_position_color (BlouEditTimeline *timeline, const GdkRGBA *color)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (color != NULL);
  
  timeline->clip_position_color = *color;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_clip_position_color:
 * @timeline: 타임라인 객체
 * @color: 반환할 색상 구조체
 *
 * 클립 위치 텍스트의 현재 색상을 가져옵니다.
 */
void
blouedit_timeline_get_clip_position_color (BlouEditTimeline *timeline, GdkRGBA *color)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (color != NULL);
  
  *color = timeline->clip_position_color;
}

/**
 * blouedit_timeline_set_clip_position_font_size:
 * @timeline: 타임라인 객체
 * @font_size: 폰트 크기
 *
 * 클립 위치 텍스트의 폰트 크기를 설정합니다.
 */
void
blouedit_timeline_set_clip_position_font_size (BlouEditTimeline *timeline, gint font_size)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (font_size > 0);
  
  timeline->clip_position_font_size = font_size;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_clip_position_font_size:
 * @timeline: 타임라인 객체
 *
 * 클립 위치 텍스트의 현재 폰트 크기를 반환합니다.
 *
 * Returns: 폰트 크기
 */
gint
blouedit_timeline_get_clip_position_font_size (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 10);
  
  return timeline->clip_position_font_size;
}

/**
 * blouedit_timeline_format_clip_position:
 * @timeline: 타임라인 객체
 * @position: 포맷할 타임라인 위치 (나노초 단위)
 *
 * 클립 위치를 현재 설정에 따라 프레임 번호 또는 타임코드로 포맷합니다.
 *
 * Returns: 포맷된 문자열 (g_free()로 해제 필요)
 */
gchar *
blouedit_timeline_format_clip_position (BlouEditTimeline *timeline, gint64 position)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  if (timeline->use_frames_for_position) {
    /* 프레임 번호로 변환 */
    gdouble fps = timeline->framerate;
    if (fps <= 0) fps = 30.0; /* 기본값 */
    
    gint frame = (gint)((gdouble)position / GST_SECOND * fps);
    return g_strdup_printf ("%d", frame);
  } else {
    /* 타임코드로 변환 */
    return blouedit_timeline_position_to_timecode (timeline, position, timeline->timecode_format);
  }
}

/**
 * blouedit_timeline_draw_clip_position:
 * @timeline: 타임라인 객체
 * @cr: 카이로 컨텍스트
 * @clip: 클립 객체
 * @clip_x: 클립의 X 좌표
 * @clip_y: 클립의 Y 좌표
 * @clip_width: 클립의 너비
 * @clip_height: 클립의 높이
 *
 * 클립에 위치 표시기를 그립니다. 이 함수는 타임라인 그리기 함수에서 각 클립마다 호출됩니다.
 */
void
blouedit_timeline_draw_clip_position (BlouEditTimeline *timeline, cairo_t *cr, GESClip *clip,
                                     gdouble clip_x, gdouble clip_y, gdouble clip_width, gdouble clip_height)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (cr != NULL);
  g_return_if_fail (GES_IS_CLIP (clip));
  
  /* 위치 표시가 비활성화되어 있으면 아무것도 하지 않음 */
  if (timeline->clip_position_mode == BLOUEDIT_CLIP_POSITION_HIDDEN)
    return;
  
  /* 클립이 너무 작으면 표시하지 않음 */
  if (clip_width < 50)
    return;
  
  /* 클립 위치 및 길이 가져오기 */
  gint64 start_position = ges_clip_get_start (clip);
  gint64 duration = ges_clip_get_duration (clip);
  gint64 end_position = start_position + duration;
  
  /* 폰트 설정 */
  cairo_select_font_face (cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  cairo_set_font_size (cr, timeline->clip_position_font_size);
  
  /* 텍스트 색상 설정 */
  cairo_set_source_rgba (cr, 
                       timeline->clip_position_color.red,
                       timeline->clip_position_color.green,
                       timeline->clip_position_color.blue,
                       timeline->clip_position_color.alpha);
  
  /* 조건에 따라 텍스트 표시 */
  gchar *start_str = NULL;
  gchar *end_str = NULL;
  gchar *duration_str = NULL;
  cairo_text_extents_t extents;
  
  switch (timeline->clip_position_mode) {
    case BLOUEDIT_CLIP_POSITION_START_ONLY:
      start_str = blouedit_timeline_format_clip_position (timeline, start_position);
      cairo_text_extents (cr, start_str, &extents);
      /* 클립 왼쪽 상단에 표시 */
      cairo_move_to (cr, clip_x + 5, clip_y + extents.height + 2);
      cairo_show_text (cr, start_str);
      g_free (start_str);
      break;
      
    case BLOUEDIT_CLIP_POSITION_END_ONLY:
      end_str = blouedit_timeline_format_clip_position (timeline, end_position);
      cairo_text_extents (cr, end_str, &extents);
      /* 클립 오른쪽 상단에 표시 */
      cairo_move_to (cr, clip_x + clip_width - extents.width - 5, clip_y + extents.height + 2);
      cairo_show_text (cr, end_str);
      g_free (end_str);
      break;
      
    case BLOUEDIT_CLIP_POSITION_BOTH:
      start_str = blouedit_timeline_format_clip_position (timeline, start_position);
      end_str = blouedit_timeline_format_clip_position (timeline, end_position);
      
      /* 시작 위치 표시 */
      cairo_text_extents (cr, start_str, &extents);
      cairo_move_to (cr, clip_x + 5, clip_y + extents.height + 2);
      cairo_show_text (cr, start_str);
      
      /* 종료 위치 표시 */
      cairo_text_extents (cr, end_str, &extents);
      cairo_move_to (cr, clip_x + clip_width - extents.width - 5, clip_y + extents.height + 2);
      cairo_show_text (cr, end_str);
      
      g_free (start_str);
      g_free (end_str);
      break;
      
    case BLOUEDIT_CLIP_POSITION_DURATION:
      if (timeline->use_frames_for_position) {
        /* 프레임 수로 표시 */
        gdouble fps = timeline->framerate;
        if (fps <= 0) fps = 30.0; /* 기본값 */
        
        gint frames = (gint)((gdouble)duration / GST_SECOND * fps);
        duration_str = g_strdup_printf ("%d frames", frames);
      } else {
        /* 시:분:초 형식으로 표시 */
        guint hours, minutes, seconds, millis;
        guint64 time = duration / GST_MSECOND; /* 밀리초 단위로 변환 */
        
        hours = time / 3600000;
        minutes = (time / 60000) % 60;
        seconds = (time / 1000) % 60;
        millis = time % 1000;
        
        if (hours > 0) {
          duration_str = g_strdup_printf ("%02u:%02u:%02u.%03u", hours, minutes, seconds, millis);
        } else {
          duration_str = g_strdup_printf ("%02u:%02u.%03u", minutes, seconds, millis);
        }
      }
      
      /* 길이 표시 */
      cairo_text_extents (cr, duration_str, &extents);
      cairo_move_to (cr, clip_x + (clip_width - extents.width) / 2, clip_y + extents.height + 2);
      cairo_show_text (cr, duration_str);
      
      g_free (duration_str);
      break;
      
    case BLOUEDIT_CLIP_POSITION_HIDDEN:
    default:
      /* 아무것도 표시하지 않음 */
      break;
  }
}

/**
 * blouedit_timeline_init_clip_position_defaults:
 * @timeline: 타임라인 객체
 *
 * 클립 위치 표시의 기본값을 초기화합니다.
 * 타임라인 객체 생성 중에 호출됩니다.
 */
void
blouedit_timeline_init_clip_position_defaults (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 기본 설정 */
  timeline->clip_position_mode = BLOUEDIT_CLIP_POSITION_BOTH;
  timeline->use_frames_for_position = FALSE;
  timeline->clip_position_font_size = 10;
  
  /* 기본 색상: 흰색 텍스트 */
  gdk_rgba_parse (&timeline->clip_position_color, "rgba(255,255,255,0.8)");
}

/**
 * blouedit_timeline_set_use_playback_range:
 * @timeline: 타임라인 객체
 * @use_range: 재생 범위 사용 여부
 *
 * 재생 범위(인/아웃 포인트)를 사용할지 여부를 설정합니다.
 * 활성화하면 타임라인 재생 시 인/아웃 포인트 사이의 구간만 재생됩니다.
 */
void
blouedit_timeline_set_use_playback_range (BlouEditTimeline *timeline, gboolean use_range)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 상태가 변경되지 않았으면 아무 작업도 하지 않음 */
  if (timeline->use_playback_range == use_range)
    return;
  
  timeline->use_playback_range = use_range;
  
  /* 파이프라인에 재생 범위 설정 */
  if (timeline->pipeline) {
    if (use_range) {
      /* 인/아웃 포인트가 유효한지 확인 */
      if (timeline->in_point >= 0 && timeline->out_point > timeline->in_point) {
        gst_element_seek_simple (timeline->pipeline, GST_FORMAT_TIME,
                                GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE,
                                timeline->in_point);
      }
    }
  }
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_use_playback_range:
 * @timeline: 타임라인 객체
 *
 * 재생 범위 사용 여부를 반환합니다.
 *
 * Returns: 재생 범위 사용 시 TRUE, 미사용 시 FALSE
 */
gboolean
blouedit_timeline_get_use_playback_range (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  return timeline->use_playback_range;
}

/**
 * blouedit_timeline_set_in_point:
 * @timeline: 타임라인 객체
 * @position: 설정할 인 포인트 위치 (타임라인 단위)
 *
 * 재생 범위의 인 포인트(시작 지점)를 설정합니다.
 */
void
blouedit_timeline_set_in_point (BlouEditTimeline *timeline, gint64 position)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 인 포인트는 0 이상이어야 함 */
  if (position < 0)
    position = 0;
  
  /* 현재 아웃 포인트보다 앞에 있어야 함 */
  if (timeline->out_point > 0 && position >= timeline->out_point) {
    g_warning ("In point must be before out point");
    return;
  }
  
  timeline->in_point = position;
  
  /* 재생 범위가 활성화되어 있고 현재 재생 중인 경우 포지션 조정 */
  if (timeline->use_playback_range && timeline->pipeline) {
    GstState state;
    gst_element_get_state (timeline->pipeline, &state, NULL, 0);
    
    if (state == GST_STATE_PLAYING) {
      gint64 current_pos;
      if (gst_element_query_position (timeline->pipeline, GST_FORMAT_TIME, &current_pos)) {
        if (current_pos < position) {
          gst_element_seek_simple (timeline->pipeline, GST_FORMAT_TIME,
                                  GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE,
                                  position);
        }
      }
    }
  }
  
  /* 변경 이벤트 발생 */
  g_signal_emit_by_name (timeline, "in-point-changed", position);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_in_point:
 * @timeline: 타임라인 객체
 *
 * 현재 설정된 인 포인트(시작 지점)를 반환합니다.
 *
 * Returns: 인 포인트 위치 (타임라인 단위)
 */
gint64
blouedit_timeline_get_in_point (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  return timeline->in_point;
}

/**
 * blouedit_timeline_set_out_point:
 * @timeline: 타임라인 객체
 * @position: 설정할 아웃 포인트 위치 (타임라인 단위)
 *
 * 재생 범위의 아웃 포인트(종료 지점)를 설정합니다.
 */
void
blouedit_timeline_set_out_point (BlouEditTimeline *timeline, gint64 position)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 타임라인 길이보다 크지 않아야 함 */
  gint64 duration = blouedit_timeline_get_duration (timeline);
  if (position > duration && duration > 0) {
    position = duration;
  }
  
  /* 현재 인 포인트보다 뒤에 있어야 함 */
  if (position <= timeline->in_point && position > 0) {
    g_warning ("Out point must be after in point");
    return;
  }
  
  timeline->out_point = position;
  
  /* 재생 범위가 활성화되어 있고 현재 재생 중인 경우 포지션 조정 */
  if (timeline->use_playback_range && timeline->pipeline) {
    GstState state;
    gst_element_get_state (timeline->pipeline, &state, NULL, 0);
    
    if (state == GST_STATE_PLAYING) {
      gint64 current_pos;
      if (gst_element_query_position (timeline->pipeline, GST_FORMAT_TIME, &current_pos)) {
        if (current_pos > position) {
          /* 아웃 포인트 이후로 재생 헤드가 이동하면 인 포인트로 돌아가기 */
          gst_element_seek_simple (timeline->pipeline, GST_FORMAT_TIME,
                                  GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE,
                                  timeline->in_point);
        }
      }
    }
  }
  
  /* 변경 이벤트 발생 */
  g_signal_emit_by_name (timeline, "out-point-changed", position);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_out_point:
 * @timeline: 타임라인 객체
 *
 * 현재 설정된 아웃 포인트(종료 지점)를 반환합니다.
 *
 * Returns: 아웃 포인트 위치 (타임라인 단위)
 */
gint64
blouedit_timeline_get_out_point (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  return timeline->out_point;
}

/**
 * blouedit_timeline_clear_playback_range:
 * @timeline: 타임라인 객체
 *
 * 재생 범위를 초기화합니다. 인/아웃 포인트가 제거되고 전체 타임라인 재생으로 돌아갑니다.
 */
void
blouedit_timeline_clear_playback_range (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  timeline->in_point = 0;
  timeline->out_point = 0;
  timeline->use_playback_range = FALSE;
  
  /* 변경 이벤트 발생 */
  g_signal_emit_by_name (timeline, "playback-range-cleared");
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_set_show_range_markers:
 * @timeline: 타임라인 객체
 * @show: 마커 표시 여부
 *
 * 타임라인 룰러에 인/아웃 포인트 마커를 표시할지 여부를 설정합니다.
 */
void
blouedit_timeline_set_show_range_markers (BlouEditTimeline *timeline, gboolean show)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 상태가 변경되지 않았으면 아무 작업도 하지 않음 */
  if (timeline->show_range_markers == show)
    return;
  
  timeline->show_range_markers = show;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_show_range_markers:
 * @timeline: 타임라인 객체
 *
 * 인/아웃 포인트 마커 표시 여부를 반환합니다.
 *
 * Returns: 마커 표시 시 TRUE, 미표시 시 FALSE
 */
gboolean
blouedit_timeline_get_show_range_markers (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  return timeline->show_range_markers;
}

/**
 * blouedit_timeline_draw_playback_range_markers:
 * @timeline: 타임라인 객체
 * @cr: 카이로 컨텍스트
 * @ruler_x: 룰러 X 좌표
 * @ruler_y: 룰러 Y 좌표
 * @ruler_width: 룰러 너비
 * @ruler_height: 룰러 높이
 *
 * 타임라인 룰러에 인/아웃 포인트 마커를 그립니다.
 * 이 함수는 타임라인 룰러 그리기 함수에서 호출됩니다.
 */
void
blouedit_timeline_draw_playback_range_markers (BlouEditTimeline *timeline, cairo_t *cr,
                                              gdouble ruler_x, gdouble ruler_y,
                                              gdouble ruler_width, gdouble ruler_height)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (cr != NULL);
  
  /* 마커 표시가 비활성화되어 있으면 아무것도 하지 않음 */
  if (!timeline->show_range_markers)
    return;
  
  /* 인/아웃 포인트가 설정되어 있지 않으면 아무것도 하지 않음 */
  if (timeline->in_point <= 0 && timeline->out_point <= 0)
    return;
  
  /* 인 포인트 마커 그리기 */
  if (timeline->in_point > 0) {
    gdouble in_x = blouedit_timeline_get_x_from_position (timeline, timeline->in_point);
    
    /* 삼각형 마커 그리기 (녹색) */
    cairo_set_source_rgba (cr, 0.2, 0.8, 0.2, 0.8);
    cairo_move_to (cr, in_x, ruler_y);
    cairo_line_to (cr, in_x - 6, ruler_y + 6);
    cairo_line_to (cr, in_x + 6, ruler_y + 6);
    cairo_close_path (cr);
    cairo_fill (cr);
    
    /* 세로선 그리기 */
    if (timeline->use_playback_range) {
      cairo_set_source_rgba (cr, 0.2, 0.8, 0.2, 0.5);
      cairo_set_line_width (cr, 1.0);
      cairo_move_to (cr, in_x, ruler_y);
      cairo_line_to (cr, in_x, ruler_y + ruler_height + 500); /* 타임라인 아래쪽까지 그리기 */
      cairo_stroke (cr);
    }
  }
  
  /* 아웃 포인트 마커 그리기 */
  if (timeline->out_point > 0) {
    gdouble out_x = blouedit_timeline_get_x_from_position (timeline, timeline->out_point);
    
    /* 삼각형 마커 그리기 (빨간색) */
    cairo_set_source_rgba (cr, 0.8, 0.2, 0.2, 0.8);
    cairo_move_to (cr, out_x, ruler_y);
    cairo_line_to (cr, out_x - 6, ruler_y + 6);
    cairo_line_to (cr, out_x + 6, ruler_y + 6);
    cairo_close_path (cr);
    cairo_fill (cr);
    
    /* 세로선 그리기 */
    if (timeline->use_playback_range) {
      cairo_set_source_rgba (cr, 0.8, 0.2, 0.2, 0.5);
      cairo_set_line_width (cr, 1.0);
      cairo_move_to (cr, out_x, ruler_y);
      cairo_line_to (cr, out_x, ruler_y + ruler_height + 500); /* 타임라인 아래쪽까지 그리기 */
      cairo_stroke (cr);
    }
  }
  
  /* 인/아웃 포인트 사이 구간 표시 (재생 범위가 활성화된 경우) */
  if (timeline->use_playback_range && timeline->in_point > 0 && timeline->out_point > 0) {
    gdouble in_x = blouedit_timeline_get_x_from_position (timeline, timeline->in_point);
    gdouble out_x = blouedit_timeline_get_x_from_position (timeline, timeline->out_point);
    
    /* 재생 범위 배경 강조 */
    cairo_set_source_rgba (cr, 0.3, 0.6, 0.9, 0.1);
    cairo_rectangle (cr, in_x, ruler_y, out_x - in_x, ruler_height + 500);
    cairo_fill (cr);
  }
}

/**
 * blouedit_timeline_init_playback_range_defaults:
 * @timeline: 타임라인 객체
 *
 * 재생 범위의 기본값을 초기화합니다.
 * 타임라인 객체 생성 중에 호출됩니다.
 */
void
blouedit_timeline_init_playback_range_defaults (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 기본 설정 */
  timeline->use_playback_range = FALSE;
  timeline->in_point = 0;
  timeline->out_point = 0;
  timeline->show_range_markers = TRUE;
} 