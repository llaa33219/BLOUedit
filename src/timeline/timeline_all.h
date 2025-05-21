#pragma once

/**
 * 타임라인 헤더 통합 파일
 * 타임라인 관련 모든 헤더 파일을 포함합니다.
 */

/* Main timeline header */
#include "timeline.h"

/* UI elements */
#include "ui/timeline_view_mode.h"
#include "ui/timeline_hand_tool.h"

/* Markers */
#include "markers/markers.h"
#include "markers/marker_color.h"
#include "markers/marker_search.h"

/* Core types */
#include "core/types.h"
#include "core/timeline.h"

/* Tracks */
#include "tracks/tracks.h"

/* Groups */
#include "groups/groups.h"

/* Keyframes */
#include "keyframes/keyframes.h"

/* Timeline minimap */
#include "timeline_minimap.h"

/* 타임라인 그룹 관련 */
#include "groups/groups.h"

/* 클립 정렬 관련 */
#include "timeline.h"  /* For BlouEditAlignmentType */

/* 기타 확장 필요한 모듈 */
// #include "ui/timeline_ui.h"
// #include "effects/effects.h"
// #include "clips/clips.h"

#include "unlimited_tracks.h"

/* 편의를 위한 매크로 */
#define BLOUEDIT_NANOS_TO_SECONDS(n) ((gdouble)(n) / GST_SECOND)
#define BLOUEDIT_SECONDS_TO_NANOS(s) ((gint64)((s) * GST_SECOND)) 