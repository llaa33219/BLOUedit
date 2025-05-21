#pragma once

#include <gtk/gtk.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 편집 기록 항목 타입 */
typedef enum {
  BLOUEDIT_HISTORY_ADD_CLIP,        /* 클립 추가 */
  BLOUEDIT_HISTORY_REMOVE_CLIP,     /* 클립 제거 */
  BLOUEDIT_HISTORY_MOVE_CLIP,       /* 클립 이동 */
  BLOUEDIT_HISTORY_TRIM_CLIP,       /* 클립 트리밍 */
  BLOUEDIT_HISTORY_SPLIT_CLIP,      /* 클립 분할 */
  BLOUEDIT_HISTORY_MERGE_CLIPS,     /* 클립 병합 */
  BLOUEDIT_HISTORY_MODIFY_EFFECT,   /* 효과 수정 */
  BLOUEDIT_HISTORY_ADD_TRACK,       /* 트랙 추가 */
  BLOUEDIT_HISTORY_REMOVE_TRACK,    /* 트랙 제거 */
  BLOUEDIT_HISTORY_MODIFY_TRACK,    /* 트랙 수정 */
  BLOUEDIT_HISTORY_GROUP_BEGIN,     /* 그룹 편집 시작 */
  BLOUEDIT_HISTORY_GROUP_END,       /* 그룹 편집 종료 */
  BLOUEDIT_HISTORY_SNAPSHOT         /* 타임라인 스냅샷 */
} BlouEditHistoryType;

/* 히스토리 관련 함수 */
void         blouedit_timeline_history_init(BlouEditTimeline *timeline);
void         blouedit_timeline_history_add_entry(BlouEditTimeline *timeline, 
                                            BlouEditHistoryType type, 
                                            gpointer data, 
                                            gpointer old_state, 
                                            gpointer new_state);
gboolean     blouedit_timeline_history_undo(BlouEditTimeline *timeline);
gboolean     blouedit_timeline_history_redo(BlouEditTimeline *timeline);
void         blouedit_timeline_history_clear(BlouEditTimeline *timeline);
GList*       blouedit_timeline_history_get_entries(BlouEditTimeline *timeline);
void         blouedit_timeline_history_begin_group(BlouEditTimeline *timeline, const gchar *group_name);
void         blouedit_timeline_history_end_group(BlouEditTimeline *timeline);
void         blouedit_timeline_history_show_detailed_log(BlouEditTimeline *timeline);

/* 타임라인 스냅샷 관련 함수 */
void         blouedit_timeline_save_snapshot(BlouEditTimeline *timeline, const gchar *snapshot_name);
GList*       blouedit_timeline_get_snapshots(BlouEditTimeline *timeline);
gboolean     blouedit_timeline_restore_snapshot(BlouEditTimeline *timeline, const gchar *snapshot_name);
void         blouedit_timeline_delete_snapshot(BlouEditTimeline *timeline, const gchar *snapshot_name);
void         blouedit_timeline_show_snapshots_dialog(BlouEditTimeline *timeline);
void         blouedit_timeline_compare_snapshots(BlouEditTimeline *timeline, 
                                             const gchar *snapshot1_name, 
                                             const gchar *snapshot2_name);

/* 자동 저장 관련 함수 */
void         blouedit_timeline_setup_auto_save(BlouEditTimeline *timeline, guint interval_seconds);
gboolean     blouedit_timeline_restore_auto_save(BlouEditTimeline *timeline);
void         blouedit_timeline_clear_auto_saves(BlouEditTimeline *timeline);

/* 변경 시각화 관련 함수 */
void         blouedit_timeline_show_changes_visualization(BlouEditTimeline *timeline);
void         blouedit_timeline_set_changes_visualization_enabled(BlouEditTimeline *timeline, gboolean enabled);
gboolean     blouedit_timeline_get_changes_visualization_enabled(BlouEditTimeline *timeline);

G_END_DECLS 