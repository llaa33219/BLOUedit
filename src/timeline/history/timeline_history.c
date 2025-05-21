#include <gtk/gtk.h>
#include <string.h>
#include <time.h>
#include "timeline_history.h"
#include "../core/types.h"
#include "../core/timeline.h"

/* 히스토리 항목 구조체 */
typedef struct {
  BlouEditHistoryType type;
  gpointer data;
  gpointer old_state;
  gpointer new_state;
  gchar *description;
  GDateTime *timestamp;
  gchar *group_name;
} BlouEditHistoryEntry;

/* 히스토리 항목 생성 함수 */
static BlouEditHistoryEntry*
history_entry_new(BlouEditHistoryType type, 
                 gpointer data, 
                 gpointer old_state, 
                 gpointer new_state)
{
  BlouEditHistoryEntry *entry = g_new0(BlouEditHistoryEntry, 1);
  
  entry->type = type;
  entry->data = data;
  entry->old_state = old_state;
  entry->new_state = new_state;
  entry->timestamp = g_date_time_new_now_local();
  
  /* 유형에 따른 설명 생성 */
  switch (type) {
    case BLOUEDIT_HISTORY_ADD_CLIP:
      entry->description = g_strdup("클립 추가");
      break;
    case BLOUEDIT_HISTORY_REMOVE_CLIP:
      entry->description = g_strdup("클립 제거");
      break;
    case BLOUEDIT_HISTORY_MOVE_CLIP:
      entry->description = g_strdup("클립 이동");
      break;
    case BLOUEDIT_HISTORY_TRIM_CLIP:
      entry->description = g_strdup("클립 트리밍");
      break;
    case BLOUEDIT_HISTORY_SPLIT_CLIP:
      entry->description = g_strdup("클립 분할");
      break;
    case BLOUEDIT_HISTORY_MERGE_CLIPS:
      entry->description = g_strdup("클립 병합");
      break;
    case BLOUEDIT_HISTORY_MODIFY_EFFECT:
      entry->description = g_strdup("효과 수정");
      break;
    case BLOUEDIT_HISTORY_ADD_TRACK:
      entry->description = g_strdup("트랙 추가");
      break;
    case BLOUEDIT_HISTORY_REMOVE_TRACK:
      entry->description = g_strdup("트랙 제거");
      break;
    case BLOUEDIT_HISTORY_MODIFY_TRACK:
      entry->description = g_strdup("트랙 수정");
      break;
    case BLOUEDIT_HISTORY_GROUP_BEGIN:
      entry->description = g_strdup("그룹 편집 시작");
      break;
    case BLOUEDIT_HISTORY_GROUP_END:
      entry->description = g_strdup("그룹 편집 종료");
      break;
    case BLOUEDIT_HISTORY_SNAPSHOT:
      entry->description = g_strdup("타임라인 스냅샷");
      break;
    default:
      entry->description = g_strdup("알 수 없는 작업");
  }
  
  return entry;
}

/* 히스토리 항목 해제 함수 */
static void
history_entry_free(BlouEditHistoryEntry *entry)
{
  if (entry == NULL) {
    return;
  }
  
  g_free(entry->description);
  g_free(entry->group_name);
  
  if (entry->timestamp != NULL) {
    g_date_time_unref(entry->timestamp);
  }
  
  /* 각 유형별 메모리 해제 처리 - 실제 구현 시 채워야 함 */
  
  g_free(entry);
}

/* 히스토리 초기화 */
void
blouedit_timeline_history_init(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 히스토리 목록 초기화 */
  timeline->history = NULL;
  timeline->redo_history = NULL;
  timeline->current_history_group = NULL;
  timeline->history_enabled = TRUE;
}

/* 히스토리 항목 추가 */
void
blouedit_timeline_history_add_entry(BlouEditTimeline *timeline, 
                                BlouEditHistoryType type, 
                                gpointer data, 
                                gpointer old_state, 
                                gpointer new_state)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 히스토리가 비활성화된 경우 무시 */
  if (!timeline->history_enabled) {
    return;
  }
  
  /* 새 항목 생성 */
  BlouEditHistoryEntry *entry = history_entry_new(type, data, old_state, new_state);
  
  /* 현재 그룹이 있으면 그룹 이름 설정 */
  if (timeline->current_history_group != NULL) {
    entry->group_name = g_strdup(timeline->current_history_group);
  }
  
  /* 히스토리에 추가 */
  timeline->history = g_list_append(timeline->history, entry);
  
  /* 다시 실행 히스토리는 모두 제거 */
  g_list_free_full(timeline->redo_history, (GDestroyNotify)history_entry_free);
  timeline->redo_history = NULL;
  
  /* 시그널 발생 */
  g_signal_emit_by_name(timeline, "history-changed");
}

/* 실행 취소 */
gboolean
blouedit_timeline_history_undo(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 히스토리가 비어있으면 아무것도 하지 않음 */
  if (timeline->history == NULL) {
    return FALSE;
  }
  
  /* 그룹 모드인 경우의 처리 */
  if (timeline->current_history_group != NULL) {
    /* 그룹 내 모든 작업을 되돌림 */
    gchar *group_name = g_strdup(timeline->current_history_group);
    
    blouedit_timeline_history_end_group(timeline);
    
    /* 그룹의 시작부터 끝까지 모두 되돌림 */
    GList *group_items = NULL;
    GList *item = g_list_last(timeline->history);
    gboolean found_group_begin = FALSE;
    
    while (item != NULL && !found_group_begin) {
      BlouEditHistoryEntry *entry = (BlouEditHistoryEntry*)item->data;
      
      if (g_strcmp0(entry->group_name, group_name) == 0) {
        if (entry->type == BLOUEDIT_HISTORY_GROUP_BEGIN) {
          found_group_begin = TRUE;
        }
        
        GList *prev = item->prev;
        
        /* 항목을 히스토리에서 제거하고 그룹 항목에 추가 */
        timeline->history = g_list_remove_link(timeline->history, item);
        group_items = g_list_concat(item, group_items);
        
        item = prev;
      } else {
        item = item->prev;
      }
    }
    
    /* 그룹 항목을 다시 실행 히스토리에 추가 */
    timeline->redo_history = g_list_concat(group_items, timeline->redo_history);
    
    g_free(group_name);
    
    /* 시그널 발생 */
    g_signal_emit_by_name(timeline, "history-changed");
    
    return found_group_begin;
  }
  
  /* 마지막 항목 가져오기 */
  GList *last = g_list_last(timeline->history);
  BlouEditHistoryEntry *entry = (BlouEditHistoryEntry*)last->data;
  
  /* 다시 실행 히스토리로 이동 */
  timeline->history = g_list_remove_link(timeline->history, last);
  timeline->redo_history = g_list_concat(last, timeline->redo_history);
  
  /* 시그널 발생 */
  g_signal_emit_by_name(timeline, "history-changed");
  
  return TRUE;
}

/* 다시 실행 */
gboolean
blouedit_timeline_history_redo(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  
  /* 다시 실행 히스토리가 비어있으면 아무것도 하지 않음 */
  if (timeline->redo_history == NULL) {
    return FALSE;
  }
  
  /* 그룹 모드인 경우의 처리 */
  BlouEditHistoryEntry *first_entry = (BlouEditHistoryEntry*)timeline->redo_history->data;
  if (first_entry->group_name != NULL) {
    /* 그룹의 모든 항목을 다시 실행 */
    gchar *group_name = g_strdup(first_entry->group_name);
    
    /* 그룹의 시작부터 끝까지 모두 다시 실행 */
    GList *group_items = NULL;
    GList *item = timeline->redo_history;
    gboolean completed_group = FALSE;
    
    while (item != NULL && !completed_group) {
      BlouEditHistoryEntry *entry = (BlouEditHistoryEntry*)item->data;
      
      if (g_strcmp0(entry->group_name, group_name) == 0) {
        if (entry->type == BLOUEDIT_HISTORY_GROUP_END) {
          completed_group = TRUE;
        }
        
        GList *next = item->next;
        
        /* 항목을 다시 실행 히스토리에서 제거하고 그룹 항목에 추가 */
        timeline->redo_history = g_list_remove_link(timeline->redo_history, item);
        group_items = g_list_concat(group_items, item);
        
        item = next;
      } else {
        break;
      }
    }
    
    /* 그룹 항목을 히스토리에 추가 */
    timeline->history = g_list_concat(timeline->history, group_items);
    
    g_free(group_name);
    
    /* 시그널 발생 */
    g_signal_emit_by_name(timeline, "history-changed");
    
    return completed_group;
  }
  
  /* 첫 번째 항목 가져오기 */
  GList *first = timeline->redo_history;
  
  /* 히스토리로 이동 */
  timeline->redo_history = g_list_remove_link(timeline->redo_history, first);
  timeline->history = g_list_concat(timeline->history, first);
  
  /* 시그널 발생 */
  g_signal_emit_by_name(timeline, "history-changed");
  
  return TRUE;
}

/* 히스토리 지우기 */
void
blouedit_timeline_history_clear(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 모든 히스토리 해제 */
  g_list_free_full(timeline->history, (GDestroyNotify)history_entry_free);
  g_list_free_full(timeline->redo_history, (GDestroyNotify)history_entry_free);
  
  timeline->history = NULL;
  timeline->redo_history = NULL;
  
  /* 시그널 발생 */
  g_signal_emit_by_name(timeline, "history-changed");
}

/* 히스토리 항목 목록 가져오기 */
GList*
blouedit_timeline_history_get_entries(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  return timeline->history;
}

/* 그룹 편집 시작 */
void
blouedit_timeline_history_begin_group(BlouEditTimeline *timeline, const gchar *group_name)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(group_name != NULL);
  
  /* 이미 그룹 모드인 경우 기존 그룹 종료 */
  if (timeline->current_history_group != NULL) {
    blouedit_timeline_history_end_group(timeline);
  }
  
  /* 그룹 이름 설정 */
  timeline->current_history_group = g_strdup(group_name);
  
  /* 그룹 시작 항목 추가 */
  blouedit_timeline_history_add_entry(timeline, 
                                  BLOUEDIT_HISTORY_GROUP_BEGIN, 
                                  NULL, NULL, NULL);
}

/* 그룹 편집 종료 */
void
blouedit_timeline_history_end_group(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  if (timeline->current_history_group == NULL) {
    return;
  }
  
  /* 그룹 종료 항목 추가 */
  blouedit_timeline_history_add_entry(timeline, 
                                  BLOUEDIT_HISTORY_GROUP_END, 
                                  NULL, NULL, NULL);
  
  /* 그룹 이름 해제 */
  g_free(timeline->current_history_group);
  timeline->current_history_group = NULL;
}

/* 액션 유형을 문자열로 변환 */
static const gchar*
get_action_type_name(BlouEditHistoryType type)
{
  switch (type) {
    case BLOUEDIT_HISTORY_ADD_CLIP:        return "클립 추가";
    case BLOUEDIT_HISTORY_REMOVE_CLIP:     return "클립 제거";
    case BLOUEDIT_HISTORY_MOVE_CLIP:       return "클립 이동";
    case BLOUEDIT_HISTORY_TRIM_CLIP:       return "클립 트리밍";
    case BLOUEDIT_HISTORY_SPLIT_CLIP:      return "클립 분할";
    case BLOUEDIT_HISTORY_MERGE_CLIPS:     return "클립 병합";
    case BLOUEDIT_HISTORY_MODIFY_EFFECT:   return "효과 수정";
    case BLOUEDIT_HISTORY_ADD_TRACK:       return "트랙 추가";
    case BLOUEDIT_HISTORY_REMOVE_TRACK:    return "트랙 제거";
    case BLOUEDIT_HISTORY_MODIFY_TRACK:    return "트랙 수정";
    case BLOUEDIT_HISTORY_GROUP_BEGIN:     return "그룹 편집 시작";
    case BLOUEDIT_HISTORY_GROUP_END:       return "그룹 편집 종료";
    case BLOUEDIT_HISTORY_SNAPSHOT:        return "타임라인 스냅샷";
    default:                             return "알 수 없는 작업";
  }
}

/* 히스토리 항목 문자열 가져오기 */
static gchar*
get_history_entry_string(BlouEditHistoryEntry *entry)
{
  g_return_val_if_fail(entry != NULL, NULL);
  
  gchar *timestamp_str = g_date_time_format(entry->timestamp, "%H:%M:%S");
  
  gchar *result = g_strdup_printf("%s - %s%s%s",
                               timestamp_str,
                               get_action_type_name(entry->type),
                               entry->group_name ? " [" : "",
                               entry->group_name ? entry->group_name : "");
  
  g_free(timestamp_str);
  
  return result;
}

/* 히스토리 로그 보기 대화상자 */
void
blouedit_timeline_history_show_detailed_log(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *scroll, *tree_view;
  GtkListStore *store;
  GtkTreeIter iter;
  GtkCellRenderer *renderer;
  GtkTreeViewColumn *column;
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("편집 기록 로그",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_닫기", GTK_RESPONSE_CLOSE,
                                     NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 700, 500);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  
  /* 스크롤 윈도우 생성 */
  scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_widget_set_hexpand(scroll, TRUE);
  gtk_widget_set_vexpand(scroll, TRUE);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                               GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_container_add(GTK_CONTAINER(content_area), scroll);
  
  /* 모델 생성 */
  store = gtk_list_store_new(4, 
                           G_TYPE_STRING,   /* 타임스탬프 */
                           G_TYPE_STRING,   /* 작업 유형 */
                           G_TYPE_STRING,   /* 설명 */
                           G_TYPE_STRING);  /* 그룹 */
  
  /* 히스토리 항목 추가 */
  for (GList *l = timeline->history; l != NULL; l = l->next) {
    BlouEditHistoryEntry *entry = (BlouEditHistoryEntry*)l->data;
    gchar *timestamp_str = g_date_time_format(entry->timestamp, "%H:%M:%S");
    
    gtk_list_store_append(store, &iter);
    gtk_list_store_set(store, &iter,
                     0, timestamp_str,
                     1, get_action_type_name(entry->type),
                     2, entry->description,
                     3, entry->group_name ? entry->group_name : "",
                     -1);
    
    g_free(timestamp_str);
  }
  
  /* 트리 뷰 생성 */
  tree_view = gtk_tree_view_new_with_model(GTK_TREE_MODEL(store));
  g_object_unref(store);
  
  /* 열 추가 */
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("시간", 
                                                 renderer, 
                                                 "text", 0, 
                                                 NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("유형", 
                                                 renderer, 
                                                 "text", 1, 
                                                 NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("설명", 
                                                 renderer, 
                                                 "text", 2, 
                                                 NULL);
  gtk_tree_view_column_set_expand(column, TRUE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  renderer = gtk_cell_renderer_text_new();
  column = gtk_tree_view_column_new_with_attributes("그룹", 
                                                 renderer, 
                                                 "text", 3, 
                                                 NULL);
  gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), column);
  
  /* 트리 뷰 설정 */
  gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(tree_view), TRUE);
  gtk_container_add(GTK_CONTAINER(scroll), tree_view);
  
  /* 대화상자 표시 및 실행 */
  gtk_widget_show_all(dialog);
  gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 정리 */
  gtk_widget_destroy(dialog);
} 