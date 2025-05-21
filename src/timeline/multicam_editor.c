#include <gtk/gtk.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include <gst/editing-services/ges-clip.h>
#include "multicam_editor.h"
#include "core/types.h"
#include "core/timeline.h"

/* 멀티캠 그룹 목록 */
static GSList *multicam_groups = NULL;
static guint next_multicam_group_id = 1;

/* 기본 색상 팔레트 (소스별 구분 색상) */
static const GdkRGBA SOURCE_COLORS[] = {
  { 0.9, 0.2, 0.2, 0.7 }, /* 빨강 */
  { 0.2, 0.6, 0.9, 0.7 }, /* 파랑 */
  { 0.2, 0.8, 0.2, 0.7 }, /* 녹색 */
  { 0.9, 0.6, 0.2, 0.7 }, /* 주황 */
  { 0.8, 0.3, 0.8, 0.7 }, /* 보라 */
  { 0.7, 0.7, 0.2, 0.7 }, /* 노랑 */
  { 0.5, 0.5, 0.5, 0.7 }, /* 회색 */
  { 0.9, 0.4, 0.6, 0.7 }  /* 분홍 */
};

/* 프리셋 멀티캠 전환 타입 */
static const gchar* TRANSITION_TYPES[] = {
  "cut",           /* 즉시 전환 */
  "crossfade",     /* 크로스페이드 */
  "wipe_right",    /* 우측 와이프 */
  "wipe_left",     /* 좌측 와이프 */
  "wipe_up",       /* 상단 와이프 */
  "wipe_down",     /* 하단 와이프 */
  "dissolve",      /* 디졸브 */
  "fade_black"     /* 블랙을 통한 페이드 */
};

/* 타임라인 멀티캠 모드 설정 함수 */
void
blouedit_timeline_set_multicam_mode(BlouEditTimeline *timeline, BlouEditMulticamMode mode)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 이전 모드가 비활성화였고 새 모드가 활성화된 경우 */
  if (timeline->edit_mode != BLOUEDIT_EDIT_MODE_MULTICAM && mode != BLOUEDIT_MULTICAM_MODE_DISABLED) {
    /* 멀티캠 모드로 전환 */
    timeline->edit_mode = BLOUEDIT_EDIT_MODE_MULTICAM;
    
    /* UI 업데이트 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
    
    /* 메시지 표시 */
    if (mode == BLOUEDIT_MULTICAM_MODE_SOURCE_VIEW) {
      blouedit_timeline_show_message(timeline, "멀티캠 소스 뷰 모드 활성화");
    } else if (mode == BLOUEDIT_MULTICAM_MODE_EDIT) {
      blouedit_timeline_show_message(timeline, "멀티캠 편집 모드 활성화");
    }
  }
  /* 이전 모드가 활성화되었고 새 모드가 비활성화인 경우 */
  else if (timeline->edit_mode == BLOUEDIT_EDIT_MODE_MULTICAM && mode == BLOUEDIT_MULTICAM_MODE_DISABLED) {
    /* 일반 편집 모드로 되돌림 */
    timeline->edit_mode = BLOUEDIT_EDIT_MODE_NORMAL;
    
    /* UI 업데이트 */
    gtk_widget_queue_draw(GTK_WIDGET(timeline));
    
    /* 메시지 표시 */
    blouedit_timeline_show_message(timeline, "멀티캠 모드 비활성화");
  }
  
  /* 멀티캠 모드 업데이트 */
  timeline->multicam_mode = mode;
}

/* 타임라인 멀티캠 모드 가져오기 함수 */
BlouEditMulticamMode
blouedit_timeline_get_multicam_mode(BlouEditTimeline *timeline)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), BLOUEDIT_MULTICAM_MODE_DISABLED);
  
  return timeline->multicam_mode;
}

/* 멀티캠 소스 해제 함수 */
static void
multicam_source_free(BlouEditMulticamSource *source)
{
  if (!source)
    return;
    
  g_free(source->name);
  
  /* 참조 해제, 실제 클립은 삭제하지 않음 */
  if (source->source_clip)
    g_object_unref(source->source_clip);
    
  g_free(source);
}

/* 멀티캠 전환 해제 함수 */
static void
multicam_switch_free(BlouEditMulticamSwitch *sw)
{
  if (!sw)
    return;
    
  g_free(sw->transition_type);
  g_free(sw);
}

/* 멀티캠 그룹 해제 함수 */
static void
multicam_group_free(BlouEditMulticamGroup *group)
{
  if (!group)
    return;
    
  g_free(group->name);
  
  /* 소스 목록 해제 */
  g_slist_free_full(group->sources, (GDestroyNotify)multicam_source_free);
  
  /* 전환 목록 해제 */
  g_slist_free_full(group->switches, (GDestroyNotify)multicam_switch_free);
  
  g_free(group);
}

/* 멀티캠 그룹 생성 함수 */
BlouEditMulticamGroup*
blouedit_timeline_create_multicam_group(BlouEditTimeline *timeline, const gchar *name)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), NULL);
  
  BlouEditMulticamGroup *group = g_new0(BlouEditMulticamGroup, 1);
  
  group->id = next_multicam_group_id++;
  group->name = g_strdup(name ? name : "멀티캠 그룹");
  group->sources = NULL;
  group->switches = NULL;
  group->next_source_id = 1;
  group->next_switch_id = 1;
  group->active_source_id = 0;  /* 활성 소스 없음 */
  group->output_track = NULL;   /* 출력 트랙 없음 */
  
  /* 그룹 목록에 추가 */
  multicam_groups = g_slist_append(multicam_groups, group);
  
  return group;
}

/* 멀티캠 그룹 제거 함수 */
void
blouedit_timeline_remove_multicam_group(BlouEditTimeline *timeline, BlouEditMulticamGroup *group)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(group != NULL);
  
  /* 그룹 목록에서 제거 */
  multicam_groups = g_slist_remove(multicam_groups, group);
  
  /* 그룹 해제 */
  multicam_group_free(group);
  
  /* 타임라인 리드로우 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
}

/* 멀티캠 소스 추가 함수 */
BlouEditMulticamSource*
blouedit_multicam_group_add_source(BlouEditMulticamGroup *group, GESClip *clip, const gchar *name)
{
  g_return_val_if_fail(group != NULL, NULL);
  g_return_val_if_fail(GES_IS_CLIP(clip), NULL);
  
  BlouEditMulticamSource *source = g_new0(BlouEditMulticamSource, 1);
  
  source->id = group->next_source_id++;
  source->name = g_strdup(name ? name : "소스");
  source->source_clip = g_object_ref(clip);
  source->active = FALSE;
  source->synced = FALSE;
  source->sync_offset = 0;
  
  /* 소스 색상 할당 (순환) */
  gint color_index = (source->id - 1) % G_N_ELEMENTS(SOURCE_COLORS);
  source->color = SOURCE_COLORS[color_index];
  
  /* 그룹에 소스 추가 */
  group->sources = g_slist_append(group->sources, source);
  
  /* 첫 번째 소스를 활성 소스로 설정 */
  if (group->active_source_id == 0) {
    group->active_source_id = source->id;
    source->active = TRUE;
  }
  
  return source;
}

/* 멀티캠 소스 제거 함수 */
void
blouedit_multicam_group_remove_source(BlouEditMulticamGroup *group, guint source_id)
{
  g_return_if_fail(group != NULL);
  
  BlouEditMulticamSource *source_to_remove = NULL;
  GSList *item;
  
  /* 제거할 소스 찾기 */
  for (item = group->sources; item; item = item->next) {
    BlouEditMulticamSource *source = (BlouEditMulticamSource *)item->data;
    if (source->id == source_id) {
      source_to_remove = source;
      break;
    }
  }
  
  if (source_to_remove) {
    /* 소스가 활성 소스인 경우 처리 */
    if (source_id == group->active_source_id) {
      /* 첫 번째 다른 소스를 활성 소스로 설정 */
      for (item = group->sources; item; item = item->next) {
        BlouEditMulticamSource *source = (BlouEditMulticamSource *)item->data;
        if (source->id != source_id) {
          group->active_source_id = source->id;
          source->active = TRUE;
          break;
        }
      }
      
      /* 더 이상 소스가 없는 경우 */
      if (item == NULL) {
        group->active_source_id = 0;
      }
    }
    
    /* 관련된 모든 전환 제거 */
    GSList *next;
    for (item = group->switches; item; item = next) {
      BlouEditMulticamSwitch *sw = (BlouEditMulticamSwitch *)item->data;
      next = item->next;
      
      if (sw->source_id == source_id) {
        group->switches = g_slist_remove(group->switches, sw);
        multicam_switch_free(sw);
      }
    }
    
    /* 소스 목록에서 제거 및 해제 */
    group->sources = g_slist_remove(group->sources, source_to_remove);
    multicam_source_free(source_to_remove);
  }
}

/* 멀티캠 전환 추가 함수 */
BlouEditMulticamSwitch*
blouedit_multicam_group_add_switch(BlouEditMulticamGroup *group, 
                                guint source_id, 
                                gint64 position,
                                gint64 duration,
                                const gchar *transition_type)
{
  g_return_val_if_fail(group != NULL, NULL);
  
  /* 소스 확인 */
  gboolean source_exists = FALSE;
  for (GSList *item = group->sources; item; item = item->next) {
    BlouEditMulticamSource *source = (BlouEditMulticamSource *)item->data;
    if (source->id == source_id) {
      source_exists = TRUE;
      break;
    }
  }
  
  if (!source_exists) {
    g_warning("멀티캠 전환 추가 실패: 소스 ID %u가 존재하지 않습니다.", source_id);
    return NULL;
  }
  
  BlouEditMulticamSwitch *sw = g_new0(BlouEditMulticamSwitch, 1);
  
  sw->id = group->next_switch_id++;
  sw->source_id = source_id;
  sw->position = position;
  sw->duration = duration;
  sw->transition_type = g_strdup(transition_type ? transition_type : "cut");
  
  /* 전환 위치에 따라 정렬하여 추가 */
  GSList *insert_pos = NULL;
  GSList *item;
  
  for (item = group->switches; item; item = item->next) {
    BlouEditMulticamSwitch *existing = (BlouEditMulticamSwitch *)item->data;
    if (existing->position > position) {
      insert_pos = item;
      break;
    }
  }
  
  if (insert_pos) {
    group->switches = g_slist_insert_before(group->switches, insert_pos, sw);
  } else {
    group->switches = g_slist_append(group->switches, sw);
  }
  
  return sw;
}

/* 멀티캠 전환 제거 함수 */
void
blouedit_multicam_group_remove_switch(BlouEditMulticamGroup *group, guint switch_id)
{
  g_return_if_fail(group != NULL);
  
  BlouEditMulticamSwitch *switch_to_remove = NULL;
  GSList *item;
  
  /* 제거할 전환 찾기 */
  for (item = group->switches; item; item = item->next) {
    BlouEditMulticamSwitch *sw = (BlouEditMulticamSwitch *)item->data;
    if (sw->id == switch_id) {
      switch_to_remove = sw;
      break;
    }
  }
  
  if (switch_to_remove) {
    /* 전환 목록에서 제거 및 해제 */
    group->switches = g_slist_remove(group->switches, switch_to_remove);
    multicam_switch_free(switch_to_remove);
  }
}

/* 멀티캠 소스 동기화 설정 함수 */
void
blouedit_multicam_source_set_sync_offset(BlouEditMulticamSource *source, gint64 offset)
{
  g_return_if_fail(source != NULL);
  
  source->sync_offset = offset;
  source->synced = TRUE;
}

/* 멀티캠 그룹에서 특정 위치에서 활성화된 소스 찾기 */
static BlouEditMulticamSource*
get_active_source_at_position(BlouEditMulticamGroup *group, gint64 position)
{
  g_return_val_if_fail(group != NULL, NULL);
  
  /* 모든 전환을 시간순으로 살펴보며 (이미 정렬되어 있음) */
  BlouEditMulticamSwitch *last_switch = NULL;
  
  for (GSList *item = group->switches; item; item = item->next) {
    BlouEditMulticamSwitch *sw = (BlouEditMulticamSwitch *)item->data;
    
    /* 현재 위치보다 나중에 있는 전환이면 중단 */
    if (sw->position > position) {
      break;
    }
    
    last_switch = sw;
  }
  
  /* 마지막으로 활성화된 전환의 소스 ID 가져오기 */
  guint active_id = last_switch ? last_switch->source_id : group->active_source_id;
  
  /* 해당 소스 찾기 */
  for (GSList *item = group->sources; item; item = item->next) {
    BlouEditMulticamSource *source = (BlouEditMulticamSource *)item->data;
    if (source->id == active_id) {
      return source;
    }
  }
  
  return NULL;
}

/**
 * 멀티캠 편집 UI 표시 함수
 */
void
blouedit_timeline_show_multicam_editor(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *notebook;
  GtkWidget *sources_page, *switches_page, *settings_page;
  GtkWidget *sources_list, *switches_list;
  GtkWidget *sources_scroll, *switches_scroll;
  GtkWidget *button_box, *add_source_button, *remove_source_button;
  GtkWidget *add_switch_button, *remove_switch_button;
  GtkWidget *sync_button, *compile_button;
  
  dialog = gtk_dialog_new_with_buttons("멀티캠 편집기",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                     "_닫기", GTK_RESPONSE_CLOSE,
                                     NULL);
  
  gtk_window_set_default_size(GTK_WINDOW(dialog), 800, 600);
  
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  
  /* 노트북 생성 (탭 컨테이너) */
  notebook = gtk_notebook_new();
  gtk_container_add(GTK_CONTAINER(content_area), notebook);
  
  /* 소스 페이지 */
  sources_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  gtk_notebook_append_page(GTK_NOTEBOOK(notebook), sources_page, gtk_label_new("소스"));
  
  /* 소스 리스트 */
  sources_scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sources_scroll),
                                GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_box_pack_start(GTK_BOX(sources_page), sources_scroll, TRUE, TRUE, 0);
  
  sources_list = gtk_tree_view_new();
  gtk_container_add(GTK_CONTAINER(sources_scroll), sources_list);
  
  /* 소스 버튼 */
  button_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout(GTK_BUTTON_BOX(button_box), GTK_BUTTONBOX_END);
  gtk_box_set_spacing(GTK_BOX(button_box), 5);
  gtk_box_pack_start(GTK_BOX(sources_page), button_box, FALSE, FALSE, 0);
  
  add_source_button = gtk_button_new_with_label("소스 추가");
  remove_source_button = gtk_button_new_with_label("소스 제거");
  sync_button = gtk_button_new_with_label("동기화 설정");
  
  gtk_container_add(GTK_CONTAINER(button_box), add_source_button);
  gtk_container_add(GTK_CONTAINER(button_box), remove_source_button);
  gtk_container_add(GTK_CONTAINER(button_box), sync_button);
  
  /* 전환 페이지 */
  switches_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  gtk_notebook_append_page(GTK_NOTEBOOK(notebook), switches_page, gtk_label_new("전환"));
  
  /* 전환 리스트 */
  switches_scroll = gtk_scrolled_window_new(NULL, NULL);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(switches_scroll),
                                GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_box_pack_start(GTK_BOX(switches_page), switches_scroll, TRUE, TRUE, 0);
  
  switches_list = gtk_tree_view_new();
  gtk_container_add(GTK_CONTAINER(switches_scroll), switches_list);
  
  /* 전환 버튼 */
  button_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout(GTK_BUTTON_BOX(button_box), GTK_BUTTONBOX_END);
  gtk_box_set_spacing(GTK_BOX(button_box), 5);
  gtk_box_pack_start(GTK_BOX(switches_page), button_box, FALSE, FALSE, 0);
  
  add_switch_button = gtk_button_new_with_label("전환 추가");
  remove_switch_button = gtk_button_new_with_label("전환 제거");
  compile_button = gtk_button_new_with_label("컴파일");
  
  gtk_container_add(GTK_CONTAINER(button_box), add_switch_button);
  gtk_container_add(GTK_CONTAINER(button_box), remove_switch_button);
  gtk_container_add(GTK_CONTAINER(button_box), compile_button);
  
  /* 설정 페이지 */
  settings_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  gtk_notebook_append_page(GTK_NOTEBOOK(notebook), settings_page, gtk_label_new("설정"));
  
  /* TODO: 설정 페이지 내용 추가 */
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
  
  /* 대화상자 응답 처리 */
  gint response = gtk_dialog_run(GTK_DIALOG(dialog));
  
  /* 대화상자 닫기 */
  gtk_widget_destroy(dialog);
}

/* 멀티캠 소스 뷰 드로잉 함수 */
void
blouedit_timeline_draw_multicam_source_view(BlouEditTimeline *timeline, cairo_t *cr, int width, int height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(cr != NULL);
  
  /* 현재 타임라인에 활성화된 멀티캠 그룹 찾기 */
  BlouEditMulticamGroup *active_group = NULL;
  
  for (GSList *item = multicam_groups; item; item = item->next) {
    BlouEditMulticamGroup *group = (BlouEditMulticamGroup *)item->data;
    
    /* 첫 번째 그룹을 활성 그룹으로 선택 (나중에 선택 기능 추가 가능) */
    active_group = group;
    break;
  }
  
  if (!active_group || !active_group->sources) {
    /* 활성화된 멀티캠 그룹이 없거나 소스가 없는 경우 */
    cairo_set_source_rgba(cr, 0.3, 0.3, 0.3, 0.8);
    cairo_rectangle(cr, 0, 0, width, height);
    cairo_fill(cr);
    
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.8);
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 14);
    
    gchar *message = "멀티캠 소스가 없습니다. 멀티캠 편집기에서 소스를 추가하세요.";
    cairo_text_extents_t extents;
    cairo_text_extents(cr, message, &extents);
    
    cairo_move_to(cr, (width - extents.width) / 2, height / 2);
    cairo_show_text(cr, message);
    
    return;
  }
  
  /* 소스 개수에 따라 그리드 레이아웃 결정 */
  gint source_count = g_slist_length(active_group->sources);
  gint cols = 2;
  gint rows = (source_count + cols - 1) / cols;
  
  /* 소스가 많을 경우 컬럼 수 조정 */
  if (source_count > 4) {
    cols = 3;
    rows = (source_count + cols - 1) / cols;
  }
  
  /* 각 소스 뷰의 크기 계산 */
  gint cell_width = width / cols;
  gint cell_height = height / rows;
  
  /* 현재 위치에서 활성화된 소스 ID 가져오기 */
  gint64 position = blouedit_timeline_get_position(timeline);
  BlouEditMulticamSource *active_source = get_active_source_at_position(active_group, position);
  
  /* 모든 소스 그리기 */
  gint index = 0;
  for (GSList *item = active_group->sources; item; item = item->next, index++) {
    BlouEditMulticamSource *source = (BlouEditMulticamSource *)item->data;
    
    /* 그리드 위치 계산 */
    gint row = index / cols;
    gint col = index % cols;
    
    /* 셀 위치 및 크기 계산 */
    gint x = col * cell_width;
    gint y = row * cell_height;
    gint w = cell_width - 4;  /* 여백 */
    gint h = cell_height - 4; /* 여백 */
    
    /* 셀 배경 그리기 */
    cairo_set_source_rgba(cr, 0.2, 0.2, 0.2, 0.8);
    cairo_rectangle(cr, x + 2, y + 2, w, h);
    cairo_fill(cr);
    
    /* 소스 테두리 그리기 */
    if (active_source && source->id == active_source->id) {
      /* 활성 소스는 굵은 테두리로 강조 */
      cairo_set_source_rgba(cr, 1.0, 1.0, 0.0, 0.8);
      cairo_set_line_width(cr, 3.0);
    } else {
      cairo_set_source_rgba(cr, source->color.red, source->color.green, source->color.blue, source->color.alpha);
      cairo_set_line_width(cr, 1.5);
    }
    
    cairo_rectangle(cr, x + 2, y + 2, w, h);
    cairo_stroke(cr);
    
    /* 소스 이름 및 정보 그리기 */
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 12);
    
    /* 소스 이름 */
    cairo_move_to(cr, x + 10, y + 20);
    cairo_show_text(cr, source->name);
    
    /* 소스 ID */
    gchar *id_str = g_strdup_printf("ID: %u", source->id);
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 10);
    cairo_move_to(cr, x + 10, y + 40);
    cairo_show_text(cr, id_str);
    g_free(id_str);
    
    /* 동기화 상태 */
    gchar *sync_str = g_strdup_printf("동기화: %s", source->synced ? "완료" : "미설정");
    cairo_move_to(cr, x + 10, y + 55);
    cairo_show_text(cr, sync_str);
    g_free(sync_str);
  }
}

/* 멀티캠 미리 컴파일 함수 (렌더링 준비) */
void
blouedit_multicam_group_compile(BlouEditTimeline *timeline, BlouEditMulticamGroup *group)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  g_return_if_fail(group != NULL);
  
  /* 출력 트랙이 없는 경우 생성 */
  if (!group->output_track) {
    /* 첫 번째 비디오 트랙을 출력 트랙으로 설정 */
    group->output_track = blouedit_timeline_get_track_by_index(timeline, GES_TRACK_TYPE_VIDEO, 0);
    
    if (!group->output_track) {
      g_warning("멀티캠 컴파일 실패: 출력할 비디오 트랙이 없습니다.");
      return;
    }
  }
  
  /* 기존 출력 트랙의 클립 제거 */
  /* TODO: 실제 GES 구현 시 필요한 코드 추가 */
  
  /* 시작과 끝 위치 계산 */
  gint64 start_pos = G_MAXINT64;
  gint64 end_pos = 0;
  
  /* 모든 소스의 시작과 끝 위치 검사하여 전체 범위 계산 */
  for (GSList *item = group->sources; item; item = item->next) {
    BlouEditMulticamSource *source = (BlouEditMulticamSource *)item->data;
    
    gint64 clip_start = ges_timeline_element_get_start(GES_TIMELINE_ELEMENT(source->source_clip));
    gint64 clip_duration = ges_timeline_element_get_duration(GES_TIMELINE_ELEMENT(source->source_clip));
    gint64 clip_end = clip_start + clip_duration;
    
    if (clip_start < start_pos)
      start_pos = clip_start;
      
    if (clip_end > end_pos)
      end_pos = clip_end;
  }
  
  if (start_pos == G_MAXINT64 || end_pos == 0) {
    g_warning("멀티캠 컴파일 실패: 유효한 소스 범위를 찾을 수 없습니다.");
    return;
  }
  
  /* 각 전환 지점마다 클립 생성 */
  gint64 current_pos = start_pos;
  guint current_source_id = group->active_source_id;
  
  for (GSList *item = group->switches; item; item = item->next) {
    BlouEditMulticamSwitch *sw = (BlouEditMulticamSwitch *)item->data;
    
    if (sw->position <= current_pos)
      continue;
      
    /* 현재 위치에서 전환 위치까지 현재 소스의 클립 생성 */
    gint64 segment_duration = sw->position - current_pos;
    
    /* TODO: 실제 GES 구현 시 필요한 클립 생성 및 추가 코드 */
    
    /* 다음 세그먼트 준비 */
    current_pos = sw->position;
    current_source_id = sw->source_id;
  }
  
  /* 마지막 세그먼트 처리 (마지막 전환 후) */
  if (current_pos < end_pos) {
    gint64 segment_duration = end_pos - current_pos;
    
    /* TODO: 실제 GES 구현 시 필요한 클립 생성 및 추가 코드 */
  }
  
  /* 타임라인 갱신 */
  gtk_widget_queue_draw(GTK_WIDGET(timeline));
  
  /* 메시지 표시 */
  blouedit_timeline_show_message(timeline, "멀티캠 편집 컴파일 완료");
} 