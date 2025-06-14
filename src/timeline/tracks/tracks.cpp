#include "../core/timeline.h"
#include "../core/types.h"
#include "tracks.h"
#include <glib/gi18n.h>

// Forward declarations
static void on_add_track_dialog_response (GtkDialog *dialog, gint response_id, gpointer user_data);
static void on_track_type_toggled (GtkToggleButton *togglebutton, gpointer user_data);
static void on_track_type_color_changed (GtkToggleButton *radio, GtkWidget *color_button);
static void on_track_mute_toggled (GtkToggleButton *toggle, gpointer user_data);
static void on_track_solo_toggled (GtkToggleButton *toggle, gpointer user_data);
static void on_track_lock_toggled (GtkToggleButton *toggle, gpointer user_data);
static void on_track_add_button_clicked (BlouEditTimeline *timeline);
static void on_track_remove_button_clicked (GtkButton *button, gpointer user_data);
static void on_remove_track_confirm (GtkDialog *dialog, gint response_id, gpointer user_data);
static void on_track_move_up_button_clicked (GtkButton *button, gpointer user_data);
static void on_track_move_down_button_clicked (GtkButton *button, gpointer user_data);
static void on_track_row_selected (GtkListBox *box, GtkListBoxRow *row, gpointer user_data);
static void on_track_properties_apply (GtkButton *button, gpointer user_data);

// Data structures
typedef struct {
  BlouEditTimeline *timeline;
  GtkWidget *name_entry;
  GtkWidget *video_radio;
  GtkWidget *color_button;
} AddTrackDialogData;

/**
 * blouedit_timeline_add_track:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형 (GES_TRACK_TYPE_AUDIO, GES_TRACK_TYPE_VIDEO 등)
 * @name: 트랙 이름 (NULL일 경우 자동 생성)
 *
 * 타임라인에 새 트랙을 추가합니다. 트랙은 비디오, 오디오, 텍스트 등의 유형이 될 수 있습니다.
 * 무제한 트랙 지원을 위해 필요한 내부 구조를 생성합니다.
 *
 * Returns: 새로 생성된 트랙 객체 또는 실패 시 NULL
 */
BlouEditTimelineTrack *
blouedit_timeline_add_track (BlouEditTimeline *timeline, GESTrackType track_type, const gchar *name)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  /* 타임라인 구조체 가져오기 */
  /* 타임라인 구조체가 private이므로 실제 구현에서는 더 많은 헤더 파일과 수정이 필요할 수 있음 */
  
  /* GES 트랙 생성 */
  GESTrack *ges_track = ges_track_new (track_type);
  if (!ges_track) {
    g_warning ("Failed to create new GES track");
    return NULL;
  }
  
  /* 새로운 트랙 객체 생성 */
  BlouEditTimelineTrack *track = g_new0 (BlouEditTimelineTrack, 1);
  if (!track) {
    g_object_unref (ges_track);
    return NULL;
  }
  
  /* 트랙 속성 설정 */
  track->ges_track = ges_track;
  
  if (name) {
    track->name = g_strdup (name);
  } else {
    /* 자동 생성된 이름 */
    if (track_type == GES_TRACK_TYPE_AUDIO) {
      gint audio_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_AUDIO);
      track->name = g_strdup_printf (_("Audio %d"), audio_count + 1);
    } else if (track_type == GES_TRACK_TYPE_VIDEO) {
      gint video_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_VIDEO);
      track->name = g_strdup_printf (_("Video %d"), video_count + 1);
    } else {
      track->name = g_strdup_printf (_("Track %d"), g_slist_length (timeline->tracks) + 1);
    }
  }
  
  /* 기본 트랙 속성 설정 */
  track->folded = FALSE;
  track->height = timeline->default_track_height;
  track->folded_height = timeline->folded_track_height;
  track->visible = TRUE;
  track->height_resizing = FALSE;
  
  /* 기본 색상 설정 (미디어 유형에 따라 다름) */
  if (track_type == GES_TRACK_TYPE_AUDIO) {
    gdk_rgba_parse (&track->color, "#5588CC");
  } else if (track_type == GES_TRACK_TYPE_VIDEO) {
    gdk_rgba_parse (&track->color, "#CC5588");
  } else {
    gdk_rgba_parse (&track->color, "#CCCCCC");
  }
  
  /* 트랙을 GES 타임라인에 추가 */
  if (!ges_timeline_add_track (timeline->ges_timeline, ges_track)) {
    g_warning ("Failed to add track to GES timeline");
    g_free (track->name);
    g_free (track);
    g_object_unref (ges_track);
    return NULL;
  }
  
  /* 트랙 목록에 추가 */
  timeline->tracks = g_slist_append (timeline->tracks, track);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
  
  return track;
}

/**
 * blouedit_timeline_remove_track:
 * @timeline: 타임라인 객체
 * @track: 제거할 트랙 객체
 *
 * 타임라인에서 트랙을 제거합니다.
 */
void
blouedit_timeline_remove_track (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return;
  }
  
  /* 선택된 트랙인 경우 선택 해제 */
  if (timeline->selected_track == track) {
    timeline->selected_track = NULL;
  }
  
  /* 크기 조절 중인 트랙인 경우 크기 조절 취소 */
  if (timeline->resizing_track == track) {
    timeline->is_resizing_track = FALSE;
    timeline->resizing_track = NULL;
  }
  
  /* 순서 변경 중인 트랙인 경우 순서 변경 취소 */
  if (timeline->reordering_track == track) {
    timeline->is_reordering_track = FALSE;
    timeline->reordering_track = NULL;
  }
  
  /* GES 타임라인에서 트랙 제거 */
  ges_timeline_remove_track (timeline->ges_timeline, track->ges_track);
  
  /* 트랙 목록에서 제거 */
  timeline->tracks = g_slist_remove (timeline->tracks, track);
  
  /* 트랙 객체 해제 */
  track_free (track);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_track_count:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형 또는 0 (모든 트랙)
 *
 * 지정된 유형의 트랙 수를 반환합니다.
 *
 * Returns: 트랙 수
 */
gint
blouedit_timeline_get_track_count (BlouEditTimeline *timeline, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  /* 트랙 유형이 0이면 모든 트랙 반환 */
  if (track_type == 0) {
    return g_slist_length (timeline->tracks);
  }
  
  /* 특정 유형의 트랙 개수 계산 */
  gint count = 0;
  GSList *l;
  
  for (l = timeline->tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)l->data;
    if (ges_track_get_track_type (track->ges_track) == track_type) {
      count++;
    }
  }
  
  return count;
}

/**
 * blouedit_timeline_get_track_by_index:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형 또는 0 (모든 트랙)
 * @index: 트랙 인덱스
 *
 * 지정된 유형과 인덱스에 해당하는 트랙을 반환합니다.
 *
 * Returns: 트랙 객체 또는 NULL
 */
BlouEditTimelineTrack *
blouedit_timeline_get_track_by_index (BlouEditTimeline *timeline, GESTrackType track_type, gint index)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (index >= 0, NULL);
  
  /* 트랙 유형이 0이면 모든 트랙에서 검색 */
  if (track_type == 0) {
    if (index >= g_slist_length (timeline->tracks)) {
      return NULL;
    }
    
    return (BlouEditTimelineTrack *)g_slist_nth_data (timeline->tracks, index);
  }
  
  /* 특정 유형의 트랙 검색 */
  gint count = 0;
  GSList *l;
  
  for (l = timeline->tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)l->data;
    if (ges_track_get_track_type (track->ges_track) == track_type) {
      if (count == index) {
        return track;
      }
      count++;
    }
  }
  
  return NULL;
}

/**
 * blouedit_timeline_create_default_tracks:
 * @timeline: 타임라인 객체
 *
 * 타임라인에 기본 트랙(비디오 1개, 오디오 1개)을 생성합니다.
 */
void
blouedit_timeline_create_default_tracks (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 기본 비디오 트랙 생성 */
  blouedit_timeline_add_track (timeline, GES_TRACK_TYPE_VIDEO, _("Video 1"));
  
  /* 기본 오디오 트랙 생성 */
  blouedit_timeline_add_track (timeline, GES_TRACK_TYPE_AUDIO, _("Audio 1"));
}

/**
 * blouedit_timeline_is_track_at_max:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형
 *
 * 지정된 유형의 트랙이 최대 수에 도달했는지 확인합니다.
 *
 * Returns: %TRUE 트랙이 최대 수에 도달한 경우, %FALSE 그렇지 않은 경우
 */
gboolean
blouedit_timeline_is_track_at_max (BlouEditTimeline *timeline, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  gint max_tracks = blouedit_timeline_get_max_tracks (timeline, track_type);
  gint current_count = blouedit_timeline_get_track_count (timeline, track_type);
  
  return (current_count >= max_tracks);
}

/**
 * blouedit_timeline_get_track_layer_for_clip:
 * @timeline: 타임라인 객체
 * @clip: GES 클립 객체
 * @track_type: 트랙 유형
 *
 * 클립이 사용하는 레이어 인덱스를 계산합니다.
 * 이는 트랙에서 클립이 표시되는 위치를 결정하는 데 사용됩니다.
 *
 * Returns: 레이어 인덱스
 */
gint
blouedit_timeline_get_track_layer_for_clip (BlouEditTimeline *timeline, GESClip *clip, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  g_return_val_if_fail (GES_IS_CLIP (clip), 0);
  
  /* 
   * 이 함수는 복잡한 구현이 필요하지만, 프레임워크 예시로 간단한 구현만 제공합니다.
   * 실제 구현에서는 클립 및 트랙과 레이어 간의 관계를 더 세밀하게 처리해야 합니다.
   */
  
  return ges_clip_get_layer_priority (clip);
}

/**
 * blouedit_timeline_get_max_tracks:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형
 *
 * 지정된 유형의 최대 트랙 수를 반환합니다.
 * 현재는 논리적 제한이 없지만, UI 성능 문제로 인해 실용적인 제한이 있을 수 있습니다.
 *
 * Returns: 최대 트랙 수
 */
gint
blouedit_timeline_get_max_tracks (BlouEditTimeline *timeline, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  /* 
   * 실제 구현에서는 성능 및 메모리 고려사항에 따라 제한될 수 있지만,
   * 이 예시에서는 높은 값을 사용하여 "무제한" 트랙을 시뮬레이션합니다.
   */
  
  /* 비디오, 오디오 트랙은 99개로 제한 (실질적으로 무제한) */
  if (track_type == GES_TRACK_TYPE_VIDEO || track_type == GES_TRACK_TYPE_AUDIO) {
    return 99;
  }
  
  /* 기타 트랙 유형은 10개로 제한 */
  return 10;
}

/**
 * blouedit_timeline_show_track_controls:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 * @x: 컨트롤을 표시할 X 좌표
 * @y: 컨트롤을 표시할 Y 좌표
 *
 * 트랙 컨트롤 팝업 메뉴를 표시합니다.
 * 트랙을 오른쪽 클릭했을 때 트랙 관련 작업을 수행할 수 있는 메뉴를 제공합니다.
 */
void
blouedit_timeline_show_track_controls (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gdouble x, gdouble y)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 
   * 이 함수는 실제 구현에서 GTK+ 메뉴를 생성하고 표시해야 합니다.
   * 이 코드 분할 예제에서는 기본 구조만 제공합니다.
   */
}

/**
 * blouedit_timeline_show_track_properties:
 * @timeline: 타임라인 객체
 *
 * 트랙 속성 대화상자를 표시합니다.
 * 선택된 트랙의 이름, 색상 등 다양한 속성을 변경할 수 있습니다.
 */
void
blouedit_timeline_show_track_properties (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  // 대화상자 생성
  GtkWidget *dialog = gtk_dialog_new_with_buttons (_("Track Properties"),
                                                   GTK_WINDOW (gtk_widget_get_root (GTK_WIDGET (timeline))),
                                                   GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                   _("Close"), GTK_RESPONSE_CLOSE,
                                                   NULL);
  
  // 대화상자 크기 설정
  gtk_window_set_default_size (GTK_WINDOW (dialog), 500, 400);
  
  // 대화상자 컨텐츠 영역 가져오기
  GtkWidget *content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
  gtk_widget_set_margin (content_area, 12);
  gtk_box_set_spacing (GTK_BOX (content_area), 6);
  
  // 메인 박스 생성
  GtkWidget *main_box = gtk_box_new (GTK_ORIENTATION_VERTICAL, 12);
  gtk_box_append (GTK_BOX (content_area), main_box);
  
  // 트랙 목록 레이블
  GtkWidget *tracks_label = gtk_label_new (_("Timeline Tracks"));
  gtk_widget_add_css_class (tracks_label, "title-3");
  gtk_widget_set_halign (tracks_label, GTK_ALIGN_START);
  gtk_box_append (GTK_BOX (main_box), tracks_label);
  
  // 트랙 리스트 스크롤 윈도우
  GtkWidget *scrolled_window = gtk_scrolled_window_new ();
  gtk_scrolled_window_set_min_content_height (GTK_SCROLLED_WINDOW (scrolled_window), 200);
  gtk_widget_set_vexpand (scrolled_window, TRUE);
  gtk_box_append (GTK_BOX (main_box), scrolled_window);
  
  // 트랙 리스트뷰 생성
  GtkWidget *listview = gtk_list_box_new ();
  gtk_list_box_set_selection_mode (GTK_LIST_BOX (listview), GTK_SELECTION_SINGLE);
  gtk_scrolled_window_set_child (GTK_SCROLLED_WINDOW (scrolled_window), listview);
  
  // 트랙 목록 채우기
  GSList *l;
  for (l = timeline->tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)l->data;
    GESTrackType track_type = ges_track_get_track_type (track->ges_track);
    
    // 트랙 행 위젯 생성
    GtkWidget *row_box = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_widget_set_margin (row_box, 6);
    
    // 색상 표시
    GtkWidget *color_button = gtk_color_button_new_with_rgba (&track->color);
    gtk_box_append (GTK_BOX (row_box), color_button);
    
    // 이름 및 타입 표시
    gchar *type_str = track_type == GES_TRACK_TYPE_VIDEO ? _("Video") : _("Audio");
    gchar *label_text = g_strdup_printf ("%s (%s)", track->name, type_str);
    GtkWidget *label = gtk_label_new (label_text);
    g_free (label_text);
    gtk_widget_set_halign (label, GTK_ALIGN_START);
    gtk_widget_set_hexpand (label, TRUE);
    gtk_box_append (GTK_BOX (row_box), label);
    
    // 음소거/솔로 토글 추가
    if (track_type == GES_TRACK_TYPE_AUDIO) {
      GtkWidget *mute_toggle = gtk_check_button_new_with_label (_("Mute"));
      gtk_check_button_set_active (GTK_CHECK_BUTTON (mute_toggle), track->muted);
      g_object_set_data (G_OBJECT (mute_toggle), "track", track);
      g_signal_connect (mute_toggle, "toggled", G_CALLBACK (on_track_mute_toggled), timeline);
      gtk_box_append (GTK_BOX (row_box), mute_toggle);
      
      GtkWidget *solo_toggle = gtk_check_button_new_with_label (_("Solo"));
      gtk_check_button_set_active (GTK_CHECK_BUTTON (solo_toggle), track->solo);
      g_object_set_data (G_OBJECT (solo_toggle), "track", track);
      g_signal_connect (solo_toggle, "toggled", G_CALLBACK (on_track_solo_toggled), timeline);
      gtk_box_append (GTK_BOX (row_box), solo_toggle);
    }
    
    // 잠금 토글
    GtkWidget *lock_toggle = gtk_check_button_new_with_label (_("Lock"));
    gtk_check_button_set_active (GTK_CHECK_BUTTON (lock_toggle), track->locked);
    g_object_set_data (G_OBJECT (lock_toggle), "track", track);
    g_signal_connect (lock_toggle, "toggled", G_CALLBACK (on_track_lock_toggled), timeline);
    gtk_box_append (GTK_BOX (row_box), lock_toggle);
    
    // 리스트에 행 추가
    GtkWidget *row = gtk_list_box_row_new ();
    gtk_list_box_row_set_child (GTK_LIST_BOX_ROW (row), row_box);
    g_object_set_data (G_OBJECT (row), "track", track);
    gtk_list_box_append (GTK_LIST_BOX (listview), row);
  }
  
  // 트랙 속성 컨트롤을 위한 버튼 상자
  GtkWidget *button_box = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 6);
  gtk_widget_set_margin_top (button_box, 12);
  gtk_box_append (GTK_BOX (main_box), button_box);
  
  // 트랙 추가 버튼
  GtkWidget *add_button = gtk_button_new_with_label (_("Add Track"));
  g_signal_connect_swapped (add_button, "clicked", G_CALLBACK (on_track_add_button_clicked), timeline);
  gtk_box_append (GTK_BOX (button_box), add_button);
  
  // 트랙 제거 버튼
  GtkWidget *remove_button = gtk_button_new_with_label (_("Remove Track"));
  g_object_set_data (G_OBJECT (remove_button), "listview", listview);
  g_signal_connect (remove_button, "clicked", G_CALLBACK (on_track_remove_button_clicked), timeline);
  gtk_box_append (GTK_BOX (button_box), remove_button);
  
  // 트랙 이동 버튼
  GtkWidget *move_up_button = gtk_button_new_with_label (_("Move Up"));
  g_object_set_data (G_OBJECT (move_up_button), "listview", listview);
  g_signal_connect (move_up_button, "clicked", G_CALLBACK (on_track_move_up_button_clicked), timeline);
  gtk_box_append (GTK_BOX (button_box), move_up_button);
  
  GtkWidget *move_down_button = gtk_button_new_with_label (_("Move Down"));
  g_object_set_data (G_OBJECT (move_down_button), "listview", listview);
  g_signal_connect (move_down_button, "clicked", G_CALLBACK (on_track_move_down_button_clicked), timeline);
  gtk_box_append (GTK_BOX (button_box), move_down_button);
  
  // 선택된 트랙 속성 편집 섹션
  GtkWidget *properties_frame = gtk_frame_new (_("Selected Track Properties"));
  gtk_widget_set_margin_top (properties_frame, 12);
  gtk_box_append (GTK_BOX (main_box), properties_frame);
  
  GtkWidget *properties_grid = gtk_grid_new ();
  gtk_widget_set_margin (properties_grid, 12);
  gtk_grid_set_row_spacing (GTK_GRID (properties_grid), 6);
  gtk_grid_set_column_spacing (GTK_GRID (properties_grid), 12);
  gtk_frame_set_child (GTK_FRAME (properties_frame), properties_grid);
  
  // 트랙 이름 편집
  GtkWidget *name_label = gtk_label_new_with_mnemonic (_("_Name:"));
  gtk_widget_set_halign (name_label, GTK_ALIGN_START);
  gtk_grid_attach (GTK_GRID (properties_grid), name_label, 0, 0, 1, 1);
  
  GtkWidget *name_entry = gtk_entry_new ();
  gtk_widget_set_hexpand (name_entry, TRUE);
  gtk_grid_attach (GTK_GRID (properties_grid), name_entry, 1, 0, 1, 1);
  
  // 트랙 색상 편집
  GtkWidget *color_label = gtk_label_new_with_mnemonic (_("_Color:"));
  gtk_widget_set_halign (color_label, GTK_ALIGN_START);
  gtk_grid_attach (GTK_GRID (properties_grid), color_label, 0, 1, 1, 1);
  
  GtkWidget *color_button = gtk_color_button_new ();
  gtk_grid_attach (GTK_GRID (properties_grid), color_button, 1, 1, 1, 1);
  
  // 트랙 높이 편집
  GtkWidget *height_label = gtk_label_new_with_mnemonic (_("_Height:"));
  gtk_widget_set_halign (height_label, GTK_ALIGN_START);
  gtk_grid_attach (GTK_GRID (properties_grid), height_label, 0, 2, 1, 1);
  
  GtkWidget *height_spin = gtk_spin_button_new_with_range (20, 200, 5);
  gtk_grid_attach (GTK_GRID (properties_grid), height_spin, 1, 2, 1, 1);
  
  // 적용 버튼
  GtkWidget *apply_button = gtk_button_new_with_label (_("Apply Changes"));
  gtk_widget_set_margin_top (apply_button, 6);
  gtk_grid_attach (GTK_GRID (properties_grid), apply_button, 0, 3, 2, 1);
  
  // 편집 위젯의 데이터 구조체 생성
  TrackPropertyData *prop_data = g_new0 (TrackPropertyData, 1);
  prop_data->timeline = timeline;
  prop_data->listview = listview;
  prop_data->name_entry = name_entry;
  prop_data->color_button = color_button;
  prop_data->height_spin = height_spin;
  
  g_object_set_data_full (G_OBJECT (dialog), "property-data", prop_data, g_free);
  
  // 선택 변경 시그널 연결
  g_signal_connect (listview, "row-selected", G_CALLBACK (on_track_row_selected), prop_data);
  
  // 속성 적용 버튼 시그널 연결
  g_signal_connect (apply_button, "clicked", G_CALLBACK (on_track_properties_apply), prop_data);
  
  // 대화상자 표시
  gtk_widget_show (dialog);
  
  // 응답 시그널 핸들러 연결
  g_signal_connect (dialog, "response", G_CALLBACK (gtk_window_destroy), NULL);
}

// 트랙 속성 데이터 구조체
typedef struct {
  BlouEditTimeline *timeline;
  GtkWidget *listview;
  GtkWidget *name_entry;
  GtkWidget *color_button;
  GtkWidget *height_spin;
  BlouEditTimelineTrack *current_track;
} TrackPropertyData;

// 트랙 뮤트 토글 콜백
static void
on_track_mute_toggled (GtkToggleButton *toggle, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  BlouEditTimelineTrack *track = g_object_get_data (G_OBJECT (toggle), "track");
  
  if (track) {
    blouedit_timeline_set_track_muted (timeline, track, gtk_toggle_button_get_active (toggle));
  }
}

// 트랙 솔로 토글 콜백
static void
on_track_solo_toggled (GtkToggleButton *toggle, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  BlouEditTimelineTrack *track = g_object_get_data (G_OBJECT (toggle), "track");
  
  if (track) {
    blouedit_timeline_set_track_solo (timeline, track, gtk_toggle_button_get_active (toggle));
  }
}

// 트랙 잠금 토글 콜백
static void
on_track_lock_toggled (GtkToggleButton *toggle, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  BlouEditTimelineTrack *track = g_object_get_data (G_OBJECT (toggle), "track");
  
  if (track) {
    blouedit_timeline_set_track_locked (timeline, track, gtk_toggle_button_get_active (toggle));
  }
}

// 트랙 추가 버튼 클릭 콜백
static void
on_track_add_button_clicked (BlouEditTimeline *timeline)
{
  blouedit_timeline_show_add_track_dialog (timeline);
}

// 트랙 제거 버튼 클릭 콜백
static void
on_track_remove_button_clicked (GtkButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  GtkWidget *listview = GTK_WIDGET (g_object_get_data (G_OBJECT (button), "listview"));
  GtkListBoxRow *selected_row = gtk_list_box_get_selected_row (GTK_LIST_BOX (listview));
  
  if (selected_row) {
    BlouEditTimelineTrack *track = g_object_get_data (G_OBJECT (selected_row), "track");
    
    if (track) {
      // 삭제 확인 대화상자
      GtkWidget *confirm_dialog = gtk_message_dialog_new (
        GTK_WINDOW (gtk_widget_get_root (GTK_WIDGET (button))),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        GTK_MESSAGE_WARNING,
        GTK_BUTTONS_YES_NO,
        _("Remove track '%s'?"),
        track->name
      );
      
      gtk_message_dialog_format_secondary_text (
        GTK_MESSAGE_DIALOG (confirm_dialog),
        _("Removing this track will delete all clips in it. This action cannot be undone.")
      );
      
      g_signal_connect (confirm_dialog, "response", G_CALLBACK (on_remove_track_confirm), track);
      g_object_set_data (G_OBJECT (confirm_dialog), "timeline", timeline);
      g_object_set_data (G_OBJECT (confirm_dialog), "listview", listview);
      
      gtk_widget_show (confirm_dialog);
    }
  }
}

// 트랙 제거 확인 콜백
static void
on_remove_track_confirm (GtkDialog *dialog, gint response_id, gpointer user_data)
{
  if (response_id == GTK_RESPONSE_YES) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)user_data;
    BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (dialog), "timeline");
    GtkWidget *listview = g_object_get_data (G_OBJECT (dialog), "listview");
    
    // 트랙 제거
    blouedit_timeline_remove_track (timeline, track);
    
    // 리스트뷰 다시 로드
    gtk_list_box_remove_all (GTK_LIST_BOX (listview));
    GSList *l;
    for (l = timeline->tracks; l != NULL; l = l->next) {
      BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)l->data;
      
      // TODO: 리스트뷰 다시 채우기 (기존 코드와 동일)
    }
  }
  
  gtk_window_destroy (GTK_WINDOW (dialog));
}

// 트랙 위로 이동 버튼 클릭 콜백
static void
on_track_move_up_button_clicked (GtkButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  GtkWidget *listview = GTK_WIDGET (g_object_get_data (G_OBJECT (button), "listview"));
  GtkListBoxRow *selected_row = gtk_list_box_get_selected_row (GTK_LIST_BOX (listview));
  
  if (selected_row) {
    BlouEditTimelineTrack *track = g_object_get_data (G_OBJECT (selected_row), "track");
    
    if (track) {
      blouedit_timeline_move_track_up (timeline, track);
      
      // 리스트뷰 다시 로드
      // TODO: 리스트뷰 업데이트
    }
  }
}

// 트랙 아래로 이동 버튼 클릭 콜백
static void
on_track_move_down_button_clicked (GtkButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  GtkWidget *listview = GTK_WIDGET (g_object_get_data (G_OBJECT (button), "listview"));
  GtkListBoxRow *selected_row = gtk_list_box_get_selected_row (GTK_LIST_BOX (listview));
  
  if (selected_row) {
    BlouEditTimelineTrack *track = g_object_get_data (G_OBJECT (selected_row), "track");
    
    if (track) {
      blouedit_timeline_move_track_down (timeline, track);
      
      // 리스트뷰 다시 로드
      // TODO: 리스트뷰 업데이트
    }
  }
}

// 트랙 행 선택 콜백
static void
on_track_row_selected (GtkListBox *box, GtkListBoxRow *row, gpointer user_data)
{
  TrackPropertyData *data = (TrackPropertyData *)user_data;
  
  if (row) {
    BlouEditTimelineTrack *track = g_object_get_data (G_OBJECT (row), "track");
    
    if (track) {
      data->current_track = track;
      
      // 현재 속성 표시
      gtk_entry_set_text (GTK_ENTRY (data->name_entry), track->name);
      gtk_color_chooser_set_rgba (GTK_COLOR_CHOOSER (data->color_button), &track->color);
      gtk_spin_button_set_value (GTK_SPIN_BUTTON (data->height_spin), track->height);
    }
  } else {
    data->current_track = NULL;
  }
}

// 트랙 속성 적용 버튼 콜백
static void
on_track_properties_apply (GtkButton *button, gpointer user_data)
{
  TrackPropertyData *data = (TrackPropertyData *)user_data;
  
  if (data->current_track) {
    // 이름 변경
    const gchar *new_name = gtk_entry_get_text (GTK_ENTRY (data->name_entry));
    blouedit_timeline_set_track_name (data->timeline, data->current_track, new_name);
    
    // 색상 변경
    GdkRGBA new_color;
    gtk_color_chooser_get_rgba (GTK_COLOR_CHOOSER (data->color_button), &new_color);
    data->current_track->color = new_color;
    
    // 높이 변경
    gint new_height = gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON (data->height_spin));
    blouedit_timeline_set_track_height (data->timeline, data->current_track, new_height);
    
    // UI 다시 그리기
    gtk_widget_queue_draw (GTK_WIDGET (data->timeline));
  }
}

/**
 * blouedit_timeline_show_add_track_dialog:
 * @timeline: 타임라인 객체
 *
 * 새 트랙을 추가하기 위한 대화상자를 표시합니다.
 * 사용자는 트랙 유형(비디오/오디오)과 이름을 선택할 수 있습니다.
 * 무제한 트랙 지원을 위해 UI가 개선되었습니다.
 */
void
blouedit_timeline_show_add_track_dialog (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  // 대화상자 생성
  GtkWidget *dialog = gtk_dialog_new_with_buttons (_("Add New Track"),
                                                  GTK_WINDOW (gtk_widget_get_root (GTK_WIDGET (timeline))),
                                                  GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                  _("Cancel"), GTK_RESPONSE_CANCEL,
                                                  _("Add"), GTK_RESPONSE_ACCEPT,
                                                  NULL);
  
  // 대화상자 컨텐츠 영역 가져오기
  GtkWidget *content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
  gtk_widget_set_margin (content_area, 12);
  gtk_box_set_spacing (GTK_BOX (content_area), 6);
  
  // 그리드 레이아웃 생성
  GtkWidget *grid = gtk_grid_new ();
  gtk_grid_set_row_spacing (GTK_GRID (grid), 6);
  gtk_grid_set_column_spacing (GTK_GRID (grid), 12);
  gtk_box_append (GTK_BOX (content_area), grid);
  
  // 트랙 이름 입력 필드
  GtkWidget *name_label = gtk_label_new_with_mnemonic (_("Track _Name:"));
  gtk_widget_set_halign (name_label, GTK_ALIGN_START);
  GtkWidget *name_entry = gtk_entry_new ();
  gtk_label_set_mnemonic_widget (GTK_LABEL (name_label), name_entry);
  gtk_grid_attach (GTK_GRID (grid), name_label, 0, 0, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), name_entry, 1, 0, 1, 1);
  
  // 라디오 버튼 그룹 (트랙 유형 선택)
  GtkWidget *video_radio = gtk_check_button_new_with_mnemonic (_("_Video Track"));
  GtkWidget *audio_radio = gtk_check_button_new_with_mnemonic (_("_Audio Track"));
  gtk_check_button_set_group (GTK_CHECK_BUTTON (audio_radio), GTK_CHECK_BUTTON (video_radio));
  gtk_check_button_set_active (GTK_CHECK_BUTTON (video_radio), TRUE);

  // 현재 트랙 수 표시 레이블 추가
  gint video_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_VIDEO);
  gint audio_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_AUDIO);
  gchar *track_count_text = g_strdup_printf(_("Current tracks: %d video, %d audio"), video_count, audio_count);
  GtkWidget *track_count_label = gtk_label_new(track_count_text);
  g_free(track_count_text);
  gtk_widget_set_margin_top(track_count_label, 12);
  gtk_widget_set_margin_bottom(track_count_label, 6);

  // 위젯을 그리드에 추가
  gtk_grid_attach (GTK_GRID (grid), video_radio, 0, 1, 2, 1);
  gtk_grid_attach (GTK_GRID (grid), audio_radio, 0, 2, 2, 1);
  gtk_grid_attach (GTK_GRID (grid), track_count_label, 0, 3, 2, 1);

  // 색상 선택기 추가
  GtkWidget *color_label = gtk_label_new_with_mnemonic (_("Track _Color:"));
  gtk_widget_set_halign (color_label, GTK_ALIGN_START);
  GtkWidget *color_button = gtk_color_button_new ();
  
  // 기본 색상 설정 (비디오 트랙용)
  GdkRGBA color;
  gdk_rgba_parse (&color, "#CC5588");
  gtk_color_chooser_set_rgba (GTK_COLOR_CHOOSER (color_button), &color);
  
  gtk_grid_attach (GTK_GRID (grid), color_label, 0, 4, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), color_button, 1, 4, 1, 1);
  
  // 트랙 유형과 색상 연결
  g_signal_connect (video_radio, "toggled", G_CALLBACK (on_track_type_toggled), name_entry);
  g_signal_connect (video_radio, "toggled", G_CALLBACK (on_track_type_color_changed), color_button);
  
  // 비디오 트랙이 기본 선택되어 있으므로, 초기 이름 설정
  gtk_entry_set_text (GTK_ENTRY (name_entry), g_strdup_printf (_("Video %d"), video_count + 1));
  
  // 대화상자 표시
  gtk_widget_show (dialog);
  
  // 응답 시그널 핸들러 연결
  g_signal_connect (dialog, "response", G_CALLBACK (on_add_track_dialog_response), timeline);
  
  // 추가 데이터 설정을 위한 구조체 생성 및 연결
  AddTrackDialogData *data = g_new0 (AddTrackDialogData, 1);
  data->timeline = timeline;
  data->name_entry = name_entry;
  data->video_radio = video_radio;
  data->color_button = color_button;
  
  g_object_set_data_full (G_OBJECT (dialog), "dialog-data", data, g_free);
}

// 추가: 대화상자 응답 처리 함수
static void
on_add_track_dialog_response (GtkDialog *dialog, gint response_id, gpointer user_data)
{
  if (response_id == GTK_RESPONSE_ACCEPT) {
    BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
    AddTrackDialogData *data = g_object_get_data (G_OBJECT (dialog), "dialog-data");
    
    // 사용자가 '추가' 버튼을 클릭한 경우
    const gchar *name = gtk_entry_get_text (GTK_ENTRY (data->name_entry));
    
    // 트랙 유형 결정
    GESTrackType track_type = gtk_check_button_get_active (GTK_CHECK_BUTTON (data->video_radio)) 
                            ? GES_TRACK_TYPE_VIDEO : GES_TRACK_TYPE_AUDIO;
    
    // 색상 가져오기
    GdkRGBA selected_color;
    gtk_color_chooser_get_rgba (GTK_COLOR_CHOOSER (data->color_button), &selected_color);
    
    // 새 트랙 추가
    BlouEditTimelineTrack *new_track = blouedit_timeline_add_track (timeline, track_type, name);
    
    if (new_track) {
      // 색상 설정
      new_track->color = selected_color;
      
      // 트랙 추가 알림
      gchar *msg = g_strdup_printf(_("Added new %s track: %s"), 
                                  track_type == GES_TRACK_TYPE_VIDEO ? "video" : "audio", 
                                  name);
      blouedit_timeline_show_message(timeline, msg);
      g_free(msg);
    }
  }
  
  // 대화상자 파괴
  gtk_window_destroy (GTK_WINDOW (dialog));
}

/**
 * on_track_type_toggled:
 * @togglebutton: 트랙 유형 토글 버튼
 * @user_data: 이름 입력 위젯
 *
 * 트랙 유형이 변경되었을 때 기본 이름을 업데이트합니다.
 */
static void
on_track_type_toggled (GtkToggleButton *togglebutton, gpointer user_data)
{
  GtkWidget *name_entry = GTK_WIDGET (user_data);
  GtkWidget *dialog = gtk_widget_get_ancestor (GTK_WIDGET (togglebutton), GTK_TYPE_DIALOG);
  BlouEditTimeline *timeline = NULL;
  
  // 대화상자 데이터에서 타임라인 가져오기
  AddTrackDialogData *data = g_object_get_data (G_OBJECT (dialog), "dialog-data");
  if (data) {
    timeline = data->timeline;
  }
  
  if (!timeline) {
    return;
  }
  
  // 현재 트랙 유형에 따라 적절한 이름 설정
  if (gtk_toggle_button_get_active (togglebutton)) {
    // 비디오 트랙
    gint video_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_VIDEO);
    gtk_entry_set_text (GTK_ENTRY (name_entry), g_strdup_printf (_("Video %d"), video_count + 1));
  } else {
    // 오디오 트랙
    gint audio_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_AUDIO);
    gtk_entry_set_text (GTK_ENTRY (name_entry), g_strdup_printf (_("Audio %d"), audio_count + 1));
  }
}

/**
 * on_track_type_color_changed:
 * @radio: 트랙 유형 라디오 버튼
 * @color_button: 색상 선택 버튼
 *
 * 트랙 유형이 변경되었을 때 색상 기본값을 업데이트합니다.
 */
static void
on_track_type_color_changed (GtkToggleButton *radio, GtkWidget *color_button)
{
  if (gtk_toggle_button_get_active (radio)) {
    // 비디오 트랙 기본 색상
    GdkRGBA color;
    gdk_rgba_parse (&color, "#CC5588");
    gtk_color_chooser_set_rgba (GTK_COLOR_CHOOSER (color_button), &color);
  } else {
    // 오디오 트랙 기본 색상
    GdkRGBA color;
    gdk_rgba_parse (&color, "#5588CC");
    gtk_color_chooser_set_rgba (GTK_COLOR_CHOOSER (color_button), &color);
  }
}

/**
 * blouedit_timeline_start_track_reorder:
 * @timeline: 타임라인 객체
 * @track: 재정렬할 트랙
 * @y: 시작 Y 좌표
 *
 * 트랙 재정렬 작업을 시작합니다.
 * 사용자가 트랙을 드래그하여 순서를 변경할 수 있게 합니다.
 */
void
blouedit_timeline_start_track_reorder (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint y)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 현재 재정렬 중인 경우 이전 작업 취소 */
  if (timeline->is_reordering_track) {
    blouedit_timeline_end_track_reorder (timeline);
  }
  
  /* 트랙 인덱스 찾기 */
  gint index = g_slist_index (timeline->tracks, track);
  if (index < 0) {
    g_warning ("Track not found in timeline");
    return;
  }
  
  /* 재정렬 상태 설정 */
  timeline->is_reordering_track = TRUE;
  timeline->reordering_track = track;
  timeline->reorder_start_y = y;
  timeline->reorder_original_index = index;
  timeline->reorder_current_index = index;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_reorder_track_to:
 * @timeline: 타임라인 객체
 * @y: 현재 Y 좌표
 *
 * 재정렬 중인 트랙을 새 위치로 이동합니다.
 * 사용자가 드래그하는 동안 트랙 위치를 업데이트합니다.
 */
void
blouedit_timeline_reorder_track_to (BlouEditTimeline *timeline, gint y)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 재정렬 중이 아니면 아무것도 하지 않음 */
  if (!timeline->is_reordering_track || !timeline->reordering_track) {
    return;
  }
  
  /* 트랙 수 확인 */
  gint track_count = g_slist_length (timeline->tracks);
  if (track_count <= 1) {
    return;
  }
  
  /* Y 좌표를 기준으로 새 인덱스 계산 */
  /* 이 부분은 실제 구현에서 더 복잡할 수 있으며, 트랙 높이 및 스크롤 위치를 고려해야 합니다. */
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_end_track_reorder:
 * @timeline: 타임라인 객체
 *
 * 트랙 재정렬 작업을 완료합니다.
 * 트랙을 새 위치에 적용하고 재정렬 상태를 초기화합니다.
 */
void
blouedit_timeline_end_track_reorder (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* 재정렬 중이 아니면 아무것도 하지 않음 */
  if (!timeline->is_reordering_track || !timeline->reordering_track) {
    return;
  }
  
  /* 원래 인덱스와 새 인덱스가 다른 경우 트랙 순서 변경 */
  if (timeline->reorder_original_index != timeline->reorder_current_index) {
    /* 트랙 목록에서 제거 */
    timeline->tracks = g_slist_remove (timeline->tracks, timeline->reordering_track);
    
    /* 새 위치에 삽입 */
    timeline->tracks = g_slist_insert (timeline->tracks, timeline->reordering_track, timeline->reorder_current_index);
  }
  
  /* 재정렬 상태 초기화 */
  timeline->is_reordering_track = FALSE;
  timeline->reordering_track = NULL;
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_move_track_up:
 * @timeline: 타임라인 객체
 * @track: 이동할 트랙
 *
 * 트랙을 한 단계 위로 이동합니다.
 * 이미 맨 위에 있는 경우 아무 일도 일어나지 않습니다.
 */
void
blouedit_timeline_move_track_up (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 트랙 인덱스 찾기 */
  gint index = g_slist_index (timeline->tracks, track);
  if (index <= 0) {
    /* 이미 맨 위에 있거나 찾을 수 없음 */
    return;
  }
  
  /* 트랙 목록에서 제거 */
  timeline->tracks = g_slist_remove (timeline->tracks, track);
  
  /* 한 단계 위에 삽입 */
  timeline->tracks = g_slist_insert (timeline->tracks, track, index - 1);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_move_track_down:
 * @timeline: 타임라인 객체
 * @track: 이동할 트랙
 *
 * 트랙을 한 단계 아래로 이동합니다.
 * 이미 맨 아래에 있는 경우 아무 일도 일어나지 않습니다.
 */
void
blouedit_timeline_move_track_down (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 트랙 인덱스 찾기 */
  gint index = g_slist_index (timeline->tracks, track);
  if (index < 0 || index >= g_slist_length (timeline->tracks) - 1) {
    /* 이미 맨 아래에 있거나 찾을 수 없음 */
    return;
  }
  
  /* 트랙 목록에서 제거 */
  timeline->tracks = g_slist_remove (timeline->tracks, track);
  
  /* 한 단계 아래에 삽입 */
  timeline->tracks = g_slist_insert (timeline->tracks, track, index + 1);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_set_track_name:
 * @timeline: 타임라인 객체
 * @track: 이름을 변경할 트랙 객체
 * @name: 새 트랙 이름
 *
 * 트랙의 이름을 사용자 지정 값으로 설정합니다.
 */
void
blouedit_timeline_set_track_name (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, const gchar *name)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  g_return_if_fail (name != NULL);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return;
  }
  
  /* 이전 이름 해제 및 새 이름 설정 */
  g_free (track->name);
  track->name = g_strdup (name);
  
  /* 이벤트 발행: 트랙 이름 변경됨 */
  g_signal_emit_by_name (timeline, "track-name-changed", track);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_track_name:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 *
 * 트랙의 현재 이름을 반환합니다.
 *
 * Returns: 트랙 이름 (해제하지 마세요)
 */
const gchar *
blouedit_timeline_get_track_name (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (track != NULL, NULL);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return NULL;
  }
  
  return track->name;
}

/**
 * blouedit_timeline_set_track_locked:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 * @locked: 잠금 상태 (TRUE: 잠금, FALSE: 잠금 해제)
 *
 * 트랙의 잠금 상태를 설정합니다. 잠긴 트랙은 편집할 수 없습니다.
 */
void
blouedit_timeline_set_track_locked (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gboolean locked)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return;
  }
  
  /* 상태가 변경되지 않으면 아무 작업도 하지 않음 */
  if (track->locked == locked)
    return;
  
  track->locked = locked;
  
  /* GES 트랙에도 잠금 상태 적용 */
  ges_track_set_restriction_caps (track->ges_track, locked ? 
                                  gst_caps_new_empty() : NULL);
  
  /* 이벤트 발행: 트랙 잠금 상태 변경됨 */
  g_signal_emit_by_name (timeline, "track-locked-changed", track);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_track_locked:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 *
 * 트랙의 현재 잠금 상태를 반환합니다.
 *
 * Returns: 트랙이 잠겨 있으면 TRUE, 그렇지 않으면 FALSE
 */
gboolean
blouedit_timeline_get_track_locked (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (track != NULL, FALSE);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return FALSE;
  }
  
  return track->locked;
}

/**
 * blouedit_timeline_set_track_muted:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 * @muted: 음소거 상태 (TRUE: 음소거, FALSE: 소리 재생)
 *
 * 트랙의 음소거 상태를 설정합니다. 음소거된 트랙은 재생 시 소리가 들리지 않습니다.
 */
void
blouedit_timeline_set_track_muted (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gboolean muted)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return;
  }
  
  /* 상태가 변경되지 않으면 아무 작업도 하지 않음 */
  if (track->muted == muted)
    return;
  
  track->muted = muted;
  
  /* GES 트랙에도 음소거 상태 적용 */
  if (ges_track_get_track_type (track->ges_track) == GES_TRACK_TYPE_AUDIO) {
    ges_track_set_mixing (track->ges_track, !muted);
  }
  
  /* 트랙이 솔로 상태이고 음소거되면 솔로 상태 해제 */
  if (muted && track->solo) {
    track->solo = FALSE;
    /* 솔로 상태 변경 이벤트 발행 */
    g_signal_emit_by_name (timeline, "track-solo-changed", track);
  }
  
  /* 음소거 상태 변경 이벤트 발행 */
  g_signal_emit_by_name (timeline, "track-muted-changed", track);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_track_muted:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 *
 * 트랙의 현재 음소거 상태를 반환합니다.
 *
 * Returns: 트랙이 음소거되어 있으면 TRUE, 그렇지 않으면 FALSE
 */
gboolean
blouedit_timeline_get_track_muted (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (track != NULL, FALSE);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return FALSE;
  }
  
  return track->muted;
}

/**
 * blouedit_timeline_set_track_solo:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 * @solo: 솔로 상태 (TRUE: 솔로 활성화, FALSE: 솔로 비활성화)
 *
 * 트랙의 솔로 상태를 설정합니다. 솔로 상태의 트랙만 재생됩니다.
 * 여러 트랙이 솔로 상태일 수 있으며, 솔로 상태의 트랙이 없으면 모든 트랙이 재생됩니다.
 */
void
blouedit_timeline_set_track_solo (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gboolean solo)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return;
  }
  
  /* 상태가 변경되지 않으면 아무 작업도 하지 않음 */
  if (track->solo == solo)
    return;
  
  track->solo = solo;
  
  /* 솔로 모드가 활성화되면 음소거 상태 해제 */
  if (solo && track->muted) {
    track->muted = FALSE;
    /* 음소거 상태 변경 이벤트 발행 */
    g_signal_emit_by_name (timeline, "track-muted-changed", track);
  }
  
  /* 현재 솔로 상태인 트랙 개수 확인 */
  gboolean any_solo = FALSE;
  GSList *l;
  for (l = timeline->tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *t = (BlouEditTimelineTrack *)l->data;
    if (t->solo) {
      any_solo = TRUE;
      break;
    }
  }
  
  /* 솔로 상태 트랙에 따라 다른 트랙의 재생 상태 조정 */
  for (l = timeline->tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *t = (BlouEditTimelineTrack *)l->data;
    if (ges_track_get_track_type (t->ges_track) == GES_TRACK_TYPE_AUDIO) {
      gboolean should_mix = !any_solo || t->solo;
      ges_track_set_mixing (t->ges_track, should_mix);
    }
  }
  
  /* 솔로 상태 변경 이벤트 발행 */
  g_signal_emit_by_name (timeline, "track-solo-changed", track);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_track_solo:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 *
 * 트랙의 현재 솔로 상태를 반환합니다.
 *
 * Returns: 트랙이 솔로 상태이면 TRUE, 그렇지 않으면 FALSE
 */
gboolean
blouedit_timeline_get_track_solo (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (track != NULL, FALSE);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return FALSE;
  }
  
  return track->solo;
}

/**
 * blouedit_track_template_new:
 * @name: 템플릿 이름
 * @type: 템플릿 유형
 * @track_type: GES 트랙 유형
 * @color: 트랙 색상
 * @height: 트랙 높이
 * @description: 템플릿 설명 (선택 사항, NULL일 수 있음)
 *
 * 새 트랙 템플릿을 생성합니다.
 *
 * Returns: 새로 생성된 트랙 템플릿 객체
 */
BlouEditTrackTemplate *
blouedit_track_template_new (const gchar *name, 
                             BlouEditTrackTemplateType type,
                             GESTrackType track_type,
                             const GdkRGBA *color,
                             gint height,
                             const gchar *description)
{
  g_return_val_if_fail (name != NULL, NULL);
  
  BlouEditTrackTemplate *template = g_new0 (BlouEditTrackTemplate, 1);
  
  template->name = g_strdup (name);
  template->type = type;
  template->track_type = track_type;
  
  if (color) {
    template->color = *color;
  } else {
    /* 기본 색상 설정 */
    if (track_type == GES_TRACK_TYPE_AUDIO) {
      gdk_rgba_parse (&template->color, "#5588CC");
    } else if (track_type == GES_TRACK_TYPE_VIDEO) {
      gdk_rgba_parse (&template->color, "#CC5588");
    } else {
      gdk_rgba_parse (&template->color, "#CCCCCC");
    }
  }
  
  template->height = height;
  
  if (description) {
    template->description = g_strdup (description);
  } else {
    template->description = g_strdup ("");
  }
  
  return template;
}

/**
 * blouedit_track_template_free:
 * @template: 해제할 트랙 템플릿
 *
 * 트랙 템플릿의 메모리를 해제합니다.
 */
void
blouedit_track_template_free (BlouEditTrackTemplate *template)
{
  g_return_if_fail (template != NULL);
  
  g_free (template->name);
  g_free (template->description);
  g_free (template);
}

/**
 * blouedit_timeline_save_track_template:
 * @timeline: 타임라인 객체
 * @track: 템플릿으로 저장할 트랙
 * @name: 템플릿의 이름
 * @description: 템플릿 설명 (선택 사항, NULL일 수 있음)
 *
 * 현재 트랙 설정을 템플릿으로 저장합니다.
 *
 * Returns: 성공하면 TRUE, 실패하면 FALSE
 */
gboolean
blouedit_timeline_save_track_template (BlouEditTimeline *timeline,
                                       BlouEditTimelineTrack *track,
                                       const gchar *name,
                                       const gchar *description)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (track != NULL, FALSE);
  g_return_val_if_fail (name != NULL, FALSE);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return FALSE;
  }
  
  /* 트랙 유형 결정 */
  BlouEditTrackTemplateType template_type;
  GESTrackType ges_track_type = ges_track_get_track_type (track->ges_track);
  
  if (ges_track_type == GES_TRACK_TYPE_AUDIO) {
    template_type = BLOUEDIT_TRACK_TEMPLATE_AUDIO;
  } else if (ges_track_type == GES_TRACK_TYPE_VIDEO) {
    template_type = BLOUEDIT_TRACK_TEMPLATE_VIDEO;
  } else if (ges_track_type == (GESTrackType)(GES_TRACK_TYPE_VIDEO | GES_TRACK_TYPE_TEXT)) {
    template_type = BLOUEDIT_TRACK_TEMPLATE_TEXT;
  } else {
    template_type = BLOUEDIT_TRACK_TEMPLATE_CUSTOM;
  }
  
  /* 새 템플릿 생성 */
  BlouEditTrackTemplate *template = blouedit_track_template_new (
    name,
    template_type,
    ges_track_type,
    &track->color,
    track->height,
    description
  );
  
  if (!template) {
    g_warning ("Failed to create track template");
    return FALSE;
  }
  
  /* 템플릿 저장 경로 가져오기 */
  gchar *template_dir = g_build_filename (g_get_user_data_dir (), "blouedit", "track_templates", NULL);
  
  /* 디렉토리가 없으면 생성 */
  if (g_mkdir_with_parents (template_dir, 0755) != 0) {
    g_warning ("Failed to create track templates directory: %s", template_dir);
    g_free (template_dir);
    blouedit_track_template_free (template);
    return FALSE;
  }
  
  /* 템플릿 파일 경로 */
  gchar *safe_name = g_strdup (name);
  /* 파일 이름으로 사용할 수 없는 문자 제거 */
  for (gchar *p = safe_name; *p; p++) {
    if (*p == '/' || *p == '\\' || *p == ':' || *p == '*' || *p == '?' || 
        *p == '"' || *p == '<' || *p == '>' || *p == '|') {
      *p = '_';
    }
  }
  
  gchar *filename = g_strdup_printf ("%s.track", safe_name);
  gchar *template_path = g_build_filename (template_dir, filename, NULL);
  
  g_free (safe_name);
  g_free (filename);
  
  /* 템플릿 파일에 저장 */
  GKeyFile *keyfile = g_key_file_new ();
  
  g_key_file_set_string (keyfile, "TrackTemplate", "Name", template->name);
  g_key_file_set_integer (keyfile, "TrackTemplate", "Type", (gint)template->type);
  g_key_file_set_integer (keyfile, "TrackTemplate", "GESTrackType", (gint)template->track_type);
  
  gchar *color_str = g_strdup_printf ("rgba(%d,%d,%d,%f)",
                                      (gint)(template->color.red * 255),
                                      (gint)(template->color.green * 255),
                                      (gint)(template->color.blue * 255),
                                      template->color.alpha);
  g_key_file_set_string (keyfile, "TrackTemplate", "Color", color_str);
  g_free (color_str);
  
  g_key_file_set_integer (keyfile, "TrackTemplate", "Height", template->height);
  g_key_file_set_string (keyfile, "TrackTemplate", "Description", template->description);
  
  /* 파일에 저장 */
  gchar *keyfile_data = g_key_file_to_data (keyfile, NULL, NULL);
  gboolean success = g_file_set_contents (template_path, keyfile_data, -1, NULL);
  
  g_free (keyfile_data);
  g_key_file_free (keyfile);
  g_free (template_dir);
  g_free (template_path);
  
  if (!success) {
    g_warning ("Failed to save track template to file");
    blouedit_track_template_free (template);
    return FALSE;
  }
  
  /* 템플릿 메모리 해제 */
  blouedit_track_template_free (template);
  
  return TRUE;
}

/**
 * blouedit_timeline_load_track_templates:
 * @timeline: 타임라인 객체
 *
 * 저장된 모든 트랙 템플릿을 로드합니다.
 *
 * Returns: 템플릿 목록 (BlouEditTrackTemplate*의 GSList*). g_slist_free_full(list, (GDestroyNotify)blouedit_track_template_free)로 해제해야 합니다.
 */
GSList *
blouedit_timeline_load_track_templates (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  GSList *templates = NULL;
  
  /* 템플릿 디렉토리 경로 */
  gchar *template_dir = g_build_filename (g_get_user_data_dir (), "blouedit", "track_templates", NULL);
  
  /* 디렉토리가 없으면 빈 목록 반환 */
  if (!g_file_test (template_dir, G_FILE_TEST_IS_DIR)) {
    g_free (template_dir);
    return NULL;
  }
  
  /* 디렉토리 내용 읽기 */
  GDir *dir = g_dir_open (template_dir, 0, NULL);
  if (!dir) {
    g_free (template_dir);
    return NULL;
  }
  
  const gchar *file;
  while ((file = g_dir_read_name (dir)) != NULL) {
    /* .track 확장자 파일만 처리 */
    if (!g_str_has_suffix (file, ".track"))
      continue;
    
    gchar *file_path = g_build_filename (template_dir, file, NULL);
    
    /* 파일 읽기 */
    GKeyFile *keyfile = g_key_file_new ();
    if (g_key_file_load_from_file (keyfile, file_path, G_KEY_FILE_NONE, NULL)) {
      /* 템플릿 정보 가져오기 */
      gchar *name = g_key_file_get_string (keyfile, "TrackTemplate", "Name", NULL);
      gint type_int = g_key_file_get_integer (keyfile, "TrackTemplate", "Type", NULL);
      gint track_type_int = g_key_file_get_integer (keyfile, "TrackTemplate", "GESTrackType", NULL);
      gchar *color_str = g_key_file_get_string (keyfile, "TrackTemplate", "Color", NULL);
      gint height = g_key_file_get_integer (keyfile, "TrackTemplate", "Height", NULL);
      gchar *description = g_key_file_get_string (keyfile, "TrackTemplate", "Description", NULL);
      
      /* 색상 파싱 */
      GdkRGBA color;
      gdk_rgba_parse (&color, color_str);
      g_free (color_str);
      
      /* 템플릿 생성 */
      BlouEditTrackTemplate *template = blouedit_track_template_new (
        name,
        (BlouEditTrackTemplateType)type_int,
        (GESTrackType)track_type_int,
        &color,
        height,
        description
      );
      
      /* 문자열 해제 */
      g_free (name);
      g_free (description);
      
      /* 템플릿 목록에 추가 */
      if (template) {
        templates = g_slist_append (templates, template);
      }
    }
    
    g_key_file_free (keyfile);
    g_free (file_path);
  }
  
  g_dir_close (dir);
  g_free (template_dir);
  
  return templates;
}

/**
 * blouedit_timeline_apply_track_template:
 * @timeline: 타임라인 객체
 * @template: 적용할 트랙 템플릿
 *
 * 템플릿을 사용하여 새 트랙을 생성합니다.
 *
 * Returns: 템플릿으로부터 생성된 새 트랙 또는 NULL
 */
BlouEditTimelineTrack *
blouedit_timeline_apply_track_template (BlouEditTimeline *timeline, BlouEditTrackTemplate *template)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (template != NULL, NULL);
  
  /* 템플릿을 사용하여 새 트랙 생성 */
  BlouEditTimelineTrack *track = blouedit_timeline_add_track (timeline, template->track_type, template->name);
  
  if (!track) {
    g_warning ("Failed to create track from template");
    return NULL;
  }
  
  /* 템플릿의 속성 적용 */
  blouedit_timeline_set_track_color (timeline, track, &template->color);
  blouedit_timeline_set_track_height (timeline, track, template->height);
  
  return track;
}

/**
 * blouedit_timeline_set_track_height:
 * @timeline: 타임라인 객체
 * @track: 트랙 객체
 * @height: 새로운 높이 (픽셀 단위)
 *
 * 트랙의 높이를 설정합니다. 트랙은 최소 높이와 최대 높이 사이에서 조절됩니다.
 */
void
blouedit_timeline_set_track_height (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint height)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* 트랙이 목록에 있는지 확인 */
  if (!g_slist_find (timeline->tracks, track)) {
    g_warning ("Track not found in timeline");
    return;
  }
  
  /* 최소/최대 높이 범위 내로 제한 */
  height = CLAMP (height, timeline->min_track_height, timeline->max_track_height);
  
  /* 높이가 변경되지 않으면 아무 작업도 하지 않음 */
  if (track->height == height)
    return;
  
  /* 새로운 높이 설정 */
  track->height = height;
  
  /* 이벤트 발행: 트랙 높이 변경됨 */
  g_signal_emit_by_name (timeline, "track-height-changed", track);
  
  /* 위젯 다시 그리기 */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_show_message:
 * @timeline: 타임라인 객체
 * @message: 표시할 메시지
 *
 * 타임라인 상단에 간단한 메시지를 잠시 표시합니다.
 * 사용자 작업 피드백을 제공하기 위해 사용됩니다.
 */
void
blouedit_timeline_show_message (BlouEditTimeline *timeline, const gchar *message)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (message != NULL);
  
  // 상위 윈도우 얻기
  GtkRoot *root = gtk_widget_get_root (GTK_WIDGET (timeline));
  
  if (GTK_IS_WINDOW (root)) {
    // 토스트 알림 표시
    GtkWidget *toast = gtk_label_new (message);
    gtk_widget_add_css_class (toast, "toast");
    
    // 윈도우에 오버레이 추가
    GtkWidget *overlay = gtk_overlay_new ();
    gtk_overlay_set_child (GTK_OVERLAY (overlay), toast);
    
    // 오버레이 위치 설정
    gtk_widget_set_halign (toast, GTK_ALIGN_CENTER);
    gtk_widget_set_valign (toast, GTK_ALIGN_START);
    gtk_widget_set_margin_top (toast, 12);
    gtk_widget_add_css_class (toast, "background");
    gtk_widget_add_css_class (toast, "card");
    gtk_widget_set_margin (toast, 12);
    
    gtk_window_present (GTK_WINDOW (root));
    
    // 3초 후 토스트 제거
    g_timeout_add_seconds (3, (GSourceFunc)gtk_widget_unparent, toast);
  } else {
    // 윈도우가 없는 경우 로그에 메시지 출력
    g_message ("%s", message);
  }
} 