#include <gtk/gtk.h>
#include <string.h>
#include "edit_mode_shortcuts.h"
#include "core/types.h"
#include "core/timeline.h"

/* 모드별 단축키 정의 구조체 */
typedef struct {
  BlouEditEditMode mode;       /* 편집 모드 */
  guint keyval;                /* 키 값 */
  GdkModifierType modifiers;   /* 수정자 키 (Ctrl, Alt 등) */
  const gchar *name;           /* 모드 이름 */
  const gchar *description;    /* 모드 설명 */
} BlouEditEditModeShortcut;

/* 기본 에딧 모드 단축키 정의 */
static const BlouEditEditModeShortcut DEFAULT_MODE_SHORTCUTS[] = {
  { BLOUEDIT_EDIT_MODE_NORMAL,     GDK_KEY_n, 0,                "일반 편집",     "기본 편집 모드" },
  { BLOUEDIT_EDIT_MODE_RIPPLE,     GDK_KEY_r, 0,                "리플 편집",     "후속 클립을 함께 이동시키는 편집 모드" },
  { BLOUEDIT_EDIT_MODE_ROLL,       GDK_KEY_o, 0,                "롤 편집",       "인접 클립의 경계를 조정하는 편집 모드" },
  { BLOUEDIT_EDIT_MODE_SLIP,       GDK_KEY_s, 0,                "슬립 편집",     "클립 길이를 유지한 채 내용을 조정하는 모드" },
  { BLOUEDIT_EDIT_MODE_SLIDE,      GDK_KEY_d, 0,                "슬라이드 편집", "인접 클립 간격을 유지하며 이동하는 모드" },
  { BLOUEDIT_EDIT_MODE_TRIM,       GDK_KEY_t, 0,                "트림 편집",     "클립 시작/끝을 정밀하게 조정하는 모드" },
  { BLOUEDIT_EDIT_MODE_RAZOR,      GDK_KEY_c, 0,                "분할 편집",     "클립을 분할하는 편집 모드" },
  { BLOUEDIT_EDIT_MODE_RANGE,      GDK_KEY_a, 0,                "범위 선택",     "여러 클립에 걸친 범위를 선택하는 모드" },
  { BLOUEDIT_EDIT_MODE_MULTICAM,   GDK_KEY_m, GDK_CONTROL_MASK, "멀티캠 편집",   "다중 카메라 앵글 전환 편집 모드" }
};

/* 현재 에딧 모드 이름 가져오기 */
const gchar*
blouedit_timeline_get_edit_mode_name(BlouEditEditMode mode)
{
  for (int i = 0; i < G_N_ELEMENTS(DEFAULT_MODE_SHORTCUTS); i++) {
    if (DEFAULT_MODE_SHORTCUTS[i].mode == mode) {
      return DEFAULT_MODE_SHORTCUTS[i].name;
    }
  }
  
  return "알 수 없는 모드";
}

/* 에딧 모드 설명 가져오기 */
static const gchar*
get_edit_mode_description(BlouEditEditMode mode)
{
  for (int i = 0; i < G_N_ELEMENTS(DEFAULT_MODE_SHORTCUTS); i++) {
    if (DEFAULT_MODE_SHORTCUTS[i].mode == mode) {
      return DEFAULT_MODE_SHORTCUTS[i].description;
    }
  }
  
  return "알 수 없는 모드 설명";
}

/* 수정자 키를 문자열로 변환 */
static gchar*
modifiers_to_string(GdkModifierType modifiers)
{
  GString *str = g_string_new(NULL);
  
  if (modifiers & GDK_SHIFT_MASK)
    g_string_append(str, "Shift+");
  
  if (modifiers & GDK_CONTROL_MASK)
    g_string_append(str, "Ctrl+");
  
  if (modifiers & GDK_MOD1_MASK)
    g_string_append(str, "Alt+");
  
  if (modifiers & GDK_META_MASK)
    g_string_append(str, "Meta+");
  
  return g_string_free(str, FALSE);
}

/* 키 값을 문자열로 변환 */
static gchar*
keyval_to_string(guint keyval)
{
  return g_strdup(gdk_keyval_name(keyval));
}

/* 에딧 모드 단축키 초기화 */
void
blouedit_timeline_init_edit_mode_shortcuts(BlouEditTimeline *timeline)
{
  /* 타임라인의 키 이벤트 핸들러 연결 등 초기화 작업 수행 */
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 필요한 경우 추가 초기화 코드 작성 */
}

/* 에딧 모드 단축키 핸들러 */
gboolean
blouedit_timeline_handle_edit_mode_shortcut(BlouEditTimeline *timeline, GdkEventKey *event)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(event != NULL, FALSE);
  
  /* 현재 포커스가 타임라인에 있는지 확인 */
  if (!gtk_widget_has_focus(GTK_WIDGET(timeline))) {
    return FALSE;
  }
  
  /* 현재 키 상태 및 누른 키 가져오기 */
  guint keyval = event->keyval;
  GdkModifierType modifiers = event->state & (GDK_SHIFT_MASK | GDK_CONTROL_MASK | GDK_MOD1_MASK | GDK_META_MASK);
  
  /* 단축키와 일치하는지 확인 */
  for (int i = 0; i < G_N_ELEMENTS(DEFAULT_MODE_SHORTCUTS); i++) {
    if (DEFAULT_MODE_SHORTCUTS[i].keyval == keyval && DEFAULT_MODE_SHORTCUTS[i].modifiers == modifiers) {
      /* 에딧 모드 변경 */
      timeline->edit_mode = DEFAULT_MODE_SHORTCUTS[i].mode;
      
      /* 모드 변경 오버레이 표시 (타임라인을 다시 그려서 오버레이 표시) */
      gtk_widget_queue_draw(GTK_WIDGET(timeline));
      
      /* 상태바에 현재 모드 표시 (상태바 위젯이 있다면) */
      gchar *msg = g_strdup_printf("에딧 모드: %s", blouedit_timeline_get_edit_mode_name(timeline->edit_mode));
      blouedit_timeline_show_message(timeline, msg);
      g_free(msg);
      
      return TRUE;  /* 이벤트 처리 완료 */
    }
  }
  
  return FALSE;  /* 이벤트 처리되지 않음 */
}

/* 에딧 모드 상태 표시 오버레이 그리기 함수 */
void
blouedit_timeline_draw_edit_mode_overlay(BlouEditTimeline *timeline, cairo_t *cr, int width, int height)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* 현재 모드 가져오기 */
  BlouEditEditMode current_mode = timeline->edit_mode;
  
  /* 기본 모드면 오버레이 표시하지 않음 */
  if (current_mode == BLOUEDIT_EDIT_MODE_NORMAL) {
    return;
  }
  
  /* 모드 이름과 설명 가져오기 */
  const gchar *mode_name = blouedit_timeline_get_edit_mode_name(current_mode);
  const gchar *mode_desc = get_edit_mode_description(current_mode);
  
  /* 오버레이 배경 그리기 */
  cairo_save(cr);
  cairo_set_source_rgba(cr, 0.2, 0.2, 0.2, 0.7);
  cairo_rectangle(cr, width - 200, height - 70, 190, 60);
  cairo_fill(cr);
  
  /* 오버레이 테두리 그리기 */
  cairo_set_source_rgba(cr, 0.8, 0.8, 0.8, 0.8);
  cairo_set_line_width(cr, 1.0);
  cairo_rectangle(cr, width - 200, height - 70, 190, 60);
  cairo_stroke(cr);
  
  /* 모드 이름 텍스트 그리기 */
  cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);
  cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size(cr, 14);
  cairo_move_to(cr, width - 190, height - 50);
  cairo_show_text(cr, mode_name);
  
  /* 모드 설명 텍스트 그리기 */
  cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  cairo_set_font_size(cr, 12);
  cairo_move_to(cr, width - 190, height - 30);
  cairo_show_text(cr, mode_desc);
  
  cairo_restore(cr);
}

/* 단축키 대화상자 응답 핸들러 */
static void
on_shortcuts_dialog_response(GtkDialog *dialog, gint response_id, gpointer user_data)
{
  /* 대화상자 닫기 */
  gtk_widget_destroy(GTK_WIDGET(dialog));
}

/* 에딧 모드 단축키 대화상자 표시 함수 */
void
blouedit_timeline_show_edit_mode_shortcuts_dialog(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  GtkWidget *dialog, *content_area, *grid;
  GtkWidget *label_header_mode, *label_header_shortcut, *label_header_desc;
  
  GtkDialogFlags flags = GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT;
  
  /* 대화상자 생성 */
  dialog = gtk_dialog_new_with_buttons("편집 모드 단축키",
                                     GTK_WINDOW(gtk_widget_get_toplevel(GTK_WIDGET(timeline))),
                                     flags,
                                     "_닫기", GTK_RESPONSE_CLOSE,
                                     NULL);
  
  /* 대화상자 크기 설정 */
  gtk_window_set_default_size(GTK_WINDOW(dialog), 500, 400);
  
  /* 콘텐츠 영역 가져오기 */
  content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
  gtk_container_set_border_width(GTK_CONTAINER(content_area), 10);
  gtk_box_set_spacing(GTK_BOX(content_area), 10);
  
  /* 그리드 생성 */
  grid = gtk_grid_new();
  gtk_grid_set_row_spacing(GTK_GRID(grid), 10);
  gtk_grid_set_column_spacing(GTK_GRID(grid), 20);
  gtk_container_add(GTK_CONTAINER(content_area), grid);
  
  /* 헤더 레이블 추가 */
  label_header_mode = gtk_label_new("<b>편집 모드</b>");
  gtk_label_set_use_markup(GTK_LABEL(label_header_mode), TRUE);
  gtk_widget_set_halign(label_header_mode, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), label_header_mode, 0, 0, 1, 1);
  
  label_header_shortcut = gtk_label_new("<b>단축키</b>");
  gtk_label_set_use_markup(GTK_LABEL(label_header_shortcut), TRUE);
  gtk_widget_set_halign(label_header_shortcut, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), label_header_shortcut, 1, 0, 1, 1);
  
  label_header_desc = gtk_label_new("<b>설명</b>");
  gtk_label_set_use_markup(GTK_LABEL(label_header_desc), TRUE);
  gtk_widget_set_halign(label_header_desc, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), label_header_desc, 2, 0, 1, 1);
  
  /* 구분선 추가 */
  GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_grid_attach(GTK_GRID(grid), separator, 0, 1, 3, 1);
  
  /* 모든 단축키 정보 추가 */
  for (int i = 0; i < G_N_ELEMENTS(DEFAULT_MODE_SHORTCUTS); i++) {
    GtkWidget *label_mode, *label_shortcut, *label_desc;
    gchar *shortcut_str;
    
    /* 모드 이름 레이블 */
    label_mode = gtk_label_new(DEFAULT_MODE_SHORTCUTS[i].name);
    gtk_widget_set_halign(label_mode, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), label_mode, 0, i + 2, 1, 1);
    
    /* 단축키 레이블 */
    gchar *mod_str = modifiers_to_string(DEFAULT_MODE_SHORTCUTS[i].modifiers);
    gchar *key_str = keyval_to_string(DEFAULT_MODE_SHORTCUTS[i].keyval);
    shortcut_str = g_strdup_printf("%s%s", mod_str, key_str);
    g_free(mod_str);
    g_free(key_str);
    
    label_shortcut = gtk_label_new(shortcut_str);
    gtk_widget_set_halign(label_shortcut, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), label_shortcut, 1, i + 2, 1, 1);
    g_free(shortcut_str);
    
    /* 설명 레이블 */
    label_desc = gtk_label_new(DEFAULT_MODE_SHORTCUTS[i].description);
    gtk_widget_set_halign(label_desc, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), label_desc, 2, i + 2, 1, 1);
  }
  
  /* 설명 추가 */
  GtkWidget *label_info = gtk_label_new("편집 모드를 변경하려면 타임라인에 포커스가 있는 상태에서 해당 단축키를 누르세요.");
  gtk_widget_set_halign(label_info, GTK_ALIGN_START);
  gtk_grid_attach(GTK_GRID(grid), label_info, 0, G_N_ELEMENTS(DEFAULT_MODE_SHORTCUTS) + 3, 3, 1);
  
  /* 대화상자 응답 핸들러 */
  g_signal_connect(dialog, "response", G_CALLBACK(on_shortcuts_dialog_response), NULL);
  
  /* 대화상자 표시 */
  gtk_widget_show_all(dialog);
} 