#pragma once

#include <gtk/gtk.h>
#include <json-glib/json-glib.h>
#include "../core/types.h"
#include "../core/timeline.h"

G_BEGIN_DECLS

/* 키프레임 템플릿 관련 함수 */
void         blouedit_timeline_save_keyframe_template(BlouEditTimeline *timeline, 
                                                  const gchar *template_name,
                                                  GList *keyframes);
                                                  
GList*       blouedit_timeline_load_keyframe_template(BlouEditTimeline *timeline, 
                                                  const gchar *template_name);
                                                  
void         blouedit_timeline_delete_keyframe_template(const gchar *template_name);

GList*       blouedit_timeline_get_keyframe_templates(void);

void         blouedit_timeline_apply_keyframe_template(BlouEditTimeline *timeline, 
                                                   const gchar *template_name,
                                                   GESClip *clip,
                                                   const gchar *property_name);
                                                   
void         blouedit_timeline_show_keyframe_templates_dialog(BlouEditTimeline *timeline,
                                                          GESClip *clip,
                                                          const gchar *property_name);

G_END_DECLS 