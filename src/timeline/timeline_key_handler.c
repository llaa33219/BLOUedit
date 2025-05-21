#include <gtk/gtk.h>
#include <string.h>
#include "timeline.h"
#include "edit_mode_shortcuts.h"
#include "core/types.h"
#include "core/timeline.h"

/**
 * blouedit_timeline_handle_key_press:
 * @timeline: The timeline
 * @event: The key press event
 *
 * Handles key press events on the timeline.
 *
 * Returns: TRUE if the event was handled, FALSE otherwise
 */
gboolean
blouedit_timeline_handle_key_press (BlouEditTimeline *timeline, GdkEventKey *event)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (event != NULL, FALSE);
  
  /* First check if this is an edit mode shortcut */
  if (blouedit_timeline_handle_edit_mode_shortcut(timeline, event)) {
    return TRUE;
  }
  
  /* Handle other timeline-specific shortcuts */
  switch (event->keyval) {
    case GDK_KEY_Delete:
    case GDK_KEY_BackSpace:
      /* Delete selected clips */
      if (timeline->selected_clips) {
        GSList *clips_to_remove = g_slist_copy(timeline->selected_clips);
        
        /* Begin group action */
        blouedit_timeline_begin_group_action(timeline, "Delete clips");
        
        /* Remove all selected clips */
        for (GSList *c = clips_to_remove; c; c = c->next) {
          GESClip *clip = GES_CLIP(c->data);
          ges_timeline_element_remove(GES_TIMELINE_ELEMENT(clip));
        }
        
        /* End group action */
        blouedit_timeline_end_group_action(timeline);
        
        /* Clear selection */
        blouedit_timeline_clear_selection(timeline);
        
        /* Free temporary list */
        g_slist_free(clips_to_remove);
        
        /* Redraw */
        gtk_widget_queue_draw(GTK_WIDGET(timeline));
        
        return TRUE;
      }
      break;
      
    case GDK_KEY_space:
      /* Toggle playback */
      /* TODO: implement play/pause functionality */
      return TRUE;
      
    case GDK_KEY_Home:
      /* Go to start */
      blouedit_timeline_set_position(timeline, 0);
      return TRUE;
      
    case GDK_KEY_End:
      /* Go to end */
      blouedit_timeline_set_position(timeline, blouedit_timeline_get_duration(timeline));
      return TRUE;
      
    case GDK_KEY_z:
      /* Undo/Redo */
      if (event->state & GDK_CONTROL_MASK) {
        if (event->state & GDK_SHIFT_MASK) {
          /* Ctrl+Shift+Z: Redo */
          blouedit_timeline_redo(timeline);
        } else {
          /* Ctrl+Z: Undo */
          blouedit_timeline_undo(timeline);
        }
        return TRUE;
      }
      break;
      
    case GDK_KEY_s:
      /* Show shortcuts dialog when Ctrl+Shift+S is pressed */
      if ((event->state & GDK_CONTROL_MASK) && (event->state & GDK_SHIFT_MASK)) {
        blouedit_timeline_show_edit_mode_shortcuts_dialog(timeline);
        return TRUE;
      }
      break;
      
    case GDK_KEY_m:
      /* Multicam editing */
      if (event->state & GDK_CONTROL_MASK) {
        /* Toggle between multicam modes */
        if (timeline->edit_mode != BLOUEDIT_EDIT_MODE_MULTICAM) {
          /* Enter multicam source view mode */
          blouedit_timeline_set_multicam_mode(timeline, BLOUEDIT_MULTICAM_MODE_SOURCE_VIEW);
        } else if (timeline->multicam_mode == BLOUEDIT_MULTICAM_MODE_SOURCE_VIEW) {
          /* Switch to multicam edit mode */
          blouedit_timeline_set_multicam_mode(timeline, BLOUEDIT_MULTICAM_MODE_EDIT);
        } else {
          /* Disable multicam mode */
          blouedit_timeline_set_multicam_mode(timeline, BLOUEDIT_MULTICAM_MODE_DISABLED);
        }
        return TRUE;
      }
      
      /* Show multicam editor */
      if ((event->state & GDK_CONTROL_MASK) && (event->state & GDK_SHIFT_MASK)) {
        blouedit_timeline_show_multicam_editor(timeline);
        return TRUE;
      }
      break;
      
    case GDK_KEY_plus:
    case GDK_KEY_equal:
      /* Zoom in */
      if (event->state & GDK_CONTROL_MASK) {
        blouedit_timeline_zoom_in(timeline);
        return TRUE;
      }
      break;
      
    case GDK_KEY_minus:
      /* Zoom out */
      if (event->state & GDK_CONTROL_MASK) {
        blouedit_timeline_zoom_out(timeline);
        return TRUE;
      }
      break;
      
    case GDK_KEY_0:
      /* Reset zoom */
      if (event->state & GDK_CONTROL_MASK) {
        blouedit_timeline_set_zoom_level(timeline, 1.0);
        return TRUE;
      }
      break;
      
    case GDK_KEY_a:
      /* Select all clips */
      if (event->state & GDK_CONTROL_MASK) {
        blouedit_timeline_select_all_clips(timeline);
        return TRUE;
      }
      break;
      
    case GDK_KEY_Left:
      /* Move playhead left */
      {
        gint64 pos = blouedit_timeline_get_position(timeline);
        gint64 step = 0;
        
        if (event->state & GDK_SHIFT_MASK) {
          /* Larger steps with Shift */
          step = GST_SECOND;
        } else {
          /* Normal steps */
          step = GST_SECOND / timeline->framerate;
        }
        
        if (pos >= step) {
          blouedit_timeline_set_position(timeline, pos - step);
        } else {
          blouedit_timeline_set_position(timeline, 0);
        }
        
        return TRUE;
      }
      break;
      
    case GDK_KEY_Right:
      /* Move playhead right */
      {
        gint64 pos = blouedit_timeline_get_position(timeline);
        gint64 step = 0;
        
        if (event->state & GDK_SHIFT_MASK) {
          /* Larger steps with Shift */
          step = GST_SECOND;
        } else {
          /* Normal steps */
          step = GST_SECOND / timeline->framerate;
        }
        
        blouedit_timeline_set_position(timeline, pos + step);
        return TRUE;
      }
      break;
      
    default:
      break;
  }
  
  return FALSE;
}

/**
 * blouedit_timeline_key_press_event:
 * @widget: The timeline widget
 * @event: The key press event
 * @user_data: User data pointer (the timeline)
 *
 * Internal key press event callback handler.
 *
 * Returns: TRUE if the event was handled, FALSE otherwise
 */
gboolean
blouedit_timeline_key_press_event (GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE(user_data);
  
  return blouedit_timeline_handle_key_press(timeline, event);
}

/**
 * blouedit_timeline_connect_key_events:
 * @timeline: The timeline
 *
 * Connect key press event handler to the timeline.
 */
void
blouedit_timeline_connect_key_events (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  g_signal_connect (GTK_WIDGET(timeline), "key-press-event",
                   G_CALLBACK (blouedit_timeline_key_press_event), timeline);
  
  /* Initialize the edit mode shortcuts */
  blouedit_timeline_init_edit_mode_shortcuts(timeline);
} 