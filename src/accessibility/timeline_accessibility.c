#include <gtk/gtk.h>
#include <glib.h>
#include <glib/gi18n.h>
#include "accessibility.h"
#include "../timeline/timeline.h"
#include "../timeline/multicam_editor.h"
#include "../timeline/edge_trimming.h"

/* Private function prototypes */
static gboolean on_timeline_key_press(GtkWidget *widget, GdkEventKey *event, BlouEditTimeline *timeline);
static void on_timeline_focus_in(GtkWidget *widget, GdkEventFocus *event, BlouEditTimeline *timeline);
static void on_timeline_focus_out(GtkWidget *widget, GdkEventFocus *event, BlouEditTimeline *timeline);
static void announce_timeline_state(BlouEditTimeline *timeline);
static void announce_position_update(BlouEditTimeline *timeline, gint64 position);
static void announce_selection_change(BlouEditTimeline *timeline);
static void announce_track_change(BlouEditTimeline *timeline, BlouEditTimelineTrack *track);
static void announce_edit_mode_change(BlouEditTimeline *timeline, BlouEditEditMode mode);

/* Enhance a timeline with accessibility features */
void 
blouedit_timeline_enhance_accessibility(BlouEditTimeline *timeline)
{
  g_return_if_fail(BLOUEDIT_IS_TIMELINE(timeline));
  
  /* Ensure the timeline can receive keyboard focus */
  gtk_widget_set_can_focus(GTK_WIDGET(timeline), TRUE);
  
  /* Connect signal handlers */
  g_signal_connect(timeline, "key-press-event", 
                  G_CALLBACK(on_timeline_key_press), timeline);
                  
  g_signal_connect(timeline, "focus-in-event",
                  G_CALLBACK(on_timeline_focus_in), timeline);
                  
  g_signal_connect(timeline, "focus-out-event",
                  G_CALLBACK(on_timeline_focus_out), timeline);
  
  /* Add accessibility roles */
  AtkObject *accessible = gtk_widget_get_accessible(GTK_WIDGET(timeline));
  atk_object_set_role(accessible, ATK_ROLE_SCROLL_PANE);
  atk_object_set_name(accessible, _("Video Timeline Editor"));
  atk_object_set_description(accessible, 
    _("Timeline for editing video clips, audio, and other media elements"));
}

/* Handle timeline keyboard navigation */
gboolean 
blouedit_timeline_handle_accessibility_key_press(BlouEditTimeline *timeline, GdkEventKey *event)
{
  g_return_val_if_fail(BLOUEDIT_IS_TIMELINE(timeline), FALSE);
  g_return_val_if_fail(event != NULL, FALSE);
  
  /* Only handle keyboard navigation if enabled */
  if (!blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION)) {
    return FALSE;
  }
  
  gint64 position = blouedit_timeline_get_position(timeline);
  gint64 frame_step = 1;
  gint64 small_step = 10;
  gint64 medium_step = 100;
  gint64 large_step = 1000;
  gboolean handled = TRUE;
  
  /* Handle timeline navigation keys */
  switch (event->keyval) {
    case GDK_KEY_Right:
      /* Move one frame forward */
      if (event->state & GDK_SHIFT_MASK) {
        /* With Shift: Move by small step */
        blouedit_timeline_set_position(timeline, position + small_step);
      } else if (event->state & GDK_CONTROL_MASK) {
        /* With Ctrl: Move by medium step */
        blouedit_timeline_set_position(timeline, position + medium_step);
      } else if ((event->state & GDK_MOD1_MASK) || (event->state & GDK_META_MASK)) {
        /* With Alt/Meta: Move by large step */
        blouedit_timeline_set_position(timeline, position + large_step);
      } else {
        /* No modifier: Move by single frame */
        blouedit_timeline_set_position(timeline, position + frame_step);
      }
      break;
      
    case GDK_KEY_Left:
      /* Move one frame backward */
      if (event->state & GDK_SHIFT_MASK) {
        /* With Shift: Move by small step */
        blouedit_timeline_set_position(timeline, position - small_step);
      } else if (event->state & GDK_CONTROL_MASK) {
        /* With Ctrl: Move by medium step */
        blouedit_timeline_set_position(timeline, position - medium_step);
      } else if ((event->state & GDK_MOD1_MASK) || (event->state & GDK_META_MASK)) {
        /* With Alt/Meta: Move by large step */
        blouedit_timeline_set_position(timeline, position - large_step);
      } else {
        /* No modifier: Move by single frame */
        blouedit_timeline_set_position(timeline, position - frame_step);
      }
      break;
      
    case GDK_KEY_Home:
      /* Go to start of timeline */
      blouedit_timeline_set_position(timeline, 0);
      blouedit_accessibility_screen_reader_announce(_("Timeline start"), 1);
      break;
      
    case GDK_KEY_End:
      /* Go to end of timeline */
      blouedit_timeline_set_position(timeline, blouedit_timeline_get_duration(timeline));
      blouedit_accessibility_screen_reader_announce(_("Timeline end"), 1);
      break;
      
    case GDK_KEY_Page_Up:
      /* Previous marker or large jump backward */
      blouedit_timeline_goto_prev_marker(timeline);
      break;
      
    case GDK_KEY_Page_Down:
      /* Next marker or large jump forward */
      blouedit_timeline_goto_next_marker(timeline);
      break;
      
    case GDK_KEY_Up:
      /* Previous track */
      if (timeline->selected_track) {
        GSList *tracks = timeline->tracks;
        gint index = g_slist_index(tracks, timeline->selected_track);
        
        if (index > 0) {
          BlouEditTimelineTrack *prev_track = g_slist_nth_data(tracks, index - 1);
          timeline->selected_track = prev_track;
          announce_track_change(timeline, prev_track);
          gtk_widget_queue_draw(GTK_WIDGET(timeline));
        }
      }
      break;
      
    case GDK_KEY_Down:
      /* Next track */
      if (timeline->selected_track) {
        GSList *tracks = timeline->tracks;
        gint index = g_slist_index(tracks, timeline->selected_track);
        
        if (index < g_slist_length(tracks) - 1) {
          BlouEditTimelineTrack *next_track = g_slist_nth_data(tracks, index + 1);
          timeline->selected_track = next_track;
          announce_track_change(timeline, next_track);
          gtk_widget_queue_draw(GTK_WIDGET(timeline));
        }
      }
      break;
      
    case GDK_KEY_space:
      /* Toggle play/pause */
      /* This should be handled by the player */
      blouedit_accessibility_screen_reader_announce(_("Toggle playback"), 1);
      break;
      
    case GDK_KEY_m:
      /* Add marker at current position */
      if (event->state & GDK_CONTROL_MASK) {
        BlouEditTimelineMarker *marker = blouedit_timeline_add_marker(
          timeline, position, BLOUEDIT_MARKER_TYPE_GENERIC, _("Marker"));
        blouedit_timeline_select_marker(timeline, marker);
        blouedit_accessibility_screen_reader_announce(_("Marker added"), 1);
      } else {
        handled = FALSE;
      }
      break;
      
    case GDK_KEY_Delete:
    case GDK_KEY_BackSpace:
      /* Delete selected marker or clip */
      if (timeline->selected_marker) {
        blouedit_timeline_remove_marker(timeline, timeline->selected_marker);
        blouedit_accessibility_screen_reader_announce(_("Marker deleted"), 1);
      } else if (timeline->selected_clips) {
        /* Delete selected clips */
        for (GSList *item = timeline->selected_clips; item; item = item->next) {
          GESClip *clip = GES_CLIP(item->data);
          blouedit_timeline_delete_clip(timeline, clip);
        }
        blouedit_accessibility_screen_reader_announce(_("Selected clips deleted"), 1);
        blouedit_timeline_clear_selection(timeline);
      } else {
        handled = FALSE;
      }
      break;
      
    case GDK_KEY_e:
      /* Cycle through edit modes */
      if (event->state & GDK_CONTROL_MASK) {
        BlouEditEditMode current_mode = blouedit_timeline_get_edit_mode(timeline);
        BlouEditEditMode next_mode = (current_mode + 1) % 5; /* 5 is the number of edit modes */
        
        blouedit_timeline_set_edit_mode(timeline, next_mode);
        announce_edit_mode_change(timeline, next_mode);
      } else {
        handled = FALSE;
      }
      break;
      
    case GDK_KEY_z:
      /* Undo */
      if (event->state & GDK_CONTROL_MASK) {
        if (blouedit_timeline_undo(timeline)) {
          blouedit_accessibility_screen_reader_announce(_("Undo"), 1);
        } else {
          blouedit_accessibility_screen_reader_announce(_("Nothing to undo"), 1);
        }
      } else {
        handled = FALSE;
      }
      break;
      
    case GDK_KEY_y:
    case GDK_KEY_Z:
      /* Redo */
      if (event->state & GDK_CONTROL_MASK) {
        if (blouedit_timeline_redo(timeline)) {
          blouedit_accessibility_screen_reader_announce(_("Redo"), 1);
        } else {
          blouedit_accessibility_screen_reader_announce(_("Nothing to redo"), 1);
        }
      } else {
        handled = FALSE;
      }
      break;
      
    case GDK_KEY_s:
      /* Toggle snap */
      if (event->state & GDK_CONTROL_MASK) {
        gboolean snap_enabled = blouedit_timeline_toggle_snap(timeline);
        if (snap_enabled) {
          blouedit_accessibility_screen_reader_announce(_("Snapping enabled"), 1);
        } else {
          blouedit_accessibility_screen_reader_announce(_("Snapping disabled"), 1);
        }
      } else {
        handled = FALSE;
      }
      break;
      
    case GDK_KEY_a:
      /* Select all clips */
      if (event->state & GDK_CONTROL_MASK) {
        blouedit_timeline_select_all_clips(timeline);
        blouedit_accessibility_screen_reader_announce(_("All clips selected"), 1);
      } else {
        handled = FALSE;
      }
      break;
      
    case GDK_KEY_F1:
      /* Announce current position and state */
      announce_timeline_state(timeline);
      break;
      
    default:
      handled = FALSE;
      break;
  }
  
  return handled;
}

/* Handle timeline key press event */
static gboolean 
on_timeline_key_press(GtkWidget *widget, GdkEventKey *event, BlouEditTimeline *timeline)
{
  return blouedit_timeline_handle_accessibility_key_press(timeline, event);
}

/* Handle timeline focus in event */
static void 
on_timeline_focus_in(GtkWidget *widget, GdkEventFocus *event, BlouEditTimeline *timeline)
{
  /* Draw focus indicators if keyboard focus visualization is enabled */
  if (blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION)) {
    gtk_widget_queue_draw(widget);
    
    /* Announce that timeline has received focus */
    blouedit_accessibility_screen_reader_announce(_("Timeline editor focused"), 1);
  }
}

/* Handle timeline focus out event */
static void 
on_timeline_focus_out(GtkWidget *widget, GdkEventFocus *event, BlouEditTimeline *timeline)
{
  /* Clear focus indicators */
  if (blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_KEYBOARD_NAVIGATION)) {
    gtk_widget_queue_draw(widget);
  }
}

/* Announce timeline state (position, selected clips/markers, etc.) */
static void 
announce_timeline_state(BlouEditTimeline *timeline)
{
  gint64 position = blouedit_timeline_get_position(timeline);
  gint64 duration = blouedit_timeline_get_duration(timeline);
  
  gchar *timecode = blouedit_timeline_position_to_timecode(
    timeline, position, blouedit_timeline_get_timecode_format(timeline));
    
  /* Get track info */
  gint video_tracks = blouedit_timeline_get_track_count(timeline, GES_TRACK_TYPE_VIDEO);
  gint audio_tracks = blouedit_timeline_get_track_count(timeline, GES_TRACK_TYPE_AUDIO);
  
  /* Build announcement string */
  GString *msg = g_string_new("");
  
  /* Add current position */
  g_string_append_printf(msg, _("Position: %s. "), timecode);
  
  /* Add timeline info */
  g_string_append_printf(msg, 
    ngettext("%d video track, ", "%d video tracks, ", video_tracks), 
    video_tracks);
    
  g_string_append_printf(msg, 
    ngettext("%d audio track. ", "%d audio tracks. ", audio_tracks), 
    audio_tracks);
  
  /* Add selection info */
  gint selected_count = g_slist_length(timeline->selected_clips);
  if (selected_count > 0) {
    g_string_append_printf(msg, 
      ngettext("%d clip selected. ", "%d clips selected. ", selected_count), 
      selected_count);
  } else if (timeline->selected_marker) {
    g_string_append_printf(msg, _("Marker '%s' selected. "), timeline->selected_marker->name);
  }
  
  /* Add edit mode info */
  switch (blouedit_timeline_get_edit_mode(timeline)) {
    case BLOUEDIT_EDIT_MODE_NORMAL:
      g_string_append(msg, _("Normal edit mode."));
      break;
    case BLOUEDIT_EDIT_MODE_RIPPLE:
      g_string_append(msg, _("Ripple edit mode."));
      break;
    case BLOUEDIT_EDIT_MODE_ROLL:
      g_string_append(msg, _("Roll edit mode."));
      break;
    case BLOUEDIT_EDIT_MODE_SLIP:
      g_string_append(msg, _("Slip edit mode."));
      break;
    case BLOUEDIT_EDIT_MODE_SLIDE:
      g_string_append(msg, _("Slide edit mode."));
      break;
    case BLOUEDIT_EDIT_MODE_MULTICAM:
      g_string_append(msg, _("Multicam edit mode."));
      break;
  }
  
  /* Announce the message */
  blouedit_accessibility_screen_reader_announce(msg->str, 1);
  
  /* Clean up */
  g_string_free(msg, TRUE);
  g_free(timecode);
}

/* Announce timeline position update */
static void 
announce_position_update(BlouEditTimeline *timeline, gint64 position)
{
  /* Only announce if screen reader is enabled and in verbose mode */
  if (!blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER) ||
      g_slist_length(timeline->markers) == 0) {
    return;
  }
  
  /* Check if we're near any markers */
  BlouEditTimelineMarker *nearest = blouedit_timeline_get_marker_at_position(timeline, position, 10);
  
  if (nearest) {
    gchar *announcement = g_strdup_printf(_("Marker: %s"), nearest->name);
    blouedit_accessibility_screen_reader_announce(announcement, 2);
    g_free(announcement);
  }
}

/* Announce selection change */
static void 
announce_selection_change(BlouEditTimeline *timeline)
{
  /* Only announce if screen reader is enabled */
  if (!blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER)) {
    return;
  }
  
  gint selected_count = g_slist_length(timeline->selected_clips);
  
  if (selected_count > 0) {
    gchar *announcement = g_strdup_printf(
      ngettext("%d clip selected", "%d clips selected", selected_count), 
      selected_count);
    blouedit_accessibility_screen_reader_announce(announcement, 1);
    g_free(announcement);
  } else if (timeline->selected_marker) {
    gchar *announcement = g_strdup_printf(_("Marker '%s' selected"), timeline->selected_marker->name);
    blouedit_accessibility_screen_reader_announce(announcement, 1);
    g_free(announcement);
  } else {
    blouedit_accessibility_screen_reader_announce(_("Nothing selected"), 1);
  }
}

/* Announce track change */
static void 
announce_track_change(BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  /* Only announce if screen reader is enabled */
  if (!blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER)) {
    return;
  }
  
  if (track) {
    gchar *track_type;
    if (track->ges_track->type == GES_TRACK_TYPE_VIDEO) {
      track_type = _("Video");
    } else if (track->ges_track->type == GES_TRACK_TYPE_AUDIO) {
      track_type = _("Audio");
    } else {
      track_type = _("Unknown");
    }
    
    gchar *announcement = g_strdup_printf(_("%s track: %s"), track_type, track->name);
    blouedit_accessibility_screen_reader_announce(announcement, 1);
    g_free(announcement);
  }
}

/* Announce edit mode change */
static void 
announce_edit_mode_change(BlouEditTimeline *timeline, BlouEditEditMode mode)
{
  /* Only announce if screen reader is enabled */
  if (!blouedit_accessibility_is_feature_enabled(BLOUEDIT_ACCESSIBILITY_FEATURE_SCREEN_READER)) {
    return;
  }
  
  const gchar *mode_name;
  
  switch (mode) {
    case BLOUEDIT_EDIT_MODE_NORMAL:
      mode_name = _("Normal edit mode");
      break;
    case BLOUEDIT_EDIT_MODE_RIPPLE:
      mode_name = _("Ripple edit mode");
      break;
    case BLOUEDIT_EDIT_MODE_ROLL:
      mode_name = _("Roll edit mode");
      break;
    case BLOUEDIT_EDIT_MODE_SLIP:
      mode_name = _("Slip edit mode");
      break;
    case BLOUEDIT_EDIT_MODE_SLIDE:
      mode_name = _("Slide edit mode");
      break;
    case BLOUEDIT_EDIT_MODE_MULTICAM:
      mode_name = _("Multicam edit mode");
      break;
    default:
      mode_name = _("Unknown edit mode");
      break;
  }
  
  blouedit_accessibility_screen_reader_announce(mode_name, 1);
} 