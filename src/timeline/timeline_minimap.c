#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>
#include "timeline.h"
#include "core/types.h"
#include "tracks/tracks.h"

/**
 * Timeline Minimap Implementation
 * 
 * This file contains the implementation of the timeline minimap feature,
 * which provides a bird's-eye view of the entire project timeline.
 */

static gboolean blouedit_timeline_minimap_draw (GtkWidget *widget, cairo_t *cr, BlouEditTimeline *timeline);
static gboolean blouedit_timeline_minimap_button_press (GtkWidget *widget, GdkEventButton *event, BlouEditTimeline *timeline);
static gboolean blouedit_timeline_minimap_button_release (GtkWidget *widget, GdkEventButton *event, BlouEditTimeline *timeline);
static gboolean blouedit_timeline_minimap_motion (GtkWidget *widget, GdkEventMotion *event, BlouEditTimeline *timeline);

struct _BlouEditTimelineMinimap {
  GtkWidget *widget;       /* The minimap widget */
  gint height;             /* Height of the minimap */
  gboolean visible;        /* Whether the minimap is visible */
  gboolean dragging;       /* Whether user is dragging the viewport in the minimap */
  gdouble drag_start_x;    /* X position where dragging started */
  gdouble viewport_start;  /* Start of viewport as ratio of total duration (0.0-1.0) */
  gdouble viewport_width;  /* Width of viewport as ratio of total duration (0.0-1.0) */
};

/**
 * blouedit_timeline_minimap_new:
 * @timeline: A BlouEditTimeline object
 *
 * Creates a new timeline minimap for the specified timeline.
 *
 * Returns: The newly created minimap widget
 */
GtkWidget *
blouedit_timeline_minimap_new (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  // Create a minimap if it doesn't exist yet
  if (!timeline->minimap) {
    // Allocate minimap structure
    timeline->minimap = g_new0 (BlouEditTimelineMinimap, 1);
    
    // Set default values
    timeline->minimap->height = 50;
    timeline->minimap->visible = TRUE;
    timeline->minimap->dragging = FALSE;
    timeline->minimap->drag_start_x = 0.0;
    timeline->minimap->viewport_start = 0.0;
    timeline->minimap->viewport_width = 1.0;
    
    // Create drawing area widget
    timeline->minimap->widget = gtk_drawing_area_new ();
    gtk_widget_set_size_request (timeline->minimap->widget, -1, timeline->minimap->height);
    
    // Connect signals
    g_signal_connect (timeline->minimap->widget, "draw",
                     G_CALLBACK (blouedit_timeline_minimap_draw), timeline);
    
    // Add mouse event handling
    gtk_widget_add_events (timeline->minimap->widget,
                          GDK_BUTTON_PRESS_MASK |
                          GDK_BUTTON_RELEASE_MASK |
                          GDK_POINTER_MOTION_MASK);
    
    g_signal_connect (timeline->minimap->widget, "button-press-event",
                     G_CALLBACK (blouedit_timeline_minimap_button_press), timeline);
    g_signal_connect (timeline->minimap->widget, "button-release-event",
                     G_CALLBACK (blouedit_timeline_minimap_button_release), timeline);
    g_signal_connect (timeline->minimap->widget, "motion-notify-event",
                     G_CALLBACK (blouedit_timeline_minimap_motion), timeline);
  }
  
  return timeline->minimap->widget;
}

/**
 * blouedit_timeline_minimap_draw:
 * @widget: The minimap drawing area widget
 * @cr: The Cairo context
 * @timeline: The timeline object
 *
 * Draws the timeline minimap, showing a condensed view of all clips and tracks.
 *
 * Returns: TRUE to stop other handlers from being invoked
 */
static gboolean
blouedit_timeline_minimap_draw (GtkWidget *widget, cairo_t *cr, BlouEditTimeline *timeline)
{
  gint width, height;
  gint64 duration;
  GSList *tracks, *l;
  int y_pos = 0;
  int track_height = 4; // Fixed small height for each track in minimap
  
  // Get widget dimensions
  width = gtk_widget_get_allocated_width (widget);
  height = gtk_widget_get_allocated_height (widget);
  
  // Get timeline duration
  duration = blouedit_timeline_get_duration (timeline);
  if (duration <= 0)
    duration = GST_SECOND * 60; // Default to 60 seconds if no duration
  
  // Draw background
  cairo_set_source_rgb (cr, 0.15, 0.15, 0.15);
  cairo_rectangle (cr, 0, 0, width, height);
  cairo_fill (cr);
  
  // Draw grid lines (every 10 seconds)
  cairo_set_source_rgba (cr, 0.4, 0.4, 0.4, 0.5);
  cairo_set_line_width (cr, 0.5);
  
  for (gint64 t = 0; t <= duration; t += GST_SECOND * 10) {
    gdouble x = (gdouble)t / duration * width;
    cairo_move_to (cr, x, 0);
    cairo_line_to (cr, x, height);
    cairo_stroke (cr);
  }
  
  // Get track list
  tracks = timeline->tracks;
  
  // Calculate total height needed
  int total_tracks = g_slist_length (tracks);
  int total_height = total_tracks * (track_height + 1);
  
  // If we have tracks but they're less than the available height, center them
  if (total_tracks > 0 && total_height < height) {
    y_pos = (height - total_height) / 2;
  }
  
  // Draw each track and its clips
  for (l = tracks; l != NULL; l = l->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)l->data;
    GESTrack *ges_track = track->ges_track;
    GESTrackType track_type = ges_track_get_track_type (ges_track);
    
    // Draw track background
    if (track_type == GES_TRACK_TYPE_VIDEO) {
      // Video track
      cairo_set_source_rgba (cr, track->color.red * 0.8, track->color.green * 0.8, track->color.blue * 0.8, 0.3);
    } else if (track_type == GES_TRACK_TYPE_AUDIO) {
      // Audio track
      cairo_set_source_rgba (cr, track->color.red * 0.8, track->color.green * 0.8, track->color.blue * 0.8, 0.3);
    } else {
      // Other track type
      cairo_set_source_rgba (cr, 0.5, 0.5, 0.5, 0.3);
    }
    
    cairo_rectangle (cr, 0, y_pos, width, track_height);
    cairo_fill (cr);
    
    // Get clips from this track
    GList *clip_list = ges_track_get_elements (ges_track);
    GList *clip;
    
    // Draw each clip
    for (clip = clip_list; clip != NULL; clip = clip->next) {
      GESClip *ges_clip = GES_CLIP (clip->data);
      
      if (!GES_IS_CLIP (ges_clip))
        continue;
      
      // Get clip position and duration
      gint64 clip_start = ges_clip_get_start (ges_clip);
      gint64 clip_duration = ges_clip_get_duration (ges_clip);
      
      // Calculate positions
      gdouble clip_x = (gdouble)clip_start / duration * width;
      gdouble clip_width = (gdouble)clip_duration / duration * width;
      
      // Ensure minimum width
      if (clip_width < 2)
        clip_width = 2;
      
      // Draw clip
      if (track_type == GES_TRACK_TYPE_VIDEO) {
        // Video clip
        cairo_set_source_rgba (cr, track->color.red, track->color.green, track->color.blue, 0.8);
      } else if (track_type == GES_TRACK_TYPE_AUDIO) {
        // Audio clip
        cairo_set_source_rgba (cr, track->color.red, track->color.green, track->color.blue, 0.8);
      }
      
      cairo_rectangle (cr, clip_x, y_pos, clip_width, track_height);
      cairo_fill (cr);
      
      // Add a border to selected clips
      if (blouedit_timeline_is_clip_selected(timeline, ges_clip)) {
        cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 0.8);
        cairo_set_line_width (cr, 1.0);
        cairo_rectangle (cr, clip_x, y_pos, clip_width, track_height);
        cairo_stroke (cr);
      }
    }
    
    // Free clip list
    g_list_free (clip_list);
    
    // Move to next track position
    y_pos += track_height + 1;
  }
  
  // Draw playhead
  gint64 position = blouedit_timeline_get_position (timeline);
  double playhead_x = (double)position / duration * width;
  
  cairo_set_source_rgb (cr, 1.0, 0.0, 0.0);
  cairo_set_line_width (cr, 2.0);
  cairo_move_to (cr, playhead_x, 0);
  cairo_line_to (cr, playhead_x, height);
  cairo_stroke (cr);
  
  // Draw viewport rectangle (represents visible part in main timeline)
  double viewport_x = timeline->minimap->viewport_start * width;
  double viewport_w = timeline->minimap->viewport_width * width;
  
  cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 0.3);
  cairo_rectangle (cr, viewport_x, 0, viewport_w, height);
  cairo_fill (cr);
  
  cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 0.8);
  cairo_set_line_width (cr, 1.0);
  cairo_rectangle (cr, viewport_x, 0, viewport_w, height);
  cairo_stroke (cr);
  
  return TRUE;
}

/**
 * blouedit_timeline_minimap_button_press:
 * @widget: The minimap widget
 * @event: The button event
 * @timeline: The timeline
 *
 * Handles button press events in the minimap, allowing the user to
 * navigate by clicking or start dragging the viewport.
 *
 * Returns: TRUE if the event was handled
 */
static gboolean
blouedit_timeline_minimap_button_press (GtkWidget *widget, GdkEventButton *event, BlouEditTimeline *timeline)
{
  int width = gtk_widget_get_allocated_width (widget);
  double viewport_x = timeline->minimap->viewport_start * width;
  double viewport_w = timeline->minimap->viewport_width * width;
  
  // Only handle left button
  if (event->button != 1)
    return FALSE;
  
  // Check if click is inside viewport (to drag)
  if (event->x >= viewport_x && event->x <= viewport_x + viewport_w) {
    // Start dragging viewport
    timeline->minimap->dragging = TRUE;
    timeline->minimap->drag_start_x = event->x;
    
    // Store original viewport position
    timeline->minimap->viewport_start = (double)viewport_x / width;
    
    // Change cursor to indicate dragging
    GdkWindow *window = gtk_widget_get_window (widget);
    GdkCursor *cursor = gdk_cursor_new_for_display (gdk_window_get_display (window), GDK_FLEUR);
    gdk_window_set_cursor (window, cursor);
    g_object_unref (cursor);
  } else {
    // Click outside viewport - jump timeline to this position
    double ratio = event->x / width;
    gint64 duration = blouedit_timeline_get_duration (timeline);
    gint64 position = ratio * duration;
    
    // Set timeline position
    blouedit_timeline_set_position (timeline, position);
    
    // Center viewport around this position if possible
    double half_viewport = timeline->minimap->viewport_width / 2;
    double new_start = ratio - half_viewport;
    
    // Keep viewport within bounds
    if (new_start < 0)
      new_start = 0;
    else if (new_start + timeline->minimap->viewport_width > 1.0)
      new_start = 1.0 - timeline->minimap->viewport_width;
    
    timeline->minimap->viewport_start = new_start;
    
    // Update horizontal scroll of main timeline
    blouedit_timeline_set_horizontal_scroll (timeline, new_start / (1.0 - timeline->minimap->viewport_width));
    
    // Redraw
    gtk_widget_queue_draw (widget);
  }
  
  return TRUE;
}

/**
 * blouedit_timeline_minimap_button_release:
 * @widget: The minimap widget
 * @event: The button event
 * @timeline: The timeline
 *
 * Handles button release in the minimap, ending dragging operations.
 *
 * Returns: TRUE if the event was handled
 */
static gboolean
blouedit_timeline_minimap_button_release (GtkWidget *widget, GdkEventButton *event, BlouEditTimeline *timeline)
{
  if (event->button != 1)
    return FALSE;
  
  if (timeline->minimap->dragging) {
    timeline->minimap->dragging = FALSE;
    
    // Restore cursor
    GdkWindow *window = gtk_widget_get_window (widget);
    gdk_window_set_cursor (window, NULL);
  }
  
  return TRUE;
}

/**
 * blouedit_timeline_minimap_motion:
 * @widget: The minimap widget
 * @event: The motion event
 * @timeline: The timeline
 *
 * Handles motion events in the minimap, updating the viewport position when dragging.
 *
 * Returns: TRUE if the event was handled
 */
static gboolean
blouedit_timeline_minimap_motion (GtkWidget *widget, GdkEventMotion *event, BlouEditTimeline *timeline)
{
  if (!timeline->minimap->dragging)
    return FALSE;
  
  int width = gtk_widget_get_allocated_width (widget);
  double delta_x = event->x - timeline->minimap->drag_start_x;
  double delta_ratio = delta_x / width;
  
  // Calculate new viewport start
  double new_start = timeline->minimap->viewport_start + delta_ratio;
  
  // Keep viewport within bounds
  if (new_start < 0)
    new_start = 0;
  else if (new_start + timeline->minimap->viewport_width > 1.0)
    new_start = 1.0 - timeline->minimap->viewport_width;
  
  // Update viewport
  timeline->minimap->viewport_start = new_start;
  
  // Update horizontal scroll of main timeline
  blouedit_timeline_set_horizontal_scroll (timeline, new_start / (1.0 - timeline->minimap->viewport_width));
  
  // Redraw
  gtk_widget_queue_draw (widget);
  
  return TRUE;
}

/**
 * blouedit_timeline_update_minimap:
 * @timeline: The timeline
 *
 * Updates the minimap to reflect changes in the timeline.
 * This should be called whenever clips, tracks, or the timeline view changes.
 */
void
blouedit_timeline_update_minimap (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (timeline->minimap && timeline->minimap->visible) {
    // Calculate current viewport from timeline's zoom and scroll position
    gdouble scroll_pos = blouedit_timeline_get_horizontal_scroll (timeline);
    gdouble zoom_level = blouedit_timeline_get_zoom_level (timeline);
    
    // Calculate viewport width based on zoom level
    // A higher zoom level = smaller viewport width
    timeline->minimap->viewport_width = 1.0 / zoom_level;
    if (timeline->minimap->viewport_width > 1.0)
      timeline->minimap->viewport_width = 1.0;
    
    // Calculate viewport start based on scroll position and zoom
    timeline->minimap->viewport_start = scroll_pos * (1.0 - timeline->minimap->viewport_width);
    
    // Redraw minimap
    gtk_widget_queue_draw (timeline->minimap->widget);
  }
}

/**
 * blouedit_timeline_show_minimap:
 * @timeline: The timeline
 * @show: Whether to show the minimap
 *
 * Shows or hides the timeline minimap.
 */
void
blouedit_timeline_show_minimap (BlouEditTimeline *timeline, gboolean show)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (!timeline->minimap)
    return;
  
  timeline->minimap->visible = show;
  
  if (show) {
    gtk_widget_show (timeline->minimap->widget);
    blouedit_timeline_update_minimap (timeline);
  } else {
    gtk_widget_hide (timeline->minimap->widget);
  }
}

/**
 * blouedit_timeline_get_minimap_visible:
 * @timeline: The timeline
 *
 * Gets whether the minimap is currently visible.
 *
 * Returns: TRUE if the minimap is visible
 */
gboolean
blouedit_timeline_get_minimap_visible (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  if (!timeline->minimap)
    return FALSE;
  
  return timeline->minimap->visible;
}

/**
 * blouedit_timeline_set_minimap_height:
 * @timeline: The timeline
 * @height: The desired height in pixels
 *
 * Sets the height of the minimap in pixels.
 */
void
blouedit_timeline_set_minimap_height (BlouEditTimeline *timeline, gint height)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (!timeline->minimap)
    return;
  
  if (height < 20)
    height = 20; // Minimum height
  else if (height > 100)
    height = 100; // Maximum height
  
  timeline->minimap->height = height;
  gtk_widget_set_size_request (timeline->minimap->widget, -1, height);
}

/**
 * blouedit_timeline_get_minimap_height:
 * @timeline: The timeline
 *
 * Gets the current height of the minimap.
 *
 * Returns: The minimap height in pixels
 */
gint
blouedit_timeline_get_minimap_height (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  if (!timeline->minimap)
    return 0;
  
  return timeline->minimap->height;
}

/**
 * blouedit_timeline_toggle_minimap:
 * @timeline: The timeline
 *
 * Toggles the visibility of the minimap.
 *
 * Returns: The new visibility state
 */
gboolean
blouedit_timeline_toggle_minimap (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  if (!timeline->minimap)
    return FALSE;
  
  blouedit_timeline_show_minimap (timeline, !timeline->minimap->visible);
  
  return timeline->minimap->visible;
} 