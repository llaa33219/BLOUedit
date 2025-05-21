#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/editing-services/ges-timeline.h>

#include "timeline.h"
#include "core/types.h"

/* Slip edit - changes clip in/out points without changing duration or position */
void
blouedit_timeline_slip_clip (BlouEditTimeline *timeline, GESClip *clip, gint64 offset)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (GES_IS_CLIP (clip));
  
  /* Get clip's current properties */
  guint64 start = ges_clip_get_start (clip);
  guint64 duration = ges_timeline_element_get_duration (GES_TIMELINE_ELEMENT (clip));
  gint64 in_point = ges_clip_get_inpoint (clip);
  
  /* Calculate new in-point */
  gint64 new_in_point = in_point + offset;
  
  /* Ensure in-point is not negative */
  if (new_in_point < 0)
    new_in_point = 0;
  
  /* Get the maximum possible in-point (depends on source media length) */
  GESAsset *asset = ges_extractable_get_asset (GES_EXTRACTABLE (clip));
  if (GES_IS_URI_CLIP_ASSET (asset)) {
    guint64 max_duration = ges_uri_clip_asset_get_duration (GES_URI_CLIP_ASSET (asset));
    
    /* Ensure we don't exceed available media */
    if (new_in_point + duration > max_duration)
      new_in_point = MAX(0, (gint64)(max_duration - duration));
  }
  
  /* If in-point hasn't changed, do nothing */
  if (new_in_point == in_point)
    return;
  
  /* Record state before change for history */
  GValue before_value = G_VALUE_INIT;
  g_value_init (&before_value, G_TYPE_INT64);
  g_value_set_int64 (&before_value, in_point);
  
  /* Update in-point */
  ges_clip_set_inpoint (clip, new_in_point);
  
  /* Record state after change for history */
  GValue after_value = G_VALUE_INIT;
  g_value_init (&after_value, G_TYPE_INT64);
  g_value_set_int64 (&after_value, new_in_point);
  
  /* Record action in timeline history */
  blouedit_timeline_record_action (timeline, BLOUEDIT_HISTORY_SLIP_CLIP, 
                                  GES_TIMELINE_ELEMENT (clip),
                                  "Slip Clip", &before_value, &after_value);
  
  /* Clean up GValue resources */
  g_value_unset (&before_value);
  g_value_unset (&after_value);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/* Slide edit - moves clip between neighbors without changing their duration */
void
blouedit_timeline_slide_clip (BlouEditTimeline *timeline, GESClip *clip, gint64 position)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (GES_IS_CLIP (clip));
  
  /* Get clip's current properties */
  guint64 old_start = ges_clip_get_start (clip);
  guint64 duration = ges_timeline_element_get_duration (GES_TIMELINE_ELEMENT (clip));
  
  /* If position hasn't changed, do nothing */
  if (position == old_start)
    return;
  
  /* Get track for this clip */
  GESTrack *track = NULL;
  GList *clip_tracks = ges_clip_get_tracks (clip);
  if (clip_tracks)
    track = GES_TRACK (clip_tracks->data);
  g_list_free (clip_tracks);
  
  if (!track)
    return;
  
  /* Find clips immediately before and after on the same track */
  GESClip *prev_clip = NULL;
  GESClip *next_clip = NULL;
  GList *clips = ges_track_get_clips (track);
  
  GList *clip_item;
  for (clip_item = clips; clip_item != NULL; clip_item = clip_item->next) {
    GESClip *current = GES_CLIP (clip_item->data);
    
    /* Skip the clip being slid */
    if (current == clip)
      continue;
    
    guint64 clip_start = ges_clip_get_start (current);
    guint64 clip_end = clip_start + ges_timeline_element_get_duration (GES_TIMELINE_ELEMENT (current));
    
    /* Check if this is clip is before our clip */
    if (clip_end <= old_start && (!prev_clip || ges_clip_get_start (prev_clip) < clip_start))
      prev_clip = current;
    
    /* Check if this clip is after our clip */
    if (clip_start >= old_start + duration && (!next_clip || ges_clip_get_start (next_clip) > clip_start))
      next_clip = current;
  }
  
  g_list_free (clips);
  
  /* Calculate valid range for slide */
  gint64 min_position = prev_clip ? ges_clip_get_start (prev_clip) + 
                        ges_timeline_element_get_duration (GES_TIMELINE_ELEMENT (prev_clip)) : 0;
  
  gint64 max_position = next_clip ? ges_clip_get_start (next_clip) - duration : G_MAXINT64;
  
  /* Clamp position to valid range */
  position = CLAMP (position, min_position, max_position);
  
  /* If the position is the same after clamping, nothing to do */
  if (position == old_start)
    return;
  
  /* Record state before change for history */
  GValue before_value = G_VALUE_INIT;
  g_value_init (&before_value, G_TYPE_INT64);
  g_value_set_int64 (&before_value, old_start);
  
  /* Update position */
  ges_timeline_element_set_start (GES_TIMELINE_ELEMENT (clip), position);
  
  /* Record state after change for history */
  GValue after_value = G_VALUE_INIT;
  g_value_init (&after_value, G_TYPE_INT64);
  g_value_set_int64 (&after_value, position);
  
  /* Record action in timeline history */
  blouedit_timeline_record_action (timeline, BLOUEDIT_HISTORY_SLIDE_CLIP, 
                                  GES_TIMELINE_ELEMENT (clip),
                                  "Slide Clip", &before_value, &after_value);
  
  /* Clean up GValue resources */
  g_value_unset (&before_value);
  g_value_unset (&after_value);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
} 