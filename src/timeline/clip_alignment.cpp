#include <gtk/gtk.h>
#include <gst/gst.h>
#include <ges/ges.h>
#include "timeline.h"

/* Helper struct for clip alignment operations */
typedef struct {
  GESClip *clip;
  gint64 start;
  gint64 end;
  gint64 duration;
  gint64 center;
} ClipAlignmentInfo;

/* Helper function to free clip alignment info */
static void
clip_alignment_info_free (ClipAlignmentInfo *info)
{
  if (info) {
    g_free (info);
  }
}

/* Helper function to collect alignment info from selected clips */
static GSList *
collect_selected_clips_info (BlouEditTimeline *timeline)
{
  GSList *info_list = NULL;
  GSList *clip_item;
  
  for (clip_item = timeline->selected_clips; clip_item != NULL; clip_item = clip_item->next) {
    GESClip *clip = GES_CLIP (clip_item->data);
    
    /* Skip locked clips */
    if (blouedit_timeline_is_clip_locked (timeline, clip))
      continue;
    
    /* Create alignment info */
    ClipAlignmentInfo *info = g_new (ClipAlignmentInfo, 1);
    info->clip = clip;
    info->start = ges_clip_get_start (clip);
    info->duration = ges_clip_get_duration (clip);
    info->end = info->start + info->duration;
    info->center = info->start + (info->duration / 2);
    
    /* Add to list */
    info_list = g_slist_append (info_list, info);
  }
  
  return info_list;
}

/* Function to compare clip start positions (for sorting) */
static gint
compare_clip_starts (gconstpointer a, gconstpointer b)
{
  const ClipAlignmentInfo *info_a = a;
  const ClipAlignmentInfo *info_b = b;
  
  return (info_a->start > info_b->start) ? 1 : ((info_a->start < info_b->start) ? -1 : 0);
}

/* Function to compare clip end positions (for sorting) */
static gint
compare_clip_ends (gconstpointer a, gconstpointer b)
{
  const ClipAlignmentInfo *info_a = a;
  const ClipAlignmentInfo *info_b = b;
  
  return (info_a->end > info_b->end) ? 1 : ((info_a->end < info_b->end) ? -1 : 0);
}

/* Align selected clips by their start, end, or center positions */
void
blouedit_timeline_align_selected_clips (BlouEditTimeline *timeline, BlouEditAlignmentType align_type)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Do nothing if less than 2 clips are selected */
  if (!timeline->selected_clips || g_slist_length (timeline->selected_clips) < 2)
    return;
  
  /* Collect clip information */
  GSList *clip_info_list = collect_selected_clips_info (timeline);
  if (!clip_info_list || g_slist_length (clip_info_list) < 2) {
    g_slist_free_full (clip_info_list, (GDestroyNotify)clip_alignment_info_free);
    return;
  }
  
  /* Begin group action for history */
  blouedit_timeline_begin_group_action (timeline, "Align Clips");
  
  /* Sort clips based on alignment type to find the reference clip */
  ClipAlignmentInfo *reference_clip_info = NULL;
  
  switch (align_type) {
    case BLOUEDIT_ALIGN_START:
      /* Sort by start time */
      clip_info_list = g_slist_sort (clip_info_list, compare_clip_starts);
      /* First clip is reference */
      reference_clip_info = (ClipAlignmentInfo *)clip_info_list->data;
      break;
    
    case BLOUEDIT_ALIGN_END:
      /* Sort by end time */
      clip_info_list = g_slist_sort (clip_info_list, compare_clip_ends);
      /* First clip is reference */
      reference_clip_info = (ClipAlignmentInfo *)clip_info_list->data;
      break;
    
    case BLOUEDIT_ALIGN_CENTER:
      /* Find clip with center closest to timeline playhead as reference */
      gint64 playhead_pos = ges_timeline_get_pipeline_time (timeline->ges_timeline);
      gint64 min_distance = G_MAXINT64;
      
      GSList *info_item;
      for (info_item = clip_info_list; info_item != NULL; info_item = info_item->next) {
        ClipAlignmentInfo *info = (ClipAlignmentInfo *)info_item->data;
        gint64 distance = ABS (info->center - playhead_pos);
        
        if (distance < min_distance) {
          min_distance = distance;
          reference_clip_info = info;
        }
      }
      break;
  }
  
  if (!reference_clip_info) {
    /* This should not happen, but just in case */
    reference_clip_info = (ClipAlignmentInfo *)clip_info_list->data;
  }
  
  /* Get reference position based on alignment type */
  gint64 reference_position;
  switch (align_type) {
    case BLOUEDIT_ALIGN_START:
      reference_position = reference_clip_info->start;
      break;
      
    case BLOUEDIT_ALIGN_END:
      reference_position = reference_clip_info->end;
      break;
      
    case BLOUEDIT_ALIGN_CENTER:
      reference_position = reference_clip_info->center;
      break;
      
    default:
      reference_position = reference_clip_info->start;
      break;
  }
  
  /* Move all clips to align with the reference position */
  GSList *info_item;
  for (info_item = clip_info_list; info_item != NULL; info_item = info_item->next) {
    ClipAlignmentInfo *info = (ClipAlignmentInfo *)info_item->data;
    gint64 new_start;
    
    /* Skip the reference clip */
    if (info == reference_clip_info)
      continue;
    
    /* Calculate new position */
    switch (align_type) {
      case BLOUEDIT_ALIGN_START:
        new_start = reference_position;
        break;
        
      case BLOUEDIT_ALIGN_END:
        new_start = reference_position - info->duration;
        break;
        
      case BLOUEDIT_ALIGN_CENTER:
        new_start = reference_position - (info->duration / 2);
        break;
        
      default:
        new_start = reference_position;
        break;
    }
    
    /* Ensure we don't go negative */
    if (new_start < 0)
      new_start = 0;
    
    /* Move the clip to the new position */
    if (ges_clip_get_start (info->clip) != new_start) {
      ges_timeline_element_set_start (GES_TIMELINE_ELEMENT (info->clip), new_start);
    }
  }
  
  /* Clean up */
  g_slist_free_full (clip_info_list, (GDestroyNotify)clip_alignment_info_free);
  
  /* End the group action */
  blouedit_timeline_end_group_action (timeline);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/* Distribute selected clips evenly between the first and last clip */
void
blouedit_timeline_distribute_selected_clips (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Need at least 3 clips to distribute (otherwise alignment is sufficient) */
  if (!timeline->selected_clips || g_slist_length (timeline->selected_clips) < 3)
    return;
  
  /* Collect clip information */
  GSList *clip_info_list = collect_selected_clips_info (timeline);
  if (!clip_info_list || g_slist_length (clip_info_list) < 3) {
    g_slist_free_full (clip_info_list, (GDestroyNotify)clip_alignment_info_free);
    return;
  }
  
  /* Sort clips by start time */
  clip_info_list = g_slist_sort (clip_info_list, compare_clip_starts);
  
  /* Get first and last clips */
  ClipAlignmentInfo *first_clip = (ClipAlignmentInfo *)clip_info_list->data;
  ClipAlignmentInfo *last_clip = (ClipAlignmentInfo *)g_slist_last (clip_info_list)->data;
  
  /* Calculate total distribution space */
  gint64 total_space = last_clip->start - first_clip->start;
  gint num_intervals = g_slist_length (clip_info_list) - 1;
  
  /* If no space to distribute, exit */
  if (total_space <= 0 || num_intervals < 1) {
    g_slist_free_full (clip_info_list, (GDestroyNotify)clip_alignment_info_free);
    return;
  }
  
  /* Begin group action for history */
  blouedit_timeline_begin_group_action (timeline, "Distribute Clips");
  
  /* Calculate and set new positions for all clips except first and last */
  gint64 interval = total_space / num_intervals;
  gint current_interval = 1;
  
  GSList *info_item = clip_info_list->next; /* Skip first clip */
  while (info_item && info_item->next) { /* Stop before last clip */
    ClipAlignmentInfo *info = (ClipAlignmentInfo *)info_item->data;
    
    /* Calculate new start position based on even distribution */
    gint64 new_start = first_clip->start + (interval * current_interval);
    
    /* Move the clip to the new position */
    if (ges_clip_get_start (info->clip) != new_start) {
      ges_timeline_element_set_start (GES_TIMELINE_ELEMENT (info->clip), new_start);
    }
    
    current_interval++;
    info_item = info_item->next;
  }
  
  /* Clean up */
  g_slist_free_full (clip_info_list, (GDestroyNotify)clip_alignment_info_free);
  
  /* End the group action */
  blouedit_timeline_end_group_action (timeline);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/* Remove gaps between selected clips (arrange them sequentially) */
void
blouedit_timeline_remove_gaps_between_selected_clips (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Need at least 2 clips to remove gaps */
  if (!timeline->selected_clips || g_slist_length (timeline->selected_clips) < 2)
    return;
  
  /* Collect clip information */
  GSList *clip_info_list = collect_selected_clips_info (timeline);
  if (!clip_info_list || g_slist_length (clip_info_list) < 2) {
    g_slist_free_full (clip_info_list, (GDestroyNotify)clip_alignment_info_free);
    return;
  }
  
  /* Sort clips by start time */
  clip_info_list = g_slist_sort (clip_info_list, compare_clip_starts);
  
  /* Begin group action for history */
  blouedit_timeline_begin_group_action (timeline, "Remove Gaps Between Clips");
  
  /* Start with the first clip's position */
  ClipAlignmentInfo *first_clip = (ClipAlignmentInfo *)clip_info_list->data;
  gint64 next_start = first_clip->start;
  
  GSList *info_item;
  for (info_item = clip_info_list; info_item != NULL; info_item = info_item->next) {
    ClipAlignmentInfo *info = (ClipAlignmentInfo *)info_item->data;
    
    /* Move clip to the next position */
    if (ges_clip_get_start (info->clip) != next_start) {
      ges_timeline_element_set_start (GES_TIMELINE_ELEMENT (info->clip), next_start);
    }
    
    /* Update next start position to be right after this clip */
    next_start += info->duration;
  }
  
  /* Clean up */
  g_slist_free_full (clip_info_list, (GDestroyNotify)clip_alignment_info_free);
  
  /* End the group action */
  blouedit_timeline_end_group_action (timeline);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
} 