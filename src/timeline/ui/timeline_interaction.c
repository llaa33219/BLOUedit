#include "timeline_interaction.h"
#include <glib/gi18n.h>

// Timeline clip structure
typedef struct {
    gint id;
    gint track_index;
    gint64 start_time;
    gint64 duration;
    char *clip_type;
    char *media_path;
    GtkWidget *widget;
    
    // Flags
    gboolean selected;
    gboolean locked;
    gboolean muted;
    
    // Group info
    gint group_id;
} TimelineClip;

// Timeline track structure
typedef struct {
    gint id;
    char *track_type;
    char *track_name;
    GtkWidget *widget;
    
    // Flags
    gboolean muted;
    gboolean solo;
    gboolean locked;
    gboolean visible;
    
    // Visual properties
    char *color;
    gint height;
} TimelineTrack;

// Timeline marker structure
typedef struct {
    gint id;
    gint64 time;
    char *marker_type;
    char *marker_name;
    char *marker_color;
    GtkWidget *widget;
} TimelineMarker;

// Timeline group structure
typedef struct {
    gint id;
    GList *clip_ids;
} TimelineGroup;

// Timeline structure
typedef struct {
    GtkWidget *widget;
    GList *clips;
    GList *tracks;
    GList *markers;
    GList *groups;
    GList *snapshots;
    
    // Current state
    gint64 duration;
    gint64 playhead_position;
    gdouble zoom_level;
    char *edit_mode;
    
    // Snap settings
    gboolean snap_to_clips;
    gboolean snap_to_markers;
    gboolean snap_to_playhead;
    gboolean snap_to_grid;
    
    // Selection
    GList *selected_clips;
    
    // Drag state
    gboolean is_dragging;
    gint drag_start_x;
    gint drag_start_y;
    GList *drag_clips;
    
    // History
    GList *undo_stack;
    GList *redo_stack;
} Timeline;

// Private functions
static TimelineClip* timeline_find_clip_by_id(Timeline *timeline, gint clip_id);
static TimelineTrack* timeline_find_track_by_index(Timeline *timeline, gint track_index);
static TimelineMarker* timeline_find_marker_by_id(Timeline *timeline, gint marker_id);
static TimelineGroup* timeline_find_group_by_id(Timeline *timeline, gint group_id);

static gint timeline_get_next_clip_id(Timeline *timeline);
static gint timeline_get_next_track_id(Timeline *timeline);
static gint timeline_get_next_marker_id(Timeline *timeline);
static gint timeline_get_next_group_id(Timeline *timeline);

static void timeline_update_duration(Timeline *timeline);
static void timeline_update_clip_widget(Timeline *timeline, TimelineClip *clip);

static void on_clip_drag_begin(GtkGestureDrag *gesture, 
                             gdouble start_x, 
                             gdouble start_y, 
                             gpointer user_data);
static void on_clip_drag_update(GtkGestureDrag *gesture, 
                              gdouble offset_x, 
                              gdouble offset_y, 
                              gpointer user_data);
static void on_clip_drag_end(GtkGestureDrag *gesture, 
                           gdouble offset_x, 
                           gdouble offset_y, 
                           gpointer user_data);

// Timeline lookup table
static GHashTable *timeline_table = NULL;

// Initialize timeline interaction
void blouedit_timeline_interaction_init(GtkWidget *timeline_widget) {
    if (timeline_table == NULL) {
        timeline_table = g_hash_table_new(g_direct_hash, g_direct_equal);
    }
    
    Timeline *timeline = g_new0(Timeline, 1);
    timeline->widget = timeline_widget;
    timeline->clips = NULL;
    timeline->tracks = NULL;
    timeline->markers = NULL;
    timeline->groups = NULL;
    timeline->snapshots = NULL;
    timeline->duration = 0;
    timeline->playhead_position = 0;
    timeline->zoom_level = 1.0;
    timeline->edit_mode = g_strdup("normal");
    timeline->snap_to_clips = TRUE;
    timeline->snap_to_markers = TRUE;
    timeline->snap_to_playhead = TRUE;
    timeline->snap_to_grid = FALSE;
    timeline->selected_clips = NULL;
    timeline->is_dragging = FALSE;
    timeline->drag_start_x = 0;
    timeline->drag_start_y = 0;
    timeline->drag_clips = NULL;
    timeline->undo_stack = NULL;
    timeline->redo_stack = NULL;
    
    g_hash_table_insert(timeline_table, timeline_widget, timeline);
    
    // Connect to signals
    // ...
}

// Get timeline from widget
static Timeline* get_timeline(GtkWidget *timeline_widget) {
    if (timeline_table == NULL) {
        return NULL;
    }
    
    return g_hash_table_lookup(timeline_table, timeline_widget);
}

// Add a clip to the timeline
gint blouedit_timeline_add_clip(GtkWidget *timeline_widget, 
                              gint track_index, 
                              gint64 start_time, 
                              gint64 duration, 
                              const char *clip_type, 
                              const char *media_path) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return -1;
    }
    
    // Check if track exists
    TimelineTrack *track = timeline_find_track_by_index(timeline, track_index);
    if (track == NULL) {
        g_warning("Track with index %d not found", track_index);
        return -1;
    }
    
    // Create new clip
    TimelineClip *clip = g_new0(TimelineClip, 1);
    clip->id = timeline_get_next_clip_id(timeline);
    clip->track_index = track_index;
    clip->start_time = start_time;
    clip->duration = duration;
    clip->clip_type = g_strdup(clip_type);
    clip->media_path = g_strdup(media_path);
    clip->selected = FALSE;
    clip->locked = FALSE;
    clip->muted = FALSE;
    clip->group_id = -1;
    
    // Create clip widget
    clip->widget = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_add_css_class(clip->widget, "blouedit-clip");
    if (g_strcmp0(clip_type, "audio") == 0) {
        gtk_widget_add_css_class(clip->widget, "blouedit-audio-clip");
    }
    
    // Add label
    char *basename = g_path_get_basename(media_path);
    GtkWidget *label = gtk_label_new(basename);
    gtk_widget_set_hexpand(label, TRUE);
    gtk_box_append(GTK_BOX(clip->widget), label);
    g_free(basename);
    
    // Set size and position
    timeline_update_clip_widget(timeline, clip);
    
    // Add clip to list
    timeline->clips = g_list_append(timeline->clips, clip);
    
    // Update timeline duration
    timeline_update_duration(timeline);
    
    return clip->id;
}

// Remove a clip from the timeline
gboolean blouedit_timeline_remove_clip(GtkWidget *timeline_widget, gint clip_id) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return FALSE;
    }
    
    // Find the clip
    TimelineClip *clip = timeline_find_clip_by_id(timeline, clip_id);
    if (clip == NULL) {
        g_warning("Clip with ID %d not found", clip_id);
        return FALSE;
    }
    
    // Remove from any group
    if (clip->group_id != -1) {
        TimelineGroup *group = timeline_find_group_by_id(timeline, clip->group_id);
        if (group != NULL) {
            group->clip_ids = g_list_remove(group->clip_ids, GINT_TO_POINTER(clip_id));
        }
    }
    
    // Remove from selected clips
    timeline->selected_clips = g_list_remove(timeline->selected_clips, clip);
    
    // Remove widget
    gtk_widget_unparent(clip->widget);
    
    // Remove from list
    timeline->clips = g_list_remove(timeline->clips, clip);
    
    // Free memory
    g_free(clip->clip_type);
    g_free(clip->media_path);
    g_free(clip);
    
    // Update timeline duration
    timeline_update_duration(timeline);
    
    return TRUE;
}

// Move a clip to a new position
gboolean blouedit_timeline_move_clip(GtkWidget *timeline_widget, 
                                  gint clip_id, 
                                  gint track_index, 
                                  gint64 start_time) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return FALSE;
    }
    
    // Find the clip
    TimelineClip *clip = timeline_find_clip_by_id(timeline, clip_id);
    if (clip == NULL) {
        g_warning("Clip with ID %d not found", clip_id);
        return FALSE;
    }
    
    // Check if track exists
    if (track_index != clip->track_index) {
        TimelineTrack *track = timeline_find_track_by_index(timeline, track_index);
        if (track == NULL) {
            g_warning("Track with index %d not found", track_index);
            return FALSE;
        }
    }
    
    // Update clip
    clip->track_index = track_index;
    clip->start_time = start_time;
    
    // Update widget
    timeline_update_clip_widget(timeline, clip);
    
    // Update timeline duration
    timeline_update_duration(timeline);
    
    return TRUE;
}

// Split a clip at the given time
gint blouedit_timeline_split_clip(GtkWidget *timeline_widget, 
                                gint clip_id, 
                                gint64 split_time) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return -1;
    }
    
    // Find the clip
    TimelineClip *clip = timeline_find_clip_by_id(timeline, clip_id);
    if (clip == NULL) {
        g_warning("Clip with ID %d not found", clip_id);
        return -1;
    }
    
    // Check if split time is within clip bounds
    if (split_time <= clip->start_time || split_time >= clip->start_time + clip->duration) {
        g_warning("Split time %ld is outside clip bounds", split_time);
        return -1;
    }
    
    // Create new clip for the second part
    gint64 first_part_duration = split_time - clip->start_time;
    gint64 second_part_duration = clip->duration - first_part_duration;
    
    // Update the original clip to be the first part
    clip->duration = first_part_duration;
    timeline_update_clip_widget(timeline, clip);
    
    // Create a new clip for the second part
    return blouedit_timeline_add_clip(timeline_widget, 
                                    clip->track_index, 
                                    split_time, 
                                    second_part_duration, 
                                    clip->clip_type, 
                                    clip->media_path);
}

// Add a new track to the timeline
gint blouedit_timeline_add_track(GtkWidget *timeline_widget, 
                               const char *track_type, 
                               const char *track_name) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return -1;
    }
    
    // Create new track
    TimelineTrack *track = g_new0(TimelineTrack, 1);
    track->id = timeline_get_next_track_id(timeline);
    track->track_type = g_strdup(track_type);
    track->track_name = g_strdup(track_name);
    track->muted = FALSE;
    track->solo = FALSE;
    track->locked = FALSE;
    track->visible = TRUE;
    track->color = g_strdup("#808080");
    track->height = 80;
    
    // Create track widget
    track->widget = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_add_css_class(track->widget, "blouedit-timeline-track");
    
    // Add track to list
    timeline->tracks = g_list_append(timeline->tracks, track);
    
    return g_list_length(timeline->tracks) - 1; // Return track index
}

// Remove a track from the timeline
gboolean blouedit_timeline_remove_track(GtkWidget *timeline_widget, gint track_index) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return FALSE;
    }
    
    // Find the track
    TimelineTrack *track = timeline_find_track_by_index(timeline, track_index);
    if (track == NULL) {
        g_warning("Track with index %d not found", track_index);
        return FALSE;
    }
    
    // Remove clips on this track
    GList *node = timeline->clips;
    while (node != NULL) {
        GList *next = node->next;
        TimelineClip *clip = (TimelineClip *)node->data;
        
        if (clip->track_index == track_index) {
            blouedit_timeline_remove_clip(timeline_widget, clip->id);
        } else if (clip->track_index > track_index) {
            clip->track_index--;
            timeline_update_clip_widget(timeline, clip);
        }
        
        node = next;
    }
    
    // Remove widget
    gtk_widget_unparent(track->widget);
    
    // Remove from list
    timeline->tracks = g_list_remove(timeline->tracks, track);
    
    // Free memory
    g_free(track->track_type);
    g_free(track->track_name);
    g_free(track->color);
    g_free(track);
    
    return TRUE;
}

// Helper functions

// Find clip by ID
static TimelineClip* timeline_find_clip_by_id(Timeline *timeline, gint clip_id) {
    for (GList *node = timeline->clips; node != NULL; node = node->next) {
        TimelineClip *clip = (TimelineClip *)node->data;
        if (clip->id == clip_id) {
            return clip;
        }
    }
    
    return NULL;
}

// Find track by index
static TimelineTrack* timeline_find_track_by_index(Timeline *timeline, gint track_index) {
    if (track_index < 0 || track_index >= g_list_length(timeline->tracks)) {
        return NULL;
    }
    
    return g_list_nth_data(timeline->tracks, track_index);
}

// Find marker by ID
static TimelineMarker* timeline_find_marker_by_id(Timeline *timeline, gint marker_id) {
    for (GList *node = timeline->markers; node != NULL; node = node->next) {
        TimelineMarker *marker = (TimelineMarker *)node->data;
        if (marker->id == marker_id) {
            return marker;
        }
    }
    
    return NULL;
}

// Find group by ID
static TimelineGroup* timeline_find_group_by_id(Timeline *timeline, gint group_id) {
    for (GList *node = timeline->groups; node != NULL; node = node->next) {
        TimelineGroup *group = (TimelineGroup *)node->data;
        if (group->id == group_id) {
            return group;
        }
    }
    
    return NULL;
}

// Get next clip ID
static gint timeline_get_next_clip_id(Timeline *timeline) {
    gint max_id = 0;
    
    for (GList *node = timeline->clips; node != NULL; node = node->next) {
        TimelineClip *clip = (TimelineClip *)node->data;
        if (clip->id > max_id) {
            max_id = clip->id;
        }
    }
    
    return max_id + 1;
}

// Get next track ID
static gint timeline_get_next_track_id(Timeline *timeline) {
    gint max_id = 0;
    
    for (GList *node = timeline->tracks; node != NULL; node = node->next) {
        TimelineTrack *track = (TimelineTrack *)node->data;
        if (track->id > max_id) {
            max_id = track->id;
        }
    }
    
    return max_id + 1;
}

// Get next marker ID
static gint timeline_get_next_marker_id(Timeline *timeline) {
    gint max_id = 0;
    
    for (GList *node = timeline->markers; node != NULL; node = node->next) {
        TimelineMarker *marker = (TimelineMarker *)node->data;
        if (marker->id > max_id) {
            max_id = marker->id;
        }
    }
    
    return max_id + 1;
}

// Get next group ID
static gint timeline_get_next_group_id(Timeline *timeline) {
    gint max_id = 0;
    
    for (GList *node = timeline->groups; node != NULL; node = node->next) {
        TimelineGroup *group = (TimelineGroup *)node->data;
        if (group->id > max_id) {
            max_id = group->id;
        }
    }
    
    return max_id + 1;
}

// Update timeline duration
static void timeline_update_duration(Timeline *timeline) {
    gint64 max_end_time = 0;
    
    for (GList *node = timeline->clips; node != NULL; node = node->next) {
        TimelineClip *clip = (TimelineClip *)node->data;
        gint64 end_time = clip->start_time + clip->duration;
        
        if (end_time > max_end_time) {
            max_end_time = end_time;
        }
    }
    
    timeline->duration = max_end_time;
}

// Update clip widget
static void timeline_update_clip_widget(Timeline *timeline, TimelineClip *clip) {
    // Position and size are calculated based on zoom level
    // In a real implementation, this would be more complex
}

// Public API implementations

void blouedit_timeline_set_track_property(GtkWidget *timeline_widget, 
                                        gint track_index, 
                                        const char *property_name, 
                                        const GValue *property_value) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return;
    }
    
    // Find the track
    TimelineTrack *track = timeline_find_track_by_index(timeline, track_index);
    if (track == NULL) {
        g_warning("Track with index %d not found", track_index);
        return;
    }
    
    // Set property based on name
    if (g_strcmp0(property_name, "muted") == 0 && G_VALUE_HOLDS_BOOLEAN(property_value)) {
        track->muted = g_value_get_boolean(property_value);
    } else if (g_strcmp0(property_name, "solo") == 0 && G_VALUE_HOLDS_BOOLEAN(property_value)) {
        track->solo = g_value_get_boolean(property_value);
    } else if (g_strcmp0(property_name, "locked") == 0 && G_VALUE_HOLDS_BOOLEAN(property_value)) {
        track->locked = g_value_get_boolean(property_value);
    } else if (g_strcmp0(property_name, "visible") == 0 && G_VALUE_HOLDS_BOOLEAN(property_value)) {
        track->visible = g_value_get_boolean(property_value);
    } else if (g_strcmp0(property_name, "color") == 0 && G_VALUE_HOLDS_STRING(property_value)) {
        g_free(track->color);
        track->color = g_strdup(g_value_get_string(property_value));
    } else if (g_strcmp0(property_name, "height") == 0 && G_VALUE_HOLDS_INT(property_value)) {
        track->height = g_value_get_int(property_value);
    } else if (g_strcmp0(property_name, "name") == 0 && G_VALUE_HOLDS_STRING(property_value)) {
        g_free(track->track_name);
        track->track_name = g_strdup(g_value_get_string(property_value));
    }
}

gint blouedit_timeline_add_marker(GtkWidget *timeline_widget, 
                                gint64 time, 
                                const char *marker_type, 
                                const char *marker_name, 
                                const char *marker_color) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return -1;
    }
    
    // Create new marker
    TimelineMarker *marker = g_new0(TimelineMarker, 1);
    marker->id = timeline_get_next_marker_id(timeline);
    marker->time = time;
    marker->marker_type = g_strdup(marker_type);
    marker->marker_name = g_strdup(marker_name);
    marker->marker_color = g_strdup(marker_color);
    
    // Create marker widget
    // In a real implementation, this would add a visual marker to the timeline
    
    // Add marker to list
    timeline->markers = g_list_append(timeline->markers, marker);
    
    return marker->id;
}

gboolean blouedit_timeline_remove_marker(GtkWidget *timeline_widget, gint marker_id) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return FALSE;
    }
    
    // Find the marker
    TimelineMarker *marker = timeline_find_marker_by_id(timeline, marker_id);
    if (marker == NULL) {
        g_warning("Marker with ID %d not found", marker_id);
        return FALSE;
    }
    
    // Remove widget if any
    if (marker->widget != NULL) {
        gtk_widget_unparent(marker->widget);
    }
    
    // Remove from list
    timeline->markers = g_list_remove(timeline->markers, marker);
    
    // Free memory
    g_free(marker->marker_type);
    g_free(marker->marker_name);
    g_free(marker->marker_color);
    g_free(marker);
    
    return TRUE;
}

gint blouedit_timeline_group_clips(GtkWidget *timeline_widget, 
                                 const gint *clip_ids, 
                                 gint n_clips) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return -1;
    }
    
    if (n_clips <= 1) {
        g_warning("Cannot group less than 2 clips");
        return -1;
    }
    
    // Create new group
    TimelineGroup *group = g_new0(TimelineGroup, 1);
    group->id = timeline_get_next_group_id(timeline);
    group->clip_ids = NULL;
    
    // Add clips to group
    for (gint i = 0; i < n_clips; i++) {
        gint clip_id = clip_ids[i];
        TimelineClip *clip = timeline_find_clip_by_id(timeline, clip_id);
        
        if (clip == NULL) {
            g_warning("Clip with ID %d not found", clip_id);
            continue;
        }
        
        // Remove from any existing group
        if (clip->group_id != -1) {
            TimelineGroup *old_group = timeline_find_group_by_id(timeline, clip->group_id);
            if (old_group != NULL) {
                old_group->clip_ids = g_list_remove(old_group->clip_ids, GINT_TO_POINTER(clip_id));
            }
        }
        
        // Add to new group
        clip->group_id = group->id;
        group->clip_ids = g_list_append(group->clip_ids, GINT_TO_POINTER(clip_id));
    }
    
    // Add group to list
    timeline->groups = g_list_append(timeline->groups, group);
    
    return group->id;
}

gboolean blouedit_timeline_ungroup_clips(GtkWidget *timeline_widget, gint group_id) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return FALSE;
    }
    
    // Find the group
    TimelineGroup *group = timeline_find_group_by_id(timeline, group_id);
    if (group == NULL) {
        g_warning("Group with ID %d not found", group_id);
        return FALSE;
    }
    
    // Ungroup clips
    for (GList *node = group->clip_ids; node != NULL; node = node->next) {
        gint clip_id = GPOINTER_TO_INT(node->data);
        TimelineClip *clip = timeline_find_clip_by_id(timeline, clip_id);
        
        if (clip != NULL) {
            clip->group_id = -1;
        }
    }
    
    // Remove from list
    timeline->groups = g_list_remove(timeline->groups, group);
    
    // Free memory
    g_list_free(group->clip_ids);
    g_free(group);
    
    return TRUE;
}

void blouedit_timeline_set_clip_property(GtkWidget *timeline_widget, 
                                       gint clip_id, 
                                       const char *property_name, 
                                       const GValue *property_value) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return;
    }
    
    // Find the clip
    TimelineClip *clip = timeline_find_clip_by_id(timeline, clip_id);
    if (clip == NULL) {
        g_warning("Clip with ID %d not found", clip_id);
        return;
    }
    
    // Set property based on name
    if (g_strcmp0(property_name, "muted") == 0 && G_VALUE_HOLDS_BOOLEAN(property_value)) {
        clip->muted = g_value_get_boolean(property_value);
    } else if (g_strcmp0(property_name, "locked") == 0 && G_VALUE_HOLDS_BOOLEAN(property_value)) {
        clip->locked = g_value_get_boolean(property_value);
    } else if (g_strcmp0(property_name, "duration") == 0 && G_VALUE_HOLDS_INT64(property_value)) {
        clip->duration = g_value_get_int64(property_value);
        timeline_update_clip_widget(timeline, clip);
        timeline_update_duration(timeline);
    }
}

void blouedit_timeline_set_edit_mode(GtkWidget *timeline_widget, const char *mode) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return;
    }
    
    g_free(timeline->edit_mode);
    timeline->edit_mode = g_strdup(mode);
}

void blouedit_timeline_set_snap_mode(GtkWidget *timeline_widget, 
                                   gboolean snap_to_clips, 
                                   gboolean snap_to_markers, 
                                   gboolean snap_to_playhead, 
                                   gboolean snap_to_grid) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return;
    }
    
    timeline->snap_to_clips = snap_to_clips;
    timeline->snap_to_markers = snap_to_markers;
    timeline->snap_to_playhead = snap_to_playhead;
    timeline->snap_to_grid = snap_to_grid;
}

gint64 blouedit_timeline_get_duration(GtkWidget *timeline_widget) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return 0;
    }
    
    return timeline->duration;
}

void blouedit_timeline_set_zoom(GtkWidget *timeline_widget, gdouble zoom_level) {
    Timeline *timeline = get_timeline(timeline_widget);
    if (timeline == NULL) {
        return;
    }
    
    timeline->zoom_level = zoom_level;
    
    // Update all clip widgets
    for (GList *node = timeline->clips; node != NULL; node = node->next) {
        TimelineClip *clip = (TimelineClip *)node->data;
        timeline_update_clip_widget(timeline, clip);
    }
} 