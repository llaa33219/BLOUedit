/**
 * @file timeline_interaction.h
 * @brief Header file for timeline interaction functionality
 */

#ifndef BLOUEDIT_TIMELINE_INTERACTION_H
#define BLOUEDIT_TIMELINE_INTERACTION_H

#include <gtk/gtk.h>

G_BEGIN_DECLS

/**
 * @brief Initialize timeline interaction
 * 
 * @param timeline The timeline widget
 */
void blouedit_timeline_interaction_init(GtkWidget *timeline);

/**
 * @brief Add a clip to the timeline
 * 
 * @param timeline The timeline widget
 * @param track_index The track index
 * @param start_time The start time in nanoseconds
 * @param duration The duration in nanoseconds
 * @param clip_type The clip type ("video", "audio", "text", "image", etc.)
 * @param media_path The media path
 * @return gint The clip ID
 */
gint blouedit_timeline_add_clip(GtkWidget *timeline, 
                              gint track_index, 
                              gint64 start_time, 
                              gint64 duration, 
                              const char *clip_type, 
                              const char *media_path);

/**
 * @brief Remove a clip from the timeline
 * 
 * @param timeline The timeline widget
 * @param clip_id The clip ID
 * @return gboolean Whether the clip was removed
 */
gboolean blouedit_timeline_remove_clip(GtkWidget *timeline, gint clip_id);

/**
 * @brief Move a clip to a new position
 * 
 * @param timeline The timeline widget
 * @param clip_id The clip ID
 * @param track_index The new track index
 * @param start_time The new start time in nanoseconds
 * @return gboolean Whether the clip was moved
 */
gboolean blouedit_timeline_move_clip(GtkWidget *timeline, 
                                  gint clip_id, 
                                  gint track_index, 
                                  gint64 start_time);

/**
 * @brief Split a clip at the given time
 * 
 * @param timeline The timeline widget
 * @param clip_id The clip ID
 * @param split_time The split time in nanoseconds
 * @return gint The ID of the newly created second part clip
 */
gint blouedit_timeline_split_clip(GtkWidget *timeline, 
                                gint clip_id, 
                                gint64 split_time);

/**
 * @brief Add a new track to the timeline
 * 
 * @param timeline The timeline widget
 * @param track_type The track type ("video", "audio", "text", etc.)
 * @param track_name The track name
 * @return gint The track ID
 */
gint blouedit_timeline_add_track(GtkWidget *timeline, 
                               const char *track_type, 
                               const char *track_name);

/**
 * @brief Remove a track from the timeline
 * 
 * @param timeline The timeline widget
 * @param track_index The track index
 * @return gboolean Whether the track was removed
 */
gboolean blouedit_timeline_remove_track(GtkWidget *timeline, gint track_index);

/**
 * @brief Set track properties
 * 
 * @param timeline The timeline widget
 * @param track_index The track index
 * @param property_name The property name
 * @param property_value The property value
 */
void blouedit_timeline_set_track_property(GtkWidget *timeline, 
                                        gint track_index, 
                                        const char *property_name, 
                                        const GValue *property_value);

/**
 * @brief Add a marker to the timeline
 * 
 * @param timeline The timeline widget
 * @param time The marker time in nanoseconds
 * @param marker_type The marker type
 * @param marker_name The marker name
 * @param marker_color The marker color
 * @return gint The marker ID
 */
gint blouedit_timeline_add_marker(GtkWidget *timeline, 
                                gint64 time, 
                                const char *marker_type, 
                                const char *marker_name, 
                                const char *marker_color);

/**
 * @brief Remove a marker from the timeline
 * 
 * @param timeline The timeline widget
 * @param marker_id The marker ID
 * @return gboolean Whether the marker was removed
 */
gboolean blouedit_timeline_remove_marker(GtkWidget *timeline, gint marker_id);

/**
 * @brief Group multiple clips
 * 
 * @param timeline The timeline widget
 * @param clip_ids Array of clip IDs to group
 * @param n_clips Number of clips to group
 * @return gint The group ID
 */
gint blouedit_timeline_group_clips(GtkWidget *timeline, 
                                 const gint *clip_ids, 
                                 gint n_clips);

/**
 * @brief Ungroup clips
 * 
 * @param timeline The timeline widget
 * @param group_id The group ID
 * @return gboolean Whether the group was ungrouped
 */
gboolean blouedit_timeline_ungroup_clips(GtkWidget *timeline, gint group_id);

/**
 * @brief Set clip properties
 * 
 * @param timeline The timeline widget
 * @param clip_id The clip ID
 * @param property_name The property name
 * @param property_value The property value
 */
void blouedit_timeline_set_clip_property(GtkWidget *timeline, 
                                       gint clip_id, 
                                       const char *property_name, 
                                       const GValue *property_value);

/**
 * @brief Set timeline mode (normal, ripple, roll, slip, slide)
 * 
 * @param timeline The timeline widget
 * @param mode The edit mode
 */
void blouedit_timeline_set_edit_mode(GtkWidget *timeline, const char *mode);

/**
 * @brief Set timeline snap mode
 * 
 * @param timeline The timeline widget
 * @param snap_to_clips Whether to snap to clips
 * @param snap_to_markers Whether to snap to markers
 * @param snap_to_playhead Whether to snap to playhead
 * @param snap_to_grid Whether to snap to grid
 */
void blouedit_timeline_set_snap_mode(GtkWidget *timeline, 
                                   gboolean snap_to_clips, 
                                   gboolean snap_to_markers, 
                                   gboolean snap_to_playhead, 
                                   gboolean snap_to_grid);

/**
 * @brief Save timeline state
 * 
 * @param timeline The timeline widget
 * @param name The snapshot name
 * @return gint The snapshot ID
 */
gint blouedit_timeline_save_snapshot(GtkWidget *timeline, const char *name);

/**
 * @brief Restore timeline state
 * 
 * @param timeline The timeline widget
 * @param snapshot_id The snapshot ID
 * @return gboolean Whether the snapshot was restored
 */
gboolean blouedit_timeline_restore_snapshot(GtkWidget *timeline, gint snapshot_id);

/**
 * @brief Get timeline duration
 * 
 * @param timeline The timeline widget
 * @return gint64 The duration in nanoseconds
 */
gint64 blouedit_timeline_get_duration(GtkWidget *timeline);

/**
 * @brief Set timeline zoom level
 * 
 * @param timeline The timeline widget
 * @param zoom_level The zoom level
 */
void blouedit_timeline_set_zoom(GtkWidget *timeline, gdouble zoom_level);

G_END_DECLS

#endif /* BLOUEDIT_TIMELINE_INTERACTION_H */ 