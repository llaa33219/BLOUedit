#include "timeline.h"

/* Forward declarations of static functions */
static void timecode_entry_activate_cb (GtkEntry *entry, BlouEditTimeline *timeline);
static void timecode_format_changed_cb (GtkComboBox *combo_box, GtkWidget *entry);
static void blouedit_timeline_draw_timecode_ruler (BlouEditTimeline *timeline, cairo_t *cr, int width, int height);
static GtkWidget *blouedit_timeline_create_timecode_entry (BlouEditTimeline *timeline);
static gboolean blouedit_timeline_draw (GtkWidget *widget, cairo_t *cr);

/* Keyframe helper functions - implementation */

static void
keyframe_free (BlouEditKeyframe *keyframe)
{
  if (keyframe) {
    g_free (keyframe);
  }
}

static void
animatable_property_free (BlouEditAnimatableProperty *property)
{
  if (property) {
    g_free (property->name);
    g_free (property->display_name);
    g_free (property->property_name);
    
    /* Free all keyframes */
    g_slist_free_full (property->keyframes, (GDestroyNotify)keyframe_free);
    
    g_free (property);
  }
}

/* Evaluates a property value at the given position using keyframe interpolation */
static gdouble
evaluate_keyframe_segment (BlouEditKeyframe *start, BlouEditKeyframe *end, gint64 position)
{
  /* If we're exactly at the start keyframe, return its value */
  if (position == start->position) {
    return start->value;
  }
  
  /* If we're exactly at the end keyframe, return its value */
  if (position == end->position) {
    return end->value;
  }
  
  /* Calculate normalized position (0.0 to 1.0) within the segment */
  gdouble t = (gdouble)(position - start->position) / (gdouble)(end->position - start->position);
  
  /* Apply different interpolation methods based on the keyframe interpolation type */
  switch (start->interpolation) {
    case BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT:
      /* Constant interpolation: use the start keyframe's value until we reach the end */
      return start->value;
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR:
      /* Linear interpolation: simple linear blend between start and end values */
      return start->value + t * (end->value - start->value);
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER: {
      /* Bezier interpolation: cubic Bezier curve with control points */
      gdouble start_x = 0.0;
      gdouble start_y = start->value;
      gdouble control1_x = start->handle_right_x;
      gdouble control1_y = start->value + start->handle_right_y;
      gdouble control2_x = 1.0 + end->handle_left_x;
      gdouble control2_y = end->value + end->handle_left_y;
      gdouble end_x = 1.0;
      gdouble end_y = end->value;
      
      /* Find y-coordinate on the cubic Bezier curve at normalized time t */
      gdouble y = (1-t)*(1-t)*(1-t)*start_y + 
                  3*(1-t)*(1-t)*t*control1_y + 
                  3*(1-t)*t*t*control2_y + 
                  t*t*t*end_y;
      
      return y;
    }
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN:
      /* Ease-in: slow at start, faster at end */
      return start->value + (1.0 - cos(t * G_PI / 2.0)) * (end->value - start->value);
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT:
      /* Ease-out: fast at start, slower at end */
      return start->value + sin(t * G_PI / 2.0) * (end->value - start->value);
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT:
      /* Ease-in-out: slow at start and end, faster in the middle */
      return start->value + (1.0 - cos(t * G_PI)) / 2.0 * (end->value - start->value);
      
    default:
      /* Default to linear interpolation */
      return start->value + t * (end->value - start->value);
  }
}

/* Update the actual property value from the keyframes */
static void
update_property_from_keyframes (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property)
{
  if (!property || !property->object || !property->keyframes) {
    return;
  }
  
  /* Calculate the current value using the timeline position */
  gint64 position = blouedit_timeline_get_position (timeline);
  gdouble value = blouedit_timeline_evaluate_property_at_position (timeline, property, position);
  
  /* Update the actual property */
  switch (property->type) {
    case BLOUEDIT_PROPERTY_TYPE_DOUBLE:
      g_object_set (property->object, property->property_name, value, NULL);
      break;
      
    case BLOUEDIT_PROPERTY_TYPE_INT:
      g_object_set (property->object, property->property_name, (gint)value, NULL);
      break;
      
    case BLOUEDIT_PROPERTY_TYPE_BOOLEAN:
      g_object_set (property->object, property->property_name, (gboolean)(value >= 0.5), NULL);
      break;
      
    case BLOUEDIT_PROPERTY_TYPE_COLOR: {
      /* For color properties, we need to get the current color and adjust it */
      GdkRGBA color;
      g_object_get (property->object, property->property_name, &color, NULL);
      
      /* Adjust the color - assumes value represents brightness/alpha */
      color.alpha = CLAMP (value, 0.0, 1.0);
      g_object_set (property->object, property->property_name, &color, NULL);
      break;
    }
      
    case BLOUEDIT_PROPERTY_TYPE_ENUM:
      g_object_set (property->object, property->property_name, (gint)value, NULL);
      break;
      
    case BLOUEDIT_PROPERTY_TYPE_POSITION:
    case BLOUEDIT_PROPERTY_TYPE_SCALE:
    case BLOUEDIT_PROPERTY_TYPE_ROTATION:
      /* These complex types would need custom handling based on the specific object */
      /* For now, we'll just update a generic double property */
      g_object_set (property->object, property->property_name, value, NULL);
      break;
      
    default:
      break;
  }
}

/* Public Functions for Keyframe Management */

/**
 * blouedit_timeline_register_property:
 * @timeline: The timeline
 * @object: The object containing the property
 * @property_name: Name of the property in the object
 * @display_name: Human-readable name for the property
 * @type: The type of the property
 * @min_value: Minimum allowed value
 * @max_value: Maximum allowed value
 * @default_value: Default value
 *
 * Registers a property for keyframe animation.
 *
 * Returns: The newly created property object
 */
BlouEditAnimatableProperty *
blouedit_timeline_register_property (BlouEditTimeline *timeline, GObject *object,
                                  const gchar *property_name, const gchar *display_name,
                                  BlouEditPropertyType type,
                                  gdouble min_value, gdouble max_value, gdouble default_value)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (G_IS_OBJECT (object), NULL);
  g_return_val_if_fail (property_name != NULL, NULL);
  
  /* Create new property object */
  BlouEditAnimatableProperty *property = g_new0 (BlouEditAnimatableProperty, 1);
  property->name = g_strdup (property_name);
  property->display_name = g_strdup (display_name ? display_name : property_name);
  property->type = type;
  property->object = object;
  property->property_name = g_strdup (property_name);
  property->keyframes = NULL;
  property->id = timeline->next_property_id++;
  property->min_value = min_value;
  property->max_value = max_value;
  property->default_value = default_value;
  property->visible = TRUE;
  property->expanded = FALSE;
  
  /* Add to timeline's list of properties */
  timeline->animatable_properties = g_slist_append (timeline->animatable_properties, property);
  
  return property;
}

/**
 * blouedit_timeline_unregister_property:
 * @timeline: The timeline
 * @property: The property to unregister
 *
 * Removes a property from keyframe animation.
 */
void
blouedit_timeline_unregister_property (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (property != NULL);
  
  /* Remove from timeline's list */
  timeline->animatable_properties = g_slist_remove (timeline->animatable_properties, property);
  
  /* If this was the selected property, clear selection */
  if (timeline->selected_property == property) {
    timeline->selected_property = NULL;
    timeline->selected_keyframe = NULL;
  }
  
  /* Free property */
  animatable_property_free (property);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_properties:
 * @timeline: The timeline
 *
 * Gets the list of registered properties.
 *
 * Returns: A list of BlouEditAnimatableProperty objects
 */
GSList *
blouedit_timeline_get_properties (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  return timeline->animatable_properties;
}

/**
 * blouedit_timeline_get_property_by_id:
 * @timeline: The timeline
 * @id: The property ID to find
 *
 * Finds a property by its ID.
 *
 * Returns: The property, or NULL if not found
 */
BlouEditAnimatableProperty *
blouedit_timeline_get_property_by_id (BlouEditTimeline *timeline, guint id)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
    BlouEditAnimatableProperty *property = (BlouEditAnimatableProperty *)prop->data;
    if (property->id == id) {
      return property;
    }
  }
  
  return NULL;
}

/**
 * blouedit_timeline_get_property_by_name:
 * @timeline: The timeline
 * @object: The object containing the property
 * @property_name: Name of the property
 *
 * Finds a property by its name and object.
 *
 * Returns: The property, or NULL if not found
 */
BlouEditAnimatableProperty *
blouedit_timeline_get_property_by_name (BlouEditTimeline *timeline, GObject *object, const gchar *property_name)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (G_IS_OBJECT (object), NULL);
  g_return_val_if_fail (property_name != NULL, NULL);
  
  for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
    BlouEditAnimatableProperty *property = (BlouEditAnimatableProperty *)prop->data;
    if (property->object == object && g_strcmp0(property->property_name, property_name) == 0) {
      return property;
    }
  }
  
  return NULL;
}

/**
 * blouedit_timeline_add_keyframe:
 * @timeline: The timeline
 * @property: The property to add a keyframe to
 * @position: Position in the timeline
 * @value: Value of the keyframe
 * @interpolation: Interpolation type
 *
 * Adds a keyframe to the property at the specified position.
 *
 * Returns: The newly created keyframe
 */
BlouEditKeyframe *
blouedit_timeline_add_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property,
                             gint64 position, gdouble value,
                             BlouEditKeyframeInterpolation interpolation)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (property != NULL, NULL);
  
  /* Check if a keyframe already exists at this position */
  BlouEditKeyframe *existing = blouedit_timeline_get_keyframe_at_position (timeline, property, position, 0);
  if (existing) {
    /* Update the existing keyframe */
    existing->value = value;
    existing->interpolation = interpolation;
    return existing;
  }
  
  /* Create new keyframe */
  BlouEditKeyframe *keyframe = g_new0 (BlouEditKeyframe, 1);
  keyframe->position = position;
  keyframe->interpolation = interpolation;
  keyframe->value = value;
  keyframe->id = timeline->next_keyframe_id++;
  
  /* Initialize Bezier handles to default values */
  keyframe->handle_left_x = -0.25;  /* 25% of segment to the left */
  keyframe->handle_left_y = 0.0;    /* Flat handle */
  keyframe->handle_right_x = 0.25;  /* 25% of segment to the right */
  keyframe->handle_right_y = 0.0;   /* Flat handle */
  
  /* Insert in sorted order by position */
  GSList *insert_before = NULL;
  
  for (GSList *k = property->keyframes; k; k = k->next) {
    BlouEditKeyframe *existing_keyframe = (BlouEditKeyframe *)k->data;
    if (existing_keyframe->position > position) {
      insert_before = k;
      break;
    }
  }
  
  if (insert_before) {
    property->keyframes = g_slist_insert_before (property->keyframes, insert_before, keyframe);
  } else {
    property->keyframes = g_slist_append (property->keyframes, keyframe);
  }
  
  /* Record for history */
  blouedit_timeline_record_action (timeline, BLOUEDIT_HISTORY_ADD_KEYFRAME,
                                 NULL, "Add keyframe", NULL, NULL);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
  
  return keyframe;
}

/**
 * blouedit_timeline_remove_keyframe:
 * @timeline: The timeline
 * @property: The property containing the keyframe
 * @keyframe: The keyframe to remove
 *
 * Removes a keyframe from a property.
 */
void
blouedit_timeline_remove_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, BlouEditKeyframe *keyframe)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (property != NULL);
  g_return_if_fail (keyframe != NULL);
  
  /* Record for history */
  blouedit_timeline_record_action (timeline, BLOUEDIT_HISTORY_REMOVE_KEYFRAME,
                                 NULL, "Remove keyframe", NULL, NULL);
  
  /* Remove from the property's list */
  property->keyframes = g_slist_remove (property->keyframes, keyframe);
  
  /* If this was the selected keyframe, clear selection */
  if (timeline->selected_keyframe == keyframe) {
    timeline->selected_keyframe = NULL;
  }
  
  /* Free the keyframe */
  keyframe_free (keyframe);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_remove_keyframe_at_position:
 * @timeline: The timeline
 * @property: The property containing the keyframe
 * @position: Position in the timeline
 * @tolerance: Tolerance in timeline units
 *
 * Removes a keyframe at the specified position with a given tolerance.
 */
void
blouedit_timeline_remove_keyframe_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position, gint64 tolerance)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (property != NULL);
  
  BlouEditKeyframe *keyframe = blouedit_timeline_get_keyframe_at_position (timeline, property, position, tolerance);
  
  if (keyframe) {
    blouedit_timeline_remove_keyframe (timeline, property, keyframe);
  }
}

/**
 * blouedit_timeline_remove_all_keyframes:
 * @timeline: The timeline
 * @property: The property to remove all keyframes from
 *
 * Removes all keyframes from a property.
 */
void
blouedit_timeline_remove_all_keyframes (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (property != NULL);
  
  /* Record for history */
  blouedit_timeline_record_action (timeline, BLOUEDIT_HISTORY_REMOVE_KEYFRAME,
                                 NULL, "Remove all keyframes", NULL, NULL);
  
  /* Free all keyframes */
  g_slist_free_full (property->keyframes, (GDestroyNotify)keyframe_free);
  property->keyframes = NULL;
  
  /* Clear selected keyframe if it was from this property */
  if (timeline->selected_property == property) {
    timeline->selected_keyframe = NULL;
  }
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_update_keyframe:
 * @timeline: The timeline
 * @property: The property containing the keyframe
 * @keyframe: The keyframe to update
 * @position: New position
 * @value: New value
 * @interpolation: New interpolation type
 *
 * Updates a keyframe's position, value, and interpolation type.
 */
void
blouedit_timeline_update_keyframe (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property,
                                BlouEditKeyframe *keyframe, gint64 position, gdouble value,
                                BlouEditKeyframeInterpolation interpolation)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (property != NULL);
  g_return_if_fail (keyframe != NULL);
  
  /* Record for history */
  blouedit_timeline_record_action (timeline, BLOUEDIT_HISTORY_EDIT_KEYFRAME,
                                 NULL, "Edit keyframe", NULL, NULL);
  
  /* Remove from the current position in the list */
  property->keyframes = g_slist_remove (property->keyframes, keyframe);
  
  /* Update keyframe properties */
  keyframe->position = position;
  keyframe->value = value;
  keyframe->interpolation = interpolation;
  
  /* Re-insert at the correct position in the sorted list */
  GSList *insert_before = NULL;
  for (GSList *k = property->keyframes; k; k = k->next) {
    BlouEditKeyframe *existing_keyframe = (BlouEditKeyframe *)k->data;
    if (existing_keyframe->position > position) {
      insert_before = k;
      break;
    }
  }
  
  if (insert_before) {
    property->keyframes = g_slist_insert_before (property->keyframes, insert_before, keyframe);
  } else {
    property->keyframes = g_slist_append (property->keyframes, keyframe);
  }
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_update_keyframe_handles:
 * @timeline: The timeline
 * @property: The property containing the keyframe
 * @keyframe: The keyframe to update
 * @handle_left_x: Left handle X coordinate
 * @handle_left_y: Left handle Y coordinate
 * @handle_right_x: Right handle X coordinate
 * @handle_right_y: Right handle Y coordinate
 *
 * Updates a keyframe's Bezier handles.
 */
void
blouedit_timeline_update_keyframe_handles (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property,
                                        BlouEditKeyframe *keyframe,
                                        gdouble handle_left_x, gdouble handle_left_y,
                                        gdouble handle_right_x, gdouble handle_right_y)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (property != NULL);
  g_return_if_fail (keyframe != NULL);
  
  /* Record for history */
  blouedit_timeline_record_action (timeline, BLOUEDIT_HISTORY_EDIT_KEYFRAME,
                                 NULL, "Edit keyframe handles", NULL, NULL);
  
  /* Update handle positions */
  keyframe->handle_left_x = handle_left_x;
  keyframe->handle_left_y = handle_left_y;
  keyframe->handle_right_x = handle_right_x;
  keyframe->handle_right_y = handle_right_y;
  
  /* Set interpolation to Bezier if it wasn't already */
  if (keyframe->interpolation != BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    keyframe->interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER;
  }
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_keyframe_at_position:
 * @timeline: The timeline
 * @property: The property to search
 * @position: Position in the timeline
 * @tolerance: Tolerance in timeline units
 *
 * Finds a keyframe at the specified position with a given tolerance.
 *
 * Returns: The keyframe, or NULL if not found
 */
BlouEditKeyframe *
blouedit_timeline_get_keyframe_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position, gint64 tolerance)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (property != NULL, NULL);
  
  for (GSList *k = property->keyframes; k; k = k->next) {
    BlouEditKeyframe *keyframe = (BlouEditKeyframe *)k->data;
    if (ABS(keyframe->position - position) <= tolerance) {
      return keyframe;
    }
  }
  
  return NULL;
}

/**
 * blouedit_timeline_get_keyframes:
 * @timeline: The timeline
 * @property: The property to get keyframes from
 *
 * Gets all keyframes for a property.
 *
 * Returns: A list of BlouEditKeyframe objects
 */
GSList *
blouedit_timeline_get_keyframes (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (property != NULL, NULL);
  
  return property->keyframes;
}

/**
 * blouedit_timeline_get_keyframes_in_range:
 * @timeline: The timeline
 * @property: The property to get keyframes from
 * @start: Start position
 * @end: End position
 *
 * Gets all keyframes within a range for a property.
 *
 * Returns: A list of BlouEditKeyframe objects
 */
GSList *
blouedit_timeline_get_keyframes_in_range (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 start, gint64 end)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (property != NULL, NULL);
  
  GSList *result = NULL;
  
  for (GSList *k = property->keyframes; k; k = k->next) {
    BlouEditKeyframe *keyframe = (BlouEditKeyframe *)k->data;
    if (keyframe->position >= start && keyframe->position <= end) {
      result = g_slist_append (result, keyframe);
    }
  }
  
  return result;
}

/**
 * blouedit_timeline_evaluate_property_at_position:
 * @timeline: The timeline
 * @property: The property to evaluate
 * @position: Position in the timeline
 *
 * Evaluates a property's value at the specified position using keyframe interpolation.
 *
 * Returns: The interpolated value
 */
gdouble
blouedit_timeline_evaluate_property_at_position (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, gint64 position)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0.0);
  g_return_val_if_fail (property != NULL, 0.0);
  
  /* If no keyframes, return default value */
  if (!property->keyframes) {
    return property->default_value;
  }
  
  /* Get first and last keyframes */
  BlouEditKeyframe *first_keyframe = (BlouEditKeyframe *)property->keyframes->data;
  BlouEditKeyframe *last_keyframe = (BlouEditKeyframe *)g_slist_last(property->keyframes)->data;
  
  /* If position is before first keyframe, return first keyframe value */
  if (position <= first_keyframe->position) {
    return first_keyframe->value;
  }
  
  /* If position is after last keyframe, return last keyframe value */
  if (position >= last_keyframe->position) {
    return last_keyframe->value;
  }
  
  /* Find the keyframes that bracket the position */
  BlouEditKeyframe *prev_keyframe = NULL;
  BlouEditKeyframe *next_keyframe = NULL;
  
  for (GSList *k = property->keyframes; k; k = k->next) {
    BlouEditKeyframe *keyframe = (BlouEditKeyframe *)k->data;
    
    if (keyframe->position == position) {
      /* Exact match */
      return keyframe->value;
    } else if (keyframe->position < position) {
      /* This keyframe is before the position */
      prev_keyframe = keyframe;
    } else if (keyframe->position > position) {
      /* This keyframe is after the position */
      next_keyframe = keyframe;
      break;
    }
  }
  
  /* If we have both a previous and next keyframe, interpolate between them */
  if (prev_keyframe && next_keyframe) {
    return evaluate_keyframe_segment(prev_keyframe, next_keyframe, position);
  }
  
  /* Should never get here if we have a valid keyframe list */
  return property->default_value;
}

/**
 * blouedit_timeline_apply_keyframes:
 * @timeline: The timeline
 *
 * Applies all current keyframe values to their respective properties.
 *
 * Returns: TRUE if values were applied
 */
gboolean
blouedit_timeline_apply_keyframes (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  gboolean applied = FALSE;
  
  /* Iterate through all properties and update their values */
  for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
    BlouEditAnimatableProperty *property = (BlouEditAnimatableProperty *)prop->data;
    if (property->keyframes) {
      update_property_from_keyframes(timeline, property);
      applied = TRUE;
    }
  }
  
  return applied;
}

struct _BlouEditTimeline
{
  GtkWidget parent_instance;
  
  GESTimeline *ges_timeline;
  GstElement *pipeline;
  
  gdouble zoom_level;           /* Current zoom level (1.0 is default) */
  gdouble min_zoom_level;       /* Minimum zoom level (zoomed out) */
  gdouble max_zoom_level;       /* Maximum zoom level (zoomed in) */
  gdouble zoom_step;            /* How much to change per zoom action */
  
  /* Snap properties */
  BlouEditSnapMode snap_mode;   /* Current snap mode */
  guint snap_distance;          /* Snap distance in pixels */
  guint grid_interval;          /* Grid interval in timeline units (e.g. frames or ms) */
  
  /* Scrubbing properties */
  gboolean is_scrubbing;        /* Whether we are currently scrubbing */
  BlouEditScrubMode scrub_mode; /* Current scrubbing mode */
  gdouble scrub_start_x;        /* X position where scrubbing started */
  gint64 scrub_start_position;  /* Timeline position when scrubbing started */
  gdouble scrub_sensitivity;    /* Scrubbing sensitivity */
  
  /* Timeline layout properties */
  gint ruler_height;            /* Height of the ruler at the top */
  gint timeline_start_x;        /* Horizontal offset where timeline starts (for labels) */
  gint playhead_x;              /* X position of the playhead */
  
  /* Track properties */
  GSList *tracks;               /* List of BlouEditTimelineTrack */
  gint default_track_height;    /* Default height for tracks */
  gint folded_track_height;     /* Height for folded tracks */
  gint track_header_width;      /* Width of track header area */
  gint track_spacing;           /* Spacing between tracks */
  BlouEditTimelineTrack *selected_track; /* Currently selected track */
  
  /* Track resizing properties */
  gboolean is_resizing_track;        /* Whether a track is being resized */
  BlouEditTimelineTrack *resizing_track; /* Track currently being resized */
  gint resize_start_y;               /* Y position where resize started */
  gint resize_start_height;          /* Original height of track before resize */
  gint min_track_height;             /* Minimum track height allowed */
  gint max_track_height;             /* Maximum track height allowed */
  
  /* Track reordering properties */
  gboolean is_reordering_track;      /* Whether a track is being reordered */
  BlouEditTimelineTrack *reordering_track; /* Track currently being reordered */
  gint reorder_start_y;              /* Y position where reorder started */
  gint reorder_original_index;       /* Original index of the track being reordered */
  gint reorder_current_index;        /* Current index during reordering */
  
  /* Marker properties */
  GSList *markers;                   /* List of BlouEditTimelineMarker */
  guint next_marker_id;              /* Next unique marker ID to assign */
  BlouEditTimelineMarker *selected_marker; /* Currently selected marker */
  gboolean show_markers;             /* Whether to show markers */
  gint marker_height;                /* Height of marker display in ruler */
  
  /* Clip editing properties */
  BlouEditEditMode edit_mode;        /* Current edit mode (normal, ripple, etc.) */
  GSList *selected_clips;            /* List of selected GESClip objects */
  gboolean is_trimming;              /* Whether we are currently trimming a clip */
  GESClip *trimming_clip;            /* Clip currently being trimmed */
  BlouEditClipEdge trimming_edge;    /* Which edge is being trimmed */
  gboolean is_moving_clip;           /* Whether we are currently moving a clip */
  GESClip *moving_clip;              /* Clip currently being moved */
  gint64 moving_start_position;      /* Original position of clip being moved */
  gdouble moving_start_x;            /* X position where move started */
  gboolean is_moving_multiple_clips; /* Whether multiple clips are being moved together */
  GSList *moving_clips_info;         /* List of BlouEditClipMovementInfo for clips being moved */
  gint64 multi_move_offset;          /* Common offset for all moving clips */
  
  /* Keyframe properties */
  GSList *animatable_properties;     /* List of BlouEditAnimatableProperty */
  guint next_property_id;            /* Next unique property ID to assign */
  guint next_keyframe_id;            /* Next unique keyframe ID to assign */
  BlouEditAnimatableProperty *selected_property; /* Currently selected property */
  BlouEditKeyframe *selected_keyframe; /* Currently selected keyframe */
  gboolean show_keyframes;           /* Whether to show keyframes */
  gboolean is_moving_keyframe;       /* Whether we are currently moving a keyframe */
  BlouEditKeyframe *moving_keyframe; /* Keyframe currently being moved */
  gint64 moving_keyframe_start_position; /* Original position of keyframe being moved */
  gdouble moving_keyframe_start_value; /* Original value of keyframe being moved */
  gdouble moving_keyframe_start_x;   /* X position where keyframe move started */
  gdouble moving_keyframe_start_y;   /* Y position where keyframe move started */
  gboolean is_editing_keyframe_handle; /* Whether we are editing a keyframe's bezier handle */
  BlouEditKeyframe *handle_keyframe; /* Keyframe whose handle is being edited */
  gboolean is_editing_left_handle;   /* Whether editing left or right handle */
  gdouble handle_start_x;            /* Original X position of handle being edited */
  gdouble handle_start_y;            /* Original Y position of handle being edited */
  GtkWidget *keyframe_editor;        /* Keyframe editor widget */
  gint keyframe_area_height;         /* Height of keyframe display area */
  gboolean show_keyframe_values;     /* Whether to show keyframe values */
  
  /* Filtering properties */
  BlouEditMediaFilterType media_filter; /* Current media filter */
  
  /* History properties */
  GSList *history;                   /* List of history actions */
  gint history_position;             /* Current position in history */
  gint max_history_size;             /* Maximum number of history items to keep */
  gboolean group_actions;            /* Whether to group actions together */
  GSList *current_group;             /* Current group of actions */
  gchar *current_group_description;  /* Description of current group */
  
  /* Timecode properties */
  BlouEditTimecodeFormat timecode_format; /* Current timecode format */
  gdouble framerate;                 /* Current framerate */
  gboolean show_timecode;            /* Whether to show timecode display */
  GtkWidget *timecode_entry;         /* Timecode entry widget */
  
  /* Autoscroll properties */
  BlouEditAutoscrollMode autoscroll_mode; /* Current autoscroll mode */
  
  /* Player connection */
  GstElement *player;                /* Connected player element */
  gulong player_position_handler;    /* Handler ID for player position updates */
  
  /* Multi-timeline properties */
  BlouEditTimelineGroup *timeline_group; /* Group this timeline belongs to */
  gchar *timeline_name;              /* Name of this timeline in the group */
  
  /* Timeline scale properties */
  BlouEditTimelineScaleMode scale_mode;  /* Scale mode for timeline ruler */
  gint64 custom_scale_interval;          /* Custom scale interval */
  
  /* Media type visualization properties */
  BlouEditMediaVisualMode media_visual_mode; /* Media type visualization mode */
  GdkRGBA media_type_colors[6];             /* Colors for each media type */
  gboolean media_type_colors_initialized;   /* Whether media type colors have been initialized */
  
  /* Multicam properties */
  BlouEditMulticamMode multicam_mode;         /* Multicam editing mode */
  BlouEditMulticamGroup *active_multicam_group; /* Active multicam group */
  
  /* Edge trimming properties */
  BlouEditEdgeTrimMode edge_trim_mode;        /* Edge trimming mode */
};

G_DEFINE_TYPE (BlouEditTimeline, blouedit_timeline, GTK_TYPE_WIDGET)

static void
track_free (BlouEditTimelineTrack *track)
{
  if (track) {
    g_free (track->name);
    g_free (track);
  }
}

static void
marker_free (BlouEditTimelineMarker *marker)
{
  if (marker) {
    g_free (marker->name);
    g_free (marker->comment);
    g_free (marker->detailed_memo);
    g_free (marker);
  }
}

static void
history_action_free (BlouEditHistoryAction *action)
{
  if (action) {
    /* Free the description string */
    g_free (action->description);
    
    /* Clear the GValues */
    g_value_unset (&action->before_value);
    g_value_unset (&action->after_value);
    
    /* Free the action struct itself */
    g_free (action);
  }
}

static void
blouedit_timeline_dispose (GObject *object)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (object);

  /* Free track list */
  g_slist_free_full (timeline->tracks, (GDestroyNotify) track_free);
  timeline->tracks = NULL;
  
  /* Free marker list */
  g_slist_free_full (timeline->markers, (GDestroyNotify) marker_free);
  timeline->markers = NULL;
  
  /* Free selected clips list */
  g_slist_free (timeline->selected_clips);
  timeline->selected_clips = NULL;
  
  /* Free history lists */
  g_slist_free_full (timeline->history, (GDestroyNotify) history_action_free);
  timeline->history = NULL;
  
  g_slist_free_full (timeline->history_redo, (GDestroyNotify) history_action_free);
  timeline->history_redo = NULL;
  
  g_slist_free_full (timeline->current_group, (GDestroyNotify) history_action_free);
  timeline->current_group = NULL;
  
  g_free (timeline->current_group_description);
  timeline->current_group_description = NULL;

  /* Clean up GES objects */
  g_clear_object (&timeline->ges_timeline);
  g_clear_object (&timeline->pipeline);
  
  G_OBJECT_CLASS (blouedit_timeline_parent_class)->dispose (object);
}

static void
blouedit_timeline_class_init (BlouEditTimelineClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  
  object_class->dispose = blouedit_timeline_dispose;
}

static void
blouedit_timeline_init (BlouEditTimeline *timeline)
{
  /* Initialize zoom parameters */
  timeline->zoom_level = 1.0;
  timeline->min_zoom_level = 0.1;
  timeline->max_zoom_level = 10.0;
  timeline->zoom_step = 0.2;
  
  /* Initialize snap parameters */
  timeline->snap_mode = BLOUEDIT_SNAP_TO_CLIPS;
  timeline->snap_distance = 10;  /* 10 pixels snap distance */
  timeline->grid_interval = 1000; /* 1 second grid interval */
  
  /* Initialize scrubbing parameters */
  timeline->is_scrubbing = FALSE;
  timeline->scrub_mode = BLOUEDIT_SCRUB_MODE_NORMAL;
  timeline->scrub_start_x = 0;
  timeline->scrub_start_position = 0;
  timeline->scrub_sensitivity = 1.0;
  
  /* Initialize layout parameters */
  timeline->ruler_height = 20;
  timeline->timeline_start_x = 80; /* Space for track labels */
  timeline->playhead_x = timeline->timeline_start_x;
  
  /* Initialize track parameters */
  timeline->tracks = NULL;
  timeline->default_track_height = 50;
  timeline->folded_track_height = 15;
  timeline->track_header_width = 80;
  timeline->track_spacing = 2;
  timeline->selected_track = NULL;
  
  /* Initialize track resizing parameters */
  timeline->is_resizing_track = FALSE;
  timeline->resizing_track = NULL;
  timeline->resize_start_y = 0;
  timeline->resize_start_height = 0;
  timeline->min_track_height = 20;
  timeline->max_track_height = 200;
  
  /* Initialize track reordering parameters */
  timeline->is_reordering_track = FALSE;
  timeline->reordering_track = NULL;
  timeline->reorder_start_y = 0;
  timeline->reorder_original_index = 0;
  timeline->reorder_current_index = 0;
  
  /* Initialize marker parameters */
  timeline->markers = NULL;
  timeline->next_marker_id = 1;
  timeline->selected_marker = NULL;
  timeline->show_markers = TRUE;
  timeline->marker_height = 15;
  
  /* Initialize clip editing parameters */
  timeline->edit_mode = BLOUEDIT_EDIT_MODE_NORMAL;
  timeline->selected_clips = NULL;
  timeline->is_trimming = FALSE;
  timeline->trimming_clip = NULL;
  timeline->trimming_edge = BLOUEDIT_EDGE_NONE;
  timeline->is_moving_clip = FALSE;
  timeline->moving_clip = NULL;
  timeline->moving_start_position = 0;
  timeline->moving_start_x = 0;
  timeline->is_moving_multiple_clips = FALSE;
  timeline->moving_clips_info = NULL;
  timeline->multi_move_offset = 0;
  
  /* Initialize keyframe parameters */
  timeline->animatable_properties = NULL;
  timeline->next_property_id = 1;
  timeline->next_keyframe_id = 1;
  timeline->selected_property = NULL;
  timeline->selected_keyframe = NULL;
  timeline->show_keyframes = TRUE;
  timeline->is_moving_keyframe = FALSE;
  timeline->moving_keyframe = NULL;
  timeline->moving_keyframe_start_position = 0;
  timeline->moving_keyframe_start_value = 0;
  timeline->moving_keyframe_start_x = 0;
  timeline->moving_keyframe_start_y = 0;
  timeline->is_editing_keyframe_handle = FALSE;
  timeline->handle_keyframe = NULL;
  timeline->is_editing_left_handle = FALSE;
  timeline->handle_start_x = 0;
  timeline->handle_start_y = 0;
  timeline->keyframe_editor = NULL;
  timeline->keyframe_area_height = 100;
  timeline->show_keyframe_values = TRUE;
  
  /* Initialize filtering parameters */
  timeline->media_filter = BLOUEDIT_FILTER_ALL;
  
  /* Initialize history parameters */
  timeline->history = NULL;
  timeline->history_position = 0;
  timeline->max_history_size = 100;
  timeline->group_actions = FALSE;
  timeline->current_group = NULL;
  timeline->current_group_description = NULL;
  
  /* Initialize timecode parameters */
  timeline->timecode_format = BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_FF;
  timeline->framerate = 30.0;
  timeline->show_timecode = TRUE;
  timeline->timecode_entry = NULL;
  
  /* Initialize autoscroll parameters */
  timeline->autoscroll_mode = BLOUEDIT_AUTOSCROLL_PAGE;
  
  /* Initialize player connections */
  timeline->player = NULL;
  timeline->player_position_handler = 0;
  
  /* Initialize multi-timeline parameters */
  timeline->timeline_group = NULL;
  timeline->timeline_name = g_strdup("Main Timeline");
  
  /* Initialize timeline scale parameters */
  timeline->scale_mode = BLOUEDIT_TIMELINE_SCALE_SECONDS;
  timeline->custom_scale_interval = 1000; /* 1 second */
  
  /* Initialize media type visualization parameters */
  timeline->media_visual_mode = BLOUEDIT_MEDIA_VISUAL_MODE_COLOR;
  timeline->media_type_colors_initialized = FALSE;
  
  /* Initialize multicam properties */
  timeline->multicam_mode = BLOUEDIT_MULTICAM_MODE_DISABLED;
  timeline->active_multicam_group = NULL;
  
  /* Initialize edge trimming properties */
  timeline->edge_trim_mode = BLOUEDIT_EDGE_TRIM_MODE_NORMAL;
  
  /* Initialize GES timeline and pipeline */
  timeline->ges_timeline = ges_timeline_new ();
  timeline->pipeline = gst_pipeline_new ("timeline-pipeline");
  
  /* Set up event masks */
  gtk_widget_set_can_focus (GTK_WIDGET (timeline), TRUE);
  gtk_widget_add_events (GTK_WIDGET (timeline),
                        GDK_BUTTON_PRESS_MASK |
                        GDK_BUTTON_RELEASE_MASK |
                        GDK_BUTTON_MOTION_MASK |
                        GDK_POINTER_MOTION_MASK |
                        GDK_KEY_PRESS_MASK |
                        GDK_SCROLL_MASK);
                        
  /* Connect key event handlers */
  g_signal_connect (GTK_WIDGET(timeline), "key-press-event",
                   G_CALLBACK (blouedit_timeline_key_press_event), timeline);
  
  /* Initialize the edit mode shortcuts */
  blouedit_timeline_init_edit_mode_shortcuts(timeline);
}

BlouEditTimeline *
blouedit_timeline_new (void)
{
  BlouEditTimeline *timeline = g_object_new (BLOUEDIT_TYPE_TIMELINE, NULL);
  
  // 타임라인이 생성된 후 기본 트랙 설정
  if (timeline) {
    // 기본 트랙 생성 (비디오 트랙 1개, 오디오 트랙 1개)
    blouedit_timeline_create_default_tracks (timeline);
  }
  
  return timeline;
}

/* Timeline zoom functions */
void
blouedit_timeline_zoom_in (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  gdouble new_zoom = timeline->zoom_level + timeline->zoom_step;
  
  /* Clamp to maximum zoom level */
  if (new_zoom > timeline->max_zoom_level)
    new_zoom = timeline->max_zoom_level;
    
  blouedit_timeline_set_zoom_level (timeline, new_zoom);
}

void
blouedit_timeline_zoom_out (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  gdouble new_zoom = timeline->zoom_level - timeline->zoom_step;
  
  /* Clamp to minimum zoom level */
  if (new_zoom < timeline->min_zoom_level)
    new_zoom = timeline->min_zoom_level;
    
  blouedit_timeline_set_zoom_level (timeline, new_zoom);
}

void
blouedit_timeline_set_zoom_level (BlouEditTimeline *timeline, gdouble zoom_level)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Clamp zoom level to valid range */
  if (zoom_level < timeline->min_zoom_level)
    zoom_level = timeline->min_zoom_level;
  else if (zoom_level > timeline->max_zoom_level)
    zoom_level = timeline->max_zoom_level;
  
  /* Only update if value changed */
  if (timeline->zoom_level != zoom_level) {
    timeline->zoom_level = zoom_level;
    
    /* Queue redraw of the timeline widget */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
    
    /* Emit signal (to be implemented) */
    // g_signal_emit (timeline, signals[ZOOM_CHANGED], 0, zoom_level);
  }
}

gdouble
blouedit_timeline_get_zoom_level (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 1.0);
  
  return timeline->zoom_level;
}

void
blouedit_timeline_zoom_fit (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Calculate the optimal zoom level to show entire timeline */
  gint64 duration = blouedit_timeline_get_duration (timeline);
  
  /* Get widget width */
  int width = gtk_widget_get_width (GTK_WIDGET (timeline));
  
  if (duration > 0 && width > 0) {
    /* Calculate zoom to fit entire timeline in view */
    gdouble ideal_zoom = (gdouble)width / (gdouble)duration;
    
    /* Apply the calculated zoom level */
    blouedit_timeline_set_zoom_level (timeline, ideal_zoom);
  }
}

/* Timeline control */
void 
blouedit_timeline_set_position (BlouEditTimeline *timeline, gint64 position)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Implementation to be added */
}

gint64 
blouedit_timeline_get_position (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  /* Implementation to be added */
  return 0;
}

gint64 
blouedit_timeline_get_duration (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  /* Implementation to be added */
  return 0;
}

/* Snap functions */
void
blouedit_timeline_set_snap_mode (BlouEditTimeline *timeline, BlouEditSnapMode mode)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (timeline->snap_mode != mode) {
    timeline->snap_mode = mode;
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
    /* Signal could be emitted here for UI updates */
  }
}

BlouEditSnapMode
blouedit_timeline_get_snap_mode (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), BLOUEDIT_SNAP_NONE);
  
  return timeline->snap_mode;
}

void
blouedit_timeline_set_snap_distance (BlouEditTimeline *timeline, guint distance)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (timeline->snap_distance != distance) {
    timeline->snap_distance = distance;
    /* No need to redraw since snap distance is only used during drag operations */
  }
}

guint
blouedit_timeline_get_snap_distance (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 10);
  
  return timeline->snap_distance;
}

gboolean
blouedit_timeline_toggle_snap (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  if (timeline->snap_mode == BLOUEDIT_SNAP_NONE) {
    /* Turn on snap (to the default mode - clips) */
    blouedit_timeline_set_snap_mode (timeline, BLOUEDIT_SNAP_TO_CLIPS);
    return TRUE;
  } else {
    /* Turn off snap */
    blouedit_timeline_set_snap_mode (timeline, BLOUEDIT_SNAP_NONE);
    return FALSE;
  }
}

/**
 * blouedit_timeline_snap_position:
 * @timeline: A #BlouEditTimeline
 * @position: The position to snap (in timeline units)
 *
 * Snaps the given position to the nearest snap point based on 
 * the current snap mode.
 *
 * Returns: The snapped position
 */
gint64
blouedit_timeline_snap_position (BlouEditTimeline *timeline, gint64 position)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), position);
  
  /* If snapping is disabled, return the original position */
  if (timeline->snap_mode == BLOUEDIT_SNAP_NONE)
    return position;
  
  gint64 snapped_position = position;
  gint64 min_distance = G_MAXINT64;
  
  /* Convert snap distance from pixels to timeline units based on zoom level */
  gint64 max_snap_distance = (gint64)(timeline->snap_distance / timeline->zoom_level);
  
  /* Snap to grid if enabled */
  if (timeline->snap_mode & BLOUEDIT_SNAP_TO_GRID) {
    gint64 grid_position = (position / timeline->grid_interval) * timeline->grid_interval;
    gint64 distance = ABS(position - grid_position);
    
    if (distance <= max_snap_distance && distance < min_distance) {
      snapped_position = grid_position;
      min_distance = distance;
    }
    
    /* Check the next grid line too */
    grid_position += timeline->grid_interval;
    distance = ABS(position - grid_position);
    
    if (distance <= max_snap_distance && distance < min_distance) {
      snapped_position = grid_position;
      min_distance = distance;
    }
  }
  
  /* Snap to markers if enabled */
  if (timeline->snap_mode & BLOUEDIT_SNAP_TO_MARKERS) {
    /* Placeholder for future implementation 
     * We would iterate through markers and find the closest one
     */
  }
  
  /* Snap to clips if enabled */
  if (timeline->snap_mode & BLOUEDIT_SNAP_TO_CLIPS) {
    /* Placeholder for future implementation
     * We would iterate through clip edges (start and end points)
     * and find the closest one
     */
  }
  
  return snapped_position;
}

/* Timeline scrubbing functions */

/**
 * blouedit_timeline_set_playhead_position_from_x:
 * @timeline: A #BlouEditTimeline
 * @x: X coordinate in widget space
 *
 * Converts an X coordinate to a timeline position and moves the playhead there.
 */
void
blouedit_timeline_set_playhead_position_from_x (BlouEditTimeline *timeline, double x)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Account for timeline offset */
  if (x < timeline->timeline_start_x)
    x = timeline->timeline_start_x;
  
  /* Convert from widget coordinates to timeline position */
  double timeline_x = x - timeline->timeline_start_x;
  gint64 position = (gint64)(timeline_x / timeline->zoom_level);
  
  /* Check if we should snap the position */
  if (timeline->snap_mode != BLOUEDIT_SNAP_NONE) {
    position = blouedit_timeline_snap_position (timeline, position);
  }
  
  /* Set the position (clamped to valid range) */
  gint64 duration = blouedit_timeline_get_duration (timeline);
  if (position < 0)
    position = 0;
  else if (duration > 0 && position > duration)
    position = duration;
  
  /* Update the timeline position */
  blouedit_timeline_set_position (timeline, position);
  
  /* Update the playhead position */
  timeline->playhead_x = timeline->timeline_start_x + (int)(position * timeline->zoom_level);
  
  /* Queue redraw */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_x_from_position:
 * @timeline: A #BlouEditTimeline
 * @position: Timeline position
 *
 * Converts a timeline position to an X coordinate in widget space.
 *
 * Returns: The X coordinate corresponding to the position
 */
double
blouedit_timeline_get_x_from_position (BlouEditTimeline *timeline, gint64 position)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0.0);
  
  return timeline->timeline_start_x + (position * timeline->zoom_level);
}

void
blouedit_timeline_set_scrub_mode (BlouEditTimeline *timeline, BlouEditScrubMode mode)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  timeline->scrub_mode = mode;
}

BlouEditScrubMode
blouedit_timeline_get_scrub_mode (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), BLOUEDIT_SCRUB_MODE_NORMAL);
  
  return timeline->scrub_mode;
}

void
blouedit_timeline_start_scrubbing (BlouEditTimeline *timeline, double x)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  timeline->is_scrubbing = TRUE;
  timeline->scrub_start_x = x;
  timeline->scrub_start_position = blouedit_timeline_get_position (timeline);
  
  /* Pause playback while scrubbing */
  gst_element_set_state (timeline->pipeline, GST_STATE_PAUSED);
  
  /* Optional: grab pointer */
  // gdk_seat_grab (...)
}

void
blouedit_timeline_scrub_to (BlouEditTimeline *timeline, double x)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (!timeline->is_scrubbing)
    return;
  
  /* Different handling based on scrub mode */
  switch (timeline->scrub_mode) {
    case BLOUEDIT_SCRUB_MODE_NORMAL:
      /* Standard scrubbing - set position based on x coordinate */
      blouedit_timeline_set_playhead_position_from_x (timeline, x);
      break;
      
    case BLOUEDIT_SCRUB_MODE_PRECISE:
      /* Precise scrubbing - slower movement for fine control */
      {
        double delta_x = (x - timeline->scrub_start_x) * 0.5 * timeline->scrub_sensitivity;
        double precise_x = timeline->timeline_start_x + 
                          ((timeline->scrub_start_position * timeline->zoom_level) + delta_x);
        blouedit_timeline_set_playhead_position_from_x (timeline, precise_x);
      }
      break;
      
    case BLOUEDIT_SCRUB_MODE_SHUTTLE:
      /* Shuttle mode - distance from start determines playback speed */
      {
        double delta_x = x - timeline->scrub_start_x;
        /* Calculate speed factor based on distance from start point */
        double speed_factor = delta_x / 100.0 * timeline->scrub_sensitivity;
        
        /* Seek in the direction and speed indicated */
        gint64 current_pos = blouedit_timeline_get_position (timeline);
        gint64 new_pos = current_pos + (gint64)(speed_factor * 30); /* 30ms increments */
        
        blouedit_timeline_set_position (timeline, new_pos);
      }
      break;
  }
}

void
blouedit_timeline_end_scrubbing (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  timeline->is_scrubbing = FALSE;
  
  /* Optional: ungrab pointer */
  // gdk_seat_ungrab (...)
}

void
blouedit_timeline_set_scrub_sensitivity (BlouEditTimeline *timeline, gdouble sensitivity)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Clamp to reasonable range */
  if (sensitivity < 0.1)
    sensitivity = 0.1;
  else if (sensitivity > 10.0)
    sensitivity = 10.0;
    
  timeline->scrub_sensitivity = sensitivity;
}

gdouble
blouedit_timeline_get_scrub_sensitivity (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 1.0);
  
  return timeline->scrub_sensitivity;
}

/* Input event handling */
/* Modify button press handler to handle marker clicks */
gboolean
blouedit_timeline_handle_button_press (BlouEditTimeline *timeline, GdkEventButton *event)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (event != NULL, FALSE);
  
  /* Get local coordinates */
  double x = event->x;
  double y = event->y;
  
  /* Check if we're in the keyframe area */
  int keyframe_area_y = gtk_widget_get_allocated_height (GTK_WIDGET (timeline)) - timeline->keyframe_area_height;
  if (timeline->show_keyframes && y >= keyframe_area_y) {
    /* We're in the keyframe area */
    
    /* First check if we're in the properties list part */
    if (x < timeline->timeline_start_x) {
      /* Handle property selection */
      int property_height = 20; /* Same as in draw function */
      int property_index = (y - keyframe_area_y) / property_height;
      
      /* Find the property at this index */
      int visible_index = 0;
      for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
        BlouEditAnimatableProperty *property = (BlouEditAnimatableProperty *)prop->data;
        
        if (!property->visible)
          continue;
        
        if (visible_index == property_index) {
          /* Select this property */
          timeline->selected_property = property;
          timeline->selected_keyframe = NULL; /* Clear keyframe selection */
          
          /* Redraw */
          gtk_widget_queue_draw (GTK_WIDGET (timeline));
          return TRUE;
        }
        
        visible_index++;
      }
      
      return TRUE;
    }
    
    /* We're in the keyframe curves area */
    
    /* Find which property row we're in */
    int property_height = 20; /* Same as in draw function */
    int property_index = (y - keyframe_area_y) / property_height;
    
    /* Find the property at this index */
    BlouEditAnimatableProperty *property = NULL;
    int visible_index = 0;
    
    for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
      BlouEditAnimatableProperty *p = (BlouEditAnimatableProperty *)prop->data;
      
      if (!p->visible)
        continue;
      
      if (visible_index == property_index) {
        property = p;
        break;
      }
      
      visible_index++;
    }
    
    if (!property)
      return TRUE;
    
    /* We have a property, now check if we clicked on a keyframe */
    
    /* Convert x position to timeline position */
    double rel_x = x - timeline->timeline_start_x;
    gint64 position = rel_x / timeline->zoom_level * GST_SECOND;
    
    /* Find a keyframe at this position */
    BlouEditKeyframe *keyframe = blouedit_timeline_get_keyframe_at_position (
        timeline, property, position, GST_SECOND / 10);
    
    /* Left click on keyframe area */
    if (event->button == 1) {
      if (keyframe) {
        /* Select this keyframe */
        timeline->selected_property = property;
        timeline->selected_keyframe = keyframe;
        
        /* Check if this is a double-click to edit */
        if (event->type == GDK_2BUTTON_PRESS) {
          /* Show keyframe editor */
          blouedit_timeline_show_keyframe_editor (timeline, property, keyframe);
        } else {
          /* Start dragging keyframe */
          timeline->is_moving_keyframe = TRUE;
          timeline->moving_keyframe = keyframe;
          timeline->moving_keyframe_start_position = keyframe->position;
          timeline->moving_keyframe_start_value = keyframe->value;
          timeline->moving_keyframe_start_x = x;
          timeline->moving_keyframe_start_y = y;
        }
      } else {
        /* Clicked on empty area - add a new keyframe if a property is selected */
        if (property) {
          /* Select the property */
          timeline->selected_property = property;
          
          /* Calculate property value at this point */
          gdouble value;
          
          if (property->keyframes) {
            /* Use interpolated value from existing keyframes */
            value = blouedit_timeline_evaluate_property_at_position (timeline, property, position);
          } else {
            /* No keyframes yet, create first one with current property value */
            g_object_get (property->object, property->property_name, &value, NULL);
          }
          
          /* Create a new keyframe */
          BlouEditKeyframe *new_keyframe = blouedit_timeline_add_keyframe (
              timeline, property, position, value, BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR);
          
          /* Select the new keyframe */
          timeline->selected_keyframe = new_keyframe;
          
          /* Apply keyframes to update property */
          blouedit_timeline_apply_keyframes (timeline);
        }
      }
      
      /* Redraw */
      gtk_widget_queue_draw (GTK_WIDGET (timeline));
      return TRUE;
    }
    /* Right click on keyframe area */
    else if (event->button == 3) {
      if (keyframe) {
        /* Select this keyframe */
        timeline->selected_property = property;
        timeline->selected_keyframe = keyframe;
        
        /* Show context menu for keyframe */
        GtkWidget *menu = gtk_menu_new ();
        GtkWidget *edit_item = gtk_menu_item_new_with_label ("Edit Keyframe");
        GtkWidget *delete_item = gtk_menu_item_new_with_label ("Delete Keyframe");
        GtkWidget *linear_item = gtk_menu_item_new_with_label ("Linear Interpolation");
        GtkWidget *bezier_item = gtk_menu_item_new_with_label ("Bezier Interpolation");
        GtkWidget *constant_item = gtk_menu_item_new_with_label ("Constant Interpolation");
        GtkWidget *ease_in_item = gtk_menu_item_new_with_label ("Ease In");
        GtkWidget *ease_out_item = gtk_menu_item_new_with_label ("Ease Out");
        GtkWidget *ease_inout_item = gtk_menu_item_new_with_label ("Ease In/Out");
        
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), edit_item);
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), delete_item);
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), gtk_separator_menu_item_new ());
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), linear_item);
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), bezier_item);
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), constant_item);
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), ease_in_item);
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), ease_out_item);
        gtk_menu_shell_append (GTK_MENU_SHELL (menu), ease_inout_item);
        
        /* Connect signals */
        g_signal_connect_swapped (edit_item, "activate", G_CALLBACK (blouedit_timeline_show_keyframe_editor),
                                  timeline);
                                  
        g_object_set_data (G_OBJECT (delete_item), "timeline", timeline);
        g_object_set_data (G_OBJECT (delete_item), "property", property);
        g_object_set_data (G_OBJECT (delete_item), "keyframe", keyframe);
        g_signal_connect (delete_item, "activate", G_CALLBACK (on_delete_keyframe), NULL);
        
        /* Interpolation type menu items */
        g_object_set_data (G_OBJECT (linear_item), "timeline", timeline);
        g_object_set_data (G_OBJECT (linear_item), "property", property);
        g_object_set_data (G_OBJECT (linear_item), "keyframe", keyframe);
        g_object_set_data (G_OBJECT (linear_item), "interpolation",
                          GINT_TO_POINTER(BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR));
        g_signal_connect (linear_item, "activate", G_CALLBACK (on_set_keyframe_interpolation), NULL);
        
        g_object_set_data (G_OBJECT (bezier_item), "timeline", timeline);
        g_object_set_data (G_OBJECT (bezier_item), "property", property);
        g_object_set_data (G_OBJECT (bezier_item), "keyframe", keyframe);
        g_object_set_data (G_OBJECT (bezier_item), "interpolation",
                          GINT_TO_POINTER(BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER));
        g_signal_connect (bezier_item, "activate", G_CALLBACK (on_set_keyframe_interpolation), NULL);
        
        g_object_set_data (G_OBJECT (constant_item), "timeline", timeline);
        g_object_set_data (G_OBJECT (constant_item), "property", property);
        g_object_set_data (G_OBJECT (constant_item), "keyframe", keyframe);
        g_object_set_data (G_OBJECT (constant_item), "interpolation",
                          GINT_TO_POINTER(BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT));
        g_signal_connect (constant_item, "activate", G_CALLBACK (on_set_keyframe_interpolation), NULL);
        
        g_object_set_data (G_OBJECT (ease_in_item), "timeline", timeline);
        g_object_set_data (G_OBJECT (ease_in_item), "property", property);
        g_object_set_data (G_OBJECT (ease_in_item), "keyframe", keyframe);
        g_object_set_data (G_OBJECT (ease_in_item), "interpolation",
                          GINT_TO_POINTER(BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN));
        g_signal_connect (ease_in_item, "activate", G_CALLBACK (on_set_keyframe_interpolation), NULL);
        
        g_object_set_data (G_OBJECT (ease_out_item), "timeline", timeline);
        g_object_set_data (G_OBJECT (ease_out_item), "property", property);
        g_object_set_data (G_OBJECT (ease_out_item), "keyframe", keyframe);
        g_object_set_data (G_OBJECT (ease_out_item), "interpolation",
                          GINT_TO_POINTER(BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT));
        g_signal_connect (ease_out_item, "activate", G_CALLBACK (on_set_keyframe_interpolation), NULL);
        
        g_object_set_data (G_OBJECT (ease_inout_item), "timeline", timeline);
        g_object_set_data (G_OBJECT (ease_inout_item), "property", property);
        g_object_set_data (G_OBJECT (ease_inout_item), "keyframe", keyframe);
        g_object_set_data (G_OBJECT (ease_inout_item), "interpolation",
                          GINT_TO_POINTER(BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT));
        g_signal_connect (ease_inout_item, "activate", G_CALLBACK (on_set_keyframe_interpolation), NULL);
        
        /* Show the menu */
        gtk_widget_show_all (menu);
        gtk_menu_popup_at_pointer (GTK_MENU (menu), (GdkEvent *)event);
      } else {
        /* Right click on empty area - property operations menu */
        if (property) {
          /* Select the property */
          timeline->selected_property = property;
          
          /* Show property context menu */
          GtkWidget *menu = gtk_menu_new ();
          GtkWidget *add_item = gtk_menu_item_new_with_label ("Add Keyframe Here");
          GtkWidget *clear_item = gtk_menu_item_new_with_label ("Clear All Keyframes");
          
          gtk_menu_shell_append (GTK_MENU_SHELL (menu), add_item);
          gtk_menu_shell_append (GTK_MENU_SHELL (menu), clear_item);
          
          /* Connect signals */
          g_object_set_data (G_OBJECT (add_item), "timeline", timeline);
          g_object_set_data (G_OBJECT (add_item), "property", property);
          g_object_set_data (G_OBJECT (add_item), "position", GINT_TO_POINTER(position));
          g_signal_connect (add_item, "activate", G_CALLBACK (on_add_keyframe), NULL);
          
          g_object_set_data (G_OBJECT (clear_item), "timeline", timeline);
          g_object_set_data (G_OBJECT (clear_item), "property", property);
          g_signal_connect (clear_item, "activate", G_CALLBACK (on_clear_keyframes), NULL);
          
          /* Show the menu */
          gtk_widget_show_all (menu);
          gtk_menu_popup_at_pointer (GTK_MENU (menu), (GdkEvent *)event);
        }
      }
      
      /* Redraw */
      gtk_widget_queue_draw (GTK_WIDGET (timeline));
      return TRUE;
    }
    
    return TRUE;
  }
  
  /* Check if we're in the timeline area (not in track headers) */
  if (x < timeline->timeline_start_x) {
    /* In track header area */
    BlouEditTimelineTrack *track = blouedit_timeline_get_track_at_y (timeline, y);
    
    if (track) {
      /* Track header clicked, select this track */
      timeline->selected_track = track;
      
      /* Check if this is a track reordering operation (Ctrl+click on track header) */
      if (event->button == 1 && (event->state & GDK_CONTROL_MASK)) {
        blouedit_timeline_start_track_reorder (timeline, track, y);
        return TRUE;
      }
      
      /* Check if the click is near the track's bottom edge for resizing */
      int track_height = blouedit_timeline_get_track_height (timeline, track);
      int track_y = 0;
      
      /* Determine track's y position */
      GSList *l;
      for (l = timeline->tracks; l != NULL; l = l->next) {
        BlouEditTimelineTrack *t = (BlouEditTimelineTrack *)l->data;
        
        if (t == track) {
          break;
        }
        
        track_y += blouedit_timeline_get_track_height (timeline, t) + timeline->track_spacing;
      }
      
      /* Check if click is near the bottom edge of the track */
      if (y >= track_y + track_height - 5 && y <= track_y + track_height + 5) {
        /* Start track resize operation */
        blouedit_timeline_start_track_resize (timeline, track, y);
        return TRUE;
      }
      
      /* Check if it's a double-click to toggle fold state */
      if (event->type == GDK_2BUTTON_PRESS) {
        blouedit_timeline_toggle_track_folded (timeline, track);
        return TRUE;
      }
    }
    
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
    return TRUE;
  }
  
  /* Handle clicks in timeline area */
  
  /* Find the track at this y position */
  BlouEditTimelineTrack *track = blouedit_timeline_get_track_at_y (timeline, y);
  if (!track)
    return FALSE;
  
  /* Convert x position to timeline position */
  double rel_x = x - timeline->timeline_start_x;
  gint64 position = rel_x / timeline->zoom_level;
  
  /* Process depending on mouse button */
  if (event->button == 1) { /* Left button */
    /* First check if this is a click on a clip */
    GESClip *clip = blouedit_timeline_get_clip_at_position (timeline, position, track);
    
    if (clip) {
      /* Check if we're clicking on a clip edge for trimming */
      BlouEditClipEdge edge = blouedit_timeline_get_clip_edge_at_position (timeline, clip, x, timeline->trim_handle_size);
      
      if (edge != BLOUEDIT_EDGE_NONE) {
        /* Start trim operation */
        timeline->is_trimming = TRUE;
        timeline->trimming_clip = clip;
        timeline->trimming_edge = edge;
        timeline->trim_start_position = edge == BLOUEDIT_EDGE_START ? 
                                       ges_clip_get_start (clip) : 
                                       ges_clip_get_start (clip) + ges_clip_get_duration (clip);
        timeline->trim_start_x = x;
        
        /* Select the clip */
        blouedit_timeline_select_clip (timeline, clip, !(event->state & GDK_SHIFT_MASK));
        
        return TRUE;
      } else {
        /* Regular clip click - start drag operation */
        
        /* Select the clip */
        /* If shift is held, add to current selection */
        blouedit_timeline_select_clip (timeline, clip, !(event->state & GDK_SHIFT_MASK));
        
        /* Check if we have multiple clips selected */
        if (timeline->selected_clips && g_slist_length(timeline->selected_clips) > 1) {
          /* Start multiple clip movement */
          blouedit_timeline_start_moving_multiple_clips (timeline, x);
        } else {
          /* Single clip movement */
          timeline->is_dragging_clip = TRUE;
          timeline->dragging_clip = clip;
          timeline->drag_start_position = ges_clip_get_start (clip);
          timeline->drag_start_x = x;
        }
        
        return TRUE;
      }
    } else {
      /* Clicking on empty timeline area */
      
      /* Set playhead position */
      blouedit_timeline_set_playhead_position_from_x (timeline, x);
      
      /* Clear selection if not holding shift */
      if (!(event->state & GDK_SHIFT_MASK)) {
        blouedit_timeline_clear_selection (timeline);
      }
      
      /* Start scrubbing */
      blouedit_timeline_start_scrubbing (timeline, x);
    }
  } else if (event->button == 3) { /* Right button */
    /* Handle right-click contextual operations */
    GESClip *clip = blouedit_timeline_get_clip_at_position (timeline, position, track);
    
    if (clip) {
      /* If the right-clicked clip isn't already selected, select it */
      if (!blouedit_timeline_is_clip_selected (timeline, clip)) {
        blouedit_timeline_select_clip (timeline, clip, !(event->state & GDK_SHIFT_MASK));
      }
      
      /* Show clip context menu */
      GtkWidget *menu = gtk_menu_new ();
      GtkWidget *item;
      
      /* Cut */
      item = gtk_menu_item_new_with_label (_("Cut"));
      g_object_set_data (G_OBJECT (item), "timeline", timeline);
      g_object_set_data (G_OBJECT (item), "clip", clip);
      gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
      
      /* Copy */
      item = gtk_menu_item_new_with_label (_("Copy"));
      g_object_set_data (G_OBJECT (item), "timeline", timeline);
      g_object_set_data (G_OBJECT (item), "clip", clip);
      gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
      
      /* Delete */
      item = gtk_menu_item_new_with_label (_("Delete"));
      g_object_set_data (G_OBJECT (item), "timeline", timeline);
      g_object_set_data (G_OBJECT (item), "clip", clip);
      gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
      
      /* Separator */
      item = gtk_separator_menu_item_new ();
      gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
      
      /* Generate Proxy */
      item = gtk_menu_item_new_with_label (_("Generate Proxy"));
      g_object_set_data (G_OBJECT (item), "timeline", timeline);
      g_object_set_data (G_OBJECT (item), "clip", clip);
      /* We need to wrap the call because the function expects different parameters than the signal provides */
      g_signal_connect (item, "activate", G_CALLBACK (on_generate_proxy_for_clip), NULL);
      gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
      
      /* Show the menu */
      gtk_widget_show_all (menu);
      gtk_menu_popup_at_pointer (GTK_MENU (menu), (GdkEvent *)event);
    } else {
      /* Right click on empty area */
      blouedit_timeline_show_context_menu (timeline, x, y);
    }
  }
  
  return TRUE;
}

gboolean
blouedit_timeline_handle_motion (BlouEditTimeline *timeline, GdkEventMotion *event)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (event != NULL, FALSE);
  
  /* Get local coordinates */
  double x = event->x;
  double y = event->y;
  
  /* Handle keyframe dragging if active */
  if (timeline->is_moving_keyframe && timeline->moving_keyframe && timeline->selected_property) {
    /* Calculate new position and value based on mouse movement */
    double rel_x = x - timeline->timeline_start_x;
    gint64 new_position = (rel_x / timeline->zoom_level) * GST_SECOND;
    
    /* Calculate the property row's y-range */
    int keyframe_area_y = gtk_widget_get_allocated_height (GTK_WIDGET (timeline)) - timeline->keyframe_area_height;
    int property_height = 20; /* Same as in draw function */
    
    /* Find the property's index to determine its y position */
    int visible_index = 0;
    for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
      BlouEditAnimatableProperty *p = (BlouEditAnimatableProperty *)prop->data;
      
      if (!p->visible)
        continue;
      
      if (p == timeline->selected_property)
        break;
      
      visible_index++;
    }
    
    int property_y = keyframe_area_y + (visible_index * property_height);
    
    /* Calculate new value based on y position */
    gdouble prop_value_range = timeline->selected_property->max_value - timeline->selected_property->min_value;
    gdouble normalized_value = 1.0 - ((y - property_y) / property_height);
    normalized_value = CLAMP(normalized_value, 0.0, 1.0);
    gdouble new_value = timeline->selected_property->min_value + (normalized_value * prop_value_range);
    
    /* Update the keyframe with new position and value */
    blouedit_timeline_update_keyframe (timeline, timeline->selected_property, 
                                     timeline->moving_keyframe, new_position, new_value,
                                     timeline->moving_keyframe->interpolation);
    
    /* Apply keyframes to update property */
    blouedit_timeline_apply_keyframes (timeline);
    
    /* Redraw timeline */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
    
    return TRUE;
  }
  
  /* Handle keyframe handle editing if active */
  if (timeline->is_editing_keyframe_handle && timeline->handle_keyframe && timeline->selected_property) {
    /* Calculate relative coordinates from the keyframe position */
    int keyframe_area_y = gtk_widget_get_allocated_height (GTK_WIDGET (timeline)) - timeline->keyframe_area_height;
    int property_height = 20; /* Same as in draw function */
    
    /* Find the property's index to determine its y position */
    int visible_index = 0;
    for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
      BlouEditAnimatableProperty *p = (BlouEditAnimatableProperty *)prop->data;
      
      if (!p->visible)
        continue;
      
      if (p == timeline->selected_property)
        break;
      
      visible_index++;
    }
    
    int property_y = keyframe_area_y + (visible_index * property_height) + property_height / 2;
    
    /* Calculate keyframe position in UI coordinates */
    int keyframe_x = timeline->timeline_start_x + 
                    (timeline->handle_keyframe->position * timeline->zoom_level / GST_SECOND);
    
    /* Calculate normalized value of the keyframe */
    gdouble normalized_value = (timeline->handle_keyframe->value - timeline->selected_property->min_value) / 
                              (timeline->selected_property->max_value - timeline->selected_property->min_value);
    int keyframe_y = property_y - (normalized_value * property_height / 2);
    
    /* Calculate handle position - relative to keyframe in normalized coordinates */
    gdouble handle_x = (x - keyframe_x) / 50.0; /* Scale to reasonable handle lengths */
    gdouble handle_y = (keyframe_y - y) / 50.0;
    
    /* Update the appropriate handle */
    if (timeline->is_editing_left_handle) {
      /* Ensure left handle's X is negative (to the left of keyframe) */
      handle_x = MIN(handle_x, -0.01);
      
      /* Update left handle */
      blouedit_timeline_update_keyframe_handles (timeline, timeline->selected_property,
                                              timeline->handle_keyframe,
                                              handle_x, handle_y,
                                              timeline->handle_keyframe->handle_right_x,
                                              timeline->handle_keyframe->handle_right_y);
    } else {
      /* Ensure right handle's X is positive (to the right of keyframe) */
      handle_x = MAX(handle_x, 0.01);
      
      /* Update right handle */
      blouedit_timeline_update_keyframe_handles (timeline, timeline->selected_property,
                                              timeline->handle_keyframe,
                                              timeline->handle_keyframe->handle_left_x,
                                              timeline->handle_keyframe->handle_left_y,
                                              handle_x, handle_y);
    }
    
    /* Apply keyframes to update property */
    blouedit_timeline_apply_keyframes (timeline);
    
    /* Redraw timeline */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
    
    return TRUE;
  }
  
  /* Handle track reordering operation if active */
  if (timeline->is_reordering_track) {
    blouedit_timeline_reorder_track_to (timeline, y);
    return TRUE;
  }
  
  /* Handle track resize if active */
  if (timeline->is_resizing_track) {
    blouedit_timeline_resize_track_to (timeline, y);
    return TRUE;
  }
  
  /* Handle clip trimming operation */
  if (timeline->is_trimming) {
    /* Calculate the new position based on mouse movement */
    double rel_x = x - timeline->timeline_start_x;
    gint64 new_position = rel_x / timeline->zoom_level;
    
    /* Apply the appropriate trim operation based on edit mode */
    if (timeline->edit_mode == BLOUEDIT_EDIT_MODE_RIPPLE) {
      blouedit_timeline_ripple_trim (timeline, timeline->trimming_clip, 
                                    timeline->trimming_edge, new_position);
    } else if (timeline->edit_mode == BLOUEDIT_EDIT_MODE_ROLL) {
      blouedit_timeline_roll_edit (timeline, timeline->trimming_clip,
                                  timeline->trimming_edge, new_position);
    } else {
      /* Normal mode */
      blouedit_timeline_trim_clip (timeline, timeline->trimming_clip,
                                  timeline->trimming_edge, new_position);
    }
    
    return TRUE;
  }
  
  /* Handle multiple clip moving operation */
  if (timeline->is_moving_multiple_clips) {
    /* Update positions of all selected clips */
    blouedit_timeline_move_multiple_clips_to (timeline, x);
    return TRUE;
  }
  
  /* Handle clip dragging operation */
  if (timeline->is_dragging_clip) {
    /* Calculate the new position based on mouse movement */
    double rel_x = x - timeline->timeline_start_x;
    gint64 new_position = rel_x / timeline->zoom_level;
    
    /* Get movement offset */
    gint64 offset = new_position - timeline->drag_start_position;
    
    /* If it's a slip operation (holding Alt key) */
    if (event->state & GDK_MOD1_MASK) {
      /* Calculate slip offset based on x movement */
      gint64 slip_offset = (x - timeline->drag_start_x) / timeline->zoom_level;
      blouedit_timeline_slip_clip (timeline, timeline->dragging_clip, slip_offset);
    } 
    /* If it's a slide operation (holding Ctrl key) */
    else if (event->state & GDK_CONTROL_MASK) {
      blouedit_timeline_slide_clip (timeline, timeline->dragging_clip, new_position);
    }
    /* Regular move operation */
    else {
      /* Move the clip */
      guint64 start = ges_clip_get_start (timeline->dragging_clip);
      
      /* Snap position if enabled */
      gint64 target_position = start + offset;
      if (timeline->snap_mode != BLOUEDIT_SNAP_NONE) {
        target_position = blouedit_timeline_snap_position (timeline, target_position);
      }
      
      /* Update position */
      if (target_position != start) {
        ges_timeline_element_set_start (GES_TIMELINE_ELEMENT (timeline->dragging_clip), target_position);
        gtk_widget_queue_draw (GTK_WIDGET (timeline));
      }
    }
    
    return TRUE;
  }
  
  /* Handle normal timeline scrubbing */
  if (timeline->is_scrubbing) {
    blouedit_timeline_scrub_to (timeline, x);
    return TRUE;
  }
  
  /* Update cursor based on what's under the pointer */
  
  /* Check if we're in the keyframe area */
  int keyframe_area_y = gtk_widget_get_allocated_height (GTK_WIDGET (timeline)) - timeline->keyframe_area_height;
  if (timeline->show_keyframes && y >= keyframe_area_y && x >= timeline->timeline_start_x) {
    /* We're in the keyframe curves area */
    
    /* Find which property row we're in */
    int property_height = 20; /* Same as in draw function */
    int property_index = (y - keyframe_area_y) / property_height;
    
    /* Find the property at this index */
    BlouEditAnimatableProperty *property = NULL;
    int visible_index = 0;
    
    for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
      BlouEditAnimatableProperty *p = (BlouEditAnimatableProperty *)prop->data;
      
      if (!p->visible)
        continue;
      
      if (visible_index == property_index) {
        property = p;
        break;
      }
      
      visible_index++;
    }
    
    if (property) {
      /* Convert x position to timeline position */
      double rel_x = x - timeline->timeline_start_x;
      gint64 position = rel_x / timeline->zoom_level * GST_SECOND;
      
      /* Find a keyframe at this position */
      BlouEditKeyframe *keyframe = blouedit_timeline_get_keyframe_at_position (
          timeline, property, position, GST_SECOND / 10);
      
      if (keyframe) {
        /* We're over a keyframe - show move cursor */
        GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
        if (window) {
          GdkCursor *cursor = gdk_cursor_new_for_display (
              gdk_window_get_display (window), GDK_FLEUR);
          gdk_window_set_cursor (window, cursor);
          g_object_unref (cursor);
        }
        return TRUE;
      } else {
        /* We're in a property row but not over a keyframe - show crosshair cursor */
        GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
        if (window) {
          GdkCursor *cursor = gdk_cursor_new_for_display (
              gdk_window_get_display (window), GDK_CROSSHAIR);
          gdk_window_set_cursor (window, cursor);
          g_object_unref (cursor);
        }
        return TRUE;
      }
    }
  }
  
  /* If we're over a clip edge, show resize cursor */
  if (x >= timeline->timeline_start_x) {
    double rel_x = x - timeline->timeline_start_x;
    gint64 position = rel_x / timeline->zoom_level;
    
    BlouEditTimelineTrack *track = blouedit_timeline_get_track_at_y (timeline, y);
    if (track) {
      GESClip *clip = blouedit_timeline_get_clip_at_position (timeline, position, track);
      if (clip) {
        BlouEditClipEdge edge = blouedit_timeline_get_clip_edge_at_position (timeline, clip, x, timeline->trim_handle_size);
        if (edge != BLOUEDIT_EDGE_NONE) {
          GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
          if (window) {
            GdkCursor *cursor = gdk_cursor_new_for_display (
                gdk_window_get_display (window), GDK_SB_H_DOUBLE_ARROW);
            gdk_window_set_cursor (window, cursor);
            g_object_unref (cursor);
            return TRUE;
          }
        }
      }
    }
  }
  
  /* Reset to default cursor if we're not over anything special */
  GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
  if (window) {
    gdk_window_set_cursor (window, NULL);
  }
  
  return FALSE;
}

/* Handle motion events for tracks, clips, etc. */
gboolean
blouedit_timeline_handle_button_release (BlouEditTimeline *timeline, GdkEventButton *event)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (event != NULL, FALSE);
  
  /* Handle keyframe dragging completion */
  if (timeline->is_moving_keyframe) {
    timeline->is_moving_keyframe = FALSE;
    timeline->moving_keyframe = NULL;
    
    /* Reset cursor */
    GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
    if (window) {
      gdk_window_set_cursor (window, NULL);
    }
    
    return TRUE;
  }
  
  /* Handle keyframe handle editing completion */
  if (timeline->is_editing_keyframe_handle) {
    timeline->is_editing_keyframe_handle = FALSE;
    timeline->handle_keyframe = NULL;
    timeline->is_editing_left_handle = FALSE;
    
    /* Reset cursor */
    GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
    if (window) {
      gdk_window_set_cursor (window, NULL);
    }
    
    return TRUE;
  }
  
  /* Handle track reordering completion */
  if (timeline->is_reordering_track) {
    blouedit_timeline_end_track_reorder (timeline);
    return TRUE;
  }
  
  /* Handle track resize completion */
  if (timeline->is_resizing_track) {
    blouedit_timeline_end_track_resize (timeline);
    return TRUE;
  }
  
  /* Handle clip trimming completion */
  if (timeline->is_trimming) {
    timeline->is_trimming = FALSE;
    timeline->trimming_clip = NULL;
    timeline->trimming_edge = BLOUEDIT_EDGE_NONE;
    return TRUE;
  }
  
  /* Handle multiple clip movement completion */
  if (timeline->is_moving_multiple_clips) {
    blouedit_timeline_end_moving_multiple_clips (timeline);
    return TRUE;
  }
  
  /* Handle clip dragging completion */
  if (timeline->is_dragging_clip) {
    timeline->is_dragging_clip = FALSE;
    timeline->dragging_clip = NULL;
    return TRUE;
  }
  
  /* End timeline scrubbing */
  if (timeline->is_scrubbing) {
    blouedit_timeline_end_scrubbing (timeline);
    return TRUE;
  }
  
  return FALSE;
}

/* Timecode functions */
gchar *
blouedit_timeline_position_to_timecode (BlouEditTimeline *timeline, gint64 position, BlouEditTimecodeFormat format)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  gdouble framerate = blouedit_timeline_get_framerate (timeline);
  gint64 frames, seconds, minutes, hours;
  gchar *timecode = NULL;
  
  /* Convert position (in timeline units, which is nanoseconds in GStreamer) to chosen format */
  switch (format) {
    case BLOUEDIT_TIMECODE_FORMAT_FRAMES:
      /* Convert to frames */
      frames = GST_CLOCK_TIME_TO_FRAMES (position, framerate);
      timecode = g_strdup_printf ("%"G_GINT64_FORMAT, frames);
      break;
      
    case BLOUEDIT_TIMECODE_FORMAT_SECONDS:
      /* Convert to seconds with decimal milliseconds */
      seconds = position / GST_SECOND;
      gint milliseconds = (position % GST_SECOND) / (GST_SECOND / 1000);
      timecode = g_strdup_printf ("%"G_GINT64_FORMAT".%03d", seconds, milliseconds);
      break;
      
    case BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_FF:
      /* Convert to HH:MM:SS:FF format */
      hours = position / (GST_SECOND * 60 * 60);
      minutes = (position % (GST_SECOND * 60 * 60)) / (GST_SECOND * 60);
      seconds = (position % (GST_SECOND * 60)) / GST_SECOND;
      frames = GST_CLOCK_TIME_TO_FRAMES (position % GST_SECOND, framerate);
      
      timecode = g_strdup_printf ("%02"G_GINT64_FORMAT":%02"G_GINT64_FORMAT":%02"G_GINT64_FORMAT":%02"G_GINT64_FORMAT,
                                 hours, minutes, seconds, frames);
      break;
      
    case BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_MS:
      /* Convert to HH:MM:SS.mmm format */
      hours = position / (GST_SECOND * 60 * 60);
      minutes = (position % (GST_SECOND * 60 * 60)) / (GST_SECOND * 60);
      seconds = (position % (GST_SECOND * 60)) / GST_SECOND;
      gint milliseconds = (position % GST_SECOND) / (GST_SECOND / 1000);
      
      timecode = g_strdup_printf ("%02"G_GINT64_FORMAT":%02"G_GINT64_FORMAT":%02"G_GINT64_FORMAT".%03d",
                                 hours, minutes, seconds, milliseconds);
      break;
      
    default:
      /* Default to seconds format */
      seconds = position / GST_SECOND;
      timecode = g_strdup_printf ("%"G_GINT64_FORMAT, seconds);
      break;
  }
  
  return timecode;
}

gint64
blouedit_timeline_timecode_to_position (BlouEditTimeline *timeline, const gchar *timecode, BlouEditTimecodeFormat format)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  g_return_val_if_fail (timecode != NULL, 0);
  
  gdouble framerate = blouedit_timeline_get_framerate (timeline);
  gint64 position = 0;
  
  switch (format) {
    case BLOUEDIT_TIMECODE_FORMAT_FRAMES:
      {
        /* Parse frames */
        gint64 frames = g_ascii_strtoll (timecode, NULL, 10);
        position = GST_FRAMES_TO_CLOCK_TIME (frames, framerate);
      }
      break;
      
    case BLOUEDIT_TIMECODE_FORMAT_SECONDS:
      {
        /* Parse seconds.milliseconds */
        gdouble seconds = g_ascii_strtod (timecode, NULL);
        position = seconds * GST_SECOND;
      }
      break;
      
    case BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_FF:
      {
        /* Parse HH:MM:SS:FF format */
        guint hours, minutes, seconds, frames;
        if (sscanf (timecode, "%u:%u:%u:%u", &hours, &minutes, &seconds, &frames) == 4) {
          position = GST_SECOND * (hours * 60 * 60 + minutes * 60 + seconds);
          position += GST_FRAMES_TO_CLOCK_TIME (frames, framerate);
        }
      }
      break;
      
    case BLOUEDIT_TIMECODE_FORMAT_HH_MM_SS_MS:
      {
        /* Parse HH:MM:SS.mmm format */
        guint hours, minutes, seconds, milliseconds;
        if (sscanf (timecode, "%u:%u:%u.%u", &hours, &minutes, &seconds, &milliseconds) == 4) {
          position = GST_SECOND * (hours * 60 * 60 + minutes * 60 + seconds);
          position += milliseconds * (GST_SECOND / 1000);
        }
      }
      break;
      
    default:
      break;
  }
  
  return position;
}

void
blouedit_timeline_goto_timecode (BlouEditTimeline *timeline, const gchar *timecode)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (timecode != NULL);
  
  /* Parse the timecode using the current format and go to that position */
  BlouEditTimecodeFormat format = blouedit_timeline_get_timecode_format (timeline);
  gint64 position = blouedit_timeline_timecode_to_position (timeline, timecode, format);
  
  /* Set timeline position */
  blouedit_timeline_set_position (timeline, position);
}

/* Draw timecode ruler function */
static void
blouedit_timeline_draw_timecode_ruler (BlouEditTimeline *timeline, cairo_t *cr, int width, int height)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Set up drawing parameters */
  int ruler_height = timeline->ruler_height;
  int timeline_start_x = timeline->timeline_start_x;
  gdouble zoom_level = timeline->zoom_level;
  
  /* Get timeline properties */
  gint64 timeline_duration = blouedit_timeline_get_duration (timeline);
  BlouEditTimecodeFormat format = blouedit_timeline_get_timecode_format (timeline);
  gdouble framerate = blouedit_timeline_get_framerate (timeline);
  
  /* Clear the ruler area */
  cairo_save (cr);
  cairo_set_source_rgb (cr, 0.2, 0.2, 0.2); /* Dark gray background */
  cairo_rectangle (cr, 0, 0, width, ruler_height);
  cairo_fill (cr);
  
  /* Draw timecode markings */
  cairo_set_source_rgb (cr, 0.8, 0.8, 0.8); /* Light gray markings */
  cairo_set_line_width (cr, 1.0);
  
  /* Draw the ruler line at the bottom */
  cairo_move_to (cr, timeline_start_x, ruler_height - 0.5);
  cairo_line_to (cr, width, ruler_height - 0.5);
  cairo_stroke (cr);
  
  /* Calculate interval for timecode markings based on zoom level */
  gint64 interval;
  if (zoom_level > 5.0)
    interval = GST_SECOND / 10; /* 100ms intervals at high zoom */
  else if (zoom_level > 2.0)
    interval = GST_SECOND / 2; /* 500ms intervals at medium zoom */
  else if (zoom_level > 0.5)
    interval = GST_SECOND; /* 1s intervals at normal zoom */
  else if (zoom_level > 0.2)
    interval = 5 * GST_SECOND; /* 5s intervals at low zoom */
  else if (zoom_level > 0.05)
    interval = 30 * GST_SECOND; /* 30s intervals at very low zoom */
  else
    interval = 60 * GST_SECOND; /* 1min intervals at extremely low zoom */
  
  /* Draw the markings and labels */
  PangoLayout *layout = pango_cairo_create_layout (cr);
  PangoFontDescription *font_desc = pango_font_description_from_string ("Sans 8");
  pango_layout_set_font_description (layout, font_desc);
  pango_font_description_free (font_desc);
  
  /* Start from 0 and go to the end of the timeline */
  for (gint64 pos = 0; pos <= timeline_duration; pos += interval) {
    /* Convert timeline position to X coordinate */
    int x = timeline_start_x + (int)(pos * zoom_level / GST_SECOND);
    
    /* Skip if outside the visible area */
    if (x < timeline_start_x)
      continue;
    if (x > width)
      break;
    
    /* Draw tick mark */
    cairo_move_to (cr, x + 0.5, ruler_height - 5);
    cairo_line_to (cr, x + 0.5, ruler_height);
    cairo_stroke (cr);
    
    /* Draw timecode label for major intervals */
    if (pos % (5 * interval) == 0 || interval >= 30 * GST_SECOND) {
      /* Convert position to timecode */
      gchar *timecode = blouedit_timeline_position_to_timecode (timeline, pos, format);
      
      /* Measure the text */
      pango_layout_set_text (layout, timecode, -1);
      int text_width, text_height;
      pango_layout_get_pixel_size (layout, &text_width, &text_height);
      
      /* Draw the timecode centered on the mark */
      cairo_move_to (cr, x - text_width/2, ruler_height - text_height - 5);
      pango_cairo_show_layout (cr, layout);
      
      g_free (timecode);
    }
  }
  
  /* Draw playhead position */
  gint64 position = blouedit_timeline_get_position (timeline);
  int playhead_x = timeline_start_x + (int)(position * zoom_level / GST_SECOND);
  
  /* Draw playhead line */
  cairo_set_source_rgb (cr, 1.0, 0.0, 0.0); /* Red playhead */
  cairo_set_line_width (cr, 2.0);
  cairo_move_to (cr, playhead_x + 0.5, 0);
  cairo_line_to (cr, playhead_x + 0.5, ruler_height);
  cairo_stroke (cr);
  
  /* Draw current timecode at the playhead */
  if (blouedit_timeline_get_show_timecode (timeline)) {
    gchar *timecode = blouedit_timeline_position_to_timecode (timeline, position, format);
    
    /* Create a background box for the timecode */
    pango_layout_set_text (layout, timecode, -1);
    int text_width, text_height;
    pango_layout_get_pixel_size (layout, &text_width, &text_height);
    
    /* Position the timecode at the playhead but keep it in view */
    int tc_x = playhead_x - text_width/2;
    if (tc_x < timeline_start_x)
      tc_x = timeline_start_x;
    if (tc_x + text_width > width)
      tc_x = width - text_width;
    
    /* Draw background box */
    cairo_set_source_rgba (cr, 0.0, 0.0, 0.0, 0.7); /* Semi-transparent black */
    cairo_rectangle (cr, tc_x - 2, 2, text_width + 4, text_height + 2);
    cairo_fill (cr);
    
    /* Draw text */
    cairo_set_source_rgb (cr, 1.0, 1.0, 1.0); /* White text */
    cairo_move_to (cr, tc_x, 3);
    pango_cairo_show_layout (cr, layout);
    
    g_free (timecode);
  }
  
  g_object_unref (layout);
  cairo_restore (cr);
}

/* Add timecode entry widget for manual navigation */
static GtkWidget *
blouedit_timeline_create_timecode_entry (BlouEditTimeline *timeline)
{
  GtkWidget *entry = gtk_entry_new ();
  
  /* Set up entry properties */
  gtk_entry_set_max_length (GTK_ENTRY (entry), 15); /* Enough for HH:MM:SS:FF */
  gtk_entry_set_width_chars (GTK_ENTRY (entry), 15);
  
  /* Set initial value to current position */
  gint64 position = blouedit_timeline_get_position (timeline);
  BlouEditTimecodeFormat format = blouedit_timeline_get_timecode_format (timeline);
  gchar *timecode = blouedit_timeline_position_to_timecode (timeline, position, format);
  gtk_entry_set_text (GTK_ENTRY (entry), timecode);
  g_free (timecode);
  
  /* Connect to "activate" signal for navigation when Enter is pressed */
  g_signal_connect_data (entry, "activate", G_CALLBACK (timecode_entry_activate_cb), timeline, NULL, 0);
  
  return entry;
}

/* Callback for timecode entry widget */
static void
timecode_entry_activate_cb (GtkEntry *entry, BlouEditTimeline *timeline)
{
  const gchar *text = gtk_entry_get_text (entry);
  
  /* Go to the entered timecode */
  blouedit_timeline_goto_timecode (timeline, text);
}

/* Update the draw function to include timecode ruler */
static gboolean
blouedit_timeline_draw (GtkWidget *widget, cairo_t *cr)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (widget);
  
  /* Get widget dimensions */
  int width = gtk_widget_get_allocated_width (widget);
  int height = gtk_widget_get_allocated_height (widget);
  
  /* Draw background */
  cairo_set_source_rgb (cr, 0.1, 0.1, 0.1); /* Dark background */
  cairo_rectangle (cr, 0, 0, width, height);
  cairo_fill (cr);
  
  /* Draw ruler at the top based on scale mode */
  if (timeline->scale_mode == BLOUEDIT_TIMELINE_SCALE_CUSTOM) {
    blouedit_timeline_draw_scale(timeline, cr, width, timeline->ruler_height);
  } else {
    blouedit_timeline_draw_timecode_ruler(timeline, cr, width, height);
  }
  
  /* If in multicam source view mode, draw the source view */
  if (timeline->edit_mode == BLOUEDIT_EDIT_MODE_MULTICAM && 
      timeline->multicam_mode == BLOUEDIT_MULTICAM_MODE_SOURCE_VIEW) {
    blouedit_timeline_draw_multicam_source_view(timeline, cr, width, height);
  }
  
  /* Draw edge trimming UI if active */
  blouedit_timeline_draw_edge_trimming_ui(timeline, cr, width, height);
  
  /* Calculate vertical positions */
  int y_offset = timeline->ruler_height;
  
  /* Draw track headers and clips here... */
  /* Note: When rendering clips, check visibility with:
   * if (blouedit_timeline_is_clip_visible (timeline, clip)) {
   *   // Draw the clip
   * }
   */
  
  /* Draw the edit mode overlay if needed */
  blouedit_timeline_draw_edit_mode_overlay(timeline, cr, width, height);
  
  /* Draw keyframes area if we have properties and keyframes are visible */
  if (timeline->show_keyframes && timeline->animatable_properties) {
    /* Draw keyframe area separator line */
    cairo_set_source_rgb (cr, 0.3, 0.3, 0.3);
    cairo_set_line_width (cr, 1.0);
    cairo_move_to (cr, 0, height - timeline->keyframe_area_height);
    cairo_line_to (cr, width, height - timeline->keyframe_area_height);
    cairo_stroke (cr);
    
    /* Draw keyframe area background */
    cairo_set_source_rgb (cr, 0.15, 0.15, 0.15);
    cairo_rectangle (cr, 0, height - timeline->keyframe_area_height, width, timeline->keyframe_area_height);
    cairo_fill (cr);
    
    /* Draw keyframe property labels */
    PangoLayout *layout = pango_cairo_create_layout (cr);
    PangoFontDescription *font_desc = pango_font_description_from_string ("Sans 9");
    pango_layout_set_font_description (layout, font_desc);
    pango_font_description_free (font_desc);
    
    int property_height = 20; /* Height per property */
    int keyframe_area_y = height - timeline->keyframe_area_height;
    int property_y = keyframe_area_y;
    
    /* Draw header for properties area */
  cairo_set_source_rgb (cr, 0.2, 0.2, 0.2);
    cairo_rectangle (cr, 0, keyframe_area_y, timeline->timeline_start_x, timeline->keyframe_area_height);
  cairo_fill (cr);
  
    cairo_set_source_rgb (cr, 0.7, 0.7, 0.7);
    pango_layout_set_text (layout, "Properties", -1);
    cairo_move_to (cr, 5, keyframe_area_y + 3);
    pango_cairo_show_layout (cr, layout);
    
    /* Draw visible properties and their keyframes */
    for (GSList *prop = timeline->animatable_properties; prop; prop = prop->next) {
      BlouEditAnimatableProperty *property = (BlouEditAnimatableProperty *)prop->data;
      
      /* Skip hidden properties */
      if (!property->visible)
        continue;
      
      /* Draw property background - highlight if selected */
      if (property == timeline->selected_property) {
        cairo_set_source_rgb (cr, 0.3, 0.3, 0.5); /* Highlight selected property */
      } else {
        cairo_set_source_rgb (cr, 0.2, 0.2, 0.2);
      }
      cairo_rectangle (cr, 0, property_y, timeline->timeline_start_x, property_height);
      cairo_fill (cr);
      
      /* Draw property name */
      cairo_set_source_rgb (cr, 0.8, 0.8, 0.8);
      pango_layout_set_text (layout, property->display_name, -1);
      cairo_move_to (cr, 5, property_y + 3);
      pango_cairo_show_layout (cr, layout);
      
      /* Draw grid lines for this property */
      cairo_set_source_rgba (cr, 0.3, 0.3, 0.3, 0.5);
      cairo_set_line_width (cr, 0.5);
      
      /* Horizontal center line */
      cairo_move_to (cr, timeline->timeline_start_x, property_y + property_height/2);
      cairo_line_to (cr, width, property_y + property_height/2);
      cairo_stroke (cr);
      
      /* Vertical grid lines at major intervals */
      gint64 timeline_duration = blouedit_timeline_get_duration (timeline);
      for (gint64 pos = 0; pos <= timeline_duration; pos += GST_SECOND) {
        int x = timeline->timeline_start_x + (pos * timeline->zoom_level / GST_SECOND);
        if (x >= timeline->timeline_start_x && x <= width) {
          cairo_move_to (cr, x, property_y);
          cairo_line_to (cr, x, property_y + property_height);
          cairo_stroke (cr);
        }
      }
      
      /* Draw keyframe value curve */
      if (property->keyframes && g_slist_length(property->keyframes) > 1) {
        cairo_set_source_rgba (cr, 0.2, 0.7, 0.9, 0.8);
        cairo_set_line_width (cr, 1.5);
        
        /* Get the value range for scaling */
        gdouble value_min = property->min_value;
        gdouble value_max = property->max_value;
        gdouble value_range = value_max - value_min;
        
        /* Skip if range is invalid */
        if (value_range <= 0)
          continue;
        
        /* Calculate number of segments to draw for smooth curves */
        int segments_per_second = MAX(10, timeline->zoom_level * 5);
        int min_segments = 20;
        
        BlouEditKeyframe *first_keyframe = (BlouEditKeyframe *)property->keyframes->data;
        BlouEditKeyframe *last_keyframe = (BlouEditKeyframe *)g_slist_last(property->keyframes)->data;
        
        /* Start the path at the first keyframe */
        gdouble first_x = timeline->timeline_start_x + (first_keyframe->position * timeline->zoom_level / GST_SECOND);
        gdouble first_y = property_y + property_height - 
                          ((first_keyframe->value - value_min) / value_range) * property_height;
        
        cairo_move_to (cr, first_x, first_y);
        
        /* For each segment between keyframes, draw the curve */
        GSList *current = property->keyframes;
        while (current && current->next) {
          BlouEditKeyframe *k1 = (BlouEditKeyframe *)current->data;
          BlouEditKeyframe *k2 = (BlouEditKeyframe *)current->next->data;
          
          /* Calculate time range between these keyframes in nanoseconds */
          gint64 time_range = k2->position - k1->position;
          
          /* Calculate number of points to draw for smooth curve */
          int num_points = MAX(min_segments, (time_range / GST_SECOND) * segments_per_second);
          
          for (int i = 1; i <= num_points; i++) {
            /* Calculate intermediate time point */
            gint64 time = k1->position + (time_range * i) / num_points;
            
            /* Evaluate curve at this time */
            gdouble value = evaluate_keyframe_segment(k1, k2, time);
            
            /* Convert to screen coordinates */
            gdouble x = timeline->timeline_start_x + (time * timeline->zoom_level / GST_SECOND);
            gdouble y = property_y + property_height - 
                        ((value - value_min) / value_range) * property_height;
            
            /* Draw line to this point */
            cairo_line_to (cr, x, y);
          }
          
          /* Move to next keyframe pair */
          current = current->next;
        }
        
        cairo_stroke (cr);
      }
      
      /* Draw the keyframes */
      for (GSList *k = property->keyframes; k; k = k->next) {
        BlouEditKeyframe *keyframe = (BlouEditKeyframe *)k->data;
        
        /* Calculate keyframe position in UI coordinates */
        int keyframe_x = timeline->timeline_start_x + 
                         (keyframe->position * timeline->zoom_level / GST_SECOND);
        
        /* Skip if outside visible area */
        if (keyframe_x < timeline->timeline_start_x || keyframe_x > width)
          continue;
        
        /* Calculate y position based on normalized value */
        gdouble normalized_value = (keyframe->value - property->min_value) / 
                                  (property->max_value - property->min_value);
        int keyframe_y = property_y + property_height - (normalized_value * property_height);
        
        /* Draw keyframe marker based on interpolation type */
        if (keyframe == timeline->selected_keyframe) {
          /* Selected keyframe - larger and highlighted */
          cairo_set_source_rgb (cr, 1.0, 0.8, 0.2); /* Yellow highlight */
          cairo_arc (cr, keyframe_x, keyframe_y, 6, 0, 2 * G_PI);
          cairo_fill (cr);
        }
        
        /* Draw different shapes based on interpolation type */
        switch (keyframe->interpolation) {
          case BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR:
            /* Diamond shape for linear */
            cairo_set_source_rgb (cr, 0.2, 0.7, 0.9);
            cairo_move_to (cr, keyframe_x, keyframe_y - 5);
            cairo_line_to (cr, keyframe_x + 5, keyframe_y);
            cairo_line_to (cr, keyframe_x, keyframe_y + 5);
            cairo_line_to (cr, keyframe_x - 5, keyframe_y);
            cairo_close_path (cr);
            cairo_fill (cr);
            break;
            
          case BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER:
            /* Circle for bezier */
            cairo_set_source_rgb (cr, 0.2, 0.9, 0.2);
            cairo_arc (cr, keyframe_x, keyframe_y, 4, 0, 2 * G_PI);
            cairo_fill (cr);
            
            /* Draw bezier handles if this is the selected keyframe */
            if (keyframe == timeline->selected_keyframe) {
              /* Draw left handle if not the first keyframe */
              if (k != property->keyframes) {
                gdouble handle_x = keyframe_x + keyframe->handle_left_x * 50;
                gdouble handle_y = keyframe_y + keyframe->handle_left_y * 50;
                
                /* Handle line */
                cairo_set_source_rgba (cr, 0.8, 0.8, 0.2, 0.7);
  cairo_set_line_width (cr, 1.0);
                cairo_move_to (cr, keyframe_x, keyframe_y);
                cairo_line_to (cr, handle_x, handle_y);
  cairo_stroke (cr);
  
                /* Handle control point */
                cairo_set_source_rgb (cr, 0.8, 0.8, 0.2);
                cairo_arc (cr, handle_x, handle_y, 3, 0, 2 * G_PI);
                cairo_fill (cr);
              }
              
              /* Draw right handle if not the last keyframe */
              if (k->next) {
                gdouble handle_x = keyframe_x + keyframe->handle_right_x * 50;
                gdouble handle_y = keyframe_y + keyframe->handle_right_y * 50;
                
                /* Handle line */
                cairo_set_source_rgba (cr, 0.8, 0.8, 0.2, 0.7);
                cairo_set_line_width (cr, 1.0);
                cairo_move_to (cr, keyframe_x, keyframe_y);
                cairo_line_to (cr, handle_x, handle_y);
                cairo_stroke (cr);
                
                /* Handle control point */
                cairo_set_source_rgb (cr, 0.8, 0.8, 0.2);
                cairo_arc (cr, handle_x, handle_y, 3, 0, 2 * G_PI);
                cairo_fill (cr);
              }
            }
            break;
            
          case BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT:
            /* Square for constant */
            cairo_set_source_rgb (cr, 0.9, 0.5, 0.2);
            cairo_rectangle (cr, keyframe_x - 4, keyframe_y - 4, 8, 8);
            cairo_fill (cr);
            break;
            
          case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN:
          case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT:
          case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT:
            /* Circles with different colors for ease functions */
            if (keyframe->interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN)
              cairo_set_source_rgb (cr, 0.9, 0.2, 0.5);
            else if (keyframe->interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT)
              cairo_set_source_rgb (cr, 0.5, 0.2, 0.9);
            else
              cairo_set_source_rgb (cr, 0.7, 0.2, 0.7);
              
            cairo_arc (cr, keyframe_x, keyframe_y, 4, 0, 2 * G_PI);
            cairo_fill (cr);
            break;
            
          default:
            /* Default circle */
            cairo_set_source_rgb (cr, 0.2, 0.7, 0.9);
            cairo_arc (cr, keyframe_x, keyframe_y, 4, 0, 2 * G_PI);
            cairo_fill (cr);
            break;
        }
        
        /* Show keyframe value if enabled */
        if (timeline->show_keyframe_values || keyframe == timeline->selected_keyframe) {
          gchar *value_text = g_strdup_printf ("%.2f", keyframe->value);
          
          /* Draw value text with background for readability */
          pango_layout_set_text (layout, value_text, -1);
          int text_width, text_height;
          pango_layout_get_pixel_size (layout, &text_width, &text_height);
          
          /* Position text above or below keyframe point based on space */
          int text_y;
          if (keyframe_y > property_y + text_height + 5)
            text_y = keyframe_y - text_height - 5;
          else
            text_y = keyframe_y + 5;
          
          /* Draw text background */
          cairo_set_source_rgba (cr, 0.1, 0.1, 0.1, 0.7);
          cairo_rectangle (cr, keyframe_x - text_width/2 - 2, text_y - 1, 
                          text_width + 4, text_height + 2);
          cairo_fill (cr);
          
          /* Draw text */
          cairo_set_source_rgb (cr, 1.0, 1.0, 1.0);
          cairo_move_to (cr, keyframe_x - text_width/2, text_y);
          pango_cairo_show_layout (cr, layout);
          
          g_free (value_text);
        }
      }
      
      /* Move to next property */
      property_y += property_height;
    }
    
    g_object_unref (layout);
  }
  
  /* Get current filter and show filter information */
  BlouEditMediaFilterType filter = blouedit_timeline_get_media_filter (timeline);
  if (filter != BLOUEDIT_FILTER_ALL) {
    /* Create filter indicator text */
    PangoLayout *layout = pango_cairo_create_layout (cr);
    PangoFontDescription *font_desc = pango_font_description_from_string ("Sans 10");
    pango_layout_set_font_description (layout, font_desc);
    pango_font_description_free (font_desc);
    
    /* Create filter description text */
    GString *filter_text = g_string_new ("Filtered: ");
    if (filter & BLOUEDIT_FILTER_VIDEO)
      g_string_append (filter_text, "Video ");
    if (filter & BLOUEDIT_FILTER_AUDIO)
      g_string_append (filter_text, "Audio ");
    if (filter & BLOUEDIT_FILTER_IMAGE)
      g_string_append (filter_text, "Image ");
    if (filter & BLOUEDIT_FILTER_TEXT)
      g_string_append (filter_text, "Text ");
    if (filter & BLOUEDIT_FILTER_EFFECT)
      g_string_append (filter_text, "Effects ");
    if (filter & BLOUEDIT_FILTER_TRANSITION)
      g_string_append (filter_text, "Transitions ");
    
    /* Set the text */
    pango_layout_set_text (layout, filter_text->str, -1);
    g_string_free (filter_text, TRUE);
    
    /* Draw the filter indicator */
    int text_width, text_height;
    pango_layout_get_pixel_size (layout, &text_width, &text_height);
    
    /* Draw background */
    cairo_set_source_rgba (cr, 0.0, 0.0, 0.0, 0.7); /* Semi-transparent black */
    cairo_rectangle (cr, width - text_width - 10, height - text_height - 10,
                    text_width + 8, text_height + 6);
    cairo_fill (cr);
    
    /* Draw text */
    cairo_set_source_rgb (cr, 1.0, 0.8, 0.2); /* Yellow-ish text for filter */
    cairo_move_to (cr, width - text_width - 6, height - text_height - 7);
    pango_cairo_show_layout (cr, layout);
    
    g_object_unref (layout);
  }
  
  return TRUE;
}

/* Add a function to show the timecode entry dialog */
void
blouedit_timeline_show_timecode_dialog (BlouEditTimeline *timeline)
{
  GtkWidget *dialog = gtk_dialog_new_with_buttons ("Go to Timecode",
                                                 NULL, /* No parent window */
                                                 GTK_DIALOG_MODAL,
                                                 "_Cancel", GTK_RESPONSE_CANCEL,
                                                 "_Go", GTK_RESPONSE_ACCEPT,
                                                 NULL);
  
  /* Create the content area */
  GtkWidget *content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
  
  GtkWidget *grid = gtk_grid_new ();
  gtk_grid_set_row_spacing (GTK_GRID (grid), 6);
  gtk_grid_set_column_spacing (GTK_GRID (grid), 12);
  gtk_container_set_border_width (GTK_CONTAINER (grid), 10);
  
  /* Add label and timecode entry */
  GtkWidget *label = gtk_label_new ("Enter timecode:");
  GtkWidget *entry = blouedit_timeline_create_timecode_entry (timeline);
  
  gtk_grid_attach (GTK_GRID (grid), label, 0, 0, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), entry, 1, 0, 1, 1);
  
  /* Add format selector */
  GtkWidget *format_label = gtk_label_new ("Format:");
  GtkWidget *format_combo = gtk_combo_box_text_new ();
  
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (format_combo), "Frames");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (format_combo), "Seconds");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (format_combo), "HH:MM:SS:FF");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (format_combo), "HH:MM:SS.mmm");
  
  /* Set active format */
  BlouEditTimecodeFormat format = blouedit_timeline_get_timecode_format (timeline);
  gtk_combo_box_set_active (GTK_COMBO_BOX (format_combo), format);
  
  gtk_grid_attach (GTK_GRID (grid), format_label, 0, 1, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), format_combo, 1, 1, 1, 1);
  
  gtk_container_add (GTK_CONTAINER (content_area), grid);
  gtk_widget_show_all (dialog);
  
  /* Connect format combo box change signal */
  g_signal_connect (format_combo, "changed", G_CALLBACK (timecode_format_changed_cb), entry);
  
  /* Store the timeline and entry in dialog data for the callback */
  g_object_set_data (G_OBJECT (dialog), "timeline", timeline);
  g_object_set_data (G_OBJECT (dialog), "entry", entry);
  g_object_set_data (G_OBJECT (dialog), "format-combo", format_combo);
  
  /* Run the dialog */
  if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT) {
    /* Get timecode and format from widgets */
    const gchar *timecode = gtk_entry_get_text (GTK_ENTRY (entry));
    BlouEditTimecodeFormat selected_format = (BlouEditTimecodeFormat) gtk_combo_box_get_active (GTK_COMBO_BOX (format_combo));
    
    /* Set the new format */
    blouedit_timeline_set_timecode_format (timeline, selected_format);
    
    /* Go to the entered timecode using the selected format */
    gint64 position = blouedit_timeline_timecode_to_position (timeline, timecode, selected_format);
    blouedit_timeline_set_position (timeline, position);
  }
  
  gtk_widget_destroy (dialog);
}

/* Callback for timecode format changes */
static void
timecode_format_changed_cb (GtkComboBox *combo_box, GtkWidget *entry)
{
  GtkWidget *dialog = gtk_widget_get_toplevel (GTK_WIDGET (combo_box));
  if (!GTK_IS_DIALOG (dialog))
    return;
  
  BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (dialog), "timeline");
  if (!timeline)
    return;
  
  /* Get current position */
  gint64 position = blouedit_timeline_get_position (timeline);
  
  /* Get selected format */
  BlouEditTimecodeFormat format = (BlouEditTimecodeFormat) gtk_combo_box_get_active (combo_box);
  
  /* Update entry text with new format */
  gchar *timecode = blouedit_timeline_position_to_timecode (timeline, position, format);
  gtk_entry_set_text (GTK_ENTRY (entry), timecode);
  g_free (timecode);
}

/* Timeline autoscroll functions */
void
blouedit_timeline_set_autoscroll_mode (BlouEditTimeline *timeline, BlouEditAutoscrollMode mode)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  static GQuark autoscroll_mode_quark = 0;
  if (!autoscroll_mode_quark)
    autoscroll_mode_quark = g_quark_from_static_string ("blouedit-autoscroll-mode");
  
  g_object_set_qdata (G_OBJECT (timeline), autoscroll_mode_quark, GINT_TO_POINTER (mode));
}

BlouEditAutoscrollMode
blouedit_timeline_get_autoscroll_mode (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), BLOUEDIT_AUTOSCROLL_PAGE);
  
  static GQuark autoscroll_mode_quark = 0;
  if (!autoscroll_mode_quark)
    autoscroll_mode_quark = g_quark_from_static_string ("blouedit-autoscroll-mode");
  
  gpointer mode_ptr = g_object_get_qdata (G_OBJECT (timeline), autoscroll_mode_quark);
  
  /* Default to page mode if not set */
  if (!mode_ptr)
    return BLOUEDIT_AUTOSCROLL_PAGE;
  
  return (BlouEditAutoscrollMode) GPOINTER_TO_INT (mode_ptr);
}

void
blouedit_timeline_set_horizontal_scroll (BlouEditTimeline *timeline, gdouble scroll_pos)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  static GQuark scroll_pos_quark = 0;
  if (!scroll_pos_quark)
    scroll_pos_quark = g_quark_from_static_string ("blouedit-horz-scroll-pos");
  
  /* Store as pointer to double, ensuring we handle the memory properly */
  gdouble *old_pos = (gdouble *) g_object_get_qdata (G_OBJECT (timeline), scroll_pos_quark);
  if (old_pos)
    g_free (old_pos);
  
  gdouble *new_pos = g_new (gdouble, 1);
  *new_pos = scroll_pos;
  g_object_set_qdata_full (G_OBJECT (timeline), scroll_pos_quark, new_pos, g_free);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

gdouble
blouedit_timeline_get_horizontal_scroll (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0.0);
  
  static GQuark scroll_pos_quark = 0;
  if (!scroll_pos_quark)
    scroll_pos_quark = g_quark_from_static_string ("blouedit-horz-scroll-pos");
  
  gdouble *scroll_pos = (gdouble *) g_object_get_qdata (G_OBJECT (timeline), scroll_pos_quark);
  
  /* Default to 0.0 if not set */
  if (!scroll_pos)
    return 0.0;
  
  return *scroll_pos;
}

/* Function to handle auto-scrolling based on playhead position */
void
blouedit_timeline_update_scroll_for_playhead (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Get current playhead position */
  gint64 position = blouedit_timeline_get_position (timeline);
  
  /* Get widget dimensions */
  int width = gtk_widget_get_allocated_width (GTK_WIDGET (timeline));
  
  /* Skip scrolling if there's no meaningful width */
  if (width <= timeline->timeline_start_x)
    return;
  
  /* Calculate playhead position in widget coordinates */
  double playhead_x = blouedit_timeline_get_x_from_position (timeline, position);
  
  /* Get current scroll position */
  gdouble scroll_pos = blouedit_timeline_get_horizontal_scroll (timeline);
  
  /* Available timeline width (excluding the track headers area) */
  int available_width = width - timeline->timeline_start_x;
  
  /* Visible area start and end position in timeline units */
  gint64 visible_start = scroll_pos * GST_SECOND / timeline->zoom_level;
  gint64 visible_end = visible_start + (available_width * GST_SECOND / timeline->zoom_level);
  
  /* Check autoscroll mode */
  BlouEditAutoscrollMode mode = blouedit_timeline_get_autoscroll_mode (timeline);
  
  switch (mode) {
    case BLOUEDIT_AUTOSCROLL_NONE:
      /* No autoscrolling */
      break;
    
    case BLOUEDIT_AUTOSCROLL_PAGE:
      /* Page scroll when playhead reaches the edge of the view */
      if (position < visible_start || position > visible_end) {
        /* Center the playhead */
        gdouble new_scroll = MAX(0, position - (available_width / 2 * GST_SECOND / timeline->zoom_level)) * timeline->zoom_level / GST_SECOND;
        blouedit_timeline_set_horizontal_scroll (timeline, new_scroll);
      }
      break;
    
    case BLOUEDIT_AUTOSCROLL_SMOOTH:
      /* Always center the playhead */
      {
        gdouble new_scroll = MAX(0, position - (available_width / 2 * GST_SECOND / timeline->zoom_level)) * timeline->zoom_level / GST_SECOND;
        blouedit_timeline_set_horizontal_scroll (timeline, new_scroll);
      }
      break;
    
    case BLOUEDIT_AUTOSCROLL_SCROLL:
      /* Scroll when playhead reaches the edge */
      {
        /* Define a margin (20% of view width) */
        int margin = available_width * 0.2;
        
        /* If playhead moves outside the visible area plus margins, scroll */
        if (playhead_x < timeline->timeline_start_x + margin) {
          /* Scroll left */
          gdouble new_scroll = MAX(0, ((position - (margin * GST_SECOND / timeline->zoom_level)) * timeline->zoom_level / GST_SECOND));
          blouedit_timeline_set_horizontal_scroll (timeline, new_scroll);
        } else if (playhead_x > width - margin) {
          /* Scroll right */
          gdouble new_scroll = MAX(0, ((position + (margin * GST_SECOND / timeline->zoom_level) - available_width * GST_SECOND / timeline->zoom_level) * timeline->zoom_level / GST_SECOND));
          blouedit_timeline_set_horizontal_scroll (timeline, new_scroll);
        }
      }
      break;
  }
}

/* Handle autoscroll based on the current mode - should be called regularly during playback */
void
blouedit_timeline_handle_autoscroll (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* This function is a simple wrapper around update_scroll_for_playhead */
  blouedit_timeline_update_scroll_for_playhead (timeline);
}

/* Track height functions */
// ... existing track height functions ...

/* Track reordering functions */
void
blouedit_timeline_start_track_reorder (BlouEditTimeline *timeline, BlouEditTimelineTrack *track, gint y)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);

  /* Don't start another reorder operation if one is in progress */
  if (timeline->is_reordering_track)
    return;
  
  timeline->is_reordering_track = TRUE;
  timeline->reordering_track = track;
  timeline->reorder_start_y = y;
  
  /* Find the original index of the track */
  timeline->reorder_original_index = g_slist_index (timeline->tracks, track);
  timeline->reorder_current_index = timeline->reorder_original_index;
  
  /* Capture pointer */
  GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
  if (window) {
    GdkDisplay *display = gdk_window_get_display (window);
    GdkCursor *cursor = gdk_cursor_new_for_display (display, GDK_HAND1);
    gdk_window_set_cursor (window, cursor);
    g_object_unref (cursor);
  }
  
  /* Redraw to show the track being reordered */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_reorder_track_to (BlouEditTimeline *timeline, gint y)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (!timeline->is_reordering_track || !timeline->reordering_track)
    return;
  
  /* Find which track is at the current y position */
  BlouEditTimelineTrack *target_track = blouedit_timeline_get_track_at_y (timeline, y);
  if (!target_track)
    return;
  
  /* Get the index of the track under cursor */
  gint target_index = g_slist_index (timeline->tracks, target_track);
  
  /* If the target index is same as current, do nothing */
  if (target_index == timeline->reorder_current_index)
    return;
  
  /* Remove the track from its current position */
  timeline->tracks = g_slist_remove (timeline->tracks, timeline->reordering_track);
  
  /* Insert it at the target position */
  timeline->tracks = g_slist_insert (timeline->tracks, timeline->reordering_track, target_index);
  
  /* Update current index */
  timeline->reorder_current_index = target_index;
  
  /* Also update the actual GES tracks if needed */
  /* This would need to update the priorities of track elements in the GES timeline */
  
  /* Redraw with the new order */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_end_track_reorder (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  if (!timeline->is_reordering_track)
    return;
  
  /* Update GES track priorities to match the new order */
  gint priority = 0;
  for (GSList *l = timeline->tracks; l != NULL; l = l->next, priority++) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)l->data;
    if (track && track->ges_track) {
      ges_track_set_priority (track->ges_track, priority);
    }
  }
  
  /* Reset reordering state */
  timeline->is_reordering_track = FALSE;
  timeline->reordering_track = NULL;
  
  /* Reset cursor */
  GdkWindow *window = gtk_widget_get_window (GTK_WIDGET (timeline));
  if (window) {
    gdk_window_set_cursor (window, NULL);
  }
  
  /* Redraw with the final order */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_move_track_up (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* Find the track's current index */
  gint index = g_slist_index (timeline->tracks, track);
  
  /* If already at the top, do nothing */
  if (index <= 0)
    return;
  
  /* Remove the track from its current position */
  timeline->tracks = g_slist_remove (timeline->tracks, track);
  
  /* Insert it one position higher */
  timeline->tracks = g_slist_insert (timeline->tracks, track, index - 1);
  
  /* Update GES track priorities */
  gint priority = 0;
  for (GSList *l = timeline->tracks; l != NULL; l = l->next, priority++) {
    BlouEditTimelineTrack *t = (BlouEditTimelineTrack *)l->data;
    if (t && t->ges_track) {
      ges_track_set_priority (t->ges_track, priority);
    }
  }
  
  /* Redraw with the new order */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_move_track_down (BlouEditTimeline *timeline, BlouEditTimelineTrack *track)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (track != NULL);
  
  /* Find the track's current index */
  gint index = g_slist_index (timeline->tracks, track);
  
  /* If already at the bottom, do nothing */
  if (index < 0 || index >= (g_slist_length (timeline->tracks) - 1))
    return;
  
  /* Remove the track from its current position */
  timeline->tracks = g_slist_remove (timeline->tracks, track);
  
  /* Insert it one position lower */
  timeline->tracks = g_slist_insert (timeline->tracks, track, index + 1);
  
  /* Update GES track priorities */
  gint priority = 0;
  for (GSList *l = timeline->tracks; l != NULL; l = l->next, priority++) {
    BlouEditTimelineTrack *t = (BlouEditTimelineTrack *)l->data;
    if (t && t->ges_track) {
      ges_track_set_priority (t->ges_track, priority);
    }
  }
  
  /* Redraw with the new order */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/* Timeline history functions */

/**
 * blouedit_timeline_record_action:
 * @timeline: The timeline to record action for
 * @type: Type of action being recorded
 * @element: The timeline element affected
 * @description: Human-readable description of the action
 * @before_value: The state before the action (or NULL)
 * @after_value: The state after the action (or NULL)
 *
 * Records an action in the timeline's history for undo/redo.
 * If a compound action group is in progress, adds to that.
 */
void 
blouedit_timeline_record_action (BlouEditTimeline *timeline, BlouEditHistoryActionType type, 
                               GESTimelineElement *element, const gchar *description,
                               const GValue *before_value, const GValue *after_value)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Don't record actions that happen during undo/redo */
  if (timeline->is_inside_history_action)
    return;
  
  /* Create new history action */
  BlouEditHistoryAction *action = g_new0 (BlouEditHistoryAction, 1);
  action->type = type;
  action->element = element;
  action->time_stamp = g_get_monotonic_time ();
  action->description = g_strdup (description);
  
  /* Initialize GValues */
  g_value_init (&action->before_value, G_VALUE_TYPE (before_value));
  g_value_init (&action->after_value, G_VALUE_TYPE (after_value));
  
  /* Copy the values */
  g_value_copy (before_value, &action->before_value);
  g_value_copy (after_value, &action->after_value);
  
  /* If we're in a compound action, add to the group */
  if (timeline->current_group) {
    timeline->current_group = g_slist_append (timeline->current_group, action);
    return;
  }
  
  /* Add to history */
  timeline->history = g_slist_prepend (timeline->history, action);
  
  /* Clear redo stack since we've made a new action */
  g_slist_free_full (timeline->history_redo, (GDestroyNotify) history_action_free);
  timeline->history_redo = NULL;
  
  /* Limit history size */
  if (timeline->max_history_size > 0) {
    while (g_slist_length (timeline->history) > (guint)timeline->max_history_size) {
      GSList *last = g_slist_last (timeline->history);
      BlouEditHistoryAction *last_action = (BlouEditHistoryAction *) last->data;
      
      /* Remove the last item */
      timeline->history = g_slist_remove_link (timeline->history, last);
      
      /* Free the action */
      history_action_free (last_action);
      g_slist_free (last);
    }
  }
}

/**
 * blouedit_timeline_begin_group_action:
 * @timeline: The timeline
 * @description: Human-readable description of the compound action
 *
 * Begins a compound action group. All actions recorded until
 * blouedit_timeline_end_group_action() is called will be treated
 * as a single undo/redo operation.
 */
void 
blouedit_timeline_begin_group_action (BlouEditTimeline *timeline, const gchar *description)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* No nested groups for now */
  if (timeline->current_group)
    return;
  
  timeline->current_group = NULL;
  timeline->current_group_description = g_strdup (description);
}

/**
 * blouedit_timeline_end_group_action:
 * @timeline: The timeline
 *
 * Ends a compound action group started with blouedit_timeline_begin_group_action().
 * The grouped actions will be treated as a single undo/redo operation.
 */
void 
blouedit_timeline_end_group_action (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Only if we have a group in progress */
  if (!timeline->current_group)
    return;
  
  /* Create a group action */
  if (timeline->current_group && g_slist_length (timeline->current_group) > 0) {
    /* Add the group to history */
    timeline->history = g_slist_prepend (timeline->history, 
                                         g_slist_copy (timeline->current_group));
    
    /* Store the description */
    g_object_set_data_full (G_OBJECT (timeline->history->data), 
                            "group-description",
                            timeline->current_group_description,
                            (GDestroyNotify) g_free);
    
    /* Reset group tracking */
    timeline->current_group_description = NULL;
    g_slist_free (timeline->current_group);
    timeline->current_group = NULL;
    
    /* Clear redo stack */
    g_slist_free_full (timeline->history_redo, (GDestroyNotify) history_action_free);
    timeline->history_redo = NULL;
  }
}

/**
 * blouedit_timeline_undo:
 * @timeline: The timeline
 *
 * Undoes the most recent action in the timeline's history.
 *
 * Returns: TRUE if an action was undone, FALSE if nothing to undo
 */
gboolean 
blouedit_timeline_undo (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  /* Check if we have anything to undo */
  if (!timeline->history)
    return FALSE;
  
  /* Set flag to prevent recording actions during undo */
  timeline->is_inside_history_action = TRUE;
  
  /* Get the top action from history */
  GSList *top = timeline->history;
  BlouEditHistoryAction *action = (BlouEditHistoryAction *) top->data;
  
  /* If it's a group action (GSList), undo each action in reverse order */
  if (G_TYPE_CHECK_INSTANCE_TYPE (action, G_TYPE_SLIST)) {
    GSList *group = (GSList*) action;
    GSList *l;
    
    /* Undo each action in the group in reverse order */
    for (l = g_slist_last (group); l != NULL; l = l->prev) {
      BlouEditHistoryAction *group_action = (BlouEditHistoryAction *) l->data;
      
      /* Undo this individual action */
      /* Apply before_value to the element */
      /* The exact implementation depends on the action type */
      /* For now, we'll just have a placeholder that restores a simple property */
      if (group_action->element) {
        /* For properties stored in GValues */
        g_object_set_property (G_OBJECT (group_action->element),
                              g_value_get_string (&group_action->before_value),
                              &group_action->after_value);
      }
    }
    
    /* Move to redo stack */
    timeline->history = g_slist_remove_link (timeline->history, top);
    timeline->history_redo = g_slist_prepend (timeline->history_redo, top->data);
    g_slist_free (top);
  } else {
    /* Single action - apply the before value */
    if (action->element) {
      /* For properties stored in GValues */
      g_object_set_property (G_OBJECT (action->element),
                            g_value_get_string (&action->before_value),
                            &action->after_value);
    }
    
    /* Move to redo stack */
    timeline->history = g_slist_remove_link (timeline->history, top);
    timeline->history_redo = g_slist_prepend (timeline->history_redo, action);
    g_slist_free (top);
  }
  
  /* Reset flag */
  timeline->is_inside_history_action = FALSE;
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
  
  return TRUE;
}

/**
 * blouedit_timeline_redo:
 * @timeline: The timeline
 *
 * Redoes the most recently undone action.
 *
 * Returns: TRUE if an action was redone, FALSE if nothing to redo
 */
gboolean 
blouedit_timeline_redo (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  /* Check if we have anything to redo */
  if (!timeline->history_redo)
    return FALSE;
  
  /* Set flag to prevent recording actions during redo */
  timeline->is_inside_history_action = TRUE;
  
  /* Get the top action from redo stack */
  GSList *top = timeline->history_redo;
  BlouEditHistoryAction *action = (BlouEditHistoryAction *) top->data;
  
  /* If it's a group action (GSList), redo each action in original order */
  if (G_TYPE_CHECK_INSTANCE_TYPE (action, G_TYPE_SLIST)) {
    GSList *group = (GSList*) action;
    GSList *l;
    
    /* Redo each action in the group in original order */
    for (l = group; l != NULL; l = l->next) {
      BlouEditHistoryAction *group_action = (BlouEditHistoryAction *) l->data;
      
      /* Redo this individual action */
      /* Apply after_value to the element */
      if (group_action->element) {
        /* For properties stored in GValues */
        g_object_set_property (G_OBJECT (group_action->element),
                              g_value_get_string (&group_action->after_value),
                              &group_action->before_value);
      }
    }
    
    /* Move back to history stack */
    timeline->history_redo = g_slist_remove_link (timeline->history_redo, top);
    timeline->history = g_slist_prepend (timeline->history, top->data);
    g_slist_free (top);
  } else {
    /* Single action - apply the after value */
    if (action->element) {
      /* For properties stored in GValues */
      g_object_set_property (G_OBJECT (action->element),
                            g_value_get_string (&action->after_value),
                            &action->before_value);
    }
    
    /* Move back to history stack */
    timeline->history_redo = g_slist_remove_link (timeline->history_redo, top);
    timeline->history = g_slist_prepend (timeline->history, action);
    g_slist_free (top);
  }
  
  /* Reset flag */
  timeline->is_inside_history_action = FALSE;
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
  
  return TRUE;
}

/**
 * blouedit_timeline_clear_history:
 * @timeline: The timeline
 *
 * Clears the entire undo/redo history.
 */
void 
blouedit_timeline_clear_history (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Free and clear undo history */
  g_slist_free_full (timeline->history, (GDestroyNotify) history_action_free);
  timeline->history = NULL;
  
  /* Free and clear redo history */
  g_slist_free_full (timeline->history_redo, (GDestroyNotify) history_action_free);
  timeline->history_redo = NULL;
}

/**
 * blouedit_timeline_set_max_history_size:
 * @timeline: The timeline
 * @max_size: Maximum number of history items to keep (0 for unlimited)
 *
 * Sets the maximum number of history items to keep.
 * Older items beyond this limit will be automatically removed.
 */
void 
blouedit_timeline_set_max_history_size (BlouEditTimeline *timeline, gint max_size)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (max_size >= 0);
  
  timeline->max_history_size = max_size;
  
  /* Trim history if needed */
  if (max_size > 0) {
    while (g_slist_length (timeline->history) > (guint)max_size) {
      GSList *last = g_slist_last (timeline->history);
      BlouEditHistoryAction *last_action = (BlouEditHistoryAction *) last->data;
      
      /* Remove the last item */
      timeline->history = g_slist_remove_link (timeline->history, last);
      
      /* Free the action */
      history_action_free (last_action);
      g_slist_free (last);
    }
  }
}

/**
 * blouedit_timeline_get_max_history_size:
 * @timeline: The timeline
 *
 * Gets the maximum number of history items to keep.
 *
 * Returns: Maximum history size, or 0 for unlimited
 */
gint 
blouedit_timeline_get_max_history_size (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  return timeline->max_history_size;
}

/**
 * blouedit_timeline_can_undo:
 * @timeline: The timeline
 *
 * Checks if undo operation is available.
 *
 * Returns: TRUE if there are actions to undo, FALSE otherwise.
 */
gboolean 
blouedit_timeline_can_undo (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  return (timeline->history != NULL);
}

/**
 * blouedit_timeline_can_redo:
 * @timeline: The timeline
 *
 * Checks if redo operation is available.
 *
 * Returns: TRUE if there are actions to redo, FALSE otherwise.
 */
gboolean 
blouedit_timeline_can_redo (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  
  return (timeline->history_redo != NULL);
}

/**
 * blouedit_timeline_get_history_actions:
 * @timeline: The timeline
 * @limit: Maximum number of actions to return (0 for all)
 *
 * Gets a list of recent history actions.
 *
 * Returns: (transfer container): A list of BlouEditHistoryAction objects.
 * The caller should free the list with g_slist_free() when done,
 * but should not free the actions themselves.
 */
GSList* 
blouedit_timeline_get_history_actions (BlouEditTimeline *timeline, gint limit)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  GSList *result = NULL;
  GSList *l;
  gint count = 0;
  
  /* Copy the history list */
  for (l = timeline->history; l != NULL && (limit == 0 || count < limit); l = l->next, count++) {
    result = g_slist_append (result, l->data);
  }
  
  return result;
}

/* History dialog callbacks */
static void
on_history_dialog_response (GtkDialog *dialog, gint response_id, gpointer user_data)
{
  /* Close the dialog */
  gtk_widget_destroy (GTK_WIDGET (dialog));
}

static void
on_history_selection_changed (GtkTreeSelection *selection, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  GtkTreeModel *model;
  GtkTreeIter iter;
  
  if (gtk_tree_selection_get_selected (selection, &model, &iter)) {
    /* Get action index and show details */
    gint index;
    gtk_tree_model_get (model, &iter, 0, &index, -1);
    
    /* Can do something with the selected action here */
  }
}

static void
on_undo_button_clicked (GtkButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  
  /* Perform undo and update the list */
  if (blouedit_timeline_undo (timeline)) {
    /* Refresh history list */
    GtkTreeView *treeview = GTK_TREE_VIEW (g_object_get_data (G_OBJECT (button), "history-treeview"));
    GtkListStore *store = GTK_LIST_STORE (gtk_tree_view_get_model (treeview));
    gtk_list_store_clear (store);
    
    /* Add items to the list */
    int i = 0;
    for (GSList *l = timeline->history; l != NULL; l = l->next, i++) {
      BlouEditHistoryAction *action = (BlouEditHistoryAction *) l->data;
      GtkTreeIter iter;
      
      gtk_list_store_append (store, &iter);
      gtk_list_store_set (store, &iter, 
                         0, i,
                         1, g_date_time_format (g_date_time_new_from_unix_local (action->time_stamp / 1000000), "%H:%M:%S"),
                         2, action->description,
                         -1);
    }
    
    /* Update buttons */
    gtk_widget_set_sensitive (GTK_WIDGET (button), blouedit_timeline_can_undo (timeline));
    gtk_widget_set_sensitive (GTK_WIDGET (g_object_get_data (G_OBJECT (button), "redo-button")), 
                             blouedit_timeline_can_redo (timeline));
  }
}

static void
on_redo_button_clicked (GtkButton *button, gpointer user_data)
{
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (user_data);
  
  /* Perform redo and update the list */
  if (blouedit_timeline_redo (timeline)) {
    /* Refresh history list */
    GtkTreeView *treeview = GTK_TREE_VIEW (g_object_get_data (G_OBJECT (button), "history-treeview"));
    GtkListStore *store = GTK_LIST_STORE (gtk_tree_view_get_model (treeview));
    gtk_list_store_clear (store);
    
    /* Add items to the list */
    int i = 0;
    for (GSList *l = timeline->history; l != NULL; l = l->next, i++) {
      BlouEditHistoryAction *action = (BlouEditHistoryAction *) l->data;
      GtkTreeIter iter;
      
      gtk_list_store_append (store, &iter);
      gtk_list_store_set (store, &iter, 
                         0, i,
                         1, g_date_time_format (g_date_time_new_from_unix_local (action->time_stamp / 1000000), "%H:%M:%S"),
                         2, action->description,
                         -1);
    }
    
    /* Update buttons */
    gtk_widget_set_sensitive (GTK_WIDGET (g_object_get_data (G_OBJECT (button), "undo-button")), 
                             blouedit_timeline_can_undo (timeline));
    gtk_widget_set_sensitive (GTK_WIDGET (button), blouedit_timeline_can_redo (timeline));
  }
}

/**
 * blouedit_timeline_show_history_dialog:
 * @timeline: The timeline
 *
 * Shows a dialog with the timeline's history and options to undo/redo.
 */
void 
blouedit_timeline_show_history_dialog (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Create the dialog */
  GtkWidget *dialog = gtk_dialog_new_with_buttons ("Timeline History",
                                                 NULL, /* No parent window */
                                                 GTK_DIALOG_MODAL,
                                                 "_Close", GTK_RESPONSE_CLOSE,
                                                 NULL);
  
  GtkWidget *content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
  gtk_container_set_border_width (GTK_CONTAINER (content_area), 12);
  
  /* Create the list model for history items */
  GtkListStore *store = gtk_list_store_new (3, 
                                          G_TYPE_INT,     /* Index */
                                          G_TYPE_STRING,  /* Time */
                                          G_TYPE_STRING); /* Description */
  
  /* Add items to the list */
  int i = 0;
  for (GSList *l = timeline->history; l != NULL; l = l->next, i++) {
    BlouEditHistoryAction *action = (BlouEditHistoryAction *) l->data;
    GtkTreeIter iter;
    
    /* Format timestamp */
    GDateTime *dt = g_date_time_new_from_unix_local (action->time_stamp / 1000000);
    gchar *time_str = g_date_time_format (dt, "%H:%M:%S");
    
    gtk_list_store_append (store, &iter);
    gtk_list_store_set (store, &iter, 
                       0, i,
                       1, time_str,
                       2, action->description,
                       -1);
    
    g_date_time_unref (dt);
    g_free (time_str);
  }
  
  /* Create the tree view */
  GtkWidget *treeview = gtk_tree_view_new_with_model (GTK_TREE_MODEL (store));
  g_object_unref (store);
  
  /* Add columns */
  GtkCellRenderer *renderer = gtk_cell_renderer_text_new ();
  GtkTreeViewColumn *column;
  
  column = gtk_tree_view_column_new_with_attributes ("#", renderer, "text", 0, NULL);
  gtk_tree_view_append_column (GTK_TREE_VIEW (treeview), column);
  
  column = gtk_tree_view_column_new_with_attributes ("Time", renderer, "text", 1, NULL);
  gtk_tree_view_append_column (GTK_TREE_VIEW (treeview), column);
  
  column = gtk_tree_view_column_new_with_attributes ("Action", renderer, "text", 2, NULL);
  gtk_tree_view_column_set_expand (column, TRUE);
  gtk_tree_view_append_column (GTK_TREE_VIEW (treeview), column);
  
  /* Set up selection */
  GtkTreeSelection *selection = gtk_tree_view_get_selection (GTK_TREE_VIEW (treeview));
  gtk_tree_selection_set_mode (selection, GTK_SELECTION_SINGLE);
  g_signal_connect (selection, "changed", G_CALLBACK (on_history_selection_changed), timeline);
  
  /* Create a scrolled window for the tree view */
  GtkWidget *scrolled_window = gtk_scrolled_window_new (NULL, NULL);
  gtk_container_add (GTK_CONTAINER (scrolled_window), treeview);
  gtk_scrolled_window_set_shadow_type (GTK_SCROLLED_WINDOW (scrolled_window), 
                                      GTK_SHADOW_ETCHED_IN);
  gtk_scrolled_window_set_policy (GTK_SCROLLED_WINDOW (scrolled_window),
                                 GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_widget_set_size_request (scrolled_window, 500, 300);
  
  /* Create action buttons */
  GtkWidget *button_box = gtk_button_box_new (GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout (GTK_BUTTON_BOX (button_box), GTK_BUTTONBOX_END);
  gtk_container_set_border_width (GTK_CONTAINER (button_box), 6);
  gtk_box_set_spacing (GTK_BOX (button_box), 6);
  
  GtkWidget *undo_button = gtk_button_new_with_mnemonic ("_Undo");
  GtkWidget *redo_button = gtk_button_new_with_mnemonic ("_Redo");
  
  /* Store references for the callbacks */
  g_object_set_data (G_OBJECT (undo_button), "history-treeview", treeview);
  g_object_set_data (G_OBJECT (redo_button), "history-treeview", treeview);
  g_object_set_data (G_OBJECT (undo_button), "redo-button", redo_button);
  g_object_set_data (G_OBJECT (redo_button), "undo-button", undo_button);
  
  /* Connect signals */
  g_signal_connect (undo_button, "clicked", G_CALLBACK (on_undo_button_clicked), timeline);
  g_signal_connect (redo_button, "clicked", G_CALLBACK (on_redo_button_clicked), timeline);
  
  /* Set initial button states */
  gtk_widget_set_sensitive (undo_button, blouedit_timeline_can_undo (timeline));
  gtk_widget_set_sensitive (redo_button, blouedit_timeline_can_redo (timeline));
  
  /* Add buttons to box */
  gtk_container_add (GTK_CONTAINER (button_box), undo_button);
  gtk_container_add (GTK_CONTAINER (button_box), redo_button);
  
  /* Add widgets to dialog */
  GtkWidget *vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 6);
  gtk_box_pack_start (GTK_BOX (vbox), scrolled_window, TRUE, TRUE, 0);
  gtk_box_pack_start (GTK_BOX (vbox), button_box, FALSE, FALSE, 0);
  
  /* Add main container to dialog */
  gtk_box_pack_start (GTK_BOX (content_area), vbox, TRUE, TRUE, 0);
  
  /* Connect dialog response signal */
  g_signal_connect (dialog, "response", G_CALLBACK (on_history_dialog_response), timeline);
  
  /* Show all widgets and run the dialog */
  gtk_widget_show_all (dialog);
}

/* Multi-timeline functions */
struct _BlouEditTimelineGroup
{
  GList *timelines;           /* List of BlouEditTimeline objects */
  GHashTable *timeline_names; /* Hash table mapping timelines to names */
  BlouEditTimeline *active;   /* Currently active timeline */
  GtkWidget *switcher;        /* Optional timeline switcher widget */
};

/**
 * blouedit_timeline_group_new:
 *
 * Creates a new timeline group for managing multiple timelines.
 *
 * Returns: A new #BlouEditTimelineGroup
 */
BlouEditTimelineGroup *
blouedit_timeline_group_new (void)
{
  BlouEditTimelineGroup *group = g_new0 (BlouEditTimelineGroup, 1);
  
  group->timelines = NULL;
  group->timeline_names = g_hash_table_new_full (g_direct_hash, g_direct_equal, NULL, g_free);
  group->active = NULL;
  group->switcher = NULL;
  
  return group;
}

/**
 * blouedit_timeline_group_free:
 * @group: A #BlouEditTimelineGroup
 *
 * Frees a timeline group.
 */
void
blouedit_timeline_group_free (BlouEditTimelineGroup *group)
{
  g_return_if_fail (group != NULL);
  
  /* Free list of timelines */
  g_list_free (group->timelines);
  
  /* Free hash table of names */
  g_hash_table_destroy (group->timeline_names);
  
  /* Free group struct */
  g_free (group);
}

/**
 * blouedit_timeline_group_get_active:
 * @group: A #BlouEditTimelineGroup
 *
 * Gets the currently active timeline in the group.
 *
 * Returns: (transfer none): The active #BlouEditTimeline
 */
BlouEditTimeline *
blouedit_timeline_group_get_active (BlouEditTimelineGroup *group)
{
  g_return_val_if_fail (group != NULL, NULL);
  
  return group->active;
}

/**
 * blouedit_timeline_group_set_active:
 * @group: A #BlouEditTimelineGroup
 * @timeline: A #BlouEditTimeline
 *
 * Sets the active timeline in the group.
 */
void
blouedit_timeline_group_set_active (BlouEditTimelineGroup *group, BlouEditTimeline *timeline)
{
  g_return_if_fail (group != NULL);
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (g_list_find (group->timelines, timeline) != NULL);
  
  if (group->active != timeline) {
    group->active = timeline;
    
    /* Update the switcher UI if it exists */
    if (group->switcher) {
      GtkComboBox *combo = GTK_COMBO_BOX (g_object_get_data (G_OBJECT (group->switcher), "timeline-combo"));
      
      if (combo) {
        gint index = g_list_index (group->timelines, timeline);
        if (index >= 0) {
          gtk_combo_box_set_active (combo, index);
        }
      }
    }
    
    /* Emit a signal (will be added in future) */
    /* g_signal_emit (group, signals[ACTIVE_TIMELINE_CHANGED], 0, timeline); */
  }
}

/**
 * blouedit_timeline_group_add:
 * @group: A #BlouEditTimelineGroup
 * @timeline: A #BlouEditTimeline to add
 * @name: Name for the timeline
 *
 * Adds a timeline to the group with the given name.
 */
void
blouedit_timeline_group_add (BlouEditTimelineGroup *group, BlouEditTimeline *timeline, const gchar *name)
{
  g_return_if_fail (group != NULL);
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (name != NULL);
  
  /* Make sure timeline isn't already in the group */
  if (g_list_find (group->timelines, timeline) != NULL)
    return;
  
  /* Add to list */
  group->timelines = g_list_append (group->timelines, timeline);
  
  /* Store name */
  g_hash_table_insert (group->timeline_names, timeline, g_strdup (name));
  
  /* If this is the first timeline, make it active */
  if (group->active == NULL) {
    group->active = timeline;
  }
  
  /* Update switcher if it exists */
  if (group->switcher) {
    GtkComboBoxText *combo = GTK_COMBO_BOX_TEXT (g_object_get_data (G_OBJECT (group->switcher), "timeline-combo"));
    
    if (combo) {
      gtk_combo_box_text_append_text (combo, name);
      
      /* If this is the active timeline, select it */
      if (timeline == group->active) {
        gint index = g_list_length (group->timelines) - 1;
        gtk_combo_box_set_active (GTK_COMBO_BOX (combo), index);
      }
    }
  }
}

/**
 * blouedit_timeline_group_remove:
 * @group: A #BlouEditTimelineGroup
 * @timeline: A #BlouEditTimeline to remove
 *
 * Removes a timeline from the group.
 */
void
blouedit_timeline_group_remove (BlouEditTimelineGroup *group, BlouEditTimeline *timeline)
{
  g_return_if_fail (group != NULL);
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Get the index before removal */
  gint index = g_list_index (group->timelines, timeline);
  
  /* If timeline isn't in the group, ignore */
  if (index < 0)
    return;
  
  /* Remove from list */
  group->timelines = g_list_remove (group->timelines, timeline);
  
  /* Remove name */
  g_hash_table_remove (group->timeline_names, timeline);
  
  /* If this was the active timeline, select a new one */
  if (group->active == timeline) {
    if (group->timelines != NULL) {
      /* Try to select the timeline at the same index, or the last one */
      gint new_index = MIN(index, g_list_length (group->timelines) - 1);
      group->active = g_list_nth_data (group->timelines, new_index);
    } else {
      group->active = NULL;
    }
  }
  
  /* Update switcher if it exists */
  if (group->switcher) {
    GtkComboBoxText *combo = GTK_COMBO_BOX_TEXT (g_object_get_data (G_OBJECT (group->switcher), "timeline-combo"));
    
    if (combo) {
      /* Remove the item at the found index */
      gtk_combo_box_text_remove (combo, index);
      
      /* If we selected a new active timeline, update selection */
      if (group->active != NULL) {
        gint new_index = g_list_index (group->timelines, group->active);
        gtk_combo_box_set_active (GTK_COMBO_BOX (combo), new_index);
      }
    }
  }
}

/**
 * blouedit_timeline_group_get_all:
 * @group: A #BlouEditTimelineGroup
 *
 * Gets all timelines in the group.
 *
 * Returns: (transfer container) (element-type BlouEditTimeline): A list of timelines
 */
GList *
blouedit_timeline_group_get_all (BlouEditTimelineGroup *group)
{
  g_return_val_if_fail (group != NULL, NULL);
  
  return g_list_copy (group->timelines);
}

/**
 * blouedit_timeline_group_get_name:
 * @group: A #BlouEditTimelineGroup
 * @timeline: A #BlouEditTimeline
 *
 * Gets the name of a timeline in the group.
 *
 * Returns: The name of the timeline, or NULL if not in group
 */
gchar *
blouedit_timeline_group_get_name (BlouEditTimelineGroup *group, BlouEditTimeline *timeline)
{
  g_return_val_if_fail (group != NULL, NULL);
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  /* Return the name if found, or NULL */
  return g_strdup (g_hash_table_lookup (group->timeline_names, timeline));
}

/**
 * blouedit_timeline_group_set_name:
 * @group: A #BlouEditTimelineGroup
 * @timeline: A #BlouEditTimeline
 * @name: The new name
 *
 * Sets the name of a timeline in the group.
 */
void
blouedit_timeline_group_set_name (BlouEditTimelineGroup *group, BlouEditTimeline *timeline, const gchar *name)
{
  g_return_if_fail (group != NULL);
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (name != NULL);
  
  /* Check if timeline is in the group */
  gint index = g_list_index (group->timelines, timeline);
  if (index < 0)
    return;
  
  /* Update name */
  g_hash_table_insert (group->timeline_names, timeline, g_strdup (name));
  
  /* Update switcher if it exists */
  if (group->switcher) {
    GtkComboBoxText *combo = GTK_COMBO_BOX_TEXT (g_object_get_data (G_OBJECT (group->switcher), "timeline-combo"));
    
    if (combo) {
      /* Replace text at index */
      gtk_combo_box_text_remove (combo, index);
      gtk_combo_box_text_insert_text (combo, index, name);
      
      /* Restore active selection if this was the active timeline */
      if (timeline == group->active) {
        gtk_combo_box_set_active (GTK_COMBO_BOX (combo), index);
      }
    }
  }
}

/**
 * blouedit_timeline_group_copy_clip:
 * @group: A #BlouEditTimelineGroup
 * @src: Source timeline
 * @dest: Destination timeline
 * @clip: The clip to copy
 *
 * Copies a clip from one timeline to another within the group.
 *
 * Returns: TRUE if successful
 */
gboolean
blouedit_timeline_group_copy_clip (BlouEditTimelineGroup *group, BlouEditTimeline *src, BlouEditTimeline *dest, GESClip *clip)
{
  g_return_val_if_fail (group != NULL, FALSE);
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (src), FALSE);
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (dest), FALSE);
  g_return_val_if_fail (GES_IS_CLIP (clip), FALSE);
  g_return_val_if_fail (g_list_find (group->timelines, src) != NULL, FALSE);
  g_return_val_if_fail (g_list_find (group->timelines, dest) != NULL, FALSE);
  
  /* Get properties from source clip */
  GESAsset *asset = ges_extractable_get_asset (GES_EXTRACTABLE (clip));
  if (!asset)
    return FALSE;
  
  guint64 start = ges_clip_get_start (clip);
  guint64 duration = ges_clip_get_duration (clip);
  
  /* Create a new clip in the destination timeline */
  GESLayer *layer = ges_timeline_get_layer (dest->ges_timeline, 0);
  if (!layer) {
    /* Create a layer if none exists */
    layer = ges_layer_new ();
    ges_timeline_add_layer (dest->ges_timeline, layer);
  }
  
  GESClip *new_clip = ges_layer_add_asset (layer, asset, start, 0, duration, GES_TRACK_TYPE_UNKNOWN);
  
  if (!new_clip)
    return FALSE;
  
  /* Copy any relevant properties */
  /* (would implement more sophisticated copying of effects, transitions, etc.) */
  
  /* Redraw destination timeline */
  gtk_widget_queue_draw (GTK_WIDGET (dest));
  
  return TRUE;
}

/**
 * blouedit_timeline_group_sync_position:
 * @group: A #BlouEditTimelineGroup
 * @src: Source timeline whose position to sync from
 *
 * Synchronizes the playhead position from one timeline to all others in the group.
 */
void
blouedit_timeline_group_sync_position (BlouEditTimelineGroup *group, BlouEditTimeline *src)
{
  g_return_if_fail (group != NULL);
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (src));
  g_return_if_fail (g_list_find (group->timelines, src) != NULL);
  
  /* Get position from source timeline */
  gint64 position = blouedit_timeline_get_position (src);
  
  /* Set position for all other timelines */
  for (GList *l = group->timelines; l != NULL; l = l->next) {
    BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (l->data);
    
    if (timeline != src) {
      blouedit_timeline_set_position (timeline, position);
    }
  }
}

/* Callback for timeline switcher combo box */
static void
on_timeline_switcher_changed (GtkComboBox *combo, BlouEditTimelineGroup *group)
{
  g_return_if_fail (group != NULL);
  
  /* Get selected index */
  gint index = gtk_combo_box_get_active (combo);
  if (index < 0)
    return;
  
  /* Get timeline at this index */
  BlouEditTimeline *timeline = g_list_nth_data (group->timelines, index);
  if (timeline == NULL)
    return;
  
  /* Set as active */
  group->active = timeline;
  
  /* Get container widget */
  GtkWidget *container = GTK_WIDGET (g_object_get_data (G_OBJECT (combo), "timeline-container"));
  if (container == NULL)
    return;
  
  /* Remove previous timeline from container */
  GList *children = gtk_container_get_children (GTK_CONTAINER (container));
  for (GList *l = children; l != NULL; l = l->next) {
    gtk_container_remove (GTK_CONTAINER (container), GTK_WIDGET (l->data));
  }
  g_list_free (children);
  
  /* Add new active timeline to container */
  gtk_container_add (GTK_CONTAINER (container), GTK_WIDGET (timeline));
  gtk_widget_show (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_group_create_switcher:
 * @group: A #BlouEditTimelineGroup
 *
 * Creates a UI widget for switching between timelines in the group.
 *
 * Returns: A new #GtkWidget containing the switcher UI
 */
GtkWidget *
blouedit_timeline_group_create_switcher (BlouEditTimelineGroup *group)
{
  g_return_val_if_fail (group != NULL, NULL);
  
  /* Create vbox for layout */
  GtkWidget *vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 6);
  gtk_container_set_border_width (GTK_CONTAINER (vbox), 6);
  
  /* Create horizontal box for controls */
  GtkWidget *controls = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 6);
  
  /* Create combo box for timeline selection */
  GtkWidget *combo = gtk_combo_box_text_new ();
  
  /* Create container for the actual timeline */
  GtkWidget *container = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
  
  /* Store references */
  g_object_set_data (G_OBJECT (combo), "timeline-container", container);
  g_object_set_data (G_OBJECT (vbox), "timeline-combo", combo);
  
  /* Add timelines to combo box */
  gint active_index = -1;
  gint index = 0;
  
  for (GList *l = group->timelines; l != NULL; l = l->next, index++) {
    BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (l->data);
    
    /* Get name */
    gchar *name = g_hash_table_lookup (group->timeline_names, timeline);
    if (name == NULL)
      name = g_strdup_printf ("Timeline %d", index + 1);
    
    /* Add to combo */
    gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (combo), name);
    
    /* If this is the active timeline, note its index */
    if (timeline == group->active)
      active_index = index;
  }
  
  /* Set active selection */
  if (active_index >= 0)
    gtk_combo_box_set_active (GTK_COMBO_BOX (combo), active_index);
  
  /* Add action buttons */
  GtkWidget *add_button = gtk_button_new_with_label ("+");
  gtk_widget_set_tooltip_text (add_button, "Add new timeline");
  gtk_widget_set_size_request (add_button, 30, -1);
  
  GtkWidget *remove_button = gtk_button_new_with_label ("-");
  gtk_widget_set_tooltip_text (remove_button, "Remove current timeline");
  gtk_widget_set_size_request (remove_button, 30, -1);
  
  GtkWidget *rename_button = gtk_button_new_with_label ("✎");
  gtk_widget_set_tooltip_text (rename_button, "Rename timeline");
  gtk_widget_set_size_request (rename_button, 30, -1);
  
  /* Add widgets to control bar */
  gtk_box_pack_start (GTK_BOX (controls), combo, TRUE, TRUE, 0);
  gtk_box_pack_start (GTK_BOX (controls), add_button, FALSE, FALSE, 0);
  gtk_box_pack_start (GTK_BOX (controls), remove_button, FALSE, FALSE, 0);
  gtk_box_pack_start (GTK_BOX (controls), rename_button, FALSE, FALSE, 0);
  
  /* Add control bar and container to main vbox */
  gtk_box_pack_start (GTK_BOX (vbox), controls, FALSE, FALSE, 0);
  gtk_box_pack_start (GTK_BOX (vbox), container, TRUE, TRUE, 0);
  
  /* Connect signals */
  g_signal_connect (combo, "changed", G_CALLBACK (on_timeline_switcher_changed), group);
  
  /* Store reference to widget in group */
  group->switcher = vbox;
  
  /* Show initial timeline */
  if (group->active != NULL) {
    gtk_container_add (GTK_CONTAINER (container), GTK_WIDGET (group->active));
    gtk_widget_show (GTK_WIDGET (group->active));
  }
  
  /* Show all widgets */
  gtk_widget_show_all (vbox);
  
  return vbox;
}

/**
 * blouedit_timeline_show_keyframe_editor:
 * @timeline: The timeline
 * @property: The property containing the keyframe
 * @keyframe: The keyframe to edit
 *
 * Shows a dialog for editing keyframe properties.
 */
void
blouedit_timeline_show_keyframe_editor (BlouEditTimeline *timeline, BlouEditAnimatableProperty *property, BlouEditKeyframe *keyframe)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (property != NULL);
  g_return_if_fail (keyframe != NULL);
  
  /* Create the dialog */
  GtkWidget *dialog = gtk_dialog_new_with_buttons ("Edit Keyframe",
                                                 NULL, /* No parent window */
                                                 GTK_DIALOG_MODAL,
                                                 "_Cancel", GTK_RESPONSE_CANCEL,
                                                 "_Apply", GTK_RESPONSE_APPLY,
                                                 NULL);
  
  GtkWidget *content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
  gtk_container_set_border_width (GTK_CONTAINER (content_area), 12);
  
  /* Create a grid layout */
  GtkWidget *grid = gtk_grid_new ();
  gtk_grid_set_row_spacing (GTK_GRID (grid), 6);
  gtk_grid_set_column_spacing (GTK_GRID (grid), 12);
  
  /* Property name label */
  GtkWidget *property_label = gtk_label_new ("Property:");
  gtk_widget_set_halign (property_label, GTK_ALIGN_START);
  GtkWidget *property_value = gtk_label_new (property->display_name);
  gtk_widget_set_halign (property_value, GTK_ALIGN_START);
  
  /* Position entry */
  GtkWidget *position_label = gtk_label_new ("Position:");
  gtk_widget_set_halign (position_label, GTK_ALIGN_START);
  
  /* Convert position to timecode */
  BlouEditTimecodeFormat format = blouedit_timeline_get_timecode_format (timeline);
  gchar *timecode = blouedit_timeline_position_to_timecode (timeline, keyframe->position, format);
  
  GtkWidget *position_entry = gtk_entry_new ();
  gtk_entry_set_text (GTK_ENTRY (position_entry), timecode);
  g_free (timecode);
  
  /* Value entry with range */
  GtkWidget *value_label = gtk_label_new ("Value:");
  gtk_widget_set_halign (value_label, GTK_ALIGN_START);
  
  GtkWidget *value_scale = gtk_scale_new_with_range (GTK_ORIENTATION_HORIZONTAL,
                                                  property->min_value,
                                                  property->max_value,
                                                  (property->max_value - property->min_value) / 100.0);
  gtk_scale_set_value_pos (GTK_SCALE (value_scale), GTK_POS_RIGHT);
  gtk_range_set_value (GTK_RANGE (value_scale), keyframe->value);
  
  /* Interpolation type combo box */
  GtkWidget *interpolation_label = gtk_label_new ("Interpolation:");
  gtk_widget_set_halign (interpolation_label, GTK_ALIGN_START);
  
  GtkWidget *interpolation_combo = gtk_combo_box_text_new ();
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (interpolation_combo), "Linear");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (interpolation_combo), "Bezier");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (interpolation_combo), "Constant");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (interpolation_combo), "Ease In");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (interpolation_combo), "Ease Out");
  gtk_combo_box_text_append_text (GTK_COMBO_BOX_TEXT (interpolation_combo), "Ease In/Out");
  
  gtk_combo_box_set_active (GTK_COMBO_BOX (interpolation_combo), keyframe->interpolation);
  
  /* Bezier handle controls (initially hidden) */
  GtkWidget *bezier_frame = gtk_frame_new ("Bezier Handles");
  GtkWidget *bezier_grid = gtk_grid_new ();
  gtk_grid_set_row_spacing (GTK_GRID (bezier_grid), 6);
  gtk_grid_set_column_spacing (GTK_GRID (bezier_grid), 12);
  gtk_container_set_border_width (GTK_CONTAINER (bezier_grid), 6);
  gtk_container_add (GTK_CONTAINER (bezier_frame), bezier_grid);
  
  /* Left handle */
  GtkWidget *left_handle_label = gtk_label_new ("Left Handle:");
  gtk_widget_set_halign (left_handle_label, GTK_ALIGN_START);
  
  GtkWidget *left_x_label = gtk_label_new ("X:");
  gtk_widget_set_halign (left_x_label, GTK_ALIGN_START);
  GtkWidget *left_x_spin = gtk_spin_button_new_with_range (-1.0, 0.0, 0.01);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (left_x_spin), keyframe->handle_left_x);
  
  GtkWidget *left_y_label = gtk_label_new ("Y:");
  gtk_widget_set_halign (left_y_label, GTK_ALIGN_START);
  GtkWidget *left_y_spin = gtk_spin_button_new_with_range (-1.0, 1.0, 0.01);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (left_y_spin), keyframe->handle_left_y);
  
  /* Right handle */
  GtkWidget *right_handle_label = gtk_label_new ("Right Handle:");
  gtk_widget_set_halign (right_handle_label, GTK_ALIGN_START);
  
  GtkWidget *right_x_label = gtk_label_new ("X:");
  gtk_widget_set_halign (right_x_label, GTK_ALIGN_START);
  GtkWidget *right_x_spin = gtk_spin_button_new_with_range (0.0, 1.0, 0.01);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (right_x_spin), keyframe->handle_right_x);
  
  GtkWidget *right_y_label = gtk_label_new ("Y:");
  gtk_widget_set_halign (right_y_label, GTK_ALIGN_START);
  GtkWidget *right_y_spin = gtk_spin_button_new_with_range (-1.0, 1.0, 0.01);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (right_y_spin), keyframe->handle_right_y);
  
  /* Add controls to bezier grid */
  gtk_grid_attach (GTK_GRID (bezier_grid), left_handle_label, 0, 0, 2, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), left_x_label, 0, 1, 1, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), left_x_spin, 1, 1, 1, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), left_y_label, 0, 2, 1, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), left_y_spin, 1, 2, 1, 1);
  
  gtk_grid_attach (GTK_GRID (bezier_grid), right_handle_label, 0, 3, 2, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), right_x_label, 0, 4, 1, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), right_x_spin, 1, 4, 1, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), right_y_label, 0, 5, 1, 1);
  gtk_grid_attach (GTK_GRID (bezier_grid), right_y_spin, 1, 5, 1, 1);
  
  /* Add preview widget for showing curve */
  GtkWidget *preview_frame = gtk_frame_new ("Preview");
  GtkWidget *preview = gtk_drawing_area_new ();
  gtk_widget_set_size_request (preview, 200, 150);
  gtk_container_add (GTK_CONTAINER (preview_frame), preview);
  
  /* Connect preview drawing function */
  g_signal_connect (preview, "draw", G_CALLBACK (keyframe_editor_preview_draw), keyframe);
  
  /* Show or hide bezier controls based on interpolation type */
  if (keyframe->interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    gtk_widget_show_all (bezier_frame);
  } else {
    gtk_widget_hide (bezier_frame);
  }
  
  /* Connect interpolation combo to show/hide bezier controls */
  g_signal_connect (interpolation_combo, "changed", G_CALLBACK (on_interpolation_changed), bezier_frame);
  
  /* Add widgets to grid */
  gtk_grid_attach (GTK_GRID (grid), property_label, 0, 0, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), property_value, 1, 0, 1, 1);
  
  gtk_grid_attach (GTK_GRID (grid), position_label, 0, 1, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), position_entry, 1, 1, 1, 1);
  
  gtk_grid_attach (GTK_GRID (grid), value_label, 0, 2, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), value_scale, 1, 2, 1, 1);
  
  gtk_grid_attach (GTK_GRID (grid), interpolation_label, 0, 3, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), interpolation_combo, 1, 3, 1, 1);
  
  gtk_grid_attach (GTK_GRID (grid), bezier_frame, 0, 4, 2, 1);
  gtk_grid_attach (GTK_GRID (grid), preview_frame, 0, 5, 2, 1);
  
  /* Add grid to dialog */
  gtk_container_add (GTK_CONTAINER (content_area), grid);
  
  /* Store data for callback access */
  g_object_set_data (G_OBJECT (dialog), "timeline", timeline);
  g_object_set_data (G_OBJECT (dialog), "property", property);
  g_object_set_data (G_OBJECT (dialog), "keyframe", keyframe);
  g_object_set_data (G_OBJECT (dialog), "position-entry", position_entry);
  g_object_set_data (G_OBJECT (dialog), "value-scale", value_scale);
  g_object_set_data (G_OBJECT (dialog), "interpolation-combo", interpolation_combo);
  g_object_set_data (G_OBJECT (dialog), "left-x-spin", left_x_spin);
  g_object_set_data (G_OBJECT (dialog), "left-y-spin", left_y_spin);
  g_object_set_data (G_OBJECT (dialog), "right-x-spin", right_x_spin);
  g_object_set_data (G_OBJECT (dialog), "right-y-spin", right_y_spin);
  g_object_set_data (G_OBJECT (dialog), "preview", preview);
  
  /* Connect value changes to preview update */
  g_signal_connect (value_scale, "value-changed", G_CALLBACK (on_keyframe_value_changed), preview);
  g_signal_connect (left_x_spin, "value-changed", G_CALLBACK (on_bezier_handle_changed), preview);
  g_signal_connect (left_y_spin, "value-changed", G_CALLBACK (on_bezier_handle_changed), preview);
  g_signal_connect (right_x_spin, "value-changed", G_CALLBACK (on_bezier_handle_changed), preview);
  g_signal_connect (right_y_spin, "value-changed", G_CALLBACK (on_bezier_handle_changed), preview);
  
  /* Show the dialog */
  gtk_widget_show_all (dialog);
  
  /* Run the dialog */
  gint response = gtk_dialog_run (GTK_DIALOG (dialog));
  
  if (response == GTK_RESPONSE_APPLY) {
    /* Get values from widgets */
    const gchar *position_text = gtk_entry_get_text (GTK_ENTRY (position_entry));
    gdouble value = gtk_range_get_value (GTK_RANGE (value_scale));
    gint interpolation = gtk_combo_box_get_active (GTK_COMBO_BOX (interpolation_combo));
    
    /* Get bezier handle values */
    gdouble left_x = gtk_spin_button_get_value (GTK_SPIN_BUTTON (left_x_spin));
    gdouble left_y = gtk_spin_button_get_value (GTK_SPIN_BUTTON (left_y_spin));
    gdouble right_x = gtk_spin_button_get_value (GTK_SPIN_BUTTON (right_x_spin));
    gdouble right_y = gtk_spin_button_get_value (GTK_SPIN_BUTTON (right_y_spin));
    
    /* Begin a compound action */
    blouedit_timeline_begin_group_action (timeline, "Edit keyframe");
    
    /* Update position */
    gint64 new_position = blouedit_timeline_timecode_to_position (timeline, position_text, format);
    
    /* Update the keyframe */
    blouedit_timeline_update_keyframe (timeline, property, keyframe, 
                                     new_position, value, 
                                     (BlouEditKeyframeInterpolation)interpolation);
    
    /* Update bezier handles if needed */
    if (interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
      blouedit_timeline_update_keyframe_handles (timeline, property, keyframe,
                                              left_x, left_y, right_x, right_y);
    }
    
    /* End the compound action */
    blouedit_timeline_end_group_action (timeline);
    
    /* Apply the keyframe values to the property */
    blouedit_timeline_apply_keyframes (timeline);
  }
  
  /* Destroy dialog */
  gtk_widget_destroy (dialog);
}

/* Callback for interpolation type changes */
static void
on_interpolation_changed (GtkComboBox *combo, GtkWidget *bezier_frame)
{
  gint interpolation = gtk_combo_box_get_active (combo);
  
  /* Show bezier controls only for bezier interpolation */
  if (interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    gtk_widget_show_all (bezier_frame);
  } else {
    gtk_widget_hide (bezier_frame);
  }
  
  /* Get preview widget and force redraw */
  GtkWidget *dialog = gtk_widget_get_toplevel (GTK_WIDGET (combo));
  GtkWidget *preview = GTK_WIDGET (g_object_get_data (G_OBJECT (dialog), "preview"));
  
  /* Update the keyframe interpolation for preview */
  BlouEditKeyframe *keyframe = g_object_get_data (G_OBJECT (dialog), "keyframe");
  if (keyframe) {
    keyframe->interpolation = (BlouEditKeyframeInterpolation)interpolation;
    gtk_widget_queue_draw (preview);
  }
}

/* Callback for value changes */
static void
on_keyframe_value_changed (GtkRange *range, GtkWidget *preview)
{
  GtkWidget *dialog = gtk_widget_get_toplevel (GTK_WIDGET (range));
  BlouEditKeyframe *keyframe = g_object_get_data (G_OBJECT (dialog), "keyframe");
  
  if (keyframe) {
    /* Update keyframe value */
    keyframe->value = gtk_range_get_value (range);
    
    /* Redraw preview */
    gtk_widget_queue_draw (preview);
  }
}

/* Callback for bezier handle changes */
static void
on_bezier_handle_changed (GtkSpinButton *spin, GtkWidget *preview)
{
  GtkWidget *dialog = gtk_widget_get_toplevel (GTK_WIDGET (spin));
  BlouEditKeyframe *keyframe = g_object_get_data (G_OBJECT (dialog), "keyframe");
  
  if (keyframe) {
    /* Identify which handle was changed and update value */
    if (spin == g_object_get_data (G_OBJECT (dialog), "left-x-spin")) {
      keyframe->handle_left_x = gtk_spin_button_get_value (spin);
    } else if (spin == g_object_get_data (G_OBJECT (dialog), "left-y-spin")) {
      keyframe->handle_left_y = gtk_spin_button_get_value (spin);
    } else if (spin == g_object_get_data (G_OBJECT (dialog), "right-x-spin")) {
      keyframe->handle_right_x = gtk_spin_button_get_value (spin);
    } else if (spin == g_object_get_data (G_OBJECT (dialog), "right-y-spin")) {
      keyframe->handle_right_y = gtk_spin_button_get_value (spin);
    }
    
    /* Redraw preview */
    gtk_widget_queue_draw (preview);
  }
}

/* Preview drawing callback */
static gboolean
keyframe_editor_preview_draw (GtkWidget *widget, cairo_t *cr, BlouEditKeyframe *keyframe)
{
  /* Get widget dimensions */
  int width = gtk_widget_get_allocated_width (widget);
  int height = gtk_widget_get_allocated_height (widget);
  
  /* Draw background */
  cairo_set_source_rgb (cr, 0.1, 0.1, 0.1);
  cairo_rectangle (cr, 0, 0, width, height);
      cairo_fill (cr);
      
  /* Draw grid */
  cairo_set_source_rgba (cr, 0.3, 0.3, 0.3, 0.5);
  cairo_set_line_width (cr, 0.5);
  
  /* Horizontal grid lines (at 0.25, 0.5, 0.75) */
  for (int i = 1; i < 4; i++) {
    double y = height * i / 4.0;
    cairo_move_to (cr, 0, y);
    cairo_line_to (cr, width, y);
    cairo_stroke (cr);
  }
  
  /* Vertical grid lines (at 0.25, 0.5, 0.75) */
  for (int i = 1; i < 4; i++) {
    double x = width * i / 4.0;
    cairo_move_to (cr, x, 0);
    cairo_line_to (cr, x, height);
    cairo_stroke (cr);
  }
  
  /* Get dialog and keyframe info from dialog */
  GtkWidget *dialog = gtk_widget_get_toplevel (widget);
  
  /* If we can't get dialog data, just draw a default preview */
  if (!GTK_IS_DIALOG (dialog)) {
    /* Draw a simple linear curve */
    cairo_set_source_rgb (cr, 0.2, 0.7, 0.9);
    cairo_set_line_width (cr, 2.0);
    cairo_move_to (cr, 0, height);
    cairo_line_to (cr, width, 0);
    cairo_stroke (cr);
    return TRUE;
  }
  
  /* Get keyframe interpolation type */
  GtkComboBox *interpolation_combo = GTK_COMBO_BOX (g_object_get_data (G_OBJECT (dialog), "interpolation-combo"));
  BlouEditKeyframeInterpolation interpolation = (BlouEditKeyframeInterpolation) gtk_combo_box_get_active (interpolation_combo);
  
  /* Get value from slider */
  GtkRange *value_scale = GTK_RANGE (g_object_get_data (G_OBJECT (dialog), "value-scale"));
  gdouble current_value = gtk_range_get_value (value_scale);
  
  /* Set default next keyframe for curve visualization */
  gdouble next_value = current_value + 0.2;
  if (next_value > 1.0) next_value = current_value - 0.2;
  
  /* Get property from dialog to know value range */
  BlouEditAnimatableProperty *property = g_object_get_data (G_OBJECT (dialog), "property");
  if (property) {
    /* Normalize values to 0.0-1.0 range for display */
    gdouble value_range = property->max_value - property->min_value;
    if (value_range > 0) {
      current_value = (current_value - property->min_value) / value_range;
    }
  }
  
  /* Create temporary keyframes for preview */
  BlouEditKeyframe start_kf = {0};
  BlouEditKeyframe end_kf = {0};
  
  start_kf.position = 0;
  start_kf.value = current_value;
  start_kf.interpolation = interpolation;
  
  /* Get bezier handle values if in bezier mode */
  if (interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    GtkSpinButton *left_x_spin = GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (dialog), "left-x-spin"));
    GtkSpinButton *left_y_spin = GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (dialog), "left-y-spin"));
    GtkSpinButton *right_x_spin = GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (dialog), "right-x-spin"));
    GtkSpinButton *right_y_spin = GTK_SPIN_BUTTON (g_object_get_data (G_OBJECT (dialog), "right-y-spin"));
    
    start_kf.handle_left_x = gtk_spin_button_get_value (left_x_spin);
    start_kf.handle_left_y = gtk_spin_button_get_value (left_y_spin);
    start_kf.handle_right_x = gtk_spin_button_get_value (right_x_spin);
    start_kf.handle_right_y = gtk_spin_button_get_value (right_y_spin);
      } else {
    /* Default handle values */
    start_kf.handle_left_x = -0.25;
    start_kf.handle_left_y = 0.0;
    start_kf.handle_right_x = 0.25;
    start_kf.handle_right_y = 0.0;
  }
  
  /* End keyframe is at normalized position 1.0 */
  end_kf.position = 1.0;
  end_kf.value = next_value;
  end_kf.interpolation = start_kf.interpolation;
  end_kf.handle_left_x = -0.25;
  end_kf.handle_left_y = 0.0;
  end_kf.handle_right_x = 0.25;
  end_kf.handle_right_y = 0.0;
  
  /* Draw curve */
  cairo_set_source_rgb (cr, 0.2, 0.7, 0.9);
  cairo_set_line_width (cr, 2.0);
  
  /* Draw a series of line segments to approximate the curve */
  int segments = 50;
  cairo_move_to (cr, 0, height - (start_kf.value * height));
  
  for (int i = 1; i <= segments; i++) {
    gdouble t = (gdouble)i / segments;
    gdouble value;
    
    /* Calculate point based on interpolation */
    switch (interpolation) {
      case BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT:
        value = start_kf.value;
        break;
        
      case BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR:
        value = start_kf.value + t * (end_kf.value - start_kf.value);
        break;
        
      case BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER: {
        /* Cubic bezier interpolation */
        gdouble p0y = start_kf.value;
        gdouble p1y = start_kf.value + start_kf.handle_right_y;
        gdouble p2y = end_kf.value + end_kf.handle_left_y;
        gdouble p3y = end_kf.value;
        
        /* Cubic Bezier formula */
        value = pow(1-t, 3) * p0y +
                3 * pow(1-t, 2) * t * p1y +
                3 * (1-t) * t * t * p2y +
                pow(t, 3) * p3y;
        break;
      }
        
      case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN:
        /* Ease in (slow at start, faster at end) */
        value = start_kf.value + (1.0 - cos(t * G_PI / 2.0)) * (end_kf.value - start_kf.value);
        break;
        
      case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT:
        /* Ease out (fast at start, slower at end) */
        value = start_kf.value + sin(t * G_PI / 2.0) * (end_kf.value - start_kf.value);
        break;
        
      case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT:
        /* Ease in-out (slow at both ends, faster in middle) */
        value = start_kf.value + (1.0 - cos(t * G_PI)) / 2.0 * (end_kf.value - start_kf.value);
        break;
        
      default:
        value = start_kf.value + t * (end_kf.value - start_kf.value);
        break;
    }
    
    /* Draw to this point */
    cairo_line_to (cr, t * width, height - (value * height));
  }
  
  cairo_stroke (cr);
  
  /* Draw start point (current keyframe) */
  cairo_set_source_rgb (cr, 1.0, 0.8, 0.2);
  cairo_arc (cr, 0, height - (start_kf.value * height), 5, 0, 2 * G_PI);
      cairo_fill (cr);
      
  /* Draw end point */
  cairo_set_source_rgb (cr, 0.7, 0.7, 0.7);
  cairo_arc (cr, width, height - (end_kf.value * height), 3, 0, 2 * G_PI);
  cairo_fill (cr);
  
  /* Draw bezier handles if in bezier mode */
  if (interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    /* Draw right handle from start point */
    gdouble handle_x = start_kf.handle_right_x * width;
    gdouble handle_y = start_kf.handle_right_y * height;
    
    cairo_set_source_rgba (cr, 0.8, 0.8, 0.2, 0.7);
      cairo_set_line_width (cr, 1.0);
    cairo_move_to (cr, 0, height - (start_kf.value * height));
    cairo_line_to (cr, handle_x, height - (start_kf.value * height) - handle_y);
      cairo_stroke (cr);
      
    /* Draw handle control point */
    cairo_set_source_rgb (cr, 0.8, 0.8, 0.2);
    cairo_arc (cr, handle_x, height - (start_kf.value * height) - handle_y, 3, 0, 2 * G_PI);
    cairo_fill (cr);
  }
  
  return TRUE;
}

/* Keyframe context menu callbacks */
static void
on_delete_keyframe (GtkMenuItem *menuitem, gpointer user_data)
{
  GtkWidget *menu_item = GTK_WIDGET (menuitem);
  BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (menu_item), "timeline");
  BlouEditAnimatableProperty *property = g_object_get_data (G_OBJECT (menu_item), "property");
  BlouEditKeyframe *keyframe = g_object_get_data (G_OBJECT (menu_item), "keyframe");
  
  if (timeline && property && keyframe) {
    /* Delete the keyframe */
    blouedit_timeline_remove_keyframe (timeline, property, keyframe);
    
    /* Apply the keyframe changes */
    blouedit_timeline_apply_keyframes (timeline);
    
    /* Redraw the timeline */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
  }
}

static void
on_set_keyframe_interpolation (GtkMenuItem *menuitem, gpointer user_data)
{
  GtkWidget *menu_item = GTK_WIDGET (menuitem);
  BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (menu_item), "timeline");
  BlouEditAnimatableProperty *property = g_object_get_data (G_OBJECT (menu_item), "property");
  BlouEditKeyframe *keyframe = g_object_get_data (G_OBJECT (menu_item), "keyframe");
  BlouEditKeyframeInterpolation interpolation = (BlouEditKeyframeInterpolation)
      GPOINTER_TO_INT (g_object_get_data (G_OBJECT (menu_item), "interpolation"));
  
  if (timeline && property && keyframe) {
    /* Start a group action */
    blouedit_timeline_begin_group_action (timeline, "Change keyframe interpolation");
    
    /* Update the keyframe with the new interpolation type */
    blouedit_timeline_update_keyframe (timeline, property, keyframe,
                                     keyframe->position, keyframe->value,
                                     interpolation);
    
    /* End the group action */
    blouedit_timeline_end_group_action (timeline);
    
    /* Apply the keyframe changes */
    blouedit_timeline_apply_keyframes (timeline);
    
    /* Redraw the timeline */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
  }
}

static void
on_add_keyframe (GtkMenuItem *menuitem, gpointer user_data)
{
  GtkWidget *menu_item = GTK_WIDGET (menuitem);
  BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (menu_item), "timeline");
  BlouEditAnimatableProperty *property = g_object_get_data (G_OBJECT (menu_item), "property");
  gint64 position = GPOINTER_TO_INT (g_object_get_data (G_OBJECT (menu_item), "position"));
  
  if (timeline && property) {
    /* Calculate property value at this point */
    gdouble value;
    
    if (property->keyframes) {
      /* Use interpolated value from existing keyframes */
      value = blouedit_timeline_evaluate_property_at_position (timeline, property, position);
    } else {
      /* No keyframes yet, create first one with current property value */
      g_object_get (property->object, property->property_name, &value, NULL);
    }
    
    /* Create a new keyframe */
    BlouEditKeyframe *new_keyframe = blouedit_timeline_add_keyframe (
        timeline, property, position, value, BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR);
    
    /* Select the new keyframe */
    timeline->selected_keyframe = new_keyframe;
    
    /* Apply keyframes to update property */
    blouedit_timeline_apply_keyframes (timeline);
    
    /* Redraw the timeline */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
  }
}

static void
on_clear_keyframes (GtkMenuItem *menuitem, gpointer user_data)
{
  GtkWidget *menu_item = GTK_WIDGET (menuitem);
  BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (menu_item), "timeline");
  BlouEditAnimatableProperty *property = g_object_get_data (G_OBJECT (menu_item), "property");
  
  if (timeline && property) {
    /* Clear all keyframes for this property */
    blouedit_timeline_remove_all_keyframes (timeline, property);
    
    /* Apply the keyframe changes */
    blouedit_timeline_apply_keyframes (timeline);
    
    /* Redraw the timeline */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
  }
}

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
  g_return_val_if_fail (timeline->ges_timeline != NULL, NULL);
  
  // GES 트랙 생성
  GESTrack *ges_track = ges_track_new (track_type);
  if (!ges_track) {
    g_warning ("Failed to create new GES track");
    return NULL;
  }
  
  // 타임라인에 트랙 추가
  if (!ges_timeline_add_track (timeline->ges_timeline, ges_track)) {
    g_warning ("Failed to add track to timeline");
    gst_object_unref (ges_track);
    return NULL;
  }
  
  // 트랙 객체 생성
  BlouEditTimelineTrack *track = g_new0 (BlouEditTimelineTrack, 1);
  track->ges_track = ges_track;
  
  // 트랙 이름 설정 - 제공되지 않은 경우 자동 생성
  if (name) {
    track->name = g_strdup (name);
  } else {
    // 트랙 타입에 따라 자동 이름 생성
    if (track_type == GES_TRACK_TYPE_VIDEO) {
      // 비디오 트랙 번호 계산
      int video_track_count = 0;
      for (GSList *t = timeline->tracks; t; t = t->next) {
        BlouEditTimelineTrack *existing = (BlouEditTimelineTrack *)t->data;
        if (ges_track_get_track_type (existing->ges_track) == GES_TRACK_TYPE_VIDEO) {
          video_track_count++;
        }
      }
      track->name = g_strdup_printf (_("Video %d"), video_track_count + 1);
    } else if (track_type == GES_TRACK_TYPE_AUDIO) {
      // 오디오 트랙 번호 계산
      int audio_track_count = 0;
      for (GSList *t = timeline->tracks; t; t = t->next) {
        BlouEditTimelineTrack *existing = (BlouEditTimelineTrack *)t->data;
        if (ges_track_get_track_type (existing->ges_track) == GES_TRACK_TYPE_AUDIO) {
          audio_track_count++;
        }
      }
      track->name = g_strdup_printf (_("Audio %d"), audio_track_count + 1);
    } else {
      // 기타 트랙 타입
      track->name = g_strdup_printf (_("Track %d"), g_slist_length (timeline->tracks) + 1);
    }
  }
  
  // 기본 속성 설정
  track->folded = FALSE;
  track->height = timeline->default_track_height;
  track->folded_height = timeline->folded_track_height;
  track->visible = TRUE;
  track->height_resizing = FALSE;
  
  // 트랙 타입에 따라 기본 색상 설정
  if (track_type == GES_TRACK_TYPE_VIDEO) {
    // 비디오 트랙은 파란색 계열
    track->color.red = 0.2;
    track->color.green = 0.4;
    track->color.blue = 0.8;
    track->color.alpha = 1.0;
  } else if (track_type == GES_TRACK_TYPE_AUDIO) {
    // 오디오 트랙은 녹색 계열
    track->color.red = 0.2;
    track->color.green = 0.7;
    track->color.blue = 0.3;
    track->color.alpha = 1.0;
  } else {
    // 기타 트랙은 회색 계열
    track->color.red = 0.5;
    track->color.green = 0.5;
    track->color.blue = 0.5;
    track->color.alpha = 1.0;
  }
  
  // 트랙 리스트에 추가
  timeline->tracks = g_slist_append (timeline->tracks, track);
  
  // 히스토리에 기록
  blouedit_timeline_record_action (timeline, 
                                BLOUEDIT_HISTORY_ADD_TRACK,
                                GES_TIMELINE_ELEMENT (ges_track),
                                g_strdup_printf (_("Add %s"), track->name),
                                NULL, NULL);
  
  // 위젯 다시 그리기
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
  
  // 트랙이 현재 선택되어 있는 경우 선택 해제
      if (timeline->selected_track == track) {
    timeline->selected_track = NULL;
  }
  
  // 트랙이 현재 크기 조절 중이거나 위치 변경 중인 경우 작업 취소
  if (timeline->is_resizing_track && timeline->resizing_track == track) {
    timeline->is_resizing_track = FALSE;
    timeline->resizing_track = NULL;
  }
  
  if (timeline->is_reordering_track && timeline->reordering_track == track) {
    timeline->is_reordering_track = FALSE;
    timeline->reordering_track = NULL;
  }
  
  // 히스토리에 기록
  blouedit_timeline_record_action (timeline, 
                                BLOUEDIT_HISTORY_REMOVE_TRACK,
                                GES_TIMELINE_ELEMENT (track->ges_track),
                                g_strdup_printf (_("Remove %s"), track->name),
                                NULL, NULL);
  
  // GES 타임라인에서 트랙 제거
  ges_timeline_remove_track (timeline->ges_timeline, track->ges_track);
  
  // 트랙 리스트에서 제거
  timeline->tracks = g_slist_remove (timeline->tracks, track);
  
  // 트랙 메모리 해제
  track_free (track);
  
  // 위젯 다시 그리기
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/**
 * blouedit_timeline_get_track_count:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형 (GES_TRACK_TYPE_AUDIO, GES_TRACK_TYPE_VIDEO 등) 또는 0으로 모든 트랙
 *
 * 타임라인에 있는 특정 유형의 트랙 수를 반환합니다.
 *
 * Returns: 트랙 수
 */
gint
blouedit_timeline_get_track_count (BlouEditTimeline *timeline, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  gint count = 0;
  
  for (GSList *t = timeline->tracks; t; t = t->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)t->data;
    
    if (track_type == 0 || ges_track_get_track_type (track->ges_track) == track_type) {
      count++;
    }
  }
  
  return count;
}

/**
 * blouedit_timeline_get_track_by_index:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형 (GES_TRACK_TYPE_AUDIO, GES_TRACK_TYPE_VIDEO 등)
 * @index: 유형 내 인덱스 (0부터 시작)
 *
 * 특정 유형 내에서 순서에 맞는 트랙을 반환합니다.
 *
 * Returns: 트랙 객체 또는 찾지 못한 경우 NULL
 */
BlouEditTimelineTrack *
blouedit_timeline_get_track_by_index (BlouEditTimeline *timeline, GESTrackType track_type, gint index)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (index >= 0, NULL);
  
  gint current = 0;
  
  for (GSList *t = timeline->tracks; t; t = t->next) {
    BlouEditTimelineTrack *track = (BlouEditTimelineTrack *)t->data;
    
    if (ges_track_get_track_type (track->ges_track) == track_type) {
      if (current == index) {
        return track;
      }
      current++;
    }
  }
  
  return NULL;
}

/**
 * blouedit_timeline_create_default_tracks:
 * @timeline: 타임라인 객체
 *
 * 타임라인에 기본 트랙 구성(비디오 트랙 1개, 오디오 트랙 1개)을 생성합니다.
 */
void
blouedit_timeline_create_default_tracks (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  // 이미 트랙이 있는 경우 생성하지 않음
  if (g_slist_length (timeline->tracks) > 0) {
    return;
  }
  
  // 비디오 트랙 생성
  blouedit_timeline_add_track (timeline, GES_TRACK_TYPE_VIDEO, _("Video 1"));
  
  // 오디오 트랙 생성
  blouedit_timeline_add_track (timeline, GES_TRACK_TYPE_AUDIO, _("Audio 1"));
}

/**
 * blouedit_timeline_add_video_track:
 * @timeline: 타임라인 객체
 *
 * 타임라인에 새 비디오 트랙을 추가하는 편의 함수입니다.
 *
 * Returns: 새로 생성된 비디오 트랙 객체
 */
BlouEditTimelineTrack *
blouedit_timeline_add_video_track (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  // 비디오 트랙 번호 계산
  int video_track_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_VIDEO);
  gchar *name = g_strdup_printf (_("Video %d"), video_track_count + 1);
  
  BlouEditTimelineTrack *track = blouedit_timeline_add_track (timeline, GES_TRACK_TYPE_VIDEO, name);
  g_free (name);
  
  return track;
}

/**
 * blouedit_timeline_add_audio_track:
 * @timeline: 타임라인 객체
 *
 * 타임라인에 새 오디오 트랙을 추가하는 편의 함수입니다.
 *
 * Returns: 새로 생성된 오디오 트랙 객체
 */
BlouEditTimelineTrack *
blouedit_timeline_add_audio_track (BlouEditTimeline *timeline)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  
  // 오디오 트랙 번호 계산
  int audio_track_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_AUDIO);
  gchar *name = g_strdup_printf (_("Audio %d"), audio_track_count + 1);
  
  BlouEditTimelineTrack *track = blouedit_timeline_add_track (timeline, GES_TRACK_TYPE_AUDIO, name);
  g_free (name);
  
  return track;
}

/**
 * blouedit_timeline_is_track_at_max:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형 (GES_TRACK_TYPE_AUDIO, GES_TRACK_TYPE_VIDEO 등)
 *
 * 특정 트랙 유형이 최대 허용치에 도달했는지 확인합니다.
 * 현재는 무제한 트랙을 지원하므로 항상 FALSE를 반환합니다.
 *
 * Returns: 최대 트랙 수에 도달했으면 TRUE, 아니면 FALSE
 */
gboolean
blouedit_timeline_is_track_at_max (BlouEditTimeline *timeline, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), TRUE);
  
  // 무제한 트랙 지원이므로 항상 FALSE 반환
  return FALSE;
}

/**
 * blouedit_timeline_get_track_layer_for_clip:
 * @timeline: 타임라인 객체
 * @clip: GES 클립 객체
 * @track_type: 트랙 유형 (GES_TRACK_TYPE_AUDIO, GES_TRACK_TYPE_VIDEO 등)
 *
 * 클립이 특정 트랙 유형에서 사용해야 할 레이어 번호를 반환합니다.
 * 여러 트랙을 지원하기 위해 클립을 적절한 트랙에 배치하는 데 사용됩니다.
 *
 * Returns: 레이어 번호
 */
gint
blouedit_timeline_get_track_layer_for_clip (BlouEditTimeline *timeline, GESClip *clip, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  g_return_val_if_fail (GES_IS_CLIP (clip), 0);
  
  // 클립의 현재 레이어 가져오기
  gint layer = ges_clip_get_layer_priority (clip);
  
  // 트랙 유형에 맞는 레이어 번호 계산
  // 비디오 트랙은 위쪽부터 (낮은 레이어 번호), 오디오 트랙은 아래쪽부터 (높은 레이어 번호)
  if (track_type == GES_TRACK_TYPE_VIDEO) {
    return layer;
  } else if (track_type == GES_TRACK_TYPE_AUDIO) {
    // 오디오 트랙의 경우 비디오 트랙 수를 감안하여 계산
    gint video_tracks = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_VIDEO);
    return video_tracks + layer;
  }
  
  return layer;
}

/**
 * blouedit_timeline_get_max_tracks:
 * @timeline: 타임라인 객체
 * @track_type: 트랙 유형 (GES_TRACK_TYPE_AUDIO, GES_TRACK_TYPE_VIDEO 등)
 *
 * 특정 트랙 유형에 대한 최대 허용 트랙 수를 반환합니다.
 * 무제한 트랙을 지원하므로 G_MAXINT를 반환합니다.
 *
 * Returns: 최대 트랙 수
 */
gint
blouedit_timeline_get_max_tracks (BlouEditTimeline *timeline, GESTrackType track_type)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), 0);
  
  // 무제한 트랙 지원 (실제로는 시스템 메모리에 따라 제한됨)
  return G_MAXINT;
}

/* 타임라인 초기화 시 추가되어야 할 코드
 * blouedit_timeline_init 함수 수정이 필요합니다.
 * 아래 코드를 기존 init 함수 내에 추가해야 합니다:
 *
 * // 기본 트랙 높이 및 간격 설정
 * timeline->default_track_height = 80;
 * timeline->folded_track_height = 20;
 * timeline->track_spacing = 2;
 * timeline->track_header_width = 150;
 *
 * // 트랙 리스트 초기화
 * timeline->tracks = NULL;
 * timeline->selected_track = NULL;
 *
 * // 트랙 관련 상태 초기화
 * timeline->is_resizing_track = FALSE;
 * timeline->resizing_track = NULL;
 * timeline->min_track_height = 20;
 * timeline->max_track_height = 200;
 * timeline->is_reordering_track = FALSE;
 * timeline->reordering_track = NULL;
 *
 * // 기본 트랙 생성
 * blouedit_timeline_create_default_tracks (timeline);
 */

/* GES 타임라인 생성 후 호출되어야 할 코드
 * 아래 함수가 GES 타임라인 생성 후 호출되어야 합니다:
 *
 * // 기존 트랙이 있다면 제거
 * while (timeline->tracks) {
 *   BlouEditTimelineTrack *track = (BlouEditTimelineTrack *) timeline->tracks->data;
 *   blouedit_timeline_remove_track (timeline, track);
 * }
 *
 * // 기본 트랙 생성
 * blouedit_timeline_create_default_tracks (timeline);
 */

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
  
  GtkWidget *menu = gtk_menu_new ();
  GtkWidget *item;
  
  // 트랙 속성 메뉴 항목
  item = gtk_menu_item_new_with_label (_("Track Properties..."));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_show_track_properties), timeline);
  g_object_set_data (G_OBJECT (item), "track", track);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 구분선
  item = gtk_separator_menu_item_new ();
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 트랙 접기/펼치기 메뉴 항목
  if (track->folded) {
    item = gtk_menu_item_new_with_label (_("Expand Track"));
  } else {
    item = gtk_menu_item_new_with_label (_("Collapse Track"));
  }
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_toggle_track_folded), timeline);
  g_object_set_data (G_OBJECT (item), "track", track);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 트랙 높이 리셋 메뉴 항목
  item = gtk_menu_item_new_with_label (_("Reset Track Height"));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_reset_track_height), timeline);
  g_object_set_data (G_OBJECT (item), "track", track);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 구분선
  item = gtk_separator_menu_item_new ();
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 모든 트랙 접기 메뉴 항목
  item = gtk_menu_item_new_with_label (_("Collapse All Tracks"));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_fold_all_tracks), timeline);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 모든 트랙 펼치기 메뉴 항목
  item = gtk_menu_item_new_with_label (_("Expand All Tracks"));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_unfold_all_tracks), timeline);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 구분선
  item = gtk_separator_menu_item_new ();
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 비디오 트랙 추가 메뉴 항목
  item = gtk_menu_item_new_with_label (_("Add Video Track"));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_add_video_track), timeline);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 오디오 트랙 추가 메뉴 항목
  item = gtk_menu_item_new_with_label (_("Add Audio Track"));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_add_audio_track), timeline);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 구분선
  item = gtk_separator_menu_item_new ();
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 트랙 삭제 메뉴 항목
  item = gtk_menu_item_new_with_label (_("Delete Track"));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_remove_track), timeline);
  g_object_set_data (G_OBJECT (item), "track", track);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  // 메뉴 표시
  gtk_widget_show_all (menu);
  gtk_menu_popup_at_pointer (GTK_MENU (menu), NULL);
}

/**
 * blouedit_timeline_show_track_properties:
 * @timeline: 타임라인 객체
 *
 * 트랙 속성 대화상자를 표시합니다.
 * 트랙 이름, 색상, 높이 등을 설정할 수 있는 대화상자를 표시합니다.
 */
void
blouedit_timeline_show_track_properties (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  // 선택된 트랙이 없는 경우 리턴
  BlouEditTimelineTrack *track = (BlouEditTimelineTrack *) g_object_get_data (G_OBJECT (gtk_get_current_event_widget ()), "track");
  if (!track) {
    return;
  }
  
  // 대화상자 생성
  GtkWidget *dialog = gtk_dialog_new_with_buttons (_("Track Properties"),
                                                  GTK_WINDOW (gtk_widget_get_toplevel (GTK_WIDGET (timeline))),
                                                  GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                  _("Cancel"), GTK_RESPONSE_CANCEL,
                                                  _("OK"), GTK_RESPONSE_ACCEPT,
                                                  NULL);
  
  // 대화상자 컨텐츠 영역 가져오기
  GtkWidget *content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
  gtk_container_set_border_width (GTK_CONTAINER (content_area), 12);
  gtk_box_set_spacing (GTK_BOX (content_area), 6);
  
  // 그리드 레이아웃 생성
  GtkWidget *grid = gtk_grid_new ();
  gtk_grid_set_row_spacing (GTK_GRID (grid), 6);
  gtk_grid_set_column_spacing (GTK_GRID (grid), 12);
  gtk_container_add (GTK_CONTAINER (content_area), grid);
  
  // 트랙 이름 입력 필드
  GtkWidget *name_label = gtk_label_new_with_mnemonic (_("_Name:"));
  gtk_widget_set_halign (name_label, GTK_ALIGN_START);
  GtkWidget *name_entry = gtk_entry_new ();
  gtk_entry_set_text (GTK_ENTRY (name_entry), track->name);
  gtk_label_set_mnemonic_widget (GTK_LABEL (name_label), name_entry);
  gtk_grid_attach (GTK_GRID (grid), name_label, 0, 0, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), name_entry, 1, 0, 1, 1);
  
  // 트랙 높이 스핀 버튼
  GtkWidget *height_label = gtk_label_new_with_mnemonic (_("_Height:"));
  gtk_widget_set_halign (height_label, GTK_ALIGN_START);
  GtkWidget *height_spin = gtk_spin_button_new_with_range (
      timeline->min_track_height, timeline->max_track_height, 1);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (height_spin), track->height);
  gtk_label_set_mnemonic_widget (GTK_LABEL (height_label), height_spin);
  gtk_grid_attach (GTK_GRID (grid), height_label, 0, 1, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), height_spin, 1, 1, 1, 1);
  
  // 트랙 색상 선택 버튼
  GtkWidget *color_label = gtk_label_new_with_mnemonic (_("_Color:"));
  gtk_widget_set_halign (color_label, GTK_ALIGN_START);
  GtkWidget *color_button = gtk_color_button_new_with_rgba (&track->color);
  gtk_label_set_mnemonic_widget (GTK_LABEL (color_label), color_button);
  gtk_grid_attach (GTK_GRID (grid), color_label, 0, 2, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), color_button, 1, 2, 1, 1);
  
  // 트랙 표시 여부 체크박스
  GtkWidget *visible_check = gtk_check_button_new_with_mnemonic (_("_Visible"));
  gtk_toggle_button_set_active (GTK_TOGGLE_BUTTON (visible_check), track->visible);
  gtk_grid_attach (GTK_GRID (grid), visible_check, 0, 3, 2, 1);
  
  // 대화상자 표시
  gtk_widget_show_all (dialog);
  gint response = gtk_dialog_run (GTK_DIALOG (dialog));
  
  // 사용자가 OK 버튼을 클릭한 경우
  if (response == GTK_RESPONSE_ACCEPT) {
    // 트랙 이름 업데이트
    const gchar *new_name = gtk_entry_get_text (GTK_ENTRY (name_entry));
    if (g_strcmp0 (track->name, new_name) != 0) {
      g_free (track->name);
      track->name = g_strdup (new_name);
    }
    
    // 트랙 높이 업데이트
    gint new_height = gtk_spin_button_get_value_as_int (GTK_SPIN_BUTTON (height_spin));
    if (track->height != new_height) {
      blouedit_timeline_set_track_height (timeline, track, new_height);
    }
    
    // 트랙 색상 업데이트
    GdkRGBA new_color;
    gtk_color_button_get_rgba (GTK_COLOR_BUTTON (color_button), &new_color);
    blouedit_timeline_set_track_color (timeline, track, &new_color);
    
    // 트랙 표시 여부 업데이트
    gboolean new_visible = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (visible_check));
    if (track->visible != new_visible) {
      blouedit_timeline_set_track_visible (timeline, track, new_visible);
    }
    
    // 위젯 다시 그리기
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
  }
  
  // 대화상자 파괴
  gtk_widget_destroy (dialog);
}

/**
 * blouedit_timeline_show_add_track_dialog:
 * @timeline: 타임라인 객체
 *
 * 트랙 추가 대화상자를 표시합니다.
 * 비디오, 오디오 등 다양한 유형의 트랙을 추가할 수 있는 대화상자를 표시합니다.
 */
void
blouedit_timeline_show_add_track_dialog (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  // 대화상자 생성
  GtkWidget *dialog = gtk_dialog_new_with_buttons (_("Add Track"),
                                                  GTK_WINDOW (gtk_widget_get_toplevel (GTK_WIDGET (timeline))),
                                                  GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
                                                  _("Cancel"), GTK_RESPONSE_CANCEL,
                                                  _("Add"), GTK_RESPONSE_ACCEPT,
                                                  NULL);
  
  // 대화상자 컨텐츠 영역 가져오기
  GtkWidget *content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
  gtk_container_set_border_width (GTK_CONTAINER (content_area), 12);
  gtk_box_set_spacing (GTK_BOX (content_area), 6);
  
  // 그리드 레이아웃 생성
  GtkWidget *grid = gtk_grid_new ();
  gtk_grid_set_row_spacing (GTK_GRID (grid), 6);
  gtk_grid_set_column_spacing (GTK_GRID (grid), 12);
  gtk_container_add (GTK_CONTAINER (content_area), grid);
  
  // 트랙 유형 라디오 버튼
  GtkWidget *type_label = gtk_label_new (_("Track Type:"));
  gtk_widget_set_halign (type_label, GTK_ALIGN_START);
  gtk_grid_attach (GTK_GRID (grid), type_label, 0, 0, 1, 1);
  
  GtkWidget *video_radio = gtk_radio_button_new_with_label (NULL, _("Video Track"));
  gtk_grid_attach (GTK_GRID (grid), video_radio, 1, 0, 1, 1);
  
  GtkWidget *audio_radio = gtk_radio_button_new_with_label_from_widget (
      GTK_RADIO_BUTTON (video_radio), _("Audio Track"));
  gtk_grid_attach (GTK_GRID (grid), audio_radio, 1, 1, 1, 1);
  
  GtkWidget *text_radio = gtk_radio_button_new_with_label_from_widget (
      GTK_RADIO_BUTTON (video_radio), _("Text Track"));
  gtk_grid_attach (GTK_GRID (grid), text_radio, 1, 2, 1, 1);
  
  // 트랙 이름 입력 필드
  GtkWidget *name_label = gtk_label_new_with_mnemonic (_("_Name:"));
  gtk_widget_set_halign (name_label, GTK_ALIGN_START);
  GtkWidget *name_entry = gtk_entry_new ();
  
  // 트랙 유형에 따라 기본 이름 설정
  g_signal_connect (video_radio, "toggled", G_CALLBACK (on_track_type_toggled), name_entry);
  g_signal_connect (audio_radio, "toggled", G_CALLBACK (on_track_type_toggled), name_entry);
  g_signal_connect (text_radio, "toggled", G_CALLBACK (on_track_type_toggled), name_entry);
  
  // 초기 기본값 설정 (비디오 트랙 선택 시)
  gint video_count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_VIDEO);
  gchar *default_name = g_strdup_printf (_("Video %d"), video_count + 1);
  gtk_entry_set_text (GTK_ENTRY (name_entry), default_name);
  g_free (default_name);
  
  gtk_label_set_mnemonic_widget (GTK_LABEL (name_label), name_entry);
  gtk_grid_attach (GTK_GRID (grid), name_label, 0, 3, 1, 1);
  gtk_grid_attach (GTK_GRID (grid), name_entry, 1, 3, 1, 1);
  
  // 대화상자 표시
  gtk_widget_show_all (dialog);
  gint response = gtk_dialog_run (GTK_DIALOG (dialog));
  
  // 사용자가 추가 버튼을 클릭한 경우
  if (response == GTK_RESPONSE_ACCEPT) {
    const gchar *name = gtk_entry_get_text (GTK_ENTRY (name_entry));
    GESTrackType track_type;
    
    // 선택된 트랙 유형 확인
    if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (audio_radio))) {
      track_type = GES_TRACK_TYPE_AUDIO;
    } else if (gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (text_radio))) {
      track_type = GES_TRACK_TYPE_TEXT;
    } else {
      track_type = GES_TRACK_TYPE_VIDEO;
    }
    
    // 트랙 추가
    blouedit_timeline_add_track (timeline, track_type, name);
  }
  
  // 대화상자 파괴
  gtk_widget_destroy (dialog);
}

/**
 * on_track_type_toggled:
 * @radio: 토글된 라디오 버튼
 * @entry: 이름 입력 엔트리
 *
 * 트랙 유형 라디오 버튼이 토글되었을 때 호출되는 콜백 함수.
 * 트랙 유형에 따라 기본 이름을 업데이트합니다.
 */
static void
on_track_type_toggled (GtkToggleButton *radio, GtkWidget *entry)
{
  // 활성화된 버튼인 경우에만 처리
  if (!gtk_toggle_button_get_active (radio)) {
    return;
  }
  
  const gchar *label = gtk_button_get_label (GTK_BUTTON (radio));
  
  // 버튼 라벨에 따라 트랙 타입 판별 및 기본 이름 설정
  BlouEditTimeline *timeline = BLOUEDIT_TIMELINE (g_object_get_data (G_OBJECT (radio), "timeline"));
  if (!timeline) {
    return;
  }
  
  gchar *default_name = NULL;
  
  if (g_strcmp0 (label, _("Video Track")) == 0) {
    gint count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_VIDEO);
    default_name = g_strdup_printf (_("Video %d"), count + 1);
  } else if (g_strcmp0 (label, _("Audio Track")) == 0) {
    gint count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_AUDIO);
    default_name = g_strdup_printf (_("Audio %d"), count + 1);
  } else if (g_strcmp0 (label, _("Text Track")) == 0) {
    gint count = blouedit_timeline_get_track_count (timeline, GES_TRACK_TYPE_TEXT);
    default_name = g_strdup_printf (_("Text %d"), count + 1);
  }
  
  if (default_name) {
    gtk_entry_set_text (GTK_ENTRY (entry), default_name);
    g_free (default_name);
  }
}

/* Clip selection functions */
void
blouedit_timeline_select_clip (BlouEditTimeline *timeline, GESClip *clip, gboolean clear_selection)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (GES_IS_CLIP (clip));
  
  /* If requested to clear selection first, remove all current selections */
  if (clear_selection) {
    blouedit_timeline_clear_selection (timeline);
  }
  
  /* If clip is already selected, do nothing */
  if (g_slist_find (timeline->selected_clips, clip))
    return;
  
  /* Add to selected clips list */
  timeline->selected_clips = g_slist_append (timeline->selected_clips, clip);
  
  /* Keep a reference to the clip */
  g_object_ref (clip);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_unselect_clip (BlouEditTimeline *timeline, GESClip *clip)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (GES_IS_CLIP (clip));
  
  /* If clip is not selected, do nothing */
  if (!g_slist_find (timeline->selected_clips, clip))
    return;
  
  /* Remove from selected clips list */
  timeline->selected_clips = g_slist_remove (timeline->selected_clips, clip);
  
  /* Release the reference */
  g_object_unref (clip);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_clear_selection (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Release references to all selected clips */
  g_slist_foreach (timeline->selected_clips, (GFunc) g_object_unref, NULL);
  
  /* Free the list */
  g_slist_free (timeline->selected_clips);
  timeline->selected_clips = NULL;
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_select_all_clips (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Clear current selection */
  blouedit_timeline_clear_selection (timeline);
  
  /* Find all layers in the timeline */
  GList *layers = ges_timeline_get_layers (timeline->ges_timeline);
  GList *layer_item;
  
  /* Iterate through all layers */
  for (layer_item = layers; layer_item != NULL; layer_item = layer_item->next) {
    GESLayer *layer = GES_LAYER (layer_item->data);
    GList *clips = ges_layer_get_clips (layer);
    GList *clip_item;
    
    /* Select all clips in this layer */
    for (clip_item = clips; clip_item != NULL; clip_item = clip_item->next) {
      GESClip *clip = GES_CLIP (clip_item->data);
      timeline->selected_clips = g_slist_append (timeline->selected_clips, clip);
      g_object_ref (clip);
    }
    
    g_list_free (clips);
  }
  
  g_list_free (layers);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

void
blouedit_timeline_select_clips_in_range (BlouEditTimeline *timeline, gint64 start, gint64 end, BlouEditTimelineTrack *track)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  g_return_if_fail (start <= end);
  
  /* Find all layers in the timeline */
  GList *layers = ges_timeline_get_layers (timeline->ges_timeline);
  GList *layer_item;
  
  /* Iterate through all layers */
  for (layer_item = layers; layer_item != NULL; layer_item = layer_item->next) {
    GESLayer *layer = GES_LAYER (layer_item->data);
    GList *clips = ges_layer_get_clips (layer);
    GList *clip_item;
    
    /* Check all clips in this layer */
    for (clip_item = clips; clip_item != NULL; clip_item = clip_item->next) {
      GESClip *clip = GES_CLIP (clip_item->data);
      gint64 clip_start = ges_clip_get_start (clip);
      gint64 clip_end = clip_start + ges_clip_get_duration (clip);
      
      /* Check if clip overlaps with the given range */
      if ((clip_start >= start && clip_start < end) || 
          (clip_end > start && clip_end <= end) ||
          (clip_start <= start && clip_end >= end)) {
        
        /* For track-specific selection, check if clip is in the track */
        if (track != NULL) {
          /* Check if clip has elements in the specified track */
          GList *tracks = ges_timeline_get_tracks (timeline->ges_timeline);
          GList *track_item;
          gboolean clip_in_track = FALSE;
          
          for (track_item = tracks; track_item != NULL; track_item = track_item->next) {
            GESTrack *ges_track = GES_TRACK (track_item->data);
            if (ges_track == track->ges_track) {
              GList *track_elements = ges_clip_get_track_elements (clip);
              GList *elem_item;
              
              for (elem_item = track_elements; elem_item != NULL; elem_item = elem_item->next) {
                GESTrackElement *element = GES_TRACK_ELEMENT (elem_item->data);
                if (ges_track_element_get_track (element) == ges_track) {
                  clip_in_track = TRUE;
                  break;
                }
              }
              
              g_list_free (track_elements);
              if (clip_in_track)
                break;
            }
          }
          
          g_list_free (tracks);
          
          if (!clip_in_track)
            continue;  /* Skip this clip if not in specified track */
        }
        
        /* Select the clip */
        timeline->selected_clips = g_slist_append (timeline->selected_clips, clip);
        g_object_ref (clip);
      }
    }
    
    g_list_free (clips);
  }
  
  g_list_free (layers);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

gboolean
blouedit_timeline_is_clip_selected (BlouEditTimeline *timeline, GESClip *clip)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), FALSE);
  g_return_val_if_fail (GES_IS_CLIP (clip), FALSE);
  
  return g_slist_find (timeline->selected_clips, clip) != NULL;
}

/* Helper function to free clip movement info */
static void
clip_movement_info_free (BlouEditClipMovementInfo *info)
{
  if (info) {
    g_free (info);
  }
}

/* Start moving multiple clips */
void
blouedit_timeline_start_moving_multiple_clips (BlouEditTimeline *timeline, gdouble start_x)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Don't do anything if no clips are selected */
  if (!timeline->selected_clips)
    return;
    
  /* Don't start another operation if already moving clips */
  if (timeline->is_moving_clip || timeline->is_moving_multiple_clips)
    return;
  
  /* Set up for multiple clip movement */
  timeline->is_moving_multiple_clips = TRUE;
  timeline->moving_start_x = start_x;
  timeline->multi_move_offset = 0;
  
  /* Clear any existing movement info */
  if (timeline->moving_clips_info) {
    g_slist_free_full (timeline->moving_clips_info, (GDestroyNotify)clip_movement_info_free);
    timeline->moving_clips_info = NULL;
  }
  
  /* Create movement info for each selected clip */
  GSList *clip_item;
  for (clip_item = timeline->selected_clips; clip_item != NULL; clip_item = clip_item->next) {
    GESClip *clip = GES_CLIP (clip_item->data);
    
    /* Skip locked clips */
    if (blouedit_timeline_is_clip_locked (timeline, clip))
      continue;
    
    /* Create movement info */
    BlouEditClipMovementInfo *info = g_new (BlouEditClipMovementInfo, 1);
    info->clip = clip;
    info->original_start_position = ges_clip_get_start (clip);
    
    /* Add to list */
    timeline->moving_clips_info = g_slist_append (timeline->moving_clips_info, info);
  }
  
  /* If no movable clips, cancel the operation */
  if (!timeline->moving_clips_info) {
    timeline->is_moving_multiple_clips = FALSE;
    return;
  }
  
  /* Record this action for history */
  blouedit_timeline_begin_group_action (timeline, "Move Multiple Clips");
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/* Update position of multiple clips being moved */
void
blouedit_timeline_move_multiple_clips_to (BlouEditTimeline *timeline, gdouble x)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Only proceed if we're in multiple clip move mode */
  if (!timeline->is_moving_multiple_clips || !timeline->moving_clips_info)
    return;
  
  /* Calculate movement in timeline units */
  double rel_x_start = timeline->moving_start_x - timeline->timeline_start_x;
  double rel_x_current = x - timeline->timeline_start_x;
  gint64 offset = (rel_x_current - rel_x_start) / timeline->zoom_level * GST_SECOND;
  
  /* Apply snap if enabled */
  if (timeline->snap_mode != BLOUEDIT_SNAP_NONE) {
    /* Find the lead clip - the earliest one in the selection */
    BlouEditClipMovementInfo *earliest_info = NULL;
    gint64 earliest_pos = G_MAXINT64;
    
    GSList *info_item;
    for (info_item = timeline->moving_clips_info; info_item != NULL; info_item = info_item->next) {
      BlouEditClipMovementInfo *info = (BlouEditClipMovementInfo *)info_item->data;
      if (info->original_start_position < earliest_pos) {
        earliest_pos = info->original_start_position;
        earliest_info = info;
      }
    }
    
    if (earliest_info) {
      /* Calculate target position with offset for the earliest clip */
      gint64 target_pos = earliest_info->original_start_position + offset;
      
      /* Snap this position */
      gint64 snapped_pos = blouedit_timeline_snap_position (timeline, target_pos);
      
      /* Adjust offset to account for snap */
      offset = offset + (snapped_pos - target_pos);
    }
  }
  
  /* Store the common offset for all clips */
  timeline->multi_move_offset = offset;
  
  /* Update position of all clips being moved */
  GSList *info_item;
  for (info_item = timeline->moving_clips_info; info_item != NULL; info_item = info_item->next) {
    BlouEditClipMovementInfo *info = (BlouEditClipMovementInfo *)info_item->data;
    
    /* Calculate new position */
    gint64 new_pos = info->original_start_position + offset;
    
    /* Ensure we don't go negative */
    if (new_pos < 0)
      new_pos = 0;
    
    /* Update clip position */
    if (ges_clip_get_start (info->clip) != new_pos) {
      ges_timeline_element_set_start (GES_TIMELINE_ELEMENT (info->clip), new_pos);
    }
  }
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/* End the multiple clip movement operation */
void
blouedit_timeline_end_moving_multiple_clips (BlouEditTimeline *timeline)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  /* Only proceed if we're in multiple clip move mode */
  if (!timeline->is_moving_multiple_clips)
    return;
  
  /* End the group action for history */
  blouedit_timeline_end_group_action (timeline);
  
  /* Clean up */
  if (timeline->moving_clips_info) {
    g_slist_free_full (timeline->moving_clips_info, (GDestroyNotify)clip_movement_info_free);
    timeline->moving_clips_info = NULL;
  }
  
  timeline->is_moving_multiple_clips = FALSE;
  timeline->multi_move_offset = 0;
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (timeline));
}

/* Snap type menu callback */
static void
on_set_snap_mode (GtkMenuItem *menuitem, gpointer user_data)
{
  GtkWidget *menu_item = GTK_WIDGET (menuitem);
  BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (menu_item), "timeline");
  BlouEditSnapMode mode = (BlouEditSnapMode) GPOINTER_TO_INT (g_object_get_data (G_OBJECT (menu_item), "snap-mode"));
  
  if (timeline) {
    /* Set the snap mode */
    blouedit_timeline_set_snap_mode (timeline, mode);
    
    /* Redraw the timeline */
    gtk_widget_queue_draw (GTK_WIDGET (timeline));
  }
}

/**
 * on_generate_proxy_for_clip:
 * @menuitem: The menu item that was activated
 * @user_data: User data (not used)
 *
 * Callback for generating a proxy for a clip from the context menu.
 */
static void
on_generate_proxy_for_clip (GtkMenuItem *menuitem, gpointer user_data)
{
  GtkWidget *menu_item = GTK_WIDGET (menuitem);
  BlouEditTimeline *timeline = g_object_get_data (G_OBJECT (menu_item), "timeline");
  GESClip *clip = g_object_get_data (G_OBJECT (menu_item), "clip");
  
  if (timeline && clip) {
    /* Generate proxy for the clip */
    blouedit_timeline_generate_proxy_for_clip (timeline, clip);
  }
}

/**
 * blouedit_timeline_show_context_menu:
 * @timeline: 타임라인 객체
 * @x: 컨텍스트 메뉴를 표시할 X 좌표
 * @y: 컨텍스트 메뉴를 표시할 Y 좌표
 *
 * 타임라인 컨텍스트 메뉴를 표시합니다.
 * 타임라인 영역을 오른쪽 클릭했을 때 나타나는 메뉴입니다.
 * 스냅 모드 선택, 편집 모드 변경 등의 기능을 제공합니다.
 */
void
blouedit_timeline_show_context_menu (BlouEditTimeline *timeline, gdouble x, gdouble y)
{
  g_return_if_fail (BLOUEDIT_IS_TIMELINE (timeline));
  
  GtkWidget *menu = gtk_menu_new ();
  GtkWidget *item;
  GtkWidget *submenu;
  
  /* 스냅 설정 메뉴 */
  submenu = gtk_menu_new ();
  
  /* 스냅 비활성화 */
  item = gtk_radio_menu_item_new_with_label (NULL, _("No Snapping"));
  if (timeline->snap_mode == BLOUEDIT_SNAP_NONE)
    gtk_check_menu_item_set_active (GTK_CHECK_MENU_ITEM (item), TRUE);
  g_object_set_data (G_OBJECT (item), "timeline", timeline);
  g_object_set_data (G_OBJECT (item), "snap-mode", GINT_TO_POINTER(BLOUEDIT_SNAP_NONE));
  g_signal_connect (item, "activate", G_CALLBACK (on_set_snap_mode), NULL);
  gtk_menu_shell_append (GTK_MENU_SHELL (submenu), item);
  
  GSList *snap_group = gtk_radio_menu_item_get_group (GTK_RADIO_MENU_ITEM (item));
  
  /* 클립에 맞추기 */
  item = gtk_radio_menu_item_new_with_label (snap_group, _("Snap to Clips"));
  if (timeline->snap_mode == BLOUEDIT_SNAP_TO_CLIPS)
    gtk_check_menu_item_set_active (GTK_CHECK_MENU_ITEM (item), TRUE);
  g_object_set_data (G_OBJECT (item), "timeline", timeline);
  g_object_set_data (G_OBJECT (item), "snap-mode", GINT_TO_POINTER(BLOUEDIT_SNAP_TO_CLIPS));
  g_signal_connect (item, "activate", G_CALLBACK (on_set_snap_mode), NULL);
  gtk_menu_shell_append (GTK_MENU_SHELL (submenu), item);
  
  snap_group = gtk_radio_menu_item_get_group (GTK_RADIO_MENU_ITEM (item));
  
  /* 마커에 맞추기 */
  item = gtk_radio_menu_item_new_with_label (snap_group, _("Snap to Markers"));
  if (timeline->snap_mode == BLOUEDIT_SNAP_TO_MARKERS)
    gtk_check_menu_item_set_active (GTK_CHECK_MENU_ITEM (item), TRUE);
  g_object_set_data (G_OBJECT (item), "timeline", timeline);
  g_object_set_data (G_OBJECT (item), "snap-mode", GINT_TO_POINTER(BLOUEDIT_SNAP_TO_MARKERS));
  g_signal_connect (item, "activate", G_CALLBACK (on_set_snap_mode), NULL);
  gtk_menu_shell_append (GTK_MENU_SHELL (submenu), item);
  
  snap_group = gtk_radio_menu_item_get_group (GTK_RADIO_MENU_ITEM (item));
  
  /* 그리드에 맞추기 */
  item = gtk_radio_menu_item_new_with_label (snap_group, _("Snap to Grid"));
  if (timeline->snap_mode == BLOUEDIT_SNAP_TO_GRID)
    gtk_check_menu_item_set_active (GTK_CHECK_MENU_ITEM (item), TRUE);
  g_object_set_data (G_OBJECT (item), "timeline", timeline);
  g_object_set_data (G_OBJECT (item), "snap-mode", GINT_TO_POINTER(BLOUEDIT_SNAP_TO_GRID));
  g_signal_connect (item, "activate", G_CALLBACK (on_set_snap_mode), NULL);
  gtk_menu_shell_append (GTK_MENU_SHELL (submenu), item);
  
  snap_group = gtk_radio_menu_item_get_group (GTK_RADIO_MENU_ITEM (item));
  
  /* 모든 요소에 맞추기 */
  item = gtk_radio_menu_item_new_with_label (snap_group, _("Snap to All"));
  if (timeline->snap_mode == BLOUEDIT_SNAP_ALL)
    gtk_check_menu_item_set_active (GTK_CHECK_MENU_ITEM (item), TRUE);
  g_object_set_data (G_OBJECT (item), "timeline", timeline);
  g_object_set_data (G_OBJECT (item), "snap-mode", GINT_TO_POINTER(BLOUEDIT_SNAP_ALL));
  g_signal_connect (item, "activate", G_CALLBACK (on_set_snap_mode), NULL);
  gtk_menu_shell_append (GTK_MENU_SHELL (submenu), item);
  
  /* 스냅 설정 메뉴 항목 */
  item = gtk_menu_item_new_with_label (_("Snap Settings"));
  gtk_menu_item_set_submenu (GTK_MENU_ITEM (item), submenu);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  /* 구분선 */
  item = gtk_separator_menu_item_new ();
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  /* 프록시 설정 */
  item = gtk_menu_item_new_with_label (_("Proxy Settings..."));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_show_proxy_settings_dialog), timeline);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  /* 성능 모드 설정 */
  item = gtk_menu_item_new_with_label (_("Performance Mode..."));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_show_performance_settings_dialog), timeline);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  /* 구분선 */
  item = gtk_separator_menu_item_new ();
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  /* 타임라인 히스토리 */
  item = gtk_menu_item_new_with_label (_("Timeline History..."));
  g_signal_connect_swapped (item, "activate", G_CALLBACK (blouedit_timeline_show_history_dialog), timeline);
  gtk_menu_shell_append (GTK_MENU_SHELL (menu), item);
  
  /* 메뉴 표시 */
  gtk_widget_show_all (menu);
  gtk_menu_popup_at_pointer (GTK_MENU (menu), NULL);
}