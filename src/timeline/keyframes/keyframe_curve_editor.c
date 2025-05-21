#include <gtk/gtk.h>
#include <gst/gst.h>

#include "../timeline.h"
#include "../core/types.h"
#include "keyframes.h"

/* Define private curve editor structure */
typedef struct _BlouEditKeyframeCurveEditor BlouEditKeyframeCurveEditor;

struct _BlouEditKeyframeCurveEditor
{
  GtkWidget *editor_widget;              /* Main widget container */
  GtkWidget *drawing_area;               /* Drawing area for curve visualization */
  GtkWidget *interpolation_combo;        /* Interpolation type selector */
  GtkWidget *value_adjustment;           /* Value adjustment slider */
  GtkWidget *bezier_handles_frame;       /* Frame containing bezier handle controls */
  GtkWidget *handle_left_x;              /* Left handle X position control */
  GtkWidget *handle_left_y;              /* Left handle Y position control */
  GtkWidget *handle_right_x;             /* Right handle X position control */
  GtkWidget *handle_right_y;             /* Right handle Y position control */
  
  BlouEditAnimatableProperty *property;  /* Property being edited */
  BlouEditKeyframe *keyframe;            /* Keyframe being edited */
  BlouEditTimeline *timeline;            /* Parent timeline */
  
  gboolean updating_ui;                  /* Flag to prevent feedback loops during updates */
};

/* Forward declaration of drawing function */
static gboolean keyframe_curve_editor_draw (GtkWidget *widget, cairo_t *cr, BlouEditKeyframeCurveEditor *editor);

/* Create a new curve editor for the given keyframe */
GtkWidget *
blouedit_keyframe_curve_editor_new (BlouEditTimeline *timeline, 
                                   BlouEditAnimatableProperty *property,
                                   BlouEditKeyframe *keyframe)
{
  g_return_val_if_fail (BLOUEDIT_IS_TIMELINE (timeline), NULL);
  g_return_val_if_fail (property != NULL, NULL);
  g_return_val_if_fail (keyframe != NULL, NULL);
  
  /* Create editor instance */
  BlouEditKeyframeCurveEditor *editor = g_new0 (BlouEditKeyframeCurveEditor, 1);
  editor->property = property;
  editor->keyframe = keyframe;
  editor->timeline = timeline;
  editor->updating_ui = FALSE;
  
  /* Create main container */
  editor->editor_widget = gtk_box_new (GTK_ORIENTATION_VERTICAL, 6);
  gtk_widget_set_margin_start (editor->editor_widget, 12);
  gtk_widget_set_margin_end (editor->editor_widget, 12);
  gtk_widget_set_margin_top (editor->editor_widget, 12);
  gtk_widget_set_margin_bottom (editor->editor_widget, 12);
  
  /* Create header with keyframe info */
  GtkWidget *header = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 6);
  
  GtkWidget *property_label = gtk_label_new (property->display_name);
  gtk_widget_set_halign (property_label, GTK_ALIGN_START);
  gtk_box_pack_start (GTK_BOX (header), property_label, FALSE, FALSE, 0);
  
  gchar *position_str = blouedit_timeline_position_to_timecode (
      timeline, keyframe->position, blouedit_timeline_get_timecode_format (timeline));
  GtkWidget *position_label = gtk_label_new (position_str);
  g_free (position_str);
  gtk_widget_set_halign (position_label, GTK_ALIGN_END);
  gtk_box_pack_end (GTK_BOX (header), position_label, FALSE, FALSE, 0);
  
  gtk_box_pack_start (GTK_BOX (editor->editor_widget), header, FALSE, FALSE, 0);
  
  /* Create curve visualization area */
  editor->drawing_area = gtk_drawing_area_new ();
  gtk_widget_set_size_request (editor->drawing_area, 300, 200);
  gtk_box_pack_start (GTK_BOX (editor->editor_widget), editor->drawing_area, TRUE, TRUE, 0);
  
  /* Connect drawing signal */
  g_signal_connect (editor->drawing_area, "draw", 
                    G_CALLBACK (keyframe_curve_editor_draw), editor);
  
  /* Create interpolation type selector */
  GtkWidget *interp_box = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 6);
  GtkWidget *interp_label = gtk_label_new ("Interpolation:");
  gtk_box_pack_start (GTK_BOX (interp_box), interp_label, FALSE, FALSE, 0);
  
  editor->interpolation_combo = gtk_combo_box_text_new ();
  gtk_combo_box_text_append (GTK_COMBO_BOX_TEXT (editor->interpolation_combo), 
                            "linear", "Linear");
  gtk_combo_box_text_append (GTK_COMBO_BOX_TEXT (editor->interpolation_combo), 
                            "bezier", "Bezier");
  gtk_combo_box_text_append (GTK_COMBO_BOX_TEXT (editor->interpolation_combo), 
                            "constant", "Constant");
  gtk_combo_box_text_append (GTK_COMBO_BOX_TEXT (editor->interpolation_combo), 
                            "ease-in", "Ease In");
  gtk_combo_box_text_append (GTK_COMBO_BOX_TEXT (editor->interpolation_combo), 
                            "ease-out", "Ease Out");
  gtk_combo_box_text_append (GTK_COMBO_BOX_TEXT (editor->interpolation_combo), 
                            "ease-in-out", "Ease In/Out");
  
  /* Set current interpolation */
  switch (keyframe->interpolation) {
    case BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "linear");
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "bezier");
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "constant");
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-in");
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-out");
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-in-out");
      break;
  }
  
  gtk_box_pack_start (GTK_BOX (interp_box), editor->interpolation_combo, TRUE, TRUE, 0);
  gtk_box_pack_start (GTK_BOX (editor->editor_widget), interp_box, FALSE, FALSE, 6);
  
  /* Create value slider */
  GtkWidget *value_box = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 6);
  GtkWidget *value_label = gtk_label_new ("Value:");
  gtk_box_pack_start (GTK_BOX (value_box), value_label, FALSE, FALSE, 0);
  
  /* Create adjustment for property range */
  GtkAdjustment *adjustment = gtk_adjustment_new (
      keyframe->value,                /* Initial value */
      property->min_value,            /* Minimum */
      property->max_value,            /* Maximum */
      (property->max_value - property->min_value) / 100.0, /* Step */
      (property->max_value - property->min_value) / 10.0,  /* Page */
      0.0);                           /* Page size (not used) */
  
  editor->value_adjustment = gtk_scale_new (GTK_ORIENTATION_HORIZONTAL, adjustment);
  gtk_scale_set_draw_value (GTK_SCALE (editor->value_adjustment), TRUE);
  gtk_scale_set_value_pos (GTK_SCALE (editor->value_adjustment), GTK_POS_RIGHT);
  gtk_box_pack_start (GTK_BOX (value_box), editor->value_adjustment, TRUE, TRUE, 0);
  
  gtk_box_pack_start (GTK_BOX (editor->editor_widget), value_box, FALSE, FALSE, 6);
  
  /* Create bezier handle controls */
  editor->bezier_handles_frame = gtk_frame_new ("Bezier Handles");
  GtkWidget *handle_grid = gtk_grid_new ();
  gtk_grid_set_column_spacing (GTK_GRID (handle_grid), 12);
  gtk_grid_set_row_spacing (GTK_GRID (handle_grid), 6);
  gtk_container_add (GTK_CONTAINER (editor->bezier_handles_frame), handle_grid);
  
  /* Left handle controls */
  GtkWidget *left_label = gtk_label_new ("Left Handle:");
  gtk_grid_attach (GTK_GRID (handle_grid), left_label, 0, 0, 1, 1);
  
  GtkWidget *left_x_label = gtk_label_new ("X:");
  gtk_grid_attach (GTK_GRID (handle_grid), left_x_label, 0, 1, 1, 1);
  
  GtkAdjustment *left_x_adj = gtk_adjustment_new (
      keyframe->handle_left_x, -1.0, 0.0, 0.01, 0.1, 0.0);
  editor->handle_left_x = gtk_spin_button_new (left_x_adj, 0.01, 2);
  gtk_grid_attach (GTK_GRID (handle_grid), editor->handle_left_x, 1, 1, 1, 1);
  
  GtkWidget *left_y_label = gtk_label_new ("Y:");
  gtk_grid_attach (GTK_GRID (handle_grid), left_y_label, 0, 2, 1, 1);
  
  GtkAdjustment *left_y_adj = gtk_adjustment_new (
      keyframe->handle_left_y, -1.0, 1.0, 0.01, 0.1, 0.0);
  editor->handle_left_y = gtk_spin_button_new (left_y_adj, 0.01, 2);
  gtk_grid_attach (GTK_GRID (handle_grid), editor->handle_left_y, 1, 2, 1, 1);
  
  /* Right handle controls */
  GtkWidget *right_label = gtk_label_new ("Right Handle:");
  gtk_grid_attach (GTK_GRID (handle_grid), right_label, 2, 0, 1, 1);
  
  GtkWidget *right_x_label = gtk_label_new ("X:");
  gtk_grid_attach (GTK_GRID (handle_grid), right_x_label, 2, 1, 1, 1);
  
  GtkAdjustment *right_x_adj = gtk_adjustment_new (
      keyframe->handle_right_x, 0.0, 1.0, 0.01, 0.1, 0.0);
  editor->handle_right_x = gtk_spin_button_new (right_x_adj, 0.01, 2);
  gtk_grid_attach (GTK_GRID (handle_grid), editor->handle_right_x, 3, 1, 1, 1);
  
  GtkWidget *right_y_label = gtk_label_new ("Y:");
  gtk_grid_attach (GTK_GRID (handle_grid), right_y_label, 2, 2, 1, 1);
  
  GtkAdjustment *right_y_adj = gtk_adjustment_new (
      keyframe->handle_right_y, -1.0, 1.0, 0.01, 0.1, 0.0);
  editor->handle_right_y = gtk_spin_button_new (right_y_adj, 0.01, 2);
  gtk_grid_attach (GTK_GRID (handle_grid), editor->handle_right_y, 3, 2, 1, 1);
  
  gtk_box_pack_start (GTK_BOX (editor->editor_widget), editor->bezier_handles_frame, FALSE, FALSE, 6);
  
  /* Show/hide bezier controls based on interpolation type */
  if (keyframe->interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    gtk_widget_show_all (editor->bezier_handles_frame);
  } else {
    gtk_widget_hide (editor->bezier_handles_frame);
  }
  
  /* Create buttons for actions */
  GtkWidget *button_box = gtk_button_box_new (GTK_ORIENTATION_HORIZONTAL);
  gtk_button_box_set_layout (GTK_BUTTON_BOX (button_box), GTK_BUTTONBOX_END);
  gtk_box_set_spacing (GTK_BOX (button_box), 6);
  
  GtkWidget *apply_button = gtk_button_new_with_label ("Apply");
  gtk_container_add (GTK_CONTAINER (button_box), apply_button);
  
  GtkWidget *reset_button = gtk_button_new_with_label ("Reset");
  gtk_container_add (GTK_CONTAINER (button_box), reset_button);
  
  gtk_box_pack_end (GTK_BOX (editor->editor_widget), button_box, FALSE, FALSE, 6);
  
  /* Connect signals */
  g_signal_connect_swapped (editor->interpolation_combo, "changed", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_update_interpolation), 
                          editor);
  
  g_signal_connect_swapped (editor->value_adjustment, "value-changed", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_update_value), 
                          editor);
  
  g_signal_connect_swapped (editor->handle_left_x, "value-changed", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_update_handles), 
                          editor);
  g_signal_connect_swapped (editor->handle_left_y, "value-changed", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_update_handles), 
                          editor);
  g_signal_connect_swapped (editor->handle_right_x, "value-changed", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_update_handles), 
                          editor);
  g_signal_connect_swapped (editor->handle_right_y, "value-changed", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_update_handles), 
                          editor);
  
  g_signal_connect_swapped (apply_button, "clicked", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_apply), 
                          editor);
  
  g_signal_connect_swapped (reset_button, "clicked", 
                          G_CALLBACK (blouedit_keyframe_curve_editor_reset), 
                          editor);
  
  /* Set cleanup handling */
  g_object_set_data_full (G_OBJECT (editor->editor_widget), "editor-data", 
                         editor, (GDestroyNotify) blouedit_keyframe_curve_editor_free);
  
  gtk_widget_show_all (editor->editor_widget);
  return editor->editor_widget;
}

/* Free editor data when widget is destroyed */
void
blouedit_keyframe_curve_editor_free (BlouEditKeyframeCurveEditor *editor)
{
  if (editor) {
    g_free (editor);
  }
}

/* Update interpolation type from combo box */
void
blouedit_keyframe_curve_editor_update_interpolation (BlouEditKeyframeCurveEditor *editor)
{
  g_return_if_fail (editor != NULL);
  
  if (editor->updating_ui)
    return;
  
  /* Get selected interpolation type */
  const gchar *interp_id = gtk_combo_box_get_active_id (
      GTK_COMBO_BOX (editor->interpolation_combo));
  
  BlouEditKeyframeInterpolation interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR;
  
  if (g_strcmp0 (interp_id, "linear") == 0) {
    interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR;
    gtk_widget_hide (editor->bezier_handles_frame);
  } else if (g_strcmp0 (interp_id, "bezier") == 0) {
    interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER;
    gtk_widget_show_all (editor->bezier_handles_frame);
  } else if (g_strcmp0 (interp_id, "constant") == 0) {
    interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT;
    gtk_widget_hide (editor->bezier_handles_frame);
  } else if (g_strcmp0 (interp_id, "ease-in") == 0) {
    interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN;
    gtk_widget_hide (editor->bezier_handles_frame);
  } else if (g_strcmp0 (interp_id, "ease-out") == 0) {
    interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT;
    gtk_widget_hide (editor->bezier_handles_frame);
  } else if (g_strcmp0 (interp_id, "ease-in-out") == 0) {
    interpolation = BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT;
    gtk_widget_hide (editor->bezier_handles_frame);
  }
  
  /* Update keyframe interpolation */
  editor->keyframe->interpolation = interpolation;
  
  /* Redraw the visualization */
  gtk_widget_queue_draw (editor->drawing_area);
}

/* Update keyframe value from slider */
void
blouedit_keyframe_curve_editor_update_value (BlouEditKeyframeCurveEditor *editor)
{
  g_return_if_fail (editor != NULL);
  
  if (editor->updating_ui)
    return;
  
  /* Get current value */
  gdouble value = gtk_range_get_value (GTK_RANGE (editor->value_adjustment));
  
  /* Update keyframe value */
  editor->keyframe->value = value;
  
  /* Redraw the visualization */
  gtk_widget_queue_draw (editor->drawing_area);
}

/* Update bezier handles from spin buttons */
void
blouedit_keyframe_curve_editor_update_handles (BlouEditKeyframeCurveEditor *editor)
{
  g_return_if_fail (editor != NULL);
  
  if (editor->updating_ui)
    return;
  
  /* Get handle values */
  gdouble left_x = gtk_spin_button_get_value (GTK_SPIN_BUTTON (editor->handle_left_x));
  gdouble left_y = gtk_spin_button_get_value (GTK_SPIN_BUTTON (editor->handle_left_y));
  gdouble right_x = gtk_spin_button_get_value (GTK_SPIN_BUTTON (editor->handle_right_x));
  gdouble right_y = gtk_spin_button_get_value (GTK_SPIN_BUTTON (editor->handle_right_y));
  
  /* Update keyframe handles */
  editor->keyframe->handle_left_x = left_x;
  editor->keyframe->handle_left_y = left_y;
  editor->keyframe->handle_right_x = right_x;
  editor->keyframe->handle_right_y = right_y;
  
  /* Redraw the visualization */
  gtk_widget_queue_draw (editor->drawing_area);
}

/* Apply changes to the timeline */
void
blouedit_keyframe_curve_editor_apply (BlouEditKeyframeCurveEditor *editor)
{
  g_return_if_fail (editor != NULL);
  
  /* Update keyframe in timeline */
  blouedit_timeline_update_keyframe (editor->timeline, editor->property,
                                   editor->keyframe, editor->keyframe->position,
                                   editor->keyframe->value,
                                   editor->keyframe->interpolation);
  
  /* Update handles if using bezier interpolation */
  if (editor->keyframe->interpolation == BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    blouedit_timeline_update_keyframe_handles (editor->timeline, editor->property,
                                            editor->keyframe,
                                            editor->keyframe->handle_left_x,
                                            editor->keyframe->handle_left_y,
                                            editor->keyframe->handle_right_x,
                                            editor->keyframe->handle_right_y);
  }
  
  /* Apply keyframes to property */
  blouedit_timeline_apply_keyframes (editor->timeline);
  
  /* Redraw timeline */
  gtk_widget_queue_draw (GTK_WIDGET (editor->timeline));
}

/* Reset changes to original values */
void
blouedit_keyframe_curve_editor_reset (BlouEditKeyframeCurveEditor *editor)
{
  g_return_if_fail (editor != NULL);
  
  editor->updating_ui = TRUE;
  
  /* Reset value slider */
  gtk_range_set_value (GTK_RANGE (editor->value_adjustment), editor->keyframe->value);
  
  /* Reset interpolation combo */
  switch (editor->keyframe->interpolation) {
    case BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "linear");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "bezier");
      gtk_widget_show_all (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "constant");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-in");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-out");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-in-out");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
  }
  
  /* Reset handle values */
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_left_x), 
                            editor->keyframe->handle_left_x);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_left_y), 
                            editor->keyframe->handle_left_y);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_right_x), 
                            editor->keyframe->handle_right_x);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_right_y), 
                            editor->keyframe->handle_right_y);
  
  editor->updating_ui = FALSE;
  
  /* Redraw visualization */
  gtk_widget_queue_draw (editor->drawing_area);
}

/* Draw curve visualization */
static gboolean
keyframe_curve_editor_draw (GtkWidget *widget, cairo_t *cr, BlouEditKeyframeCurveEditor *editor)
{
  g_return_val_if_fail (editor != NULL, FALSE);
  
  /* Get drawing area dimensions */
  gint width = gtk_widget_get_allocated_width (widget);
  gint height = gtk_widget_get_allocated_height (widget);
  
  /* Clear background */
  cairo_set_source_rgb (cr, 0.2, 0.2, 0.2);
  cairo_paint (cr);
  
  /* Draw grid */
  cairo_set_source_rgba (cr, 0.5, 0.5, 0.5, 0.3);
  cairo_set_line_width (cr, 1.0);
  
  /* Vertical grid lines */
  for (int i = 0; i <= 10; i++) {
    double x = i * (width / 10.0);
    cairo_move_to (cr, x, 0);
    cairo_line_to (cr, x, height);
  }
  
  /* Horizontal grid lines */
  for (int i = 0; i <= 10; i++) {
    double y = i * (height / 10.0);
    cairo_move_to (cr, 0, y);
    cairo_line_to (cr, width, y);
  }
  
  cairo_stroke (cr);
  
  /* Draw curve */
  cairo_set_source_rgb (cr, 0.0, 0.7, 1.0);
  cairo_set_line_width (cr, 2.0);
  
  switch (editor->keyframe->interpolation) {
    case BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR:
      /* Linear: straight line */
      cairo_move_to (cr, 0, height);
      cairo_line_to (cr, width, 0);
      break;
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER:
      /* Bezier curve */
      cairo_move_to (cr, 0, height);
      
      /* Convert handle coordinates to drawing area coordinates */
      double h1x = width * (1.0 + editor->keyframe->handle_left_x);  /* -1..0 -> 0..width */
      double h1y = height * (1.0 - editor->keyframe->handle_left_y) / 2.0;  /* -1..1 -> height..0 */
      double h2x = width * editor->keyframe->handle_right_x;         /* 0..1 -> 0..width */
      double h2y = height * (1.0 - editor->keyframe->handle_right_y) / 2.0; /* -1..1 -> height..0 */
      
      cairo_curve_to (cr, h1x, h1y, h2x, h2y, width, 0);
      
      /* Draw handles as small circles */
      cairo_stroke (cr);
      cairo_set_source_rgba (cr, 1.0, 0.5, 0.0, 0.8);
      
      /* Left handle */
      cairo_arc (cr, h1x, h1y, 4.0, 0, 2 * G_PI);
      cairo_fill (cr);
      
      /* Draw line from start to left handle */
      cairo_move_to (cr, 0, height);
      cairo_line_to (cr, h1x, h1y);
      cairo_stroke (cr);
      
      /* Right handle */
      cairo_arc (cr, h2x, h2y, 4.0, 0, 2 * G_PI);
      cairo_fill (cr);
      
      /* Draw line from end to right handle */
      cairo_move_to (cr, width, 0);
      cairo_line_to (cr, h2x, h2y);
      cairo_stroke (cr);
      
      /* Reset color for other drawing */
      cairo_set_source_rgb (cr, 0.0, 0.7, 1.0);
      break;
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT:
      /* Constant: horizontal then vertical */
      cairo_move_to (cr, 0, height);
      cairo_line_to (cr, width * 0.9, height);
      cairo_line_to (cr, width * 0.9, 0);
      cairo_line_to (cr, width, 0);
      break;
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN:
      /* Ease in: start slow, end fast */
      cairo_move_to (cr, 0, height);
      cairo_curve_to (cr, width * 0.6, height * 0.9, width * 0.9, height * 0.3, width, 0);
      break;
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT:
      /* Ease out: start fast, end slow */
      cairo_move_to (cr, 0, height);
      cairo_curve_to (cr, width * 0.1, height * 0.3, width * 0.4, height * 0.1, width, 0);
      break;
      
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT:
      /* Ease in/out: start and end slow, fast in middle */
      cairo_move_to (cr, 0, height);
      cairo_curve_to (cr, width * 0.25, height * 0.9, width * 0.75, height * 0.1, width, 0);
      break;
  }
  
  /* Draw the main curve path (if not bezier which is already drawn) */
  if (editor->keyframe->interpolation != BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER) {
    cairo_stroke (cr);
  }
  
  /* Draw keyframe points */
  cairo_set_source_rgb (cr, 1.0, 1.0, 0.0);
  
  /* Start point (previous keyframe) */
  cairo_arc (cr, 0, height, 5.0, 0, 2 * G_PI);
  cairo_fill (cr);
  
  /* End point (current keyframe) */
  cairo_arc (cr, width, 0, 5.0, 0, 2 * G_PI);
  cairo_fill (cr);
  
  return FALSE;
}

/* Update the editor with current keyframe values */
void
blouedit_keyframe_curve_editor_update (GtkWidget *editor_widget, 
                                      BlouEditKeyframe *keyframe)
{
  g_return_if_fail (GTK_IS_WIDGET (editor_widget));
  g_return_if_fail (keyframe != NULL);
  
  BlouEditKeyframeCurveEditor *editor = g_object_get_data (G_OBJECT (editor_widget), "editor-data");
  if (!editor)
    return;
  
  editor->updating_ui = TRUE;
  
  /* Update value slider */
  gtk_range_set_value (GTK_RANGE (editor->value_adjustment), keyframe->value);
  
  /* Update interpolation combo */
  switch (keyframe->interpolation) {
    case BLOUEDIT_KEYFRAME_INTERPOLATION_LINEAR:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "linear");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_BEZIER:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "bezier");
      gtk_widget_show_all (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_CONSTANT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "constant");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-in");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_OUT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-out");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
    case BLOUEDIT_KEYFRAME_INTERPOLATION_EASE_IN_OUT:
      gtk_combo_box_set_active_id (GTK_COMBO_BOX (editor->interpolation_combo), "ease-in-out");
      gtk_widget_hide (editor->bezier_handles_frame);
      break;
  }
  
  /* Update handle values */
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_left_x), keyframe->handle_left_x);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_left_y), keyframe->handle_left_y);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_right_x), keyframe->handle_right_x);
  gtk_spin_button_set_value (GTK_SPIN_BUTTON (editor->handle_right_y), keyframe->handle_right_y);
  
  editor->updating_ui = FALSE;
  
  /* Redraw visualization */
  gtk_widget_queue_draw (editor->drawing_area);
} 