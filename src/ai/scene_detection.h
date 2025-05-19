#pragma once

#include <glib-object.h>
#include <gtk/gtk.h>

G_BEGIN_DECLS

#define BLOUEDIT_TYPE_SCENE_DETECTOR (blouedit_scene_detector_get_type())

G_DECLARE_FINAL_TYPE (BlouEditSceneDetector, blouedit_scene_detector, BLOUEDIT, SCENE_DETECTOR, GObject)

BlouEditSceneDetector *blouedit_scene_detector_new (void);

/* Start scene detection on a file */
gboolean blouedit_scene_detector_process_file (BlouEditSceneDetector *detector, 
                                              const char *video_uri,
                                              GCancellable *cancellable,
                                              GError **error);

/* Get detected scene changes */
GArray *blouedit_scene_detector_get_scene_changes (BlouEditSceneDetector *detector);

/* Set sensitivity threshold (0.0 to 1.0) */
void blouedit_scene_detector_set_threshold (BlouEditSceneDetector *detector, double threshold);
double blouedit_scene_detector_get_threshold (BlouEditSceneDetector *detector);

/* Enable or disable ML-based scene detection */
void blouedit_scene_detector_set_use_ml (BlouEditSceneDetector *detector, gboolean use_ml);
gboolean blouedit_scene_detector_get_use_ml (BlouEditSceneDetector *detector);

G_END_DECLS 