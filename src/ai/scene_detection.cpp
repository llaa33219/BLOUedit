#include "scene_detection.h"
#include <glib/gi18n.h>
#include <json-glib/json-glib.h>
#include <stdlib.h>
#include <string.h>

struct _BlouEditSceneDetector
{
  GObject parent_instance;

  /* Properties */
  double threshold;
  gboolean use_ml;
  
  /* Python binding */
  char *python_path;
  
  /* Results */
  GArray *scene_changes; /* Array of gint64 timestamps in ms */
};

G_DEFINE_TYPE (BlouEditSceneDetector, blouedit_scene_detector, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_THRESHOLD,
  PROP_USE_ML,
  N_PROPS
};

static GParamSpec *props[N_PROPS] = { NULL, };

static void
blouedit_scene_detector_finalize (GObject *object)
{
  BlouEditSceneDetector *self = BLOUEDIT_SCENE_DETECTOR (object);

  g_clear_pointer (&self->python_path, g_free);
  
  if (self->scene_changes)
    g_array_unref (self->scene_changes);

  G_OBJECT_CLASS (blouedit_scene_detector_parent_class)->finalize (object);
}

static void
blouedit_scene_detector_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  BlouEditSceneDetector *self = BLOUEDIT_SCENE_DETECTOR (object);

  switch (prop_id)
    {
    case PROP_THRESHOLD:
      blouedit_scene_detector_set_threshold (self, g_value_get_double (value));
      break;
    case PROP_USE_ML:
      blouedit_scene_detector_set_use_ml (self, g_value_get_boolean (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
blouedit_scene_detector_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  BlouEditSceneDetector *self = BLOUEDIT_SCENE_DETECTOR (object);

  switch (prop_id)
    {
    case PROP_THRESHOLD:
      g_value_set_double (value, self->threshold);
      break;
    case PROP_USE_ML:
      g_value_set_boolean (value, self->use_ml);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
blouedit_scene_detector_class_init (BlouEditSceneDetectorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = blouedit_scene_detector_finalize;
  object_class->set_property = blouedit_scene_detector_set_property;
  object_class->get_property = blouedit_scene_detector_get_property;

  props[PROP_THRESHOLD] =
    g_param_spec_double ("threshold",
                         "Threshold",
                         "Sensitivity threshold for scene detection",
                         0.0, 1.0, 0.5,
                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
                         
  props[PROP_USE_ML] =
    g_param_spec_boolean ("use-ml",
                          "Use ML",
                          "Whether to use machine learning for scene detection",
                          TRUE,
                          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);

  g_object_class_install_properties (object_class, N_PROPS, props);
}

static void
blouedit_scene_detector_init (BlouEditSceneDetector *self)
{
  self->threshold = 0.5;
  self->use_ml = TRUE;
  self->scene_changes = g_array_new (FALSE, FALSE, sizeof (gint64));
  
  // Set default Python path
  const char *datadir = g_getenv ("BLOUEDIT_DATADIR");
  if (datadir != NULL)
    self->python_path = g_build_filename (datadir, "ai", "python", NULL);
  else
    self->python_path = g_strdup ("/app/share/blouedit/ai/python");
}

BlouEditSceneDetector *
blouedit_scene_detector_new (void)
{
  return g_object_new (BLOUEDIT_TYPE_SCENE_DETECTOR, NULL);
}

static gboolean
parse_scene_detection_json (BlouEditSceneDetector *self,
                           const char           *json_str,
                           GError              **error)
{
  g_return_val_if_fail (BLOUEDIT_IS_SCENE_DETECTOR (self), FALSE);
  g_return_val_if_fail (json_str != NULL, FALSE);
  
  GError *err = NULL;
  JsonParser *parser = json_parser_new ();
  
  if (!json_parser_load_from_data (parser, json_str, -1, &err))
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   "Failed to parse scene detection results: %s",
                   err->message);
      g_error_free (err);
      g_object_unref (parser);
      return FALSE;
    }
  
  JsonNode *root = json_parser_get_root (parser);
  JsonObject *obj = json_node_get_object (root);
  
  if (json_object_has_member (obj, "status"))
    {
      const char *status = json_object_get_string_member (obj, "status");
      
      if (g_strcmp0 (status, "success") == 0)
        {
          // Clear previous results
          g_array_remove_range (self->scene_changes, 0, self->scene_changes->len);
          
          if (json_object_has_member (obj, "scenes"))
            {
              JsonArray *scenes = json_object_get_array_member (obj, "scenes");
              
              for (guint i = 0; i < json_array_get_length (scenes); i++)
                {
                  gint64 timestamp = json_array_get_int_element (scenes, i);
                  g_array_append_val (self->scene_changes, timestamp);
                }
              
              g_object_unref (parser);
              return TRUE;
            }
        }
      else if (json_object_has_member (obj, "error"))
        {
          const char *error_msg = json_object_get_string_member (obj, "error");
          g_set_error (error,
                       G_IO_ERROR,
                       G_IO_ERROR_FAILED,
                       "Scene detection failed: %s",
                       error_msg);
          g_object_unref (parser);
          return FALSE;
        }
    }
  
  g_set_error (error,
               G_IO_ERROR,
               G_IO_ERROR_FAILED,
               "Invalid scene detection results format");
  g_object_unref (parser);
  return FALSE;
}

gboolean
blouedit_scene_detector_process_file (BlouEditSceneDetector *detector,
                                     const char           *video_uri,
                                     GCancellable         *cancellable,
                                     GError              **error)
{
  g_return_val_if_fail (BLOUEDIT_IS_SCENE_DETECTOR (detector), FALSE);
  g_return_val_if_fail (video_uri != NULL, FALSE);
  
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) err = NULL;
  g_autofree char *stdout_buf = NULL;
  g_autofree char *stderr_buf = NULL;
  g_autofree char *script_path = NULL;
  
  // Prepare Python script path
  script_path = g_build_filename (detector->python_path, "bridge.py", NULL);
  
  if (!g_file_test (script_path, G_FILE_TEST_EXISTS))
    {
      g_set_error (error,
                  G_IO_ERROR,
                  G_IO_ERROR_NOT_FOUND,
                  "Python script not found: %s",
                  script_path);
      return FALSE;
    }
  
  // Build command arguments
  GPtrArray *argv = g_ptr_array_new ();
  g_ptr_array_add (argv, "python3");
  g_ptr_array_add (argv, script_path);
  g_ptr_array_add (argv, "scene-detection");
  g_ptr_array_add (argv, "--video");
  g_ptr_array_add (argv, (gpointer)video_uri);
  g_ptr_array_add (argv, "--threshold");
  
  g_autofree char *threshold_str = g_strdup_printf ("%g", detector->threshold);
  g_ptr_array_add (argv, threshold_str);
  
  if (!detector->use_ml)
    g_ptr_array_add (argv, "--no-ml");
  
  g_ptr_array_add (argv, NULL);
  
  // Create subprocess
  subprocess = g_subprocess_newv ((const char * const *)argv->pdata,
                                 G_SUBPROCESS_FLAGS_STDOUT_PIPE |
                                 G_SUBPROCESS_FLAGS_STDERR_PIPE,
                                 &err);
  g_ptr_array_free (argv, TRUE);
  
  if (subprocess == NULL)
    {
      g_propagate_prefixed_error (error,
                                 g_steal_pointer (&err),
                                 "Failed to launch scene detection: ");
      return FALSE;
    }
  
  // Communicate with subprocess
  if (!g_subprocess_communicate_utf8 (subprocess, NULL, cancellable,
                                     &stdout_buf, &stderr_buf, &err))
    {
      g_propagate_prefixed_error (error,
                                 g_steal_pointer (&err),
                                 "Failed to communicate with scene detection process: ");
      return FALSE;
    }
  
  // Check exit status
  int exit_status = g_subprocess_get_exit_status (subprocess);
  if (exit_status != 0)
    {
      g_set_error (error,
                  G_IO_ERROR,
                  G_IO_ERROR_FAILED,
                  "Scene detection process failed (exit code %d): %s",
                  exit_status, stderr_buf ? stderr_buf : "Unknown error");
      return FALSE;
    }
  
  // Parse JSON output
  if (stdout_buf != NULL && *stdout_buf != '\0')
    {
      return parse_scene_detection_json (detector, stdout_buf, error);
    }
  
  g_set_error (error,
              G_IO_ERROR,
              G_IO_ERROR_FAILED,
              "No output from scene detection process");
  return FALSE;
}

GArray *
blouedit_scene_detector_get_scene_changes (BlouEditSceneDetector *detector)
{
  g_return_val_if_fail (BLOUEDIT_IS_SCENE_DETECTOR (detector), NULL);
  
  return detector->scene_changes;
}

void
blouedit_scene_detector_set_threshold (BlouEditSceneDetector *detector, double threshold)
{
  g_return_if_fail (BLOUEDIT_IS_SCENE_DETECTOR (detector));
  
  // Clamp to valid range
  threshold = CLAMP (threshold, 0.0, 1.0);
  
  if (detector->threshold != threshold)
    {
      detector->threshold = threshold;
      g_object_notify_by_pspec (G_OBJECT (detector), props[PROP_THRESHOLD]);
    }
}

double
blouedit_scene_detector_get_threshold (BlouEditSceneDetector *detector)
{
  g_return_val_if_fail (BLOUEDIT_IS_SCENE_DETECTOR (detector), 0.5);
  
  return detector->threshold;
}

void
blouedit_scene_detector_set_use_ml (BlouEditSceneDetector *detector, gboolean use_ml)
{
  g_return_if_fail (BLOUEDIT_IS_SCENE_DETECTOR (detector));
  
  if (detector->use_ml != use_ml)
    {
      detector->use_ml = use_ml;
      g_object_notify_by_pspec (G_OBJECT (detector), props[PROP_USE_ML]);
    }
}

gboolean
blouedit_scene_detector_get_use_ml (BlouEditSceneDetector *detector)
{
  g_return_val_if_fail (BLOUEDIT_IS_SCENE_DETECTOR (detector), TRUE);
  
  return detector->use_ml;
} 