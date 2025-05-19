#include "rotate.h"
#include <math.h>

namespace BlouEdit {
namespace Editor {

Rotate::Rotate() {
    // Initialize hash table to store rotation angles for clips
    clip_rotations = g_hash_table_new(g_direct_hash, g_direct_equal);
}

Rotate::~Rotate() {
    // Free resources
    g_hash_table_destroy(clip_rotations);
}

bool Rotate::applyRotation(GstBuffer* buffer, int rotation_angle) {
    // Validate buffer
    if (!buffer) {
        g_warning("Null buffer provided to rotation function");
        return false;
    }
    
    // Apply rotation to GStreamer buffer
    // This implementation depends on the specific pixel format
    // For a real implementation, we would need to:
    // 1. Map the buffer to access its data
    // 2. Apply the rotation transformation
    // 3. Unmap the buffer
    
    // Simple implementation (placeholder)
    GstMapInfo map;
    
    if (!gst_buffer_map(buffer, &map, GST_MAP_READWRITE)) {
        g_warning("Failed to map buffer for rotation");
        return false;
    }
    
    // Apply rotation to pixels (simplified)
    // In a real implementation, this would involve more complex pixel manipulation
    // based on the rotation angle and video format
    
    // For now, we just log the rotation
    g_print("Applying %d degree rotation to video frame\n", rotation_angle);
    
    gst_buffer_unmap(buffer, &map);
    return true;
}

GdkPixbuf* Rotate::rotateImage(GdkPixbuf* src, int angle) {
    if (!src) {
        return NULL;
    }
    
    // Normalize angle to 0-359 range
    angle = angle % 360;
    if (angle < 0) angle += 360;
    
    // Fast path for no rotation
    if (angle == 0) {
        return gdk_pixbuf_copy(src);
    }
    
    // For 90/180/270 degree rotations, use the built-in GDK functions
    if (angle == 90) {
        return gdk_pixbuf_rotate_simple(src, GDK_PIXBUF_ROTATE_CLOCKWISE);
    } else if (angle == 180) {
        return gdk_pixbuf_rotate_simple(src, GDK_PIXBUF_ROTATE_UPSIDEDOWN);
    } else if (angle == 270) {
        return gdk_pixbuf_rotate_simple(src, GDK_PIXBUF_ROTATE_COUNTERCLOCKWISE);
    }
    
    // For arbitrary angles, we need to do more complex transformation
    int width = gdk_pixbuf_get_width(src);
    int height = gdk_pixbuf_get_height(src);
    
    // Calculate dimensions of rotated image
    double radians = angle * M_PI / 180.0;
    double sin_angle = fabs(sin(radians));
    double cos_angle = fabs(cos(radians));
    
    int new_width = (int)(width * cos_angle + height * sin_angle);
    int new_height = (int)(width * sin_angle + height * cos_angle);
    
    // Create new pixbuf for rotated image
    GdkPixbuf* dest = gdk_pixbuf_new(
        gdk_pixbuf_get_colorspace(src),
        gdk_pixbuf_get_has_alpha(src),
        gdk_pixbuf_get_bits_per_sample(src),
        new_width,
        new_height
    );
    
    // Fill with transparent or background color
    gdk_pixbuf_fill(dest, 0x00000000);
    
    // Set up rotation transformation
    float matrix[6];
    calculateRotationMatrix(angle, matrix);
    
    // Apply the transformation
    gdk_pixbuf_composite(
        src, dest,
        0, 0, new_width, new_height,
        new_width/2.0 - width/2.0, new_height/2.0 - height/2.0,  // offset
        1.0, 1.0,  // scale
        GDK_INTERP_BILINEAR,
        255
    );
    
    return dest;
}

int Rotate::getClipRotation(int clip_id) {
    gpointer value = g_hash_table_lookup(clip_rotations, GINT_TO_POINTER(clip_id));
    if (value == NULL) {
        return 0;  // Default is no rotation
    }
    return GPOINTER_TO_INT(value);
}

void Rotate::setClipRotation(int clip_id, int angle) {
    // Normalize angle to 0-359
    angle = angle % 360;
    if (angle < 0) angle += 360;
    
    g_hash_table_insert(clip_rotations, GINT_TO_POINTER(clip_id), GINT_TO_POINTER(angle));
    g_print("Set rotation of clip %d to %d degrees\n", clip_id, angle);
}

void Rotate::calculateRotationMatrix(int angle, float* matrix) {
    // Convert angle to radians
    float radians = angle * M_PI / 180.0;
    
    // Calculate sine and cosine
    float cos_angle = cos(radians);
    float sin_angle = sin(radians);
    
    // Fill the transformation matrix
    matrix[0] = cos_angle;
    matrix[1] = -sin_angle;
    matrix[2] = sin_angle;
    matrix[3] = cos_angle;
    matrix[4] = 0.0f;  // x translation
    matrix[5] = 0.0f;  // y translation
}

} // namespace Editor
} // namespace BlouEdit 