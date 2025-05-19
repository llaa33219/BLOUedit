#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>

namespace BlouEdit {
namespace Editor {

class Rotate {
public:
    Rotate();
    ~Rotate();

    // Apply rotation to a video frame
    bool applyRotation(GstBuffer* buffer, int rotation_angle);
    
    // Rotate image by specific angle
    GdkPixbuf* rotateImage(GdkPixbuf* src, int angle);
    
    // Get rotation angle for a specific clip
    int getClipRotation(int clip_id);
    
    // Set rotation angle for a specific clip
    void setClipRotation(int clip_id, int angle);

private:
    // Internal storage for clip rotation angles
    GHashTable* clip_rotations;
    
    // Rotation transformation matrix calculation
    void calculateRotationMatrix(int angle, float* matrix);
};

} // namespace Editor
} // namespace BlouEdit 