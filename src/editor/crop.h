#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>

namespace BlouEdit {
namespace Editor {

// Structure to represent a crop rectangle
typedef struct {
    int x;      // Left position
    int y;      // Top position
    int width;  // Width of crop area
    int height; // Height of crop area
} CropRect;

class Crop {
public:
    Crop();
    ~Crop();

    // Apply crop to a video frame
    bool applyCrop(GstBuffer* buffer, CropRect rect, int original_width, int original_height);
    
    // Crop image based on rectangle
    GdkPixbuf* cropImage(GdkPixbuf* src, CropRect rect);
    
    // Get crop settings for a specific clip
    CropRect getClipCrop(int clip_id);
    
    // Set crop settings for a specific clip
    void setClipCrop(int clip_id, CropRect rect);
    
    // Calculate default crop rect (no crop)
    static CropRect getDefaultCropRect(int width, int height);

private:
    // Internal storage for clip crop rectangles
    GHashTable* clip_crops;
    
    // Validate crop rect is within bounds
    bool validateCropRect(CropRect rect, int width, int height);
};

} // namespace Editor
} // namespace BlouEdit 