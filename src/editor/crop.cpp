#include "crop.h"
#include <string.h>

namespace BlouEdit {
namespace Editor {

// Helper function to create a CropRect
static CropRect* create_crop_rect(int x, int y, int width, int height) {
    CropRect* rect = g_new(CropRect, 1);
    rect->x = x;
    rect->y = y;
    rect->width = width;
    rect->height = height;
    return rect;
}

// Helper function to free a CropRect
static void free_crop_rect(gpointer data) {
    g_free(data);
}

Crop::Crop() {
    // Initialize hash table to store crop rectangles for clips
    clip_crops = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, free_crop_rect);
}

Crop::~Crop() {
    // Free resources
    g_hash_table_destroy(clip_crops);
}

bool Crop::applyCrop(GstBuffer* buffer, CropRect rect, int original_width, int original_height) {
    // Validate buffer
    if (!buffer) {
        g_warning("Null buffer provided to crop function");
        return false;
    }
    
    // Validate crop rectangle
    if (!validateCropRect(rect, original_width, original_height)) {
        g_warning("Invalid crop rectangle (%d,%d,%d,%d) for dimensions %dx%d",
                 rect.x, rect.y, rect.width, rect.height, original_width, original_height);
        return false;
    }
    
    // No crop needed if using the entire frame
    if (rect.x == 0 && rect.y == 0 && 
        rect.width == original_width && rect.height == original_height) {
        return true;
    }
    
    // Apply crop to GStreamer buffer
    // This implementation depends on the specific pixel format
    // For a real implementation, we would need to:
    // 1. Map the buffer to access its data
    // 2. Apply the crop transformation
    // 3. Unmap the buffer
    
    // Simple implementation (placeholder)
    GstMapInfo map;
    
    if (!gst_buffer_map(buffer, &map, GST_MAP_READWRITE)) {
        g_warning("Failed to map buffer for crop operation");
        return false;
    }
    
    // Apply crop to pixels (simplified)
    // In a real implementation, this would involve extracting the crop region
    // and updating the buffer dimensions
    
    // For now, we just log the crop
    g_print("Applying crop (%d,%d,%d,%d) to video frame of size %dx%d\n", 
           rect.x, rect.y, rect.width, rect.height, original_width, original_height);
    
    gst_buffer_unmap(buffer, &map);
    return true;
}

GdkPixbuf* Crop::cropImage(GdkPixbuf* src, CropRect rect) {
    if (!src) {
        return NULL;
    }
    
    int width = gdk_pixbuf_get_width(src);
    int height = gdk_pixbuf_get_height(src);
    
    // Validate crop rectangle
    if (!validateCropRect(rect, width, height)) {
        g_warning("Invalid crop rectangle (%d,%d,%d,%d) for image size %dx%d",
                 rect.x, rect.y, rect.width, rect.height, width, height);
        return gdk_pixbuf_copy(src);
    }
    
    // No crop needed if using the entire image
    if (rect.x == 0 && rect.y == 0 && rect.width == width && rect.height == height) {
        return gdk_pixbuf_copy(src);
    }
    
    // Create a new pixbuf for the cropped portion of the image
    GdkPixbuf* cropped = gdk_pixbuf_new_subpixbuf(
        src,
        rect.x, rect.y,
        rect.width, rect.height
    );
    
    // Create a copy of the subpixbuf to ensure it has its own memory
    GdkPixbuf* result = gdk_pixbuf_copy(cropped);
    g_object_unref(cropped);
    
    return result;
}

CropRect Crop::getClipCrop(int clip_id) {
    CropRect* rect = (CropRect*)g_hash_table_lookup(clip_crops, GINT_TO_POINTER(clip_id));
    
    if (rect == NULL) {
        // Return a default crop rect (no crop)
        CropRect default_rect = {0, 0, 0, 0};
        return default_rect;
    }
    
    return *rect;
}

void Crop::setClipCrop(int clip_id, CropRect rect) {
    // Create a copy of the crop rect for storage
    CropRect* stored_rect = create_crop_rect(rect.x, rect.y, rect.width, rect.height);
    
    // Remove any existing entry first
    g_hash_table_remove(clip_crops, GINT_TO_POINTER(clip_id));
    
    // Store the new crop rect
    g_hash_table_insert(clip_crops, GINT_TO_POINTER(clip_id), stored_rect);
    
    g_print("Set crop of clip %d to (%d,%d,%d,%d)\n", 
           clip_id, rect.x, rect.y, rect.width, rect.height);
}

CropRect Crop::getDefaultCropRect(int width, int height) {
    CropRect rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = width;
    rect.height = height;
    return rect;
}

bool Crop::validateCropRect(CropRect rect, int width, int height) {
    // Check that the rect is within bounds
    if (rect.x < 0 || rect.y < 0) {
        return false;
    }
    
    if (rect.width <= 0 || rect.height <= 0) {
        return false;
    }
    
    if (rect.x + rect.width > width || rect.y + rect.height > height) {
        return false;
    }
    
    return true;
}

} // namespace Editor
} // namespace BlouEdit 