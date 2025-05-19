#include "flip.h"

namespace BlouEdit {
namespace Editor {

Flip::Flip() {
    // Initialize hash table to store flip directions for clips
    clip_flips = g_hash_table_new(g_direct_hash, g_direct_equal);
}

Flip::~Flip() {
    // Free resources
    g_hash_table_destroy(clip_flips);
}

bool Flip::applyFlip(GstBuffer* buffer, FlipDirection direction) {
    // Validate buffer
    if (!buffer) {
        g_warning("Null buffer provided to flip function");
        return false;
    }
    
    // No flip needed
    if (direction == FLIP_NONE) {
        return true;
    }
    
    // Apply flip to GStreamer buffer
    // This implementation depends on the specific pixel format
    // For a real implementation, we would need to:
    // 1. Map the buffer to access its data
    // 2. Apply the flip transformation
    // 3. Unmap the buffer
    
    // Simple implementation (placeholder)
    GstMapInfo map;
    
    if (!gst_buffer_map(buffer, &map, GST_MAP_READWRITE)) {
        g_warning("Failed to map buffer for flip operation");
        return false;
    }
    
    // Apply flip to pixels (simplified)
    // In a real implementation, this would involve more complex pixel manipulation
    // based on the flip direction and video format
    
    // For now, we just log the flip
    const char* direction_str = "unknown";
    switch (direction) {
        case FLIP_HORIZONTAL: direction_str = "horizontal"; break;
        case FLIP_VERTICAL: direction_str = "vertical"; break;
        case FLIP_BOTH: direction_str = "both horizontal and vertical"; break;
        default: direction_str = "unknown"; break;
    }
    
    g_print("Applying %s flip to video frame\n", direction_str);
    
    gst_buffer_unmap(buffer, &map);
    return true;
}

GdkPixbuf* Flip::flipImage(GdkPixbuf* src, FlipDirection direction) {
    if (!src) {
        return NULL;
    }
    
    // Fast path for no flip
    if (direction == FLIP_NONE) {
        return gdk_pixbuf_copy(src);
    }
    
    int width = gdk_pixbuf_get_width(src);
    int height = gdk_pixbuf_get_height(src);
    int rowstride = gdk_pixbuf_get_rowstride(src);
    int channels = gdk_pixbuf_get_n_channels(src);
    guchar* pixels = gdk_pixbuf_get_pixels(src);
    gboolean has_alpha = gdk_pixbuf_get_has_alpha(src);
    
    // Create new pixbuf for flipped image
    GdkPixbuf* dest = gdk_pixbuf_new(
        gdk_pixbuf_get_colorspace(src),
        has_alpha,
        gdk_pixbuf_get_bits_per_sample(src),
        width,
        height
    );
    
    guchar* dest_pixels = gdk_pixbuf_get_pixels(dest);
    int dest_rowstride = gdk_pixbuf_get_rowstride(dest);
    
    // Apply flip transformation
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate source coordinates based on flip direction
            int src_x = x;
            int src_y = y;
            
            if (direction & FLIP_HORIZONTAL) {
                src_x = width - 1 - x;
            }
            
            if (direction & FLIP_VERTICAL) {
                src_y = height - 1 - y;
            }
            
            // Copy pixel data
            guchar* src_pixel = pixels + src_y * rowstride + src_x * channels;
            guchar* dest_pixel = dest_pixels + y * dest_rowstride + x * channels;
            
            for (int c = 0; c < channels; c++) {
                dest_pixel[c] = src_pixel[c];
            }
        }
    }
    
    return dest;
}

FlipDirection Flip::getClipFlip(int clip_id) {
    gpointer value = g_hash_table_lookup(clip_flips, GINT_TO_POINTER(clip_id));
    if (value == NULL) {
        return FLIP_NONE;  // Default is no flip
    }
    return (FlipDirection)GPOINTER_TO_INT(value);
}

void Flip::setClipFlip(int clip_id, FlipDirection direction) {
    g_hash_table_insert(clip_flips, GINT_TO_POINTER(clip_id), GINT_TO_POINTER(direction));
    
    const char* direction_str = "unknown";
    switch (direction) {
        case FLIP_NONE: direction_str = "none"; break;
        case FLIP_HORIZONTAL: direction_str = "horizontal"; break;
        case FLIP_VERTICAL: direction_str = "vertical"; break;
        case FLIP_BOTH: direction_str = "both horizontal and vertical"; break;
        default: direction_str = "unknown"; break;
    }
    
    g_print("Set flip of clip %d to %s\n", clip_id, direction_str);
}

} // namespace Editor
} // namespace BlouEdit 