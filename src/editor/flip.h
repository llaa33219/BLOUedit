#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>

namespace BlouEdit {
namespace Editor {

enum FlipDirection {
    FLIP_NONE = 0,
    FLIP_HORIZONTAL = 1,
    FLIP_VERTICAL = 2,
    FLIP_BOTH = 3
};

class Flip {
public:
    Flip();
    ~Flip();

    // Apply flip to a video frame
    bool applyFlip(GstBuffer* buffer, FlipDirection direction);
    
    // Flip image horizontally or vertically
    GdkPixbuf* flipImage(GdkPixbuf* src, FlipDirection direction);
    
    // Get flip direction for a specific clip
    FlipDirection getClipFlip(int clip_id);
    
    // Set flip direction for a specific clip
    void setClipFlip(int clip_id, FlipDirection direction);

private:
    // Internal storage for clip flip directions
    GHashTable* clip_flips;
};

} // namespace Editor
} // namespace BlouEdit 