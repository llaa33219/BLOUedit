#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <vector>
#include <string>

namespace BlouEdit {
namespace Editor {

// Structure to represent a sticker on the timeline
typedef struct {
    int id;                // Unique identifier
    int clip_id;           // Clip the sticker is attached to
    std::string path;      // Path to sticker image
    int x;                 // X position (%)
    int y;                 // Y position (%)
    int width;             // Width (%)
    int height;            // Height (%)
    int start_time;        // Start time in clip (ms)
    int duration;          // Duration in clip (ms)
    float rotation;        // Rotation angle (degrees)
    float opacity;         // Opacity (0.0 - 1.0)
    GdkPixbuf* pixbuf;     // Cached pixbuf
} Sticker;

class StickerManager {
public:
    StickerManager();
    ~StickerManager();

    // Add a sticker to a clip
    int addSticker(int clip_id, const std::string& sticker_path, 
                   int start_time, int duration);
    
    // Remove a sticker
    bool removeSticker(int sticker_id);
    
    // Update sticker properties
    bool updateStickerPosition(int sticker_id, int x, int y);
    bool updateStickerSize(int sticker_id, int width, int height);
    bool updateStickerTiming(int sticker_id, int start_time, int duration);
    bool updateStickerRotation(int sticker_id, float rotation);
    bool updateStickerOpacity(int sticker_id, float opacity);
    
    // Get stickers for a clip at a specific time
    std::vector<Sticker*> getStickersAtTime(int clip_id, int time_ms);
    
    // Get all stickers for a clip
    std::vector<Sticker*> getStickersForClip(int clip_id);
    
    // Render stickers onto a video frame
    bool renderStickersOnFrame(GstBuffer* buffer, int clip_id, 
                               int time_ms, int width, int height);
    
    // Load built-in sticker packs
    void loadStickerPacks();
    
    // Get available sticker categories
    std::vector<std::string> getStickerCategories();
    
    // Get stickers in a category
    std::vector<std::string> getStickersInCategory(const std::string& category);

private:
    // Internal sticker storage
    std::vector<Sticker*> stickers;
    int next_sticker_id;
    
    // Sticker resource paths
    std::vector<std::string> sticker_categories;
    std::vector<std::vector<std::string>> sticker_paths;
    
    // Helper methods
    Sticker* findStickerById(int sticker_id);
    GdkPixbuf* loadStickerImage(const std::string& path);
};

} // namespace Editor
} // namespace BlouEdit 