#include "stickers.h"
#include <algorithm>
#include <glib/gstdio.h>
#include <cairo.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

namespace BlouEdit {
namespace Editor {

StickerManager::StickerManager() : next_sticker_id(1) {
    // Initialize sticker categories
    sticker_categories = {
        "Emojis",
        "Animals",
        "Shapes",
        "Text Bubbles",
        "Decorative"
    };
    
    // Initialize with empty paths until loadStickerPacks is called
    sticker_paths.resize(sticker_categories.size());
}

StickerManager::~StickerManager() {
    // Clean up all stickers
    for (Sticker* sticker : stickers) {
        if (sticker->pixbuf) {
            g_object_unref(sticker->pixbuf);
        }
        delete sticker;
    }
    stickers.clear();
}

int StickerManager::addSticker(int clip_id, const std::string& sticker_path, 
                             int start_time, int duration) {
    // Create new sticker
    Sticker* sticker = new Sticker();
    sticker->id = next_sticker_id++;
    sticker->clip_id = clip_id;
    sticker->path = sticker_path;
    sticker->x = 50;        // Default center position
    sticker->y = 50;
    sticker->width = 20;    // Default 20% of frame
    sticker->height = 20;
    sticker->start_time = start_time;
    sticker->duration = duration;
    sticker->rotation = 0.0f;
    sticker->opacity = 1.0f;
    
    // Load the sticker image
    sticker->pixbuf = loadStickerImage(sticker_path);
    
    if (!sticker->pixbuf) {
        g_warning("Failed to load sticker from path: %s", sticker_path.c_str());
        delete sticker;
        return -1;
    }
    
    // Add to collection
    stickers.push_back(sticker);
    
    g_print("Added sticker %d to clip %d at time %d ms for %d ms\n", 
           sticker->id, clip_id, start_time, duration);
    
    return sticker->id;
}

bool StickerManager::removeSticker(int sticker_id) {
    auto it = std::find_if(stickers.begin(), stickers.end(),
                         [sticker_id](Sticker* s) { return s->id == sticker_id; });
    
    if (it != stickers.end()) {
        Sticker* sticker = *it;
        
        // Free resources
        if (sticker->pixbuf) {
            g_object_unref(sticker->pixbuf);
        }
        
        // Remove from collection
        stickers.erase(it);
        delete sticker;
        
        g_print("Removed sticker %d\n", sticker_id);
        return true;
    }
    
    g_warning("Sticker %d not found for removal", sticker_id);
    return false;
}

bool StickerManager::updateStickerPosition(int sticker_id, int x, int y) {
    Sticker* sticker = findStickerById(sticker_id);
    if (!sticker) {
        return false;
    }
    
    // Clamp values to valid range (0-100%)
    sticker->x = std::max(0, std::min(100, x));
    sticker->y = std::max(0, std::min(100, y));
    
    g_print("Updated sticker %d position to (%d%%, %d%%)\n", 
           sticker_id, sticker->x, sticker->y);
    
    return true;
}

bool StickerManager::updateStickerSize(int sticker_id, int width, int height) {
    Sticker* sticker = findStickerById(sticker_id);
    if (!sticker) {
        return false;
    }
    
    // Clamp values to valid range (1-100%)
    sticker->width = std::max(1, std::min(100, width));
    sticker->height = std::max(1, std::min(100, height));
    
    g_print("Updated sticker %d size to (%d%%, %d%%)\n", 
           sticker_id, sticker->width, sticker->height);
    
    return true;
}

bool StickerManager::updateStickerTiming(int sticker_id, int start_time, int duration) {
    Sticker* sticker = findStickerById(sticker_id);
    if (!sticker) {
        return false;
    }
    
    // Ensure valid duration
    if (duration <= 0) {
        g_warning("Invalid duration %d for sticker %d", duration, sticker_id);
        return false;
    }
    
    sticker->start_time = start_time;
    sticker->duration = duration;
    
    g_print("Updated sticker %d timing to start=%d ms, duration=%d ms\n", 
           sticker_id, start_time, duration);
    
    return true;
}

bool StickerManager::updateStickerRotation(int sticker_id, float rotation) {
    Sticker* sticker = findStickerById(sticker_id);
    if (!sticker) {
        return false;
    }
    
    sticker->rotation = rotation;
    
    g_print("Updated sticker %d rotation to %.1f degrees\n", 
           sticker_id, rotation);
    
    return true;
}

bool StickerManager::updateStickerOpacity(int sticker_id, float opacity) {
    Sticker* sticker = findStickerById(sticker_id);
    if (!sticker) {
        return false;
    }
    
    // Clamp opacity to valid range [0.0, 1.0]
    sticker->opacity = std::max(0.0f, std::min(1.0f, opacity));
    
    g_print("Updated sticker %d opacity to %.2f\n", 
           sticker_id, sticker->opacity);
    
    return true;
}

std::vector<Sticker*> StickerManager::getStickersAtTime(int clip_id, int time_ms) {
    std::vector<Sticker*> result;
    
    for (Sticker* sticker : stickers) {
        if (sticker->clip_id == clip_id &&
            time_ms >= sticker->start_time &&
            time_ms < sticker->start_time + sticker->duration) {
            result.push_back(sticker);
        }
    }
    
    return result;
}

std::vector<Sticker*> StickerManager::getStickersForClip(int clip_id) {
    std::vector<Sticker*> result;
    
    for (Sticker* sticker : stickers) {
        if (sticker->clip_id == clip_id) {
            result.push_back(sticker);
        }
    }
    
    return result;
}

bool StickerManager::renderStickersOnFrame(GstBuffer* buffer, int clip_id, 
                                         int time_ms, int width, int height) {
    // Get stickers that should be visible at this time
    std::vector<Sticker*> visible_stickers = getStickersAtTime(clip_id, time_ms);
    
    if (visible_stickers.empty()) {
        return true;  // Nothing to render
    }
    
    // Map buffer for writing
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READWRITE)) {
        g_warning("Failed to map buffer for sticker rendering");
        return false;
    }
    
    // Create a cairo surface for drawing
    cairo_surface_t* surface = cairo_image_surface_create_for_data(
        map.data,
        CAIRO_FORMAT_ARGB32,
        width, height,
        width * 4  // Assuming 4 bytes per pixel
    );
    
    cairo_t* cr = cairo_create(surface);
    
    // Render each sticker
    for (Sticker* sticker : visible_stickers) {
        // Calculate sticker position and size in pixels
        int sticker_x = width * sticker->x / 100;
        int sticker_y = height * sticker->y / 100;
        int sticker_width = width * sticker->width / 100;
        int sticker_height = height * sticker->height / 100;
        
        // Create a temporary scaled pixbuf for the sticker
        GdkPixbuf* scaled = gdk_pixbuf_scale_simple(
            sticker->pixbuf,
            sticker_width,
            sticker_height,
            GDK_INTERP_BILINEAR
        );
        
        // Save cairo state before transformations
        cairo_save(cr);
        
        // Set opacity
        cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
        cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, sticker->opacity);
        
        // Apply rotation if needed
        if (sticker->rotation != 0.0f) {
            // Translate to center of sticker
            cairo_translate(cr, 
                           sticker_x + sticker_width / 2.0,
                           sticker_y + sticker_height / 2.0);
            
            // Rotate
            cairo_rotate(cr, sticker->rotation * M_PI / 180.0);
            
            // Translate back
            cairo_translate(cr, 
                           -(sticker_width / 2.0),
                           -(sticker_height / 2.0));
            
            // Draw sticker
            gdk_cairo_set_source_pixbuf(cr, scaled, 0, 0);
        } else {
            // Draw sticker directly at position
            gdk_cairo_set_source_pixbuf(cr, scaled, sticker_x, sticker_y);
        }
        
        cairo_paint_with_alpha(cr, sticker->opacity);
        
        // Restore cairo state
        cairo_restore(cr);
        
        // Clean up
        g_object_unref(scaled);
    }
    
    // Clean up
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    
    gst_buffer_unmap(buffer, &map);
    
    return true;
}

void StickerManager::loadStickerPacks() {
    // In a real implementation, this would scan directories for sticker packs
    // For this example, we'll just populate with some placeholders
    
    // Clear previous paths
    for (auto& category : sticker_paths) {
        category.clear();
    }
    
    // Define some dummy sticker paths
    const char* resource_dir = "/usr/share/blouedit/stickers/";
    
    // Emojis
    sticker_paths[0] = {
        std::string(resource_dir) + "emojis/smile.png",
        std::string(resource_dir) + "emojis/heart.png",
        std::string(resource_dir) + "emojis/thumbsup.png",
        std::string(resource_dir) + "emojis/star.png"
    };
    
    // Animals
    sticker_paths[1] = {
        std::string(resource_dir) + "animals/cat.png",
        std::string(resource_dir) + "animals/dog.png",
        std::string(resource_dir) + "animals/unicorn.png"
    };
    
    // Shapes
    sticker_paths[2] = {
        std::string(resource_dir) + "shapes/circle.png",
        std::string(resource_dir) + "shapes/square.png",
        std::string(resource_dir) + "shapes/triangle.png"
    };
    
    // Text Bubbles
    sticker_paths[3] = {
        std::string(resource_dir) + "bubbles/speech.png",
        std::string(resource_dir) + "bubbles/thought.png",
        std::string(resource_dir) + "bubbles/shout.png"
    };
    
    // Decorative
    sticker_paths[4] = {
        std::string(resource_dir) + "decorative/star.png",
        std::string(resource_dir) + "decorative/heart.png",
        std::string(resource_dir) + "decorative/flower.png"
    };
    
    g_print("Loaded sticker packs with %zu categories\n", sticker_categories.size());
}

std::vector<std::string> StickerManager::getStickerCategories() {
    return sticker_categories;
}

std::vector<std::string> StickerManager::getStickersInCategory(const std::string& category) {
    // Find the category index
    auto it = std::find(sticker_categories.begin(), sticker_categories.end(), category);
    
    if (it == sticker_categories.end()) {
        g_warning("Invalid sticker category: %s", category.c_str());
        return {};
    }
    
    int index = it - sticker_categories.begin();
    return sticker_paths[index];
}

Sticker* StickerManager::findStickerById(int sticker_id) {
    auto it = std::find_if(stickers.begin(), stickers.end(),
                         [sticker_id](Sticker* s) { return s->id == sticker_id; });
    
    if (it != stickers.end()) {
        return *it;
    }
    
    g_warning("Sticker with ID %d not found", sticker_id);
    return nullptr;
}

GdkPixbuf* StickerManager::loadStickerImage(const std::string& path) {
    GError* error = NULL;
    GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(path.c_str(), &error);
    
    if (!pixbuf) {
        g_warning("Failed to load sticker image: %s", error->message);
        g_error_free(error);
        return NULL;
    }
    
    return pixbuf;
}

} // namespace Editor
} // namespace BlouEdit 