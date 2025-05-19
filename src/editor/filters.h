#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace BLOUedit {

enum class FilterType {
    GRAYSCALE,
    SEPIA,
    INVERT,
    BRIGHTNESS,
    CONTRAST,
    SATURATION,
    HUE,
    BLUR,
    SHARPEN,
    VIGNETTE,
    CUSTOM
};

class VideoFilter {
private:
    FilterType type;
    
    // Filter parameters (used differently based on filter type)
    double param1; // Primary parameter (e.g., brightness level, blur radius)
    double param2; // Secondary parameter (e.g., contrast, sharpen intensity)
    double param3; // Tertiary parameter (e.g., vignette radius)
    
    std::string customScript; // For custom filters
    bool enabled;
    
public:
    VideoFilter();
    explicit VideoFilter(FilterType type);
    ~VideoFilter();
    
    void setType(FilterType type);
    FilterType getType() const;
    
    void setParam1(double value);
    double getParam1() const;
    
    void setParam2(double value);
    double getParam2() const;
    
    void setParam3(double value);
    double getParam3() const;
    
    void setCustomScript(const std::string &script);
    std::string getCustomScript() const;
    
    void setEnabled(bool enable);
    bool isEnabled() const;
    
    // Apply filter to a video frame
    bool applyFilter(GstBuffer *inFrame, GstBuffer *outFrame);
    
    // Create a GStreamer element for this filter
    GstElement* createFilterElement();
    
    // Create parameter adjustment widget
    GtkWidget* createControlWidget();
    
    // Static utility methods
    static std::string getTypeName(FilterType type);
    static std::vector<FilterType> getAvailableFilters();
    static std::string getFilterDescription(FilterType type);
};

// Filter Chain to manage a sequence of filters
class FilterChain {
private:
    std::vector<std::shared_ptr<VideoFilter>> filters;
    
public:
    FilterChain();
    
    void addFilter(std::shared_ptr<VideoFilter> filter);
    void removeFilter(size_t index);
    void moveFilterUp(size_t index);
    void moveFilterDown(size_t index);
    
    std::shared_ptr<VideoFilter> getFilter(size_t index);
    size_t getFilterCount() const;
    
    // Apply all enabled filters in sequence
    bool applyFilters(GstBuffer *inFrame, GstBuffer *outFrame);
    
    // Create a GStreamer bin containing all filter elements
    GstElement* createFilterBin();
    
    // Create UI for the filter chain
    GtkWidget* createFilterChainWidget(std::function<void()> updateCallback);
    
    // Save/load filters
    bool saveToJson(const std::string &path);
    bool loadFromJson(const std::string &path);
    
    // Clear all filters
    void clear();
};

} // namespace BLOUedit 