#include "filters.h"
#include <iostream>
#include <gst/video/video.h>

namespace BLOUedit {

// VideoFilter implementation
VideoFilter::VideoFilter() 
    : type(FilterType::GRAYSCALE), param1(0.0), param2(0.0), param3(0.0), enabled(true) {}

VideoFilter::VideoFilter(FilterType type) 
    : type(type), param1(0.0), param2(0.0), param3(0.0), enabled(true) {
    
    // Set default values based on filter type
    switch (type) {
        case FilterType::BRIGHTNESS: param1 = 0.0; break; 
        case FilterType::CONTRAST: param1 = 1.0; break;   
        case FilterType::SATURATION: param1 = 1.0; break; 
        case FilterType::HUE: param1 = 0.0; break;        
        case FilterType::BLUR: param1 = 5.0; break;       
        case FilterType::SHARPEN: param1 = 0.5; break;    
        case FilterType::VIGNETTE: param1 = 0.3; param2 = 0.7; break;
        default: break;
    }
}

VideoFilter::~VideoFilter() {}

// Getters and setters
void VideoFilter::setType(FilterType t) { type = t; }
FilterType VideoFilter::getType() const { return type; }
void VideoFilter::setParam1(double value) { param1 = value; }
double VideoFilter::getParam1() const { return param1; }
void VideoFilter::setParam2(double value) { param2 = value; }
double VideoFilter::getParam2() const { return param2; }
void VideoFilter::setParam3(double value) { param3 = value; }
double VideoFilter::getParam3() const { return param3; }
void VideoFilter::setCustomScript(const std::string &script) { customScript = script; }
std::string VideoFilter::getCustomScript() const { return customScript; }
void VideoFilter::setEnabled(bool enable) { enabled = enable; }
bool VideoFilter::isEnabled() const { return enabled; }

GstElement* VideoFilter::createFilterElement() {
    GstElement *element = nullptr;
    
    switch (type) {
        case FilterType::GRAYSCALE:
            element = gst_element_factory_make("videobalance", "grayscale");
            g_object_set(G_OBJECT(element), "saturation", 0.0, NULL);
            break;
        case FilterType::SEPIA:
            element = gst_element_factory_make("gleffects", "sepia");
            g_object_set(G_OBJECT(element), "effect", 18, NULL);
            break;
        case FilterType::INVERT:
            element = gst_element_factory_make("videoflip", "invert");
            g_object_set(G_OBJECT(element), "video-direction", 4, NULL);
            break;
        case FilterType::BRIGHTNESS:
            element = gst_element_factory_make("videobalance", "brightness");
            g_object_set(G_OBJECT(element), "brightness", param1, NULL);
            break;
        case FilterType::CONTRAST:
            element = gst_element_factory_make("videobalance", "contrast");
            g_object_set(G_OBJECT(element), "contrast", param1, NULL);
            break;
        case FilterType::SATURATION:
            element = gst_element_factory_make("videobalance", "saturation");
            g_object_set(G_OBJECT(element), "saturation", param1, NULL);
            break;
        case FilterType::HUE:
            element = gst_element_factory_make("videobalance", "hue");
            g_object_set(G_OBJECT(element), "hue", param1, NULL);
            break;
        case FilterType::BLUR:
            element = gst_element_factory_make("gaussianblur", "blur");
            if (element) g_object_set(G_OBJECT(element), "sigma", param1 * 0.1, NULL);
            break;
        case FilterType::SHARPEN:
            element = gst_element_factory_make("frei0r-filter-sharpen", "sharpen");
            if (element) g_object_set(G_OBJECT(element), "amount", param1, NULL);
            break;
        case FilterType::VIGNETTE:
            element = gst_element_factory_make("frei0r-filter-vignette", "vignette");
            if (element) {
                g_object_set(G_OBJECT(element), "aspect", 1.0, "clarity", param1, "radius", param2, NULL);
            }
            break;
        case FilterType::CUSTOM:
            if (!customScript.empty()) {
                element = gst_element_factory_make("glshader", "custom-shader");
                if (element) g_object_set(G_OBJECT(element), "fragment", customScript.c_str(), NULL);
            }
            break;
    }
    
    return element;
}

bool VideoFilter::applyFilter(GstBuffer *inFrame, GstBuffer *outFrame) {
    if (!enabled) {
        gst_buffer_copy_into(outFrame, inFrame, GST_BUFFER_COPY_ALL, 0, -1);
        return true;
    }
    
    GstVideoFrame inVideoFrame, outVideoFrame;
    GstVideoInfo info;
    
    gst_video_info_init(&info);
    gst_video_info_set_format(&info, GST_VIDEO_FORMAT_RGBA, 1920, 1080);
    
    if (!gst_video_frame_map(&inVideoFrame, &info, inFrame, GST_MAP_READ)) {
        std::cerr << "Failed to map input frame" << std::endl;
        return false;
    }
    
    if (!gst_video_frame_map(&outVideoFrame, &info, outFrame, GST_MAP_WRITE)) {
        gst_video_frame_unmap(&inVideoFrame);
        std::cerr << "Failed to map output frame" << std::endl;
        return false;
    }
    
    uint8_t *src = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&inVideoFrame, 0);
    uint8_t *dest = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&outVideoFrame, 0);
    int srcStride = GST_VIDEO_FRAME_PLANE_STRIDE(&inVideoFrame, 0);
    int destStride = GST_VIDEO_FRAME_PLANE_STRIDE(&outVideoFrame, 0);
    int width = GST_VIDEO_FRAME_WIDTH(&inVideoFrame);
    int height = GST_VIDEO_FRAME_HEIGHT(&inVideoFrame);
    
    // Apply filter effect (implementation for grayscale as an example)
    if (type == FilterType::GRAYSCALE) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int offset = y * srcStride + x * 4;
                uint8_t r = src[offset];
                uint8_t g = src[offset + 1];
                uint8_t b = src[offset + 2];
                uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
                
                dest[offset] = gray;
                dest[offset + 1] = gray;
                dest[offset + 2] = gray;
                dest[offset + 3] = src[offset + 3]; // Keep original alpha
            }
        }
    } else {
        // For simplicity, just copy for other filters
        for (int y = 0; y < height; y++) {
            memcpy(dest + y * destStride, src + y * srcStride, width * 4);
        }
    }
    
    gst_video_frame_unmap(&inVideoFrame);
    gst_video_frame_unmap(&outVideoFrame);
    
    return true;
}

GtkWidget* VideoFilter::createControlWidget() {
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    
    // Add filter type label
    GtkWidget *labelType = gtk_label_new(getTypeName(type).c_str());
    gtk_box_append(GTK_BOX(vbox), labelType);
    
    // Add enable/disable checkbox
    GtkWidget *checkEnable = gtk_check_button_new_with_label("Enable");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(checkEnable), enabled);
    g_signal_connect(checkEnable, "toggled", G_CALLBACK(+[](GtkCheckButton *button, gpointer data) {
        VideoFilter *filter = static_cast<VideoFilter*>(data);
        filter->setEnabled(gtk_check_button_get_active(button));
    }), this);
    gtk_box_append(GTK_BOX(vbox), checkEnable);
    
    // Add basic parameter slider
    GtkWidget *scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 1.0, 0.01);
    gtk_range_set_value(GTK_RANGE(scale), param1);
    g_signal_connect(scale, "value-changed", G_CALLBACK(+[](GtkRange *range, gpointer data) {
        VideoFilter *filter = static_cast<VideoFilter*>(data);
        filter->setParam1(gtk_range_get_value(range));
    }), this);
    gtk_box_append(GTK_BOX(vbox), scale);
    
    return vbox;
}

std::string VideoFilter::getTypeName(FilterType type) {
    switch (type) {
        case FilterType::GRAYSCALE: return "Grayscale";
        case FilterType::SEPIA: return "Sepia";
        case FilterType::INVERT: return "Invert Colors";
        case FilterType::BRIGHTNESS: return "Brightness";
        case FilterType::CONTRAST: return "Contrast";
        case FilterType::SATURATION: return "Saturation";
        case FilterType::HUE: return "Hue";
        case FilterType::BLUR: return "Blur";
        case FilterType::SHARPEN: return "Sharpen";
        case FilterType::VIGNETTE: return "Vignette";
        case FilterType::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

std::vector<FilterType> VideoFilter::getAvailableFilters() {
    return {
        FilterType::GRAYSCALE, FilterType::SEPIA, FilterType::INVERT,
        FilterType::BRIGHTNESS, FilterType::CONTRAST, FilterType::SATURATION,
        FilterType::HUE, FilterType::BLUR, FilterType::SHARPEN,
        FilterType::VIGNETTE, FilterType::CUSTOM
    };
}

std::string VideoFilter::getFilterDescription(FilterType type) {
    switch (type) {
        case FilterType::GRAYSCALE: return "Convert video to black and white";
        case FilterType::SEPIA: return "Apply a vintage sepia tone effect";
        case FilterType::INVERT: return "Invert all colors in the video";
        case FilterType::BRIGHTNESS: return "Adjust video brightness";
        case FilterType::CONTRAST: return "Adjust video contrast";
        case FilterType::SATURATION: return "Adjust color saturation";
        case FilterType::HUE: return "Shift the colors in the video";
        case FilterType::BLUR: return "Apply a gaussian blur effect";
        case FilterType::SHARPEN: return "Enhance video details";
        case FilterType::VIGNETTE: return "Apply a dark border around the video";
        case FilterType::CUSTOM: return "Create a custom GLSL shader filter";
        default: return "";
    }
}

// FilterChain implementation
FilterChain::FilterChain() {}

void FilterChain::addFilter(std::shared_ptr<VideoFilter> filter) {
    filters.push_back(filter);
}

void FilterChain::removeFilter(size_t index) {
    if (index < filters.size()) {
        filters.erase(filters.begin() + index);
    }
}

void FilterChain::moveFilterUp(size_t index) {
    if (index > 0 && index < filters.size()) {
        std::swap(filters[index], filters[index - 1]);
    }
}

void FilterChain::moveFilterDown(size_t index) {
    if (index < filters.size() - 1) {
        std::swap(filters[index], filters[index + 1]);
    }
}

std::shared_ptr<VideoFilter> FilterChain::getFilter(size_t index) {
    if (index < filters.size()) {
        return filters[index];
    }
    return nullptr;
}

size_t FilterChain::getFilterCount() const {
    return filters.size();
}

bool FilterChain::applyFilters(GstBuffer *inFrame, GstBuffer *outFrame) {
    if (filters.empty()) {
        gst_buffer_copy_into(outFrame, inFrame, GST_BUFFER_COPY_ALL, 0, -1);
        return true;
    }
    
    // Simple implementation for a single filter
    if (filters.size() == 1 && filters[0]->isEnabled()) {
        return filters[0]->applyFilter(inFrame, outFrame);
    }
    
    // For multiple filters, we'd need to implement filter chaining with temp buffers
    // Simplified for brevity
    gst_buffer_copy_into(outFrame, inFrame, GST_BUFFER_COPY_ALL, 0, -1);
    return true;
}

GstElement* FilterChain::createFilterBin() {
    if (filters.empty()) return nullptr;
    
    GstElement *bin = gst_bin_new("filter-bin");
    GstElement *element = nullptr;
    
    // For simplicity, just add the first enabled filter
    for (auto &filter : filters) {
        if (filter->isEnabled()) {
            element = filter->createFilterElement();
            if (element) {
                gst_bin_add(GST_BIN(bin), element);
                
                // Add ghost pads
                GstPad *sinkpad = gst_element_get_static_pad(element, "sink");
                GstPad *srcpad = gst_element_get_static_pad(element, "src");
                
                gst_element_add_pad(bin, gst_ghost_pad_new("sink", sinkpad));
                gst_element_add_pad(bin, gst_ghost_pad_new("src", srcpad));
                
                gst_object_unref(sinkpad);
                gst_object_unref(srcpad);
                
                return bin;
            }
        }
    }
    
    gst_object_unref(bin);
    return nullptr;
}

GtkWidget* FilterChain::createFilterChainWidget(std::function<void()> updateCallback) {
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    
    // Add filters list label
    GtkWidget *label = gtk_label_new("Filters");
    gtk_box_append(GTK_BOX(vbox), label);
    
    // Add button to add a new filter
    GtkWidget *addBtn = gtk_button_new_with_label("Add Filter");
    g_signal_connect(addBtn, "clicked", G_CALLBACK(+[](GtkButton*, gpointer data) {
        auto chain = static_cast<FilterChain*>(data);
        chain->addFilter(std::make_shared<VideoFilter>(FilterType::GRAYSCALE));
        // In real implementation, would update UI here
    }), this);
    
    gtk_box_append(GTK_BOX(vbox), addBtn);
    
    return vbox;
}

bool FilterChain::saveToJson(const std::string &path) {
    // Simplified placeholder implementation
    return true;
}

bool FilterChain::loadFromJson(const std::string &path) {
    // Simplified placeholder implementation
    return true;
}

void FilterChain::clear() {
    filters.clear();
}

} // namespace BLOUedit
