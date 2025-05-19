#include "transitions.h"
#include <iostream>
#include <gst/video/video.h>
#include <cmath>

namespace BLOUedit {

Transition::Transition()
    : type(TransitionType::DISSOLVE),
      direction(TransitionDirection::LEFT_TO_RIGHT),
      durationFrames(30),
      progress(0.0),
      softness(0.5),
      rotation(0.0) {
}

Transition::Transition(TransitionType type)
    : type(type),
      direction(TransitionDirection::LEFT_TO_RIGHT),
      durationFrames(30),
      progress(0.0),
      softness(0.5),
      rotation(0.0) {
}

void Transition::setType(TransitionType newType) {
    type = newType;
}

TransitionType Transition::getType() const {
    return type;
}

void Transition::setDirection(TransitionDirection newDirection) {
    direction = newDirection;
}

TransitionDirection Transition::getDirection() const {
    return direction;
}

void Transition::setDuration(int frames) {
    durationFrames = frames;
    if (durationFrames < 1) durationFrames = 1;
}

int Transition::getDuration() const {
    return durationFrames;
}

void Transition::setProgress(double value) {
    progress = value;
    if (progress < 0.0) progress = 0.0;
    if (progress > 1.0) progress = 1.0;
}

double Transition::getProgress() const {
    return progress;
}

void Transition::setSoftness(double value) {
    softness = value;
    if (softness < 0.0) softness = 0.0;
    if (softness > 1.0) softness = 1.0;
}

double Transition::getSoftness() const {
    return softness;
}

void Transition::setRotation(double value) {
    rotation = value;
    // Normalize to 0-360 range
    while (rotation < 0.0) rotation += 360.0;
    while (rotation >= 360.0) rotation -= 360.0;
}

double Transition::getRotation() const {
    return rotation;
}

void Transition::setCustomPattern(const std::string &pattern) {
    customPattern = pattern;
}

std::string Transition::getCustomPattern() const {
    return customPattern;
}

std::string Transition::getTypeName(TransitionType type) {
    switch (type) {
        case TransitionType::DISSOLVE:
            return "Dissolve";
        case TransitionType::WIPE:
            return "Wipe";
        case TransitionType::SLIDE:
            return "Slide";
        case TransitionType::ZOOM:
            return "Zoom";
        case TransitionType::PUSH:
            return "Push";
        case TransitionType::SPIN:
            return "Spin";
        case TransitionType::CUSTOM:
            return "Custom";
        default:
            return "Unknown";
    }
}

std::vector<TransitionType> Transition::getAvailableTypes() {
    return {
        TransitionType::DISSOLVE,
        TransitionType::WIPE,
        TransitionType::SLIDE,
        TransitionType::ZOOM,
        TransitionType::PUSH,
        TransitionType::SPIN,
        TransitionType::CUSTOM
    };
}

std::vector<TransitionDirection> Transition::getAvailableDirections() {
    return {
        TransitionDirection::LEFT_TO_RIGHT,
        TransitionDirection::RIGHT_TO_LEFT,
        TransitionDirection::TOP_TO_BOTTOM,
        TransitionDirection::BOTTOM_TO_TOP,
        TransitionDirection::DIAGONAL_TOP_LEFT,
        TransitionDirection::DIAGONAL_TOP_RIGHT
    };
}

bool Transition::applyTransition(GstBuffer *inFrame1, GstBuffer *inFrame2, GstBuffer *outFrame) {
    // This is a simplified implementation
    // A real implementation would apply the specific transition effect based on the type
    
    // For now, we'll just implement a basic dissolve (cross-fade) transition
    // In a real implementation, each transition type would be handled separately
    
    GstVideoFrame frame1, frame2, outF;
    GstVideoInfo info;
    
    // Map buffers to frames
    gst_video_info_init(&info);
    gst_video_info_set_format(&info, GST_VIDEO_FORMAT_RGBA, 1920, 1080); // Assuming fixed format for simplicity
    
    if (!gst_video_frame_map(&frame1, &info, inFrame1, GST_MAP_READ)) {
        std::cerr << "Failed to map input frame 1" << std::endl;
        return false;
    }
    
    if (!gst_video_frame_map(&frame2, &info, inFrame2, GST_MAP_READ)) {
        gst_video_frame_unmap(&frame1);
        std::cerr << "Failed to map input frame 2" << std::endl;
        return false;
    }
    
    if (!gst_video_frame_map(&outF, &info, outFrame, GST_MAP_WRITE)) {
        gst_video_frame_unmap(&frame1);
        gst_video_frame_unmap(&frame2);
        std::cerr << "Failed to map output frame" << std::endl;
        return false;
    }
    
    // Get frame data pointers
    uint8_t *src1 = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame1, 0);
    uint8_t *src2 = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame2, 0);
    uint8_t *dest = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&outF, 0);
    
    // Get frame stride (bytes per row)
    int stride1 = GST_VIDEO_FRAME_PLANE_STRIDE(&frame1, 0);
    int stride2 = GST_VIDEO_FRAME_PLANE_STRIDE(&frame2, 0);
    int strideOut = GST_VIDEO_FRAME_PLANE_STRIDE(&outF, 0);
    
    // Frame dimensions
    int width = GST_VIDEO_FRAME_WIDTH(&frame1);
    int height = GST_VIDEO_FRAME_HEIGHT(&frame1);
    
    // Apply transition based on type
    switch (type) {
        case TransitionType::DISSOLVE: {
            // Simple cross-fade/dissolve transition
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < 4; c++) { // RGBA
                        int offset = y * stride1 + x * 4 + c;
                        int offset2 = y * stride2 + x * 4 + c;
                        int offsetOut = y * strideOut + x * 4 + c;
                        
                        dest[offsetOut] = static_cast<uint8_t>(
                            src1[offset] * (1.0 - progress) + src2[offset2] * progress
                        );
                    }
                }
            }
            break;
        }
        
        case TransitionType::WIPE: {
            // Directional wipe transition
            int boundary = 0;
            
            switch (direction) {
                case TransitionDirection::LEFT_TO_RIGHT:
                    boundary = static_cast<int>(width * progress);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int offset1 = y * stride1 + x * 4;
                            int offset2 = y * stride2 + x * 4;
                            int offsetOut = y * strideOut + x * 4;
                            
                            if (x < boundary) {
                                // Copy from second frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src2[offset2 + c];
                                }
                            } else {
                                // Copy from first frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src1[offset1 + c];
                                }
                            }
                        }
                    }
                    break;
                    
                case TransitionDirection::RIGHT_TO_LEFT:
                    boundary = static_cast<int>(width * (1.0 - progress));
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int offset1 = y * stride1 + x * 4;
                            int offset2 = y * stride2 + x * 4;
                            int offsetOut = y * strideOut + x * 4;
                            
                            if (x > boundary) {
                                // Copy from second frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src2[offset2 + c];
                                }
                            } else {
                                // Copy from first frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src1[offset1 + c];
                                }
                            }
                        }
                    }
                    break;
                    
                case TransitionDirection::TOP_TO_BOTTOM:
                    boundary = static_cast<int>(height * progress);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int offset1 = y * stride1 + x * 4;
                            int offset2 = y * stride2 + x * 4;
                            int offsetOut = y * strideOut + x * 4;
                            
                            if (y < boundary) {
                                // Copy from second frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src2[offset2 + c];
                                }
                            } else {
                                // Copy from first frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src1[offset1 + c];
                                }
                            }
                        }
                    }
                    break;
                    
                case TransitionDirection::BOTTOM_TO_TOP:
                    boundary = static_cast<int>(height * (1.0 - progress));
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            int offset1 = y * stride1 + x * 4;
                            int offset2 = y * stride2 + x * 4;
                            int offsetOut = y * strideOut + x * 4;
                            
                            if (y > boundary) {
                                // Copy from second frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src2[offset2 + c];
                                }
                            } else {
                                // Copy from first frame
                                for (int c = 0; c < 4; c++) {
                                    dest[offsetOut + c] = src1[offset1 + c];
                                }
                            }
                        }
                    }
                    break;
                    
                default:
                    // For other directions, fall back to dissolve
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            for (int c = 0; c < 4; c++) {
                                int offset = y * stride1 + x * 4 + c;
                                int offset2 = y * stride2 + x * 4 + c;
                                int offsetOut = y * strideOut + x * 4 + c;
                                
                                dest[offsetOut] = static_cast<uint8_t>(
                                    src1[offset] * (1.0 - progress) + src2[offset2] * progress
                                );
                            }
                        }
                    }
                    break;
            }
            break;
        }
        
        // For the other transition types, we would implement specific algorithms
        // For now, we'll fall back to dissolve for all of them
        default:
            // Default to dissolve for all other transition types
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < 4; c++) {
                        int offset = y * stride1 + x * 4 + c;
                        int offset2 = y * stride2 + x * 4 + c;
                        int offsetOut = y * strideOut + x * 4 + c;
                        
                        dest[offsetOut] = static_cast<uint8_t>(
                            src1[offset] * (1.0 - progress) + src2[offset2] * progress
                        );
                    }
                }
            }
            break;
    }
    
    // Unmap frames
    gst_video_frame_unmap(&frame1);
    gst_video_frame_unmap(&frame2);
    gst_video_frame_unmap(&outF);
    
    return true;
}

// TransitionManager implementation

TransitionManager::TransitionManager() {
}

bool TransitionManager::addTransition(const std::string &fromClipId, const std::string &toClipId, const Transition &transition) {
    std::string key = fromClipId + ":" + toClipId;
    transitions[key] = transition;
    return true;
}

bool TransitionManager::removeTransition(const std::string &fromClipId, const std::string &toClipId) {
    std::string key = fromClipId + ":" + toClipId;
    auto it = transitions.find(key);
    
    if (it != transitions.end()) {
        transitions.erase(it);
        return true;
    }
    
    return false;
}

Transition* TransitionManager::getTransition(const std::string &fromClipId, const std::string &toClipId) {
    std::string key = fromClipId + ":" + toClipId;
    auto it = transitions.find(key);
    
    if (it != transitions.end()) {
        return &(it->second);
    }
    
    return nullptr;
}

bool TransitionManager::saveToProject(const std::string &projectPath) {
    // In a real implementation, this would serialize the transitions map to the project file
    // For now, we'll just return success
    return true;
}

bool TransitionManager::loadFromProject(const std::string &projectPath) {
    // In a real implementation, this would deserialize transitions from the project file
    // For now, we'll just return success
    return true;
}

} // namespace BLOUedit 