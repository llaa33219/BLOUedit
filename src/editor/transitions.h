#pragma once

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace BLOUedit {

enum class TransitionType {
    DISSOLVE,   // Fade transition
    WIPE,       // Directional wipe
    SLIDE,      // Sliding transition
    ZOOM,       // Zoom in/out transition
    PUSH,       // Push transition
    SPIN,       // Spin transition
    CUSTOM      // Custom transition
};

enum class TransitionDirection {
    LEFT_TO_RIGHT,
    RIGHT_TO_LEFT,
    TOP_TO_BOTTOM,
    BOTTOM_TO_TOP,
    DIAGONAL_TOP_LEFT,
    DIAGONAL_TOP_RIGHT
};

class Transition {
private:
    TransitionType type;
    TransitionDirection direction;
    int durationFrames;
    double progress; // 0.0 to 1.0
    
    // Transition parameters
    double softness;  // Edge softness (0.0-1.0)
    double rotation;  // For spin/rotate transition
    std::string customPattern; // For custom pattern transitions
    
public:
    Transition();
    explicit Transition(TransitionType type);
    
    void setType(TransitionType newType);
    TransitionType getType() const;
    
    void setDirection(TransitionDirection newDirection);
    TransitionDirection getDirection() const;
    
    void setDuration(int frames);
    int getDuration() const;
    
    void setProgress(double value);
    double getProgress() const;
    
    void setSoftness(double value);
    double getSoftness() const;
    
    void setRotation(double value);
    double getRotation() const;
    
    void setCustomPattern(const std::string &pattern);
    std::string getCustomPattern() const;
    
    // Returns translated type name
    static std::string getTypeName(TransitionType type);
    
    // Returns all available transition types
    static std::vector<TransitionType> getAvailableTypes();
    
    // Returns all available direction types
    static std::vector<TransitionDirection> getAvailableDirections();
    
    // Apply transition effect between two frames
    bool applyTransition(GstBuffer *inFrame1, GstBuffer *inFrame2, GstBuffer *outFrame);
};

// TransitionManager to handle transitions in the project
class TransitionManager {
private:
    std::map<std::string, Transition> transitions; // key is clipID pair ("clip1:clip2")
    
public:
    TransitionManager();
    
    bool addTransition(const std::string &fromClipId, const std::string &toClipId, const Transition &transition);
    bool removeTransition(const std::string &fromClipId, const std::string &toClipId);
    
    Transition* getTransition(const std::string &fromClipId, const std::string &toClipId);
    
    // Save/load transitions from project file
    bool saveToProject(const std::string &projectPath);
    bool loadFromProject(const std::string &projectPath);
};

} // namespace BLOUedit 