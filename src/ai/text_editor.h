#pragma once

#include <gtk/gtk.h>
#include <string>
#include <vector>
#include <functional>

namespace BlouEdit {
namespace AI {

// Callback type for when text editing is complete
typedef std::function<void(bool success, const std::string& result)> TextEditCallback;

enum TextEditOperation {
    TEXT_EDIT_TRIM,             // Trim video based on text
    TEXT_EDIT_HIGHLIGHT,        // Highlight segments based on text
    TEXT_EDIT_SUMMARIZE,        // Summarize video content
    TEXT_EDIT_EXTRACT_CLIPS,    // Extract clips based on keywords
    TEXT_EDIT_GENERATE_TITLE,   // Generate title from content
    TEXT_EDIT_GENERATE_SCRIPT   // Generate a script from video
};

class TextEditor {
public:
    TextEditor();
    ~TextEditor();

    // Initialize the AI model
    bool initialize();
    
    // Edit video based on text description
    void editFromText(const std::string& text_description, 
                      TextEditCallback callback);
    
    // Suggest edits based on video content
    void suggestEdits(const std::string& video_path,
                      TextEditCallback callback);
    
    // Perform specific operation based on text
    void performOperation(TextEditOperation operation,
                          const std::string& input_text,
                          const std::string& video_path,
                          TextEditCallback callback);
    
    // Generate text description from video
    void generateTextFromVideo(const std::string& video_path,
                              TextEditCallback callback);
                              
    // Intelligent scene description
    void describeScenes(const std::string& video_path,
                       TextEditCallback callback);
    
    // Check if the model is ready
    bool isModelReady() const;
    
    // Cancel ongoing operation
    void cancelOperation();

private:
    void* model_handle;         // Opaque handle to the AI model
    bool model_ready;           // Flag indicating if model is initialized
    bool operation_in_progress; // Flag indicating if an operation is running
    
    // Helper methods
    bool loadModel();
    void unloadModel();
    std::string processTextPrompt(const std::string& prompt);
    std::vector<std::string> extractKeywords(const std::string& text);
    std::string generateEditXML(const std::string& text_description);
};

} // namespace AI
} // namespace BlouEdit 