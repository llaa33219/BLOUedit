#include "image_generator.h"
#include <iostream>
#include <cstdlib>
#include <filesystem>

// Python binding headers
#include <Python.h>
#include <numpy/arrayobject.h>

namespace BlouEdit {
namespace AI {

// Structure to hold Python state
struct PythonState {
    PyObject* pModule;
    PyObject* pGenerator;
};

ImageGenerator::ImageGenerator() : python_state(nullptr), is_initialized(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

ImageGenerator::~ImageGenerator() {
    if (python_state) {
        PythonState* state = static_cast<PythonState*>(python_state);
        Py_XDECREF(state->pModule);
        Py_XDECREF(state->pGenerator);
        delete state;
        
        // Finalize Python interpreter
        Py_Finalize();
    }
}

bool ImageGenerator::initialize_python_environment() {
    // Initialize Python interpreter
    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cerr << "Failed to initialize Python interpreter" << std::endl;
        return false;
    }
    
    // Import NumPy functionality
    import_array();
    
    // Add our custom module directory to Python path
    PyRun_SimpleString("import sys\n"
                       "sys.path.append('src/ai/python')\n");
    
    // Import our module
    PyObject* pName = PyUnicode_DecodeFSDefault("image_generator_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python module" << std::endl;
        return false;
    }
    
    // Get the generator class
    PyObject* pGeneratorClass = PyObject_GetAttrString(pModule, "ImageGenerator");
    if (pGeneratorClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get generator class" << std::endl;
        return false;
    }
    
    // Create an instance of the generator
    PyObject* pGenerator = PyObject_CallObject(pGeneratorClass, nullptr);
    Py_DECREF(pGeneratorClass);
    
    if (pGenerator == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create generator instance" << std::endl;
        return false;
    }
    
    // Store state
    PythonState* state = new PythonState();
    state->pModule = pModule;
    state->pGenerator = pGenerator;
    python_state = state;
    
    return true;
}

bool ImageGenerator::load_model(ImageGenerationModel model) {
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        return false;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Convert enum to string
    const char* model_name;
    switch (model) {
        case ImageGenerationModel::STABLE_DIFFUSION:
            model_name = "stable-diffusion";
            break;
        case ImageGenerationModel::DALL_E:
            model_name = "dall-e";
            break;
        case ImageGenerationModel::MIDJOURNEY_COMPATIBLE:
            model_name = "midjourney";
            break;
        default:
            model_name = "stable-diffusion";
    }
    
    // Call load_model method
    PyObject* pResult = PyObject_CallMethod(state->pGenerator, "load_model", "s", model_name);
    if (pResult == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load model" << std::endl;
        return false;
    }
    
    bool success = PyObject_IsTrue(pResult);
    Py_DECREF(pResult);
    
    return success;
}

bool ImageGenerator::generate_image(const ImageGenerationParams& params, 
                                   const std::string& output_path,
                                   std::function<void(bool, const std::string&)> callback) {
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (callback) callback(false, "Python environment not initialized");
        return false;
    }
    
    // Ensure model is loaded
    if (!load_model(params.model)) {
        if (callback) callback(false, "Failed to load model");
        return false;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Prepare arguments
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(params.prompt.c_str()));
    PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(params.negative_prompt.c_str()));
    PyTuple_SetItem(pArgs, 2, PyLong_FromLong(params.width));
    PyTuple_SetItem(pArgs, 3, PyLong_FromLong(params.height));
    PyTuple_SetItem(pArgs, 4, PyLong_FromLong(params.num_inference_steps));
    PyTuple_SetItem(pArgs, 5, PyFloat_FromDouble(params.guidance_scale));
    
    // Call generate_image method
    PyObject* pResult = PyObject_CallMethod(state->pGenerator, "generate_image", "Os", 
                                            pArgs, output_path.c_str());
    Py_DECREF(pArgs);
    
    if (pResult == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to generate image" << std::endl;
        if (callback) callback(false, "Failed to generate image");
        return false;
    }
    
    bool success = PyObject_IsTrue(pResult);
    Py_DECREF(pResult);
    
    if (callback) {
        if (success) {
            callback(true, output_path);
        } else {
            callback(false, "Image generation failed");
        }
    }
    
    return success;
}

bool ImageGenerator::generate_thumbnail(const std::string& video_path,
                                        const std::string& prompt,
                                        const std::string& output_path,
                                        std::function<void(bool, const std::string&)> callback) {
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (callback) callback(false, "Python environment not initialized");
        return false;
    }
    
    // Ensure model is loaded
    if (!load_model(ImageGenerationModel::STABLE_DIFFUSION)) {
        if (callback) callback(false, "Failed to load model");
        return false;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Call generate_thumbnail method
    PyObject* pResult = PyObject_CallMethod(state->pGenerator, "generate_thumbnail", "sss",
                                            video_path.c_str(), prompt.c_str(), output_path.c_str());
    
    if (pResult == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to generate thumbnail" << std::endl;
        if (callback) callback(false, "Failed to generate thumbnail");
        return false;
    }
    
    bool success = PyObject_IsTrue(pResult);
    Py_DECREF(pResult);
    
    if (callback) {
        if (success) {
            callback(true, output_path);
        } else {
            callback(false, "Thumbnail generation failed");
        }
    }
    
    return success;
}

// Create a GTK widget for the image generator UI
GtkWidget* ImageGenerator::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 이미지 생성기");
    PangoAttrList* attr_list = pango_attr_list_new();
    pango_attr_list_insert(attr_list, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attr_list, pango_attr_scale_new(1.2));
    gtk_label_set_attributes(GTK_LABEL(title), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_append(GTK_BOX(container), title);
    
    // Prompt input
    GtkWidget* prompt_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* prompt_label = gtk_label_new("프롬프트:");
    GtkWidget* prompt_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(prompt_entry), "이미지를 설명하세요...");
    gtk_widget_set_hexpand(prompt_entry, TRUE);
    gtk_box_append(GTK_BOX(prompt_box), prompt_label);
    gtk_box_append(GTK_BOX(prompt_box), prompt_entry);
    gtk_box_append(GTK_BOX(container), prompt_box);
    
    // Negative prompt input
    GtkWidget* neg_prompt_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* neg_prompt_label = gtk_label_new("제외 프롬프트:");
    GtkWidget* neg_prompt_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(neg_prompt_entry), "원치 않는 요소를 설명하세요...");
    gtk_widget_set_hexpand(neg_prompt_entry, TRUE);
    gtk_box_append(GTK_BOX(neg_prompt_box), neg_prompt_label);
    gtk_box_append(GTK_BOX(neg_prompt_box), neg_prompt_entry);
    gtk_box_append(GTK_BOX(container), neg_prompt_box);
    
    // Size options
    GtkWidget* size_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* size_label = gtk_label_new("크기:");
    GtkWidget* size_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "512x512");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "768x768");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "1024x1024");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "512x768");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(size_combo), NULL, "768x512");
    gtk_combo_box_set_active(GTK_COMBO_BOX(size_combo), 0);
    gtk_box_append(GTK_BOX(size_box), size_label);
    gtk_box_append(GTK_BOX(size_box), size_combo);
    gtk_box_append(GTK_BOX(container), size_box);
    
    // Model selection
    GtkWidget* model_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* model_label = gtk_label_new("모델:");
    GtkWidget* model_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(model_combo), NULL, "Stable Diffusion");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(model_combo), NULL, "DALL-E");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(model_combo), NULL, "Midjourney 호환");
    gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo), 0);
    gtk_box_append(GTK_BOX(model_box), model_label);
    gtk_box_append(GTK_BOX(model_box), model_combo);
    gtk_box_append(GTK_BOX(container), model_box);
    
    // Steps slider
    GtkWidget* steps_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* steps_label = gtk_label_new("추론 단계:");
    GtkWidget* steps_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 20, 100, 1);
    gtk_range_set_value(GTK_RANGE(steps_scale), 50);
    gtk_widget_set_hexpand(steps_scale, TRUE);
    gtk_box_append(GTK_BOX(steps_box), steps_label);
    gtk_box_append(GTK_BOX(steps_box), steps_scale);
    gtk_box_append(GTK_BOX(container), steps_box);
    
    // Guidance scale slider
    GtkWidget* guidance_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* guidance_label = gtk_label_new("가이던스 강도:");
    GtkWidget* guidance_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 1.0, 20.0, 0.5);
    gtk_range_set_value(GTK_RANGE(guidance_scale), 7.5);
    gtk_widget_set_hexpand(guidance_scale, TRUE);
    gtk_box_append(GTK_BOX(guidance_box), guidance_label);
    gtk_box_append(GTK_BOX(guidance_box), guidance_scale);
    gtk_box_append(GTK_BOX(container), guidance_box);
    
    // Generate button
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* generate_button = gtk_button_new_with_label("이미지 생성");
    gtk_widget_set_hexpand(generate_button, TRUE);
    gtk_box_append(GTK_BOX(button_box), generate_button);
    gtk_box_append(GTK_BOX(container), button_box);
    
    // Preview area
    GtkWidget* preview_frame = gtk_frame_new("미리보기");
    GtkWidget* preview_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget* preview_image = gtk_image_new();
    gtk_widget_set_size_request(preview_image, 512, 512);
    gtk_box_append(GTK_BOX(preview_box), preview_image);
    gtk_frame_set_child(GTK_FRAME(preview_frame), preview_box);
    gtk_box_append(GTK_BOX(container), preview_frame);
    
    // TODO: Connect signals to callbacks
    
    return container;
}

} // namespace AI
} // namespace BlouEdit 