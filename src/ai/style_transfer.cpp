#include "style_transfer.h"
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <mutex>

// Python binding headers
#include <Python.h>
#include <numpy/arrayobject.h>

namespace BlouEdit {
namespace AI {

// Structure to hold Python state
struct PythonState {
    PyObject* pModule;
    PyObject* pStyleEngine;
    std::thread processing_thread;
    std::mutex mutex;
    bool cancel_requested;
};

StyleTransfer::StyleTransfer() : python_state(nullptr), is_initialized(false), is_processing_active(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

StyleTransfer::~StyleTransfer() {
    if (python_state) {
        PythonState* state = static_cast<PythonState*>(python_state);
        
        // Ensure any ongoing processing is canceled
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->cancel_requested = true;
        }
        
        // Wait for processing thread to finish if active
        if (state->processing_thread.joinable()) {
            state->processing_thread.join();
        }
        
        // Clean up Python objects
        Py_XDECREF(state->pModule);
        Py_XDECREF(state->pStyleEngine);
        delete state;
        
        // Don't finalize Python here as it might be used by other modules
        // Py_Finalize();
    }
}

bool StyleTransfer::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool StyleTransfer::initialize_python_environment() {
    // Initialize Python interpreter
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "Failed to initialize Python interpreter" << std::endl;
            return false;
        }
    }
    
    // Import NumPy functionality
    import_array();
    
    // Add our custom module directory to Python path
    PyRun_SimpleString("import sys\n"
                      "sys.path.append('src/ai/python')\n");
    
    // Import our module
    PyObject* pName = PyUnicode_DecodeFSDefault("style_transfer_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python style transfer module" << std::endl;
        return false;
    }
    
    // Get the style engine class
    PyObject* pStyleClass = PyObject_GetAttrString(pModule, "StyleTransferEngine");
    if (pStyleClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get style transfer engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the style engine
    PyObject* pStyleEngine = PyObject_CallObject(pStyleClass, nullptr);
    Py_DECREF(pStyleClass);
    
    if (pStyleEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create style transfer engine instance" << std::endl;
        return false;
    }
    
    // Store state
    PythonState* state = new PythonState();
    state->pModule = pModule;
    state->pStyleEngine = pStyleEngine;
    state->cancel_requested = false;
    python_state = state;
    
    return true;
}

std::vector<StylePreset> StyleTransfer::get_available_styles(const StyleCategory& category) {
    std::vector<StylePreset> styles;
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        return styles;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Convert category enum to string
    const char* category_str;
    switch (category) {
        case StyleCategory::PAINTING:
            category_str = "painting";
            break;
        case StyleCategory::PHOTO:
            category_str = "photo";
            break;
        case StyleCategory::ABSTRACT:
            category_str = "abstract";
            break;
        case StyleCategory::CARTOON:
            category_str = "cartoon";
            break;
        case StyleCategory::CINEMATIC:
            category_str = "cinematic";
            break;
        case StyleCategory::CUSTOM:
            category_str = "custom";
            break;
        default:
            category_str = "painting";
    }
    
    // Call get_available_styles method
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(category_str));
    
    PyObject* pResult = PyObject_CallMethod(state->pStyleEngine, "get_available_styles", "O", pArgs);
    Py_DECREF(pArgs);
    
    if (pResult == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to get available styles" << std::endl;
        return styles;
    }
    
    // Parse the result (list of style dictionaries)
    if (PyList_Check(pResult)) {
        Py_ssize_t size = PyList_Size(pResult);
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* pStyle = PyList_GetItem(pResult, i);
            if (PyDict_Check(pStyle)) {
                StylePreset style;
                
                // Get id
                PyObject* pId = PyDict_GetItemString(pStyle, "id");
                if (pId && PyUnicode_Check(pId)) {
                    style.id = PyUnicode_AsUTF8(pId);
                }
                
                // Get name
                PyObject* pName = PyDict_GetItemString(pStyle, "name");
                if (pName && PyUnicode_Check(pName)) {
                    style.name = PyUnicode_AsUTF8(pName);
                }
                
                // Get category
                PyObject* pCategory = PyDict_GetItemString(pStyle, "category");
                if (pCategory && PyUnicode_Check(pCategory)) {
                    std::string cat_str = PyUnicode_AsUTF8(pCategory);
                    if (cat_str == "painting") {
                        style.category = StyleCategory::PAINTING;
                    } else if (cat_str == "photo") {
                        style.category = StyleCategory::PHOTO;
                    } else if (cat_str == "abstract") {
                        style.category = StyleCategory::ABSTRACT;
                    } else if (cat_str == "cartoon") {
                        style.category = StyleCategory::CARTOON;
                    } else if (cat_str == "cinematic") {
                        style.category = StyleCategory::CINEMATIC;
                    } else if (cat_str == "custom") {
                        style.category = StyleCategory::CUSTOM;
                    }
                }
                
                // Get preview path
                PyObject* pPreviewPath = PyDict_GetItemString(pStyle, "preview_path");
                if (pPreviewPath && PyUnicode_Check(pPreviewPath)) {
                    style.preview_path = PyUnicode_AsUTF8(pPreviewPath);
                }
                
                // Get strength default
                PyObject* pStrengthDefault = PyDict_GetItemString(pStyle, "strength_default");
                if (pStrengthDefault && PyFloat_Check(pStrengthDefault)) {
                    style.strength_default = (float)PyFloat_AsDouble(pStrengthDefault);
                }
                
                styles.push_back(style);
            }
        }
    }
    
    Py_DECREF(pResult);
    
    return styles;
}

bool StyleTransfer::apply_style(
    const StyleTransferParameters& params,
    std::function<void(float)> progress_callback,
    std::function<void(bool, const std::string&)> completion_callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (completion_callback) completion_callback(false, "Python environment not initialized");
        return false;
    }
    
    if (is_processing_active) {
        std::cerr << "Style transfer is already in progress" << std::endl;
        if (completion_callback) completion_callback(false, "Style transfer is already in progress");
        return false;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->cancel_requested = false;
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::path output_dir = std::filesystem::path(params.output_path).parent_path();
    if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }
    
    // Start processing in a separate thread to avoid blocking the UI
    is_processing_active = true;
    
    state->processing_thread = std::thread([this, params, progress_callback, completion_callback, state]() {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        bool success = false;
        std::string result_message;
        
        try {
            // Create dictionary of parameters
            PyObject* pDict = PyDict_New();
            PyDict_SetItemString(pDict, "input_path", PyUnicode_FromString(params.input_path.c_str()));
            PyDict_SetItemString(pDict, "output_path", PyUnicode_FromString(params.output_path.c_str()));
            PyDict_SetItemString(pDict, "style_id", PyUnicode_FromString(params.style_id.c_str()));
            PyDict_SetItemString(pDict, "strength", PyFloat_FromDouble(params.strength));
            PyDict_SetItemString(pDict, "preserve_colors", PyBool_FromLong(params.preserve_colors ? 1 : 0));
            PyDict_SetItemString(pDict, "process_audio", PyBool_FromLong(params.process_audio ? 1 : 0));
            PyDict_SetItemString(pDict, "target_fps", PyLong_FromLong(params.target_fps));
            
            // Set segments list
            PyObject* pSegments = PyList_New(params.segments.size());
            for (size_t i = 0; i < params.segments.size(); i++) {
                PyObject* pSegment = PyTuple_New(2);
                PyTuple_SetItem(pSegment, 0, PyLong_FromLong(params.segments[i].first));
                PyTuple_SetItem(pSegment, 1, PyLong_FromLong(params.segments[i].second));
                PyList_SetItem(pSegments, i, pSegment);
            }
            PyDict_SetItemString(pDict, "segments", pSegments);
            
            // Define a progress callback function for Python to call
            PyObject* pProgressFunc = nullptr;
            if (progress_callback) {
                pProgressFunc = PyCFunction_New(&(PyMethodDef{"progress_callback", 
                    [](PyObject* self, PyObject* args) -> PyObject* {
                        float progress = 0.0f;
                        if (!PyArg_ParseTuple(args, "f", &progress)) {
                            return nullptr;
                        }
                        
                        // Get the C++ callback from the capsule
                        auto callback_ptr = static_cast<std::function<void(float)>*>(
                            PyCapsule_GetPointer(self, "progress_callback")
                        );
                        
                        // Call the C++ callback
                        (*callback_ptr)(progress);
                        
                        Py_RETURN_NONE;
                    }, 
                    METH_VARARGS, ""
                }}, nullptr, PyCapsule_New(new std::function<void(float)>(progress_callback), "progress_callback", 
                    [](PyObject* cap) {
                        delete static_cast<std::function<void(float)>*>(
                            PyCapsule_GetPointer(cap, "progress_callback")
                        );
                    }
                ));
                
                PyDict_SetItemString(pDict, "progress_callback", pProgressFunc);
            }
            
            // Call apply_style method
            PyObject* pResult = PyObject_CallMethod(state->pStyleEngine, "apply_style", "O", pDict);
            
            Py_XDECREF(pProgressFunc);
            Py_DECREF(pDict);
            
            if (pResult == nullptr) {
                PyErr_Print();
                result_message = "Failed to apply style transfer";
                success = false;
            } else {
                // Check if result is a tuple (success, message) or just a boolean
                if (PyTuple_Check(pResult) && PyTuple_Size(pResult) == 2) {
                    success = PyObject_IsTrue(PyTuple_GetItem(pResult, 0));
                    PyObject* pMessage = PyTuple_GetItem(pResult, 1);
                    if (PyUnicode_Check(pMessage)) {
                        result_message = PyUnicode_AsUTF8(pMessage);
                    }
                } else {
                    success = PyObject_IsTrue(pResult);
                    result_message = success ? params.output_path : "Style transfer failed";
                }
                
                Py_DECREF(pResult);
            }
        } catch (const std::exception& e) {
            result_message = std::string("Exception: ") + e.what();
            success = false;
        }
        
        // Mark processing as complete
        is_processing_active = false;
        
        // Call completion callback
        if (completion_callback) {
            completion_callback(success, result_message);
        }
        
        PyGILState_Release(gstate);
    });
    
    return true;
}

void StyleTransfer::cancel() {
    if (!is_processing_active || !python_state) {
        return;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Set cancel flag
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->cancel_requested = true;
    }
    
    // Try to cancel via Python
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* pResult = PyObject_CallMethod(state->pStyleEngine, "cancel", "");
    Py_XDECREF(pResult);
    PyGILState_Release(gstate);
}

bool StyleTransfer::is_processing() const {
    return is_processing_active;
}

GtkWidget* StyleTransfer::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 스타일 트랜스퍼");
    PangoAttrList* attr_list = pango_attr_list_new();
    pango_attr_list_insert(attr_list, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attr_list, pango_attr_scale_new(1.2));
    gtk_label_set_attributes(GTK_LABEL(title), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_append(GTK_BOX(container), title);
    
    // Input file selection
    GtkWidget* input_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* input_label = gtk_label_new("입력 파일:");
    GtkWidget* input_entry = gtk_entry_new();
    GtkWidget* input_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(input_entry, TRUE);
    gtk_box_append(GTK_BOX(input_box), input_label);
    gtk_box_append(GTK_BOX(input_box), input_entry);
    gtk_box_append(GTK_BOX(input_box), input_button);
    gtk_box_append(GTK_BOX(container), input_box);
    
    // Output file selection
    GtkWidget* output_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* output_label = gtk_label_new("출력 파일:");
    GtkWidget* output_entry = gtk_entry_new();
    GtkWidget* output_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(output_entry, TRUE);
    gtk_box_append(GTK_BOX(output_box), output_label);
    gtk_box_append(GTK_BOX(output_box), output_entry);
    gtk_box_append(GTK_BOX(output_box), output_button);
    gtk_box_append(GTK_BOX(container), output_box);
    
    // Style category selection
    GtkWidget* category_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* category_label = gtk_label_new("스타일 카테고리:");
    GtkWidget* category_combo = gtk_combo_box_text_new();
    
    // Add style categories
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(category_combo), NULL, "페인팅");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(category_combo), NULL, "사진");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(category_combo), NULL, "추상화");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(category_combo), NULL, "만화");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(category_combo), NULL, "영화");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(category_combo), NULL, "사용자 정의");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(category_combo), 0);
    gtk_widget_set_hexpand(category_combo, TRUE);
    gtk_box_append(GTK_BOX(category_box), category_label);
    gtk_box_append(GTK_BOX(category_box), category_combo);
    gtk_box_append(GTK_BOX(container), category_box);
    
    // Style selection grid
    GtkWidget* style_frame = gtk_frame_new("스타일 선택");
    GtkWidget* style_scroll = gtk_scrolled_window_new();
    GtkWidget* style_grid = gtk_grid_new();
    
    gtk_grid_set_row_spacing(GTK_GRID(style_grid), 10);
    gtk_grid_set_column_spacing(GTK_GRID(style_grid), 10);
    gtk_grid_set_row_homogeneous(GTK_GRID(style_grid), TRUE);
    gtk_grid_set_column_homogeneous(GTK_GRID(style_grid), TRUE);
    
    // Add some placeholder style tiles
    for (int i = 0; i < 12; i++) {
        GtkWidget* style_tile = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
        GtkWidget* style_image = gtk_image_new_from_icon_name("image-missing");
        GtkWidget* style_label = gtk_label_new(("스타일 " + std::to_string(i+1)).c_str());
        
        gtk_image_set_pixel_size(GTK_IMAGE(style_image), 100);
        gtk_widget_set_size_request(style_tile, 120, 120);
        
        gtk_box_append(GTK_BOX(style_tile), style_image);
        gtk_box_append(GTK_BOX(style_tile), style_label);
        
        gtk_grid_attach(GTK_GRID(style_grid), style_tile, i % 4, i / 4, 1, 1);
    }
    
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(style_scroll), style_grid);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(style_scroll), 
                                  GTK_POLICY_NEVER, 
                                  GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(style_scroll, -1, 250);
    gtk_frame_set_child(GTK_FRAME(style_frame), style_scroll);
    gtk_box_append(GTK_BOX(container), style_frame);
    
    // Style strength slider
    GtkWidget* strength_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* strength_label = gtk_label_new("스타일 강도:");
    GtkWidget* strength_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 1.0, 0.05);
    gtk_range_set_value(GTK_RANGE(strength_scale), 0.75);
    gtk_widget_set_hexpand(strength_scale, TRUE);
    gtk_box_append(GTK_BOX(strength_box), strength_label);
    gtk_box_append(GTK_BOX(strength_box), strength_scale);
    gtk_box_append(GTK_BOX(container), strength_box);
    
    // Options frame
    GtkWidget* options_frame = gtk_frame_new("옵션");
    GtkWidget* options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    
    // Preserve colors checkbox
    GtkWidget* preserve_colors_check = gtk_check_button_new_with_label("원본 색상 유지");
    gtk_box_append(GTK_BOX(options_box), preserve_colors_check);
    
    // Process audio checkbox
    GtkWidget* process_audio_check = gtk_check_button_new_with_label("오디오 처리");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(process_audio_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), process_audio_check);
    
    // Target FPS box
    GtkWidget* fps_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* fps_label = gtk_label_new("출력 FPS:");
    GtkWidget* fps_spin = gtk_spin_button_new_with_range(0, 60, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(fps_spin), 0);
    gtk_widget_set_tooltip_text(fps_spin, "0 = 원본과 동일");
    gtk_widget_set_hexpand(fps_spin, TRUE);
    gtk_box_append(GTK_BOX(fps_box), fps_label);
    gtk_box_append(GTK_BOX(fps_box), fps_spin);
    gtk_box_append(GTK_BOX(options_box), fps_box);
    
    gtk_frame_set_child(GTK_FRAME(options_frame), options_box);
    gtk_box_append(GTK_BOX(container), options_frame);
    
    // Process button
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* process_button = gtk_button_new_with_label("스타일 적용 시작");
    GtkWidget* cancel_button = gtk_button_new_with_label("취소");
    gtk_widget_set_hexpand(process_button, TRUE);
    gtk_widget_set_sensitive(cancel_button, FALSE);
    gtk_box_append(GTK_BOX(button_box), process_button);
    gtk_box_append(GTK_BOX(button_box), cancel_button);
    gtk_box_append(GTK_BOX(container), button_box);
    
    // Progress bar
    GtkWidget* progress_bar = gtk_progress_bar_new();
    gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(progress_bar), TRUE);
    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), "대기 중...");
    gtk_box_append(GTK_BOX(container), progress_bar);
    
    // Status label
    GtkWidget* status_label = gtk_label_new("");
    gtk_box_append(GTK_BOX(container), status_label);
    
    // TODO: Connect signals to callbacks
    
    return container;
}

} // namespace AI
} // namespace BlouEdit 