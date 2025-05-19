#include "smart_cutout.h"
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
    PyObject* pEngine;
    std::thread processing_thread;
    std::mutex mutex;
    bool cancel_requested;
};

SmartCutout::SmartCutout() : python_state(nullptr), is_initialized(false), is_processing_active(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

SmartCutout::~SmartCutout() {
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
        Py_XDECREF(state->pEngine);
        delete state;
    }
}

bool SmartCutout::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool SmartCutout::initialize_python_environment() {
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
    PyObject* pName = PyUnicode_DecodeFSDefault("smart_cutout_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python smart cutout module" << std::endl;
        return false;
    }
    
    // Get the smart cutout engine class
    PyObject* pClass = PyObject_GetAttrString(pModule, "SmartCutoutEngine");
    if (pClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get smart cutout engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the smart cutout engine
    PyObject* pEngine = PyObject_CallObject(pClass, nullptr);
    Py_DECREF(pClass);
    
    if (pEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create smart cutout engine instance" << std::endl;
        return false;
    }
    
    // Store state
    PythonState* state = new PythonState();
    state->pModule = pModule;
    state->pEngine = pEngine;
    state->cancel_requested = false;
    python_state = state;
    
    return true;
}

bool SmartCutout::apply_cutout(
    const CutoutParameters& params,
    std::function<void(float)> progress_callback,
    std::function<void(bool, const std::string&)> completion_callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (completion_callback) completion_callback(false, "Python environment not initialized");
        return false;
    }
    
    if (is_processing_active) {
        std::cerr << "Cutout operation already in progress" << std::endl;
        if (completion_callback) completion_callback(false, "Cutout operation already in progress");
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
            PyDict_SetItemString(pDict, "mask_path", PyUnicode_FromString(params.mask_path.c_str()));
            
            // Set cutout method
            const char* method_str;
            switch (params.method) {
                case CutoutMethod::AUTOMATIC: method_str = "automatic"; break;
                case CutoutMethod::SALIENT: method_str = "salient"; break;
                case CutoutMethod::PORTRAIT: method_str = "portrait"; break;
                case CutoutMethod::INTERACTIVE: method_str = "interactive"; break;
                default: method_str = "automatic";
            }
            PyDict_SetItemString(pDict, "method", PyUnicode_FromString(method_str));
            
            // Set mask type
            const char* mask_type_str;
            switch (params.mask_type) {
                case MaskType::BINARY: mask_type_str = "binary"; break;
                case MaskType::ALPHA: mask_type_str = "alpha"; break;
                case MaskType::TRIMAP: mask_type_str = "trimap"; break;
                default: mask_type_str = "alpha";
            }
            PyDict_SetItemString(pDict, "mask_type", PyUnicode_FromString(mask_type_str));
            
            // Set other parameters
            PyDict_SetItemString(pDict, "threshold", PyFloat_FromDouble(params.threshold));
            PyDict_SetItemString(pDict, "invert_mask", PyBool_FromLong(params.invert_mask ? 1 : 0));
            PyDict_SetItemString(pDict, "apply_feathering", PyBool_FromLong(params.apply_feathering ? 1 : 0));
            PyDict_SetItemString(pDict, "feather_amount", PyFloat_FromDouble(params.feather_amount));
            PyDict_SetItemString(pDict, "add_shadow", PyBool_FromLong(params.add_shadow ? 1 : 0));
            PyDict_SetItemString(pDict, "background_path", PyUnicode_FromString(params.background_path.c_str()));
            PyDict_SetItemString(pDict, "process_audio", PyBool_FromLong(params.process_audio ? 1 : 0));
            
            // Set segments list
            PyObject* pSegments = PyList_New(params.segments.size());
            for (size_t i = 0; i < params.segments.size(); i++) {
                PyObject* pSegment = PyTuple_New(2);
                PyTuple_SetItem(pSegment, 0, PyLong_FromLong(params.segments[i].first));
                PyTuple_SetItem(pSegment, 1, PyLong_FromLong(params.segments[i].second));
                PyList_SetItem(pSegments, i, pSegment);
            }
            PyDict_SetItemString(pDict, "segments", pSegments);
            
            // Set interactive cutout parameters
            PyDict_SetItemString(pDict, "initial_mask_path", PyUnicode_FromString(params.initial_mask_path.c_str()));
            
            PyObject* pForegroundPoints = PyList_New(params.foreground_points.size());
            for (size_t i = 0; i < params.foreground_points.size(); i++) {
                PyObject* pPoint = PyTuple_New(2);
                PyTuple_SetItem(pPoint, 0, PyLong_FromLong(params.foreground_points[i].first));
                PyTuple_SetItem(pPoint, 1, PyLong_FromLong(params.foreground_points[i].second));
                PyList_SetItem(pForegroundPoints, i, pPoint);
            }
            PyDict_SetItemString(pDict, "foreground_points", pForegroundPoints);
            
            PyObject* pBackgroundPoints = PyList_New(params.background_points.size());
            for (size_t i = 0; i < params.background_points.size(); i++) {
                PyObject* pPoint = PyTuple_New(2);
                PyTuple_SetItem(pPoint, 0, PyLong_FromLong(params.background_points[i].first));
                PyTuple_SetItem(pPoint, 1, PyLong_FromLong(params.background_points[i].second));
                PyList_SetItem(pBackgroundPoints, i, pPoint);
            }
            PyDict_SetItemString(pDict, "background_points", pBackgroundPoints);
            
            // Define progress callback
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
            
            // Call apply_cutout method
            PyObject* pResult = PyObject_CallMethod(state->pEngine, "apply_cutout", "O", pDict);
            
            Py_XDECREF(pProgressFunc);
            Py_DECREF(pDict);
            
            if (pResult == nullptr) {
                PyErr_Print();
                result_message = "Failed to apply smart cutout";
                success = false;
            } else {
                // Parse result
                if (PyTuple_Check(pResult) && PyTuple_Size(pResult) == 2) {
                    success = PyObject_IsTrue(PyTuple_GetItem(pResult, 0));
                    PyObject* pMessage = PyTuple_GetItem(pResult, 1);
                    if (PyUnicode_Check(pMessage)) {
                        result_message = PyUnicode_AsUTF8(pMessage);
                    }
                } else {
                    success = PyObject_IsTrue(pResult);
                    result_message = success ? params.output_path : "Smart cutout failed";
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

void SmartCutout::cancel() {
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
    PyObject* pResult = PyObject_CallMethod(state->pEngine, "cancel", "");
    Py_XDECREF(pResult);
    PyGILState_Release(gstate);
}

bool SmartCutout::is_processing() const {
    return is_processing_active;
}

GtkWidget* SmartCutout::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 스마트 컷아웃");
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
    
    // Mask output file selection
    GtkWidget* mask_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* mask_label = gtk_label_new("마스크 파일:");
    GtkWidget* mask_entry = gtk_entry_new();
    GtkWidget* mask_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(mask_entry, TRUE);
    gtk_box_append(GTK_BOX(mask_box), mask_label);
    gtk_box_append(GTK_BOX(mask_box), mask_entry);
    gtk_box_append(GTK_BOX(mask_box), mask_button);
    gtk_box_append(GTK_BOX(container), mask_box);
    
    // Method selection
    GtkWidget* method_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* method_label = gtk_label_new("컷아웃 방식:");
    GtkWidget* method_combo = gtk_combo_box_text_new();
    
    // Add cutout methods
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(method_combo), NULL, "자동 (Automatic)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(method_combo), NULL, "현저한 객체 (Salient)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(method_combo), NULL, "인물 모드 (Portrait)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(method_combo), NULL, "대화형 (Interactive)");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(method_combo), 0);
    gtk_widget_set_hexpand(method_combo, TRUE);
    gtk_box_append(GTK_BOX(method_box), method_label);
    gtk_box_append(GTK_BOX(method_box), method_combo);
    gtk_box_append(GTK_BOX(container), method_box);
    
    // Mask type selection
    GtkWidget* mask_type_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* mask_type_label = gtk_label_new("마스크 유형:");
    GtkWidget* mask_type_combo = gtk_combo_box_text_new();
    
    // Add mask types
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mask_type_combo), NULL, "알파 (Alpha)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mask_type_combo), NULL, "이진 (Binary)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mask_type_combo), NULL, "트라이맵 (Trimap)");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(mask_type_combo), 0);
    gtk_widget_set_hexpand(mask_type_combo, TRUE);
    gtk_box_append(GTK_BOX(mask_type_box), mask_type_label);
    gtk_box_append(GTK_BOX(mask_type_box), mask_type_combo);
    gtk_box_append(GTK_BOX(container), mask_type_box);
    
    // Threshold slider
    GtkWidget* threshold_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* threshold_label = gtk_label_new("임계값:");
    GtkWidget* threshold_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.1, 0.9, 0.05);
    gtk_range_set_value(GTK_RANGE(threshold_scale), 0.5);
    gtk_widget_set_hexpand(threshold_scale, TRUE);
    gtk_box_append(GTK_BOX(threshold_box), threshold_label);
    gtk_box_append(GTK_BOX(threshold_box), threshold_scale);
    gtk_box_append(GTK_BOX(container), threshold_box);
    
    // Background selection
    GtkWidget* bg_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* bg_label = gtk_label_new("배경 이미지:");
    GtkWidget* bg_entry = gtk_entry_new();
    GtkWidget* bg_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(bg_entry, TRUE);
    gtk_box_append(GTK_BOX(bg_box), bg_label);
    gtk_box_append(GTK_BOX(bg_box), bg_entry);
    gtk_box_append(GTK_BOX(bg_box), bg_button);
    gtk_box_append(GTK_BOX(container), bg_box);
    
    // Options frame
    GtkWidget* options_frame = gtk_frame_new("옵션");
    GtkWidget* options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(options_box, 5);
    gtk_widget_set_margin_end(options_box, 5);
    gtk_widget_set_margin_top(options_box, 5);
    gtk_widget_set_margin_bottom(options_box, 5);
    
    // Invert mask checkbox
    GtkWidget* invert_check = gtk_check_button_new_with_label("마스크 반전");
    gtk_box_append(GTK_BOX(options_box), invert_check);
    
    // Apply feathering checkbox
    GtkWidget* feather_check = gtk_check_button_new_with_label("가장자리 페더링");
    gtk_box_append(GTK_BOX(options_box), feather_check);
    
    // Feather amount slider
    GtkWidget* feather_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* feather_label = gtk_label_new("페더링 정도:");
    GtkWidget* feather_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.5, 10.0, 0.5);
    gtk_range_set_value(GTK_RANGE(feather_scale), 2.0);
    gtk_widget_set_hexpand(feather_scale, TRUE);
    gtk_box_append(GTK_BOX(feather_box), feather_label);
    gtk_box_append(GTK_BOX(feather_box), feather_scale);
    gtk_box_append(GTK_BOX(options_box), feather_box);
    
    // Add shadow checkbox
    GtkWidget* shadow_check = gtk_check_button_new_with_label("그림자 추가");
    gtk_box_append(GTK_BOX(options_box), shadow_check);
    
    // Process audio checkbox (for videos)
    GtkWidget* audio_check = gtk_check_button_new_with_label("오디오 처리 (비디오)");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(audio_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), audio_check);
    
    gtk_frame_set_child(GTK_FRAME(options_frame), options_box);
    gtk_box_append(GTK_BOX(container), options_frame);
    
    // Interactive cutout options
    GtkWidget* interactive_frame = gtk_frame_new("대화형 컷아웃 도구");
    GtkWidget* interactive_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(interactive_box, 5);
    gtk_widget_set_margin_end(interactive_box, 5);
    gtk_widget_set_margin_top(interactive_box, 5);
    gtk_widget_set_margin_bottom(interactive_box, 5);
    
    // Load initial mask
    GtkWidget* init_mask_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* init_mask_label = gtk_label_new("초기 마스크:");
    GtkWidget* init_mask_entry = gtk_entry_new();
    GtkWidget* init_mask_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(init_mask_entry, TRUE);
    gtk_box_append(GTK_BOX(init_mask_box), init_mask_label);
    gtk_box_append(GTK_BOX(init_mask_box), init_mask_entry);
    gtk_box_append(GTK_BOX(init_mask_box), init_mask_button);
    gtk_box_append(GTK_BOX(interactive_box), init_mask_box);
    
    // Add interactive mode buttons
    GtkWidget* interactive_buttons = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* edit_button = gtk_button_new_with_label("마스크 편집");
    GtkWidget* preview_button = gtk_button_new_with_label("미리보기");
    gtk_widget_set_hexpand(edit_button, TRUE);
    gtk_widget_set_hexpand(preview_button, TRUE);
    gtk_box_append(GTK_BOX(interactive_buttons), edit_button);
    gtk_box_append(GTK_BOX(interactive_buttons), preview_button);
    gtk_box_append(GTK_BOX(interactive_box), interactive_buttons);
    
    gtk_frame_set_child(GTK_FRAME(interactive_frame), interactive_box);
    gtk_box_append(GTK_BOX(container), interactive_frame);
    
    // Process button
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* process_button = gtk_button_new_with_label("컷아웃 시작");
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