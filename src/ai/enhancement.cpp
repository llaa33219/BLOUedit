#include "enhancement.h"
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

Enhancement::Enhancement() : python_state(nullptr), is_initialized(false), is_processing_active(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

Enhancement::~Enhancement() {
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

bool Enhancement::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool Enhancement::initialize_python_environment() {
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
    PyObject* pName = PyUnicode_DecodeFSDefault("enhancement_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python enhancement module" << std::endl;
        return false;
    }
    
    // Get the enhancement engine class
    PyObject* pClass = PyObject_GetAttrString(pModule, "EnhancementEngine");
    if (pClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get enhancement engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the enhancement engine
    PyObject* pEngine = PyObject_CallObject(pClass, nullptr);
    Py_DECREF(pClass);
    
    if (pEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create enhancement engine instance" << std::endl;
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

bool Enhancement::apply_enhancement(
    const EnhancementParameters& params,
    std::function<void(float)> progress_callback,
    std::function<void(bool, const std::string&)> completion_callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (completion_callback) completion_callback(false, "Python environment not initialized");
        return false;
    }
    
    if (is_processing_active) {
        std::cerr << "Enhancement is already in progress" << std::endl;
        if (completion_callback) completion_callback(false, "Enhancement is already in progress");
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
            
            // Set enhancement type
            const char* type_str;
            switch (params.type) {
                case EnhancementType::UPSCALE:
                    type_str = "upscale";
                    break;
                case EnhancementType::DENOISING:
                    type_str = "denoising";
                    break;
                case EnhancementType::STABILIZATION:
                    type_str = "stabilization";
                    break;
                case EnhancementType::FRAME_INTERP:
                    type_str = "frame_interp";
                    break;
                case EnhancementType::COLOR_CORRECT:
                    type_str = "color_correct";
                    break;
                case EnhancementType::SHARPEN:
                    type_str = "sharpen";
                    break;
                case EnhancementType::LIGHTING:
                    type_str = "lighting";
                    break;
                default:
                    type_str = "upscale";
            }
            PyDict_SetItemString(pDict, "type", PyUnicode_FromString(type_str));
            
            // Set common parameters
            PyDict_SetItemString(pDict, "strength", PyFloat_FromDouble(params.strength));
            PyDict_SetItemString(pDict, "target_width", PyLong_FromLong(params.target_resolution_width));
            PyDict_SetItemString(pDict, "target_height", PyLong_FromLong(params.target_resolution_height));
            PyDict_SetItemString(pDict, "maintain_aspect_ratio", PyBool_FromLong(params.maintain_aspect_ratio ? 1 : 0));
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
            
            // Set specific parameters based on enhancement type
            if (params.type == EnhancementType::UPSCALE) {
                PyDict_SetItemString(pDict, "upscale_factor", PyLong_FromLong(params.upscale_factor));
            } else if (params.type == EnhancementType::DENOISING) {
                PyDict_SetItemString(pDict, "noise_level", PyFloat_FromDouble(params.noise_level));
            } else if (params.type == EnhancementType::STABILIZATION) {
                PyDict_SetItemString(pDict, "stability_level", PyFloat_FromDouble(params.stability_level));
            } else if (params.type == EnhancementType::FRAME_INTERP) {
                PyDict_SetItemString(pDict, "target_fps", PyLong_FromLong(params.target_fps));
            } else if (params.type == EnhancementType::COLOR_CORRECT) {
                PyDict_SetItemString(pDict, "brightness", PyFloat_FromDouble(params.brightness));
                PyDict_SetItemString(pDict, "contrast", PyFloat_FromDouble(params.contrast));
                PyDict_SetItemString(pDict, "saturation", PyFloat_FromDouble(params.saturation));
                PyDict_SetItemString(pDict, "temperature", PyFloat_FromDouble(params.temperature));
                PyDict_SetItemString(pDict, "tint", PyFloat_FromDouble(params.tint));
            } else if (params.type == EnhancementType::SHARPEN) {
                PyDict_SetItemString(pDict, "sharpness", PyFloat_FromDouble(params.sharpness));
            } else if (params.type == EnhancementType::LIGHTING) {
                PyDict_SetItemString(pDict, "exposure", PyFloat_FromDouble(params.exposure));
                PyDict_SetItemString(pDict, "shadows", PyFloat_FromDouble(params.shadows));
                PyDict_SetItemString(pDict, "highlights", PyFloat_FromDouble(params.highlights));
            }
            
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
            
            // Call apply_enhancement method
            PyObject* pResult = PyObject_CallMethod(state->pEngine, "apply_enhancement", "O", pDict);
            
            Py_XDECREF(pProgressFunc);
            Py_DECREF(pDict);
            
            if (pResult == nullptr) {
                PyErr_Print();
                result_message = "Failed to apply enhancement";
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
                    result_message = success ? params.output_path : "Enhancement failed";
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

void Enhancement::cancel() {
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

bool Enhancement::is_processing() const {
    return is_processing_active;
}

GtkWidget* Enhancement::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 화질 향상");
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
    
    // Enhancement type selection
    GtkWidget* type_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* type_label = gtk_label_new("향상 유형:");
    GtkWidget* type_combo = gtk_combo_box_text_new();
    
    // Add enhancement types
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "업스케일링");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "노이즈 제거");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "안정화");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "프레임 보간");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "색상 보정");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "선명도 향상");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "조명 보정");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(type_combo), 0);
    gtk_widget_set_hexpand(type_combo, TRUE);
    gtk_box_append(GTK_BOX(type_box), type_label);
    gtk_box_append(GTK_BOX(type_box), type_combo);
    gtk_box_append(GTK_BOX(container), type_box);
    
    // Create a notebook for different enhancement types
    GtkWidget* notebook = gtk_notebook_new();
    gtk_widget_set_hexpand(notebook, TRUE);
    gtk_widget_set_vexpand(notebook, TRUE);
    gtk_box_append(GTK_BOX(container), notebook);
    
    // 1. Upscale tab
    GtkWidget* upscale_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(upscale_box, 10);
    gtk_widget_set_margin_end(upscale_box, 10);
    gtk_widget_set_margin_top(upscale_box, 10);
    gtk_widget_set_margin_bottom(upscale_box, 10);
    
    // Upscale factor
    GtkWidget* factor_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* factor_label = gtk_label_new("업스케일 배율:");
    GtkWidget* factor_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(factor_combo), NULL, "2x");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(factor_combo), NULL, "4x");
    gtk_combo_box_set_active(GTK_COMBO_BOX(factor_combo), 0);
    gtk_widget_set_hexpand(factor_combo, TRUE);
    gtk_box_append(GTK_BOX(factor_box), factor_label);
    gtk_box_append(GTK_BOX(factor_box), factor_combo);
    gtk_box_append(GTK_BOX(upscale_box), factor_box);
    
    // Target resolution
    GtkWidget* res_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* res_label = gtk_label_new("목표 해상도:");
    GtkWidget* width_spin = gtk_spin_button_new_with_range(0, 7680, 16);
    GtkWidget* res_x_label = gtk_label_new("x");
    GtkWidget* height_spin = gtk_spin_button_new_with_range(0, 4320, 16);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(width_spin), 0);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(height_spin), 0);
    gtk_widget_set_tooltip_text(width_spin, "0 = 자동");
    gtk_widget_set_tooltip_text(height_spin, "0 = 자동");
    gtk_widget_set_hexpand(width_spin, TRUE);
    gtk_widget_set_hexpand(height_spin, TRUE);
    gtk_box_append(GTK_BOX(res_box), res_label);
    gtk_box_append(GTK_BOX(res_box), width_spin);
    gtk_box_append(GTK_BOX(res_box), res_x_label);
    gtk_box_append(GTK_BOX(res_box), height_spin);
    gtk_box_append(GTK_BOX(upscale_box), res_box);
    
    // Maintain aspect ratio
    GtkWidget* aspect_check = gtk_check_button_new_with_label("화면비 유지");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(aspect_check), TRUE);
    gtk_box_append(GTK_BOX(upscale_box), aspect_check);
    
    // Add upscale tab to notebook
    GtkWidget* upscale_label = gtk_label_new("업스케일링");
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), upscale_box, upscale_label);
    
    // Common section at the bottom for all tabs
    GtkWidget* common_frame = gtk_frame_new("공통 옵션");
    GtkWidget* common_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(common_box, 5);
    gtk_widget_set_margin_end(common_box, 5);
    gtk_widget_set_margin_top(common_box, 5);
    gtk_widget_set_margin_bottom(common_box, 5);
    
    // Strength slider
    GtkWidget* strength_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* strength_label = gtk_label_new("효과 강도:");
    GtkWidget* strength_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 1.0, 0.05);
    gtk_range_set_value(GTK_RANGE(strength_scale), 1.0);
    gtk_widget_set_hexpand(strength_scale, TRUE);
    gtk_box_append(GTK_BOX(strength_box), strength_label);
    gtk_box_append(GTK_BOX(strength_box), strength_scale);
    gtk_box_append(GTK_BOX(common_box), strength_box);
    
    // Process audio checkbox
    GtkWidget* audio_check = gtk_check_button_new_with_label("오디오 처리");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(audio_check), TRUE);
    gtk_box_append(GTK_BOX(common_box), audio_check);
    
    gtk_frame_set_child(GTK_FRAME(common_frame), common_box);
    gtk_box_append(GTK_BOX(container), common_frame);
    
    // Process button
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* process_button = gtk_button_new_with_label("향상 시작");
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