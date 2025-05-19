#include "face_mosaic.h"
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

FaceMosaic::FaceMosaic() : python_state(nullptr), is_initialized(false), is_processing_active(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

FaceMosaic::~FaceMosaic() {
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

bool FaceMosaic::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool FaceMosaic::initialize_python_environment() {
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
    PyObject* pName = PyUnicode_DecodeFSDefault("face_mosaic_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python face mosaic module" << std::endl;
        return false;
    }
    
    // Get the face mosaic engine class
    PyObject* pClass = PyObject_GetAttrString(pModule, "FaceMosaicEngine");
    if (pClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get face mosaic engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the face mosaic engine
    PyObject* pEngine = PyObject_CallObject(pClass, nullptr);
    Py_DECREF(pClass);
    
    if (pEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create face mosaic engine instance" << std::endl;
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

bool FaceMosaic::detect_faces(
    const std::string& input_path,
    std::vector<FaceRect>& faces,
    float detection_threshold,
    std::function<void(float)> progress_callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        return false;
    }
    
    if (is_processing_active) {
        std::cerr << "Face mosaic operation already in progress" << std::endl;
        return false;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Reset cancel flag
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->cancel_requested = false;
    }
    
    // Mark as processing
    is_processing_active = true;
    
    bool success = false;
    
    // Use Python GIL in this thread
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    try {
        // Create parameters
        PyObject* pArgs = PyDict_New();
        PyDict_SetItemString(pArgs, "input_path", PyUnicode_FromString(input_path.c_str()));
        PyDict_SetItemString(pArgs, "detection_threshold", PyFloat_FromDouble(detection_threshold));
        
        // Create progress callback function
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
            
            PyDict_SetItemString(pArgs, "progress_callback", pProgressFunc);
        }
        
        // Call detect_faces method
        PyObject* pResult = PyObject_CallMethod(state->pEngine, "detect_faces", "O", pArgs);
        
        Py_XDECREF(pProgressFunc);
        Py_DECREF(pArgs);
        
        if (pResult == nullptr) {
            PyErr_Print();
            std::cerr << "Failed to detect faces" << std::endl;
            success = false;
        } else if (PyTuple_Check(pResult) && PyTuple_Size(pResult) == 2) {
            // Parse the result tuple (success, faces_list)
            PyObject* pSuccess = PyTuple_GetItem(pResult, 0);
            PyObject* pFaces = PyTuple_GetItem(pResult, 1);
            
            success = PyObject_IsTrue(pSuccess);
            
            if (success && PyList_Check(pFaces)) {
                // Parse face rectangles
                Py_ssize_t num_faces = PyList_Size(pFaces);
                faces.clear();
                
                for (Py_ssize_t i = 0; i < num_faces; i++) {
                    PyObject* pFace = PyList_GetItem(pFaces, i);
                    if (PyDict_Check(pFace)) {
                        FaceRect rect;
                        
                        PyObject* pX = PyDict_GetItemString(pFace, "x");
                        PyObject* pY = PyDict_GetItemString(pFace, "y");
                        PyObject* pWidth = PyDict_GetItemString(pFace, "width");
                        PyObject* pHeight = PyDict_GetItemString(pFace, "height");
                        PyObject* pConfidence = PyDict_GetItemString(pFace, "confidence");
                        PyObject* pTrackingId = PyDict_GetItemString(pFace, "tracking_id");
                        
                        if (pX && PyLong_Check(pX)) rect.x = PyLong_AsLong(pX);
                        if (pY && PyLong_Check(pY)) rect.y = PyLong_AsLong(pY);
                        if (pWidth && PyLong_Check(pWidth)) rect.width = PyLong_AsLong(pWidth);
                        if (pHeight && PyLong_Check(pHeight)) rect.height = PyLong_AsLong(pHeight);
                        if (pConfidence && PyFloat_Check(pConfidence)) rect.confidence = (float)PyFloat_AsDouble(pConfidence);
                        if (pTrackingId && PyLong_Check(pTrackingId)) rect.tracking_id = PyLong_AsLong(pTrackingId);
                        
                        faces.push_back(rect);
                    }
                }
            }
            
            Py_DECREF(pResult);
        } else {
            Py_DECREF(pResult);
            success = false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during face detection: " << e.what() << std::endl;
        success = false;
    }
    
    // Mark as not processing
    is_processing_active = false;
    
    // Release GIL
    PyGILState_Release(gstate);
    
    return success;
}

bool FaceMosaic::apply_mosaic(
    const FaceMosaicParameters& params,
    std::function<void(float)> progress_callback,
    std::function<void(bool, const std::string&)> completion_callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (completion_callback) completion_callback(false, "Python environment not initialized");
        return false;
    }
    
    if (is_processing_active) {
        std::cerr << "Face mosaic operation already in progress" << std::endl;
        if (completion_callback) completion_callback(false, "Face mosaic operation already in progress");
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
            
            // Set mosaic type
            const char* type_str;
            switch (params.mosaic_type) {
                case MosaicType::BLUR:
                    type_str = "blur";
                    break;
                case MosaicType::PIXELATE:
                    type_str = "pixelate";
                    break;
                case MosaicType::BLACK:
                    type_str = "black";
                    break;
                case MosaicType::EMOJI:
                    type_str = "emoji";
                    break;
                case MosaicType::CUSTOM_IMAGE:
                    type_str = "custom_image";
                    break;
                default:
                    type_str = "blur";
            }
            PyDict_SetItemString(pDict, "mosaic_type", PyUnicode_FromString(type_str));
            
            // Set additional parameters
            PyDict_SetItemString(pDict, "effect_intensity", PyFloat_FromDouble(params.effect_intensity));
            PyDict_SetItemString(pDict, "track_faces", PyBool_FromLong(params.track_faces ? 1 : 0));
            PyDict_SetItemString(pDict, "process_audio", PyBool_FromLong(params.process_audio ? 1 : 0));
            PyDict_SetItemString(pDict, "detect_only", PyBool_FromLong(params.detect_only ? 1 : 0));
            PyDict_SetItemString(pDict, "custom_image_path", PyUnicode_FromString(params.custom_image_path.c_str()));
            PyDict_SetItemString(pDict, "emoji_type", PyUnicode_FromString(params.emoji_type.c_str()));
            PyDict_SetItemString(pDict, "detection_threshold", PyFloat_FromDouble(params.detection_threshold));
            PyDict_SetItemString(pDict, "auto_expand_rect", PyBool_FromLong(params.auto_expand_rect ? 1 : 0));
            PyDict_SetItemString(pDict, "expansion_factor", PyFloat_FromDouble(params.expansion_factor));
            
            // Set segments list
            PyObject* pSegments = PyList_New(params.segments.size());
            for (size_t i = 0; i < params.segments.size(); i++) {
                PyObject* pSegment = PyTuple_New(2);
                PyTuple_SetItem(pSegment, 0, PyLong_FromLong(params.segments[i].first));
                PyTuple_SetItem(pSegment, 1, PyLong_FromLong(params.segments[i].second));
                PyList_SetItem(pSegments, i, pSegment);
            }
            PyDict_SetItemString(pDict, "segments", pSegments);
            
            // Set custom rects if any
            PyObject* pRects = PyList_New(params.custom_rects.size());
            for (size_t i = 0; i < params.custom_rects.size(); i++) {
                const FaceRect& rect = params.custom_rects[i];
                PyObject* pRect = PyDict_New();
                PyDict_SetItemString(pRect, "x", PyLong_FromLong(rect.x));
                PyDict_SetItemString(pRect, "y", PyLong_FromLong(rect.y));
                PyDict_SetItemString(pRect, "width", PyLong_FromLong(rect.width));
                PyDict_SetItemString(pRect, "height", PyLong_FromLong(rect.height));
                PyDict_SetItemString(pRect, "confidence", PyFloat_FromDouble(rect.confidence));
                PyDict_SetItemString(pRect, "tracking_id", PyLong_FromLong(rect.tracking_id));
                PyList_SetItem(pRects, i, pRect);
            }
            PyDict_SetItemString(pDict, "custom_rects", pRects);
            
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
            
            // Call apply_mosaic method
            PyObject* pResult = PyObject_CallMethod(state->pEngine, "apply_mosaic", "O", pDict);
            
            Py_XDECREF(pProgressFunc);
            Py_DECREF(pDict);
            
            if (pResult == nullptr) {
                PyErr_Print();
                result_message = "Failed to apply face mosaic";
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
                    result_message = success ? params.output_path : "Face mosaic failed";
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

void FaceMosaic::cancel() {
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

bool FaceMosaic::is_processing() const {
    return is_processing_active;
}

GtkWidget* FaceMosaic::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 얼굴 모자이크");
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
    
    // Mosaic type selection
    GtkWidget* type_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* type_label = gtk_label_new("모자이크 유형:");
    GtkWidget* type_combo = gtk_combo_box_text_new();
    
    // Add mosaic types
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "흐림 (Blur)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "픽셀화 (Pixelate)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "검은색 (Black)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "이모지 (Emoji)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(type_combo), NULL, "사용자 정의 이미지");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(type_combo), 0);
    gtk_widget_set_hexpand(type_combo, TRUE);
    gtk_box_append(GTK_BOX(type_box), type_label);
    gtk_box_append(GTK_BOX(type_box), type_combo);
    gtk_box_append(GTK_BOX(container), type_box);
    
    // Effect intensity slider
    GtkWidget* intensity_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* intensity_label = gtk_label_new("효과 강도:");
    GtkWidget* intensity_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 1.0, 50.0, 1.0);
    gtk_range_set_value(GTK_RANGE(intensity_scale), 15.0);
    gtk_widget_set_hexpand(intensity_scale, TRUE);
    gtk_box_append(GTK_BOX(intensity_box), intensity_label);
    gtk_box_append(GTK_BOX(intensity_box), intensity_scale);
    gtk_box_append(GTK_BOX(container), intensity_box);
    
    // Detection threshold slider
    GtkWidget* threshold_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* threshold_label = gtk_label_new("인식 임계값:");
    GtkWidget* threshold_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.1, 0.9, 0.05);
    gtk_range_set_value(GTK_RANGE(threshold_scale), 0.5);
    gtk_widget_set_hexpand(threshold_scale, TRUE);
    gtk_box_append(GTK_BOX(threshold_box), threshold_label);
    gtk_box_append(GTK_BOX(threshold_box), threshold_scale);
    gtk_box_append(GTK_BOX(container), threshold_box);
    
    // Emoji selection (initially hidden)
    GtkWidget* emoji_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* emoji_label = gtk_label_new("이모지:");
    GtkWidget* emoji_combo = gtk_combo_box_text_new();
    
    // Add emoji options
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(emoji_combo), NULL, "🙂 스마일");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(emoji_combo), NULL, "😀 웃음");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(emoji_combo), NULL, "😎 멋짐");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(emoji_combo), NULL, "🤔 생각중");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(emoji_combo), NULL, "😷 마스크");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(emoji_combo), 0);
    gtk_widget_set_hexpand(emoji_combo, TRUE);
    gtk_box_append(GTK_BOX(emoji_box), emoji_label);
    gtk_box_append(GTK_BOX(emoji_box), emoji_combo);
    gtk_widget_set_visible(emoji_box, FALSE);  // Hidden by default
    gtk_box_append(GTK_BOX(container), emoji_box);
    
    // Custom image selection (initially hidden)
    GtkWidget* custom_image_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* custom_image_label = gtk_label_new("이미지:");
    GtkWidget* custom_image_entry = gtk_entry_new();
    GtkWidget* custom_image_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(custom_image_entry, TRUE);
    gtk_box_append(GTK_BOX(custom_image_box), custom_image_label);
    gtk_box_append(GTK_BOX(custom_image_box), custom_image_entry);
    gtk_box_append(GTK_BOX(custom_image_box), custom_image_button);
    gtk_widget_set_visible(custom_image_box, FALSE);  // Hidden by default
    gtk_box_append(GTK_BOX(container), custom_image_box);
    
    // Options frame
    GtkWidget* options_frame = gtk_frame_new("옵션");
    GtkWidget* options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(options_box, 5);
    gtk_widget_set_margin_end(options_box, 5);
    gtk_widget_set_margin_top(options_box, 5);
    gtk_widget_set_margin_bottom(options_box, 5);
    
    // Track faces checkbox
    GtkWidget* track_faces_check = gtk_check_button_new_with_label("얼굴 추적 (비디오)");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(track_faces_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), track_faces_check);
    
    // Process audio checkbox
    GtkWidget* process_audio_check = gtk_check_button_new_with_label("오디오 처리 (비디오)");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(process_audio_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), process_audio_check);
    
    // Auto expand rect checkbox
    GtkWidget* expand_rect_check = gtk_check_button_new_with_label("얼굴 영역 자동 확장");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(expand_rect_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), expand_rect_check);
    
    // Detection only checkbox
    GtkWidget* detect_only_check = gtk_check_button_new_with_label("얼굴 감지만 수행");
    gtk_box_append(GTK_BOX(options_box), detect_only_check);
    
    gtk_frame_set_child(GTK_FRAME(options_frame), options_box);
    gtk_box_append(GTK_BOX(container), options_frame);
    
    // Manual selection button - for adding rectangles manually
    GtkWidget* manual_button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* manual_button = gtk_button_new_with_label("수동 선택");
    gtk_widget_set_hexpand(manual_button, TRUE);
    gtk_box_append(GTK_BOX(manual_button_box), manual_button);
    gtk_box_append(GTK_BOX(container), manual_button_box);
    
    // Process buttons
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* detect_button = gtk_button_new_with_label("얼굴 감지");
    GtkWidget* mosaic_button = gtk_button_new_with_label("모자이크 적용");
    GtkWidget* cancel_button = gtk_button_new_with_label("취소");
    gtk_widget_set_hexpand(detect_button, TRUE);
    gtk_widget_set_hexpand(mosaic_button, TRUE);
    gtk_widget_set_sensitive(cancel_button, FALSE);
    gtk_box_append(GTK_BOX(button_box), detect_button);
    gtk_box_append(GTK_BOX(button_box), mosaic_button);
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