#include "auto_subtitle.h"
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

AutoSubtitle::AutoSubtitle() : python_state(nullptr), is_initialized(false), is_processing_active(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

AutoSubtitle::~AutoSubtitle() {
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

bool AutoSubtitle::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool AutoSubtitle::initialize_python_environment() {
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
    PyObject* pName = PyUnicode_DecodeFSDefault("auto_subtitle_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python auto subtitle module" << std::endl;
        return false;
    }
    
    // Get the auto subtitle engine class
    PyObject* pClass = PyObject_GetAttrString(pModule, "AutoSubtitleEngine");
    if (pClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get auto subtitle engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the auto subtitle engine
    PyObject* pEngine = PyObject_CallObject(pClass, nullptr);
    Py_DECREF(pClass);
    
    if (pEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create auto subtitle engine instance" << std::endl;
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

bool AutoSubtitle::generate_subtitles(
    const AutoSubtitleParameters& params,
    std::function<void(float)> progress_callback,
    std::function<void(bool, const std::string&)> completion_callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (completion_callback) completion_callback(false, "Python environment not initialized");
        return false;
    }
    
    if (is_processing_active) {
        std::cerr << "Subtitle generation already in progress" << std::endl;
        if (completion_callback) completion_callback(false, "Subtitle generation already in progress");
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
            
            // Set subtitle format
            const char* format_str;
            switch (params.format) {
                case SubtitleFormat::SRT: format_str = "srt"; break;
                case SubtitleFormat::VTT: format_str = "vtt"; break;
                case SubtitleFormat::ASS: format_str = "ass"; break;
                case SubtitleFormat::TXT: format_str = "txt"; break;
                case SubtitleFormat::JSON: format_str = "json"; break;
                default: format_str = "srt";
            }
            PyDict_SetItemString(pDict, "format", PyUnicode_FromString(format_str));
            
            // Set translation mode
            const char* translation_mode_str;
            switch (params.translation_mode) {
                case TranslationMode::NONE: translation_mode_str = "none"; break;
                case TranslationMode::AUTO: translation_mode_str = "auto"; break;
                case TranslationMode::MANUAL: translation_mode_str = "manual"; break;
                default: translation_mode_str = "none";
            }
            PyDict_SetItemString(pDict, "translation_mode", PyUnicode_FromString(translation_mode_str));
            
            // Set basic parameters
            PyDict_SetItemString(pDict, "extract_audio_only", PyBool_FromLong(params.extract_audio_only ? 1 : 0));
            PyDict_SetItemString(pDict, "extracted_audio_path", PyUnicode_FromString(params.extracted_audio_path.c_str()));
            PyDict_SetItemString(pDict, "language", PyUnicode_FromString(params.language.c_str()));
            PyDict_SetItemString(pDict, "include_confidence", PyBool_FromLong(params.include_confidence ? 1 : 0));
            PyDict_SetItemString(pDict, "filter_profanity", PyBool_FromLong(params.filter_profanity ? 1 : 0));
            PyDict_SetItemString(pDict, "max_chars_per_line", PyLong_FromLong(params.max_chars_per_line));
            PyDict_SetItemString(pDict, "max_lines_per_subtitle", PyLong_FromLong(params.max_lines_per_subtitle));
            PyDict_SetItemString(pDict, "min_segment_duration_ms", PyLong_FromLong(params.min_segment_duration_ms));
            PyDict_SetItemString(pDict, "max_segment_duration_ms", PyLong_FromLong(params.max_segment_duration_ms));
            PyDict_SetItemString(pDict, "merge_short_segments", PyBool_FromLong(params.merge_short_segments ? 1 : 0));
            PyDict_SetItemString(pDict, "split_long_segments", PyBool_FromLong(params.split_long_segments ? 1 : 0));
            PyDict_SetItemString(pDict, "adjust_timing_to_scene_changes", PyBool_FromLong(params.adjust_timing_to_scene_changes ? 1 : 0));
            
            // Set translation parameters
            PyDict_SetItemString(pDict, "source_language", PyUnicode_FromString(params.source_language.c_str()));
            PyDict_SetItemString(pDict, "target_language", PyUnicode_FromString(params.target_language.c_str()));
            
            // Set styling options
            PyDict_SetItemString(pDict, "font_name", PyUnicode_FromString(params.font_name.c_str()));
            PyDict_SetItemString(pDict, "font_size", PyLong_FromLong(params.font_size));
            PyDict_SetItemString(pDict, "font_color", PyUnicode_FromString(params.font_color.c_str()));
            PyDict_SetItemString(pDict, "background_color", PyUnicode_FromString(params.background_color.c_str()));
            PyDict_SetItemString(pDict, "outline_color", PyUnicode_FromString(params.outline_color.c_str()));
            PyDict_SetItemString(pDict, "outline_width", PyLong_FromLong(params.outline_width));
            PyDict_SetItemString(pDict, "position", PyUnicode_FromString(params.position.c_str()));
            
            // Process time segments if provided
            if (!params.segments.empty()) {
                PyObject* pSegmentsList = PyList_New(params.segments.size());
                for (size_t i = 0; i < params.segments.size(); i++) {
                    PyObject* pSegment = PyTuple_New(2);
                    PyTuple_SetItem(pSegment, 0, PyLong_FromLong(params.segments[i].first));
                    PyTuple_SetItem(pSegment, 1, PyLong_FromLong(params.segments[i].second));
                    PyList_SetItem(pSegmentsList, i, pSegment);
                }
                PyDict_SetItemString(pDict, "segments", pSegmentsList);
            }
            
            // Define progress callback function
            PyObject* pProgressFunc = nullptr;
            if (progress_callback) {
                pProgressFunc = PyCFunction_New(&(PyMethodDef{"progress_callback", [](PyObject* self, PyObject* args) -> PyObject* {
                    float progress;
                    if (!PyArg_ParseTuple(args, "f", &progress)) {
                        return nullptr;
                    }
                    
                    // Get the C++ callback from captured data
                    auto* callback_ptr = static_cast<std::function<void(float)>*>(PyCapsule_GetPointer(self, "callback_ptr"));
                    if (callback_ptr) {
                        (*callback_ptr)(progress);
                    }
                    
                    Py_RETURN_NONE;
                }, METH_VARARGS, "Progress callback"}), 
                PyCapsule_New(&progress_callback, "callback_ptr", nullptr));
            } else {
                Py_INCREF(Py_None);
                pProgressFunc = Py_None;
            }
            
            // Call the process_subtitles method on the Python engine
            PyObject* pEngine = state->pEngine;
            PyObject* pProcessMethod = PyObject_GetAttrString(pEngine, "process_subtitles");
            if (!pProcessMethod) {
                PyErr_Print();
                throw std::runtime_error("Failed to get process_subtitles method");
            }
            
            PyObject* pArgs = PyTuple_New(2);
            PyTuple_SetItem(pArgs, 0, pDict);
            PyTuple_SetItem(pArgs, 1, pProgressFunc);
            
            // Regular check for cancellation during processing
            bool is_cancelled = false;
            std::thread cancellation_check_thread([&is_cancelled, state, &completion_callback]() {
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        if (state->cancel_requested) {
                            is_cancelled = true;
                            break;
                        }
                    }
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
                if (is_cancelled && completion_callback) {
                    completion_callback(false, "Operation cancelled by user");
                }
            });
            
            // Call the Python method
            PyObject* pResult = PyObject_CallObject(pProcessMethod, pArgs);
            Py_DECREF(pProcessMethod);
            Py_DECREF(pArgs);
            
            // Stop the cancellation check thread
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->cancel_requested = true;
            }
            if (cancellation_check_thread.joinable()) {
                cancellation_check_thread.join();
            }
            
            // Process the result
            if (pResult) {
                // Check if it's a dictionary
                if (PyDict_Check(pResult)) {
                    // Extract success status
                    PyObject* pSuccess = PyDict_GetItemString(pResult, "success");
                    if (pSuccess && PyBool_Check(pSuccess)) {
                        success = (pSuccess == Py_True);
                    }
                    
                    // Extract message
                    PyObject* pMessage = PyDict_GetItemString(pResult, "message");
                    if (pMessage && PyUnicode_Check(pMessage)) {
                        result_message = PyUnicode_AsUTF8(pMessage);
                    }
                    
                    // Extract additional data if needed
                    // ...
                }
                Py_DECREF(pResult);
            } else {
                PyErr_Print();
                success = false;
                result_message = "Python error occurred during subtitle generation";
            }
            
        } catch (const std::exception& e) {
            success = false;
            result_message = std::string("Exception during subtitle generation: ") + e.what();
        } catch (...) {
            success = false;
            result_message = "Unknown exception during subtitle generation";
        }
        
        // Mark processing as complete
        is_processing_active = false;
        
        // Call completion callback if provided
        if (completion_callback) {
            completion_callback(success, result_message);
        }
        
        PyGILState_Release(gstate);
    });
    
         return true;
            PyDict_SetItemString(pDict, "font_name", PyUnicode_FromString(params.font_name.c_str()));
            PyDict_SetItemString(pDict, "font_size", PyLong_FromLong(params.font_size));
            PyDict_SetItemString(pDict, "font_color", PyUnicode_FromString(params.font_color.c_str()));
            PyDict_SetItemString(pDict, "background_color", PyUnicode_FromString(params.background_color.c_str()));
            PyDict_SetItemString(pDict, "outline_color", PyUnicode_FromString(params.outline_color.c_str()));
            PyDict_SetItemString(pDict, "outline_width", PyLong_FromLong(params.outline_width));
            PyDict_SetItemString(pDict, "position", PyUnicode_FromString(params.position.c_str()));
            
            // Set segments list
            PyObject* pSegments = PyList_New(params.segments.size());
            for (size_t i = 0; i < params.segments.size(); i++) {
                PyObject* pSegment = PyTuple_New(2);
                PyTuple_SetItem(pSegment, 0, PyLong_FromLong(params.segments[i].first));
                PyTuple_SetItem(pSegment, 1, PyLong_FromLong(params.segments[i].second));
                PyList_SetItem(pSegments, i, pSegment);
            }
            PyDict_SetItemString(pDict, "segments", pSegments);
            
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
            
            // Call generate_subtitles method
            PyObject* pResult = PyObject_CallMethod(state->pEngine, "generate_subtitles", "O", pDict);
            
            Py_XDECREF(pProgressFunc);
            Py_DECREF(pDict);
            
            if (pResult == nullptr) {
                PyErr_Print();
                result_message = "Failed to generate subtitles";
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
                    result_message = success ? params.output_path : "Subtitle generation failed";
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

void AutoSubtitle::cancel() {
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

bool AutoSubtitle::is_processing() const {
    return is_processing_active;
}

GtkWidget* AutoSubtitle::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 자동 자막 생성");
    PangoAttrList* attr_list = pango_attr_list_new();
    pango_attr_list_insert(attr_list, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attr_list, pango_attr_scale_new(1.2));
    gtk_label_set_attributes(GTK_LABEL(title), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_append(GTK_BOX(container), title);
    
    // Input file selection
    GtkWidget* input_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* input_label = gtk_label_new("입력 비디오:");
    GtkWidget* input_entry = gtk_entry_new();
    GtkWidget* input_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(input_entry, TRUE);
    gtk_box_append(GTK_BOX(input_box), input_label);
    gtk_box_append(GTK_BOX(input_box), input_entry);
    gtk_box_append(GTK_BOX(input_box), input_button);
    gtk_box_append(GTK_BOX(container), input_box);
    
    // Output file selection
    GtkWidget* output_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* output_label = gtk_label_new("출력 자막:");
    GtkWidget* output_entry = gtk_entry_new();
    GtkWidget* output_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(output_entry, TRUE);
    gtk_box_append(GTK_BOX(output_box), output_label);
    gtk_box_append(GTK_BOX(output_box), output_entry);
    gtk_box_append(GTK_BOX(output_box), output_button);
    gtk_box_append(GTK_BOX(container), output_box);
    
    // Format selection
    GtkWidget* format_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* format_label = gtk_label_new("자막 형식:");
    GtkWidget* format_combo = gtk_combo_box_text_new();
    
    // Add formats
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "SubRip (SRT)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "WebVTT (VTT)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "Advanced SubStation Alpha (ASS)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "Plain Text (TXT)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "JSON");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(format_combo), 0);
    gtk_widget_set_hexpand(format_combo, TRUE);
    gtk_box_append(GTK_BOX(format_box), format_label);
    gtk_box_append(GTK_BOX(format_box), format_combo);
    gtk_box_append(GTK_BOX(container), format_box);
    
    // Language selection
    GtkWidget* lang_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* lang_label = gtk_label_new("음성 언어:");
    GtkWidget* lang_combo = gtk_combo_box_text_new();
    
    // Add languages
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "자동 감지");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "한국어 (ko)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "영어 (en)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "일본어 (ja)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "중국어 (zh)");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(lang_combo), 0);
    gtk_widget_set_hexpand(lang_combo, TRUE);
    gtk_box_append(GTK_BOX(lang_box), lang_label);
    gtk_box_append(GTK_BOX(lang_box), lang_combo);
    gtk_box_append(GTK_BOX(container), lang_box);
    
    // Translation mode
    GtkWidget* translation_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* translation_label = gtk_label_new("번역 모드:");
    GtkWidget* translation_combo = gtk_combo_box_text_new();
    
    // Add translation modes
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(translation_combo), NULL, "번역 안함");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(translation_combo), NULL, "자동 언어 감지 및 번역");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(translation_combo), NULL, "수동 언어 선택");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(translation_combo), 0);
    gtk_widget_set_hexpand(translation_combo, TRUE);
    gtk_box_append(GTK_BOX(translation_box), translation_label);
    gtk_box_append(GTK_BOX(translation_box), translation_combo);
    gtk_box_append(GTK_BOX(container), translation_box);
    
    // Target language (initially disabled, activated when translation is enabled)
    GtkWidget* target_lang_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* target_lang_label = gtk_label_new("대상 언어:");
    GtkWidget* target_lang_combo = gtk_combo_box_text_new();
    
    // Add target languages
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(target_lang_combo), NULL, "한국어 (ko)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(target_lang_combo), NULL, "영어 (en)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(target_lang_combo), NULL, "일본어 (ja)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(target_lang_combo), NULL, "중국어 (zh)");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(target_lang_combo), 0);
    gtk_widget_set_hexpand(target_lang_combo, TRUE);
    gtk_widget_set_sensitive(target_lang_label, FALSE);
    gtk_widget_set_sensitive(target_lang_combo, FALSE);
    gtk_box_append(GTK_BOX(target_lang_box), target_lang_label);
    gtk_box_append(GTK_BOX(target_lang_box), target_lang_combo);
    gtk_box_append(GTK_BOX(container), target_lang_box);
    
    // Options frame
    GtkWidget* options_frame = gtk_frame_new("옵션");
    GtkWidget* options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(options_box, 5);
    gtk_widget_set_margin_end(options_box, 5);
    gtk_widget_set_margin_top(options_box, 5);
    gtk_widget_set_margin_bottom(options_box, 5);
    
    // Include confidence checkbox
    GtkWidget* confidence_check = gtk_check_button_new_with_label("신뢰도 점수 포함");
    gtk_box_append(GTK_BOX(options_box), confidence_check);
    
    // Filter profanity checkbox
    GtkWidget* profanity_check = gtk_check_button_new_with_label("비속어 필터링");
    gtk_box_append(GTK_BOX(options_box), profanity_check);
    
    // Merge short segments checkbox
    GtkWidget* merge_check = gtk_check_button_new_with_label("짧은 세그먼트 병합");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(merge_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), merge_check);
    
    // Split long segments checkbox
    GtkWidget* split_check = gtk_check_button_new_with_label("긴 세그먼트 분할");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(split_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), split_check);
    
    // Adjust timing to scene changes checkbox
    GtkWidget* scene_check = gtk_check_button_new_with_label("장면 전환에 타이밍 조정");
    gtk_box_append(GTK_BOX(options_box), scene_check);
    
    // Max characters per line
    GtkWidget* max_chars_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* max_chars_label = gtk_label_new("줄당 최대 문자 수:");
    GtkWidget* max_chars_spin = gtk_spin_button_new_with_range(10, 80, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(max_chars_spin), 42);
    gtk_widget_set_hexpand(max_chars_spin, TRUE);
    gtk_box_append(GTK_BOX(max_chars_box), max_chars_label);
    gtk_box_append(GTK_BOX(max_chars_box), max_chars_spin);
    gtk_box_append(GTK_BOX(options_box), max_chars_box);
    
    gtk_frame_set_child(GTK_FRAME(options_frame), options_box);
    gtk_box_append(GTK_BOX(container), options_frame);
    
    // Process buttons
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* extract_button = gtk_button_new_with_label("오디오만 추출");
    GtkWidget* generate_button = gtk_button_new_with_label("자막 생성");
    GtkWidget* cancel_button = gtk_button_new_with_label("취소");
    gtk_widget_set_hexpand(generate_button, TRUE);
    gtk_box_append(GTK_BOX(button_box), extract_button);
    gtk_box_append(GTK_BOX(button_box), generate_button);
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
    
    // Store references to important widgets
    g_object_set_data(G_OBJECT(container), "input_entry", input_entry);
    g_object_set_data(G_OBJECT(container), "output_entry", output_entry);
    g_object_set_data(G_OBJECT(container), "format_combo", format_combo);
    g_object_set_data(G_OBJECT(container), "lang_combo", lang_combo);
    g_object_set_data(G_OBJECT(container), "translation_combo", translation_combo);
    g_object_set_data(G_OBJECT(container), "target_lang_combo", target_lang_combo);
    g_object_set_data(G_OBJECT(container), "confidence_check", confidence_check);
    g_object_set_data(G_OBJECT(container), "profanity_check", profanity_check);
    g_object_set_data(G_OBJECT(container), "merge_check", merge_check);
    g_object_set_data(G_OBJECT(container), "split_check", split_check);
    g_object_set_data(G_OBJECT(container), "scene_check", scene_check);
    g_object_set_data(G_OBJECT(container), "max_chars_spin", max_chars_spin);
    g_object_set_data(G_OBJECT(container), "progress_bar", progress_bar);
    g_object_set_data(G_OBJECT(container), "status_label", status_label);
    g_object_set_data(G_OBJECT(container), "auto_subtitle", this);
    
    // Connect input file browse button
    g_signal_connect(input_button, "clicked", G_CALLBACK([](GtkButton*, gpointer user_data) {
        GtkWidget* container = GTK_WIDGET(user_data);
        GtkWidget* entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "input_entry"));
        
        GtkWidget* dialog = gtk_file_chooser_dialog_new("Select Input Video",
                                                      GTK_WINDOW(gtk_widget_get_root(container)),
                                                      GTK_FILE_CHOOSER_ACTION_OPEN,
                                                      "_Cancel", GTK_RESPONSE_CANCEL,
                                                      "_Open", GTK_RESPONSE_ACCEPT,
                                                      NULL);
        
        GtkFileFilter* filter = gtk_file_filter_new();
        gtk_file_filter_set_name(filter, "Video Files");
        gtk_file_filter_add_mime_type(filter, "video/*");
        gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
        
        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            gtk_entry_set_text(GTK_ENTRY(entry), filename);
            g_free(filename);
            
            // Update output filename with same basename but different extension
            GtkWidget* output_entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "output_entry"));
            GtkWidget* format_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "format_combo"));
            
            std::string input_path = gtk_entry_get_text(GTK_ENTRY(entry));
            std::string basename = std::filesystem::path(input_path).stem().string();
            
            // Get selected format
            int format_index = gtk_combo_box_get_active(GTK_COMBO_BOX(format_combo));
            std::string extension;
            switch (format_index) {
                case 0: extension = ".srt"; break;
                case 1: extension = ".vtt"; break;
                case 2: extension = ".ass"; break;
                case 3: extension = ".txt"; break;
                case 4: extension = ".json"; break;
                default: extension = ".srt"; break;
            }
            
            std::string output_path = basename + extension;
            gtk_entry_set_text(GTK_ENTRY(output_entry), output_path.c_str());
        }
        
        gtk_widget_destroy(dialog);
    }), container);
    
    // Connect output file browse button
    g_signal_connect(output_button, "clicked", G_CALLBACK([](GtkButton*, gpointer user_data) {
        GtkWidget* container = GTK_WIDGET(user_data);
        GtkWidget* entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "output_entry"));
        
        GtkWidget* dialog = gtk_file_chooser_dialog_new("Select Output Subtitle File",
                                                      GTK_WINDOW(gtk_widget_get_root(container)),
                                                      GTK_FILE_CHOOSER_ACTION_SAVE,
                                                      "_Cancel", GTK_RESPONSE_CANCEL,
                                                      "_Save", GTK_RESPONSE_ACCEPT,
                                                      NULL);
        
        gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);
        
        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            gtk_entry_set_text(GTK_ENTRY(entry), filename);
            g_free(filename);
        }
        
        gtk_widget_destroy(dialog);
    }), container);
    
    // Connect format combo to update output extension
    g_signal_connect(format_combo, "changed", G_CALLBACK([](GtkComboBox* combo, gpointer user_data) {
        GtkWidget* container = GTK_WIDGET(user_data);
        GtkWidget* output_entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "output_entry"));
        
        std::string output_path = gtk_entry_get_text(GTK_ENTRY(output_entry));
        std::string basename = std::filesystem::path(output_path).stem().string();
        
        // Get selected format
        int format_index = gtk_combo_box_get_active(combo);
        std::string extension;
        switch (format_index) {
            case 0: extension = ".srt"; break;
            case 1: extension = ".vtt"; break;
            case 2: extension = ".ass"; break;
            case 3: extension = ".txt"; break;
            case 4: extension = ".json"; break;
            default: extension = ".srt"; break;
        }
        
        std::string new_output_path = basename + extension;
        gtk_entry_set_text(GTK_ENTRY(output_entry), new_output_path.c_str());
    }), container);
    
    // Connect translation combo to enable/disable target language
    g_signal_connect(translation_combo, "changed", G_CALLBACK([](GtkComboBox* combo, gpointer user_data) {
        GtkWidget* container = GTK_WIDGET(user_data);
        GtkWidget* target_lang_label = gtk_widget_get_prev_sibling(GTK_WIDGET(g_object_get_data(G_OBJECT(container), "target_lang_combo")));
        GtkWidget* target_lang_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "target_lang_combo"));
        
        int translation_index = gtk_combo_box_get_active(combo);
        bool enable_target = (translation_index > 0);
        
        gtk_widget_set_sensitive(target_lang_label, enable_target);
        gtk_widget_set_sensitive(target_lang_combo, enable_target);
    }), container);
    
    // Connect extract button
    g_signal_connect(extract_button, "clicked", G_CALLBACK([](GtkButton*, gpointer user_data) {
        GtkWidget* container = GTK_WIDGET(user_data);
        AutoSubtitle* auto_subtitle = static_cast<AutoSubtitle*>(g_object_get_data(G_OBJECT(container), "auto_subtitle"));
        
        // Get widget references
        GtkWidget* input_entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "input_entry"));
        GtkWidget* output_entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "output_entry"));
        GtkWidget* progress_bar = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "progress_bar"));
        GtkWidget* status_label = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "status_label"));
        
        // Get input and output paths
        std::string input_path = gtk_entry_get_text(GTK_ENTRY(input_entry));
        std::string output_path = gtk_entry_get_text(GTK_ENTRY(output_entry));
        
        // Validate inputs
        if (input_path.empty()) {
            gtk_label_set_text(GTK_LABEL(status_label), "Error: Please select an input video file");
            return;
        }
        
        if (output_path.empty()) {
            output_path = std::filesystem::path(input_path).stem().string() + ".wav";
            gtk_entry_set_text(GTK_ENTRY(output_entry), output_path.c_str());
        }
        
        // Set parameters
        AutoSubtitleParameters params;
        params.input_path = input_path;
        params.output_path = output_path;
        params.extract_audio_only = true;
        params.extracted_audio_path = output_path;
        
        // Update UI
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar), 0.0);
        gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), "추출 중...");
        gtk_label_set_text(GTK_LABEL(status_label), "오디오 추출 시작...");
        
        // Start processing
        auto_subtitle->generate_subtitles(
            params,
            [progress_bar](float progress) {
                g_idle_add([](gpointer user_data) -> gboolean {
                    auto* data = static_cast<std::pair<GtkWidget*, float>*>(user_data);
                    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(data->first), data->second);
                    delete data;
                    return FALSE;
                }, new std::pair<GtkWidget*, float>(progress_bar, progress));
            },
            [status_label, progress_bar](bool success, const std::string& message) {
                g_idle_add([](gpointer user_data) -> gboolean {
                    auto* data = static_cast<std::tuple<GtkWidget*, GtkWidget*, bool, std::string>*>(user_data);
                    GtkWidget* status_label = std::get<0>(*data);
                    GtkWidget* progress_bar = std::get<1>(*data);
                    bool success = std::get<2>(*data);
                    std::string message = std::get<3>(*data);
                    
                    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar), 1.0);
                    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), success ? "완료" : "실패");
                    gtk_label_set_text(GTK_LABEL(status_label), message.c_str());
                    
                    delete data;
                    return FALSE;
                }, new std::tuple<GtkWidget*, GtkWidget*, bool, std::string>(status_label, progress_bar, success, message));
            }
        );
    }), container);
    
    // Connect generate button
    g_signal_connect(generate_button, "clicked", G_CALLBACK([](GtkButton*, gpointer user_data) {
        GtkWidget* container = GTK_WIDGET(user_data);
        AutoSubtitle* auto_subtitle = static_cast<AutoSubtitle*>(g_object_get_data(G_OBJECT(container), "auto_subtitle"));
        
        // Get widget references
        GtkWidget* input_entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "input_entry"));
        GtkWidget* output_entry = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "output_entry"));
        GtkWidget* format_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "format_combo"));
        GtkWidget* lang_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "lang_combo"));
        GtkWidget* translation_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "translation_combo"));
        GtkWidget* target_lang_combo = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "target_lang_combo"));
        GtkWidget* confidence_check = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "confidence_check"));
        GtkWidget* profanity_check = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "profanity_check"));
        GtkWidget* merge_check = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "merge_check"));
        GtkWidget* split_check = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "split_check"));
        GtkWidget* scene_check = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "scene_check"));
        GtkWidget* max_chars_spin = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "max_chars_spin"));
        GtkWidget* progress_bar = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "progress_bar"));
        GtkWidget* status_label = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "status_label"));
        
        // Get input and output paths
        std::string input_path = gtk_entry_get_text(GTK_ENTRY(input_entry));
        std::string output_path = gtk_entry_get_text(GTK_ENTRY(output_entry));
        
        // Validate inputs
        if (input_path.empty()) {
            gtk_label_set_text(GTK_LABEL(status_label), "Error: Please select an input video file");
            return;
        }
        
        if (output_path.empty()) {
            gtk_label_set_text(GTK_LABEL(status_label), "Error: Please specify an output subtitle file");
            return;
        }
        
        // Set parameters
        AutoSubtitleParameters params;
        params.input_path = input_path;
        params.output_path = output_path;
        params.extract_audio_only = false;
        
        // Set format
        int format_index = gtk_combo_box_get_active(GTK_COMBO_BOX(format_combo));
        switch (format_index) {
            case 0: params.format = SubtitleFormat::SRT; break;
            case 1: params.format = SubtitleFormat::VTT; break;
            case 2: params.format = SubtitleFormat::ASS; break;
            case 3: params.format = SubtitleFormat::TXT; break;
            case 4: params.format = SubtitleFormat::JSON; break;
            default: params.format = SubtitleFormat::SRT; break;
        }
        
        // Set language
        int lang_index = gtk_combo_box_get_active(GTK_COMBO_BOX(lang_combo));
        switch (lang_index) {
            case 0: params.language = "auto"; break;
            case 1: params.language = "ko"; break;
            case 2: params.language = "en"; break;
            case 3: params.language = "ja"; break;
            case 4: params.language = "zh"; break;
            default: params.language = "auto"; break;
        }
        
        // Set translation mode
        int translation_index = gtk_combo_box_get_active(GTK_COMBO_BOX(translation_combo));
        switch (translation_index) {
            case 0: params.translation_mode = TranslationMode::NONE; break;
            case 1: params.translation_mode = TranslationMode::AUTO; break;
            case 2: params.translation_mode = TranslationMode::MANUAL; break;
            default: params.translation_mode = TranslationMode::NONE; break;
        }
        
        // Set target language if translation is enabled
        if (params.translation_mode != TranslationMode::NONE) {
            int target_lang_index = gtk_combo_box_get_active(GTK_COMBO_BOX(target_lang_combo));
            switch (target_lang_index) {
                case 0: params.target_language = "ko"; break;
                case 1: params.target_language = "en"; break;
                case 2: params.target_language = "ja"; break;
                case 3: params.target_language = "zh"; break;
                default: params.target_language = "en"; break;
            }
        }
        
        // Set other options
        params.include_confidence = gtk_check_button_get_active(GTK_CHECK_BUTTON(confidence_check));
        params.filter_profanity = gtk_check_button_get_active(GTK_CHECK_BUTTON(profanity_check));
        params.merge_short_segments = gtk_check_button_get_active(GTK_CHECK_BUTTON(merge_check));
        params.split_long_segments = gtk_check_button_get_active(GTK_CHECK_BUTTON(split_check));
        params.adjust_timing_to_scene_changes = gtk_check_button_get_active(GTK_CHECK_BUTTON(scene_check));
        params.max_chars_per_line = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(max_chars_spin));
        
        // Update UI
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar), 0.0);
        gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), "처리 중...");
        gtk_label_set_text(GTK_LABEL(status_label), "자막 생성 시작...");
        
        // Start processing
        auto_subtitle->generate_subtitles(
            params,
            [progress_bar](float progress) {
                g_idle_add([](gpointer user_data) -> gboolean {
                    auto* data = static_cast<std::pair<GtkWidget*, float>*>(user_data);
                    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(data->first), data->second);
                    delete data;
                    return FALSE;
                }, new std::pair<GtkWidget*, float>(progress_bar, progress));
            },
            [status_label, progress_bar](bool success, const std::string& message) {
                g_idle_add([](gpointer user_data) -> gboolean {
                    auto* data = static_cast<std::tuple<GtkWidget*, GtkWidget*, bool, std::string>*>(user_data);
                    GtkWidget* status_label = std::get<0>(*data);
                    GtkWidget* progress_bar = std::get<1>(*data);
                    bool success = std::get<2>(*data);
                    std::string message = std::get<3>(*data);
                    
                    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar), 1.0);
                    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(progress_bar), success ? "완료" : "실패");
                    gtk_label_set_text(GTK_LABEL(status_label), message.c_str());
                    
                    delete data;
                    return FALSE;
                }, new std::tuple<GtkWidget*, GtkWidget*, bool, std::string>(status_label, progress_bar, success, message));
            }
        );
    }), container);
    
    // Connect cancel button
    g_signal_connect(cancel_button, "clicked", G_CALLBACK([](GtkButton*, gpointer user_data) {
        GtkWidget* container = GTK_WIDGET(user_data);
        AutoSubtitle* auto_subtitle = static_cast<AutoSubtitle*>(g_object_get_data(G_OBJECT(container), "auto_subtitle"));
        GtkWidget* status_label = GTK_WIDGET(g_object_get_data(G_OBJECT(container), "status_label"));
        
        auto_subtitle->cancel();
        gtk_label_set_text(GTK_LABEL(status_label), "작업이 취소되었습니다.");
    }), container);
    
    return container;
}

} // namespace AI
} // namespace BlouEdit 