#include "speech_to_text.h"
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
    PyObject* pSttEngine;
};

SpeechToText::SpeechToText() : python_state(nullptr), is_initialized(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

SpeechToText::~SpeechToText() {
    if (python_state) {
        PythonState* state = static_cast<PythonState*>(python_state);
        Py_XDECREF(state->pModule);
        Py_XDECREF(state->pSttEngine);
        delete state;
        
        // Don't finalize Python here as it might be used by other modules
        // Py_Finalize();
    }
}

bool SpeechToText::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool SpeechToText::initialize_python_environment() {
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
    PyObject* pName = PyUnicode_DecodeFSDefault("stt_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python STT module" << std::endl;
        return false;
    }
    
    // Get the STT engine class
    PyObject* pSttClass = PyObject_GetAttrString(pModule, "SpeechToTextEngine");
    if (pSttClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get STT engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the STT engine
    PyObject* pSttEngine = PyObject_CallObject(pSttClass, nullptr);
    Py_DECREF(pSttClass);
    
    if (pSttEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create STT engine instance" << std::endl;
        return false;
    }
    
    // Store state
    PythonState* state = new PythonState();
    state->pModule = pModule;
    state->pSttEngine = pSttEngine;
    python_state = state;
    
    return true;
}

bool SpeechToText::transcribe(
    const SttParameters& params,
    SttResult& result,
    std::function<void(bool, const SttResult&)> callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (callback) callback(false, result);
        return false;
    }
    
    if (!std::filesystem::exists(params.audio_path)) {
        std::cerr << "Audio file not found: " << params.audio_path << std::endl;
        if (callback) callback(false, result);
        return false;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Create dictionary of parameters
    PyObject* pDict = PyDict_New();
    PyDict_SetItemString(pDict, "audio_path", PyUnicode_FromString(params.audio_path.c_str()));
    PyDict_SetItemString(pDict, "language", PyUnicode_FromString(params.language.c_str()));
    PyDict_SetItemString(pDict, "add_timestamps", PyBool_FromLong(params.add_timestamps ? 1 : 0));
    PyDict_SetItemString(pDict, "add_punctuation", PyBool_FromLong(params.add_punctuation ? 1 : 0));
    
    // Convert engine enum to string
    const char* engine_str;
    switch (params.engine) {
        case SttEngine::WHISPER:
            engine_str = "whisper";
            break;
        case SttEngine::GOOGLE:
            engine_str = "google";
            break;
        case SttEngine::AZURE:
            engine_str = "azure";
            break;
        case SttEngine::VOSK:
            engine_str = "vosk";
            break;
        default:
            engine_str = "whisper";
    }
    PyDict_SetItemString(pDict, "engine", PyUnicode_FromString(engine_str));
    
    // Call transcribe method
    PyObject* pResult = PyObject_CallMethod(
        state->pSttEngine, "transcribe", "O",
        pDict
    );
    
    Py_DECREF(pDict);
    
    if (pResult == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to transcribe speech" << std::endl;
        if (callback) callback(false, result);
        return false;
    }
    
    // Check if result is a tuple or dict (success, result) or (result_dict)
    bool success = true;
    PyObject* pResultData = nullptr;
    
    if (PyTuple_Check(pResult) && PyTuple_Size(pResult) == 2) {
        // (success, result) format
        success = PyObject_IsTrue(PyTuple_GetItem(pResult, 0));
        pResultData = PyTuple_GetItem(pResult, 1);
    } else if (PyDict_Check(pResult)) {
        // Just the result dictionary
        pResultData = pResult;
    } else {
        Py_DECREF(pResult);
        std::cerr << "Unexpected result format from STT engine" << std::endl;
        if (callback) callback(false, result);
        return false;
    }
    
    // Parse the result dictionary
    if (success && pResultData && PyDict_Check(pResultData)) {
        // Get the full text
        PyObject* pText = PyDict_GetItemString(pResultData, "text");
        if (pText && PyUnicode_Check(pText)) {
            result.text = PyUnicode_AsUTF8(pText);
        }
        
        // Get segments
        PyObject* pSegments = PyDict_GetItemString(pResultData, "segments");
        if (pSegments && PyList_Check(pSegments)) {
            Py_ssize_t size = PyList_Size(pSegments);
            result.segments.clear();
            result.timestamps.clear();
            result.confidence.clear();
            
            for (Py_ssize_t i = 0; i < size; i++) {
                PyObject* pSegment = PyList_GetItem(pSegments, i);
                if (PyDict_Check(pSegment)) {
                    // Get segment text
                    PyObject* pSegText = PyDict_GetItemString(pSegment, "text");
                    if (pSegText && PyUnicode_Check(pSegText)) {
                        result.segments.push_back(PyUnicode_AsUTF8(pSegText));
                    } else {
                        result.segments.push_back("");
                    }
                    
                    // Get start and end time
                    PyObject* pStart = PyDict_GetItemString(pSegment, "start");
                    PyObject* pEnd = PyDict_GetItemString(pSegment, "end");
                    double start = 0.0, end = 0.0;
                    
                    if (pStart && PyFloat_Check(pStart)) {
                        start = PyFloat_AsDouble(pStart);
                    } else if (pStart && PyLong_Check(pStart)) {
                        start = PyLong_AsDouble(pStart);
                    }
                    
                    if (pEnd && PyFloat_Check(pEnd)) {
                        end = PyFloat_AsDouble(pEnd);
                    } else if (pEnd && PyLong_Check(pEnd)) {
                        end = PyLong_AsDouble(pEnd);
                    }
                    
                    result.timestamps.push_back(std::make_pair(start, end));
                    
                    // Get confidence
                    PyObject* pConf = PyDict_GetItemString(pSegment, "confidence");
                    float confidence = 1.0f;
                    
                    if (pConf && PyFloat_Check(pConf)) {
                        confidence = (float)PyFloat_AsDouble(pConf);
                    }
                    
                    result.confidence.push_back(confidence);
                }
            }
        }
    }
    
    if (pResult != pResultData) {
        Py_DECREF(pResult);
    } else {
        Py_DECREF(pResultData);
    }
    
    if (callback) {
        callback(success, result);
    }
    
    return success;
}

GtkWidget* SpeechToText::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 음성 → 텍스트 변환");
    PangoAttrList* attr_list = pango_attr_list_new();
    pango_attr_list_insert(attr_list, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attr_list, pango_attr_scale_new(1.2));
    gtk_label_set_attributes(GTK_LABEL(title), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_append(GTK_BOX(container), title);
    
    // Audio file selection
    GtkWidget* audio_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* audio_label = gtk_label_new("오디오 파일:");
    GtkWidget* audio_entry = gtk_entry_new();
    GtkWidget* audio_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(audio_entry, TRUE);
    gtk_box_append(GTK_BOX(audio_box), audio_label);
    gtk_box_append(GTK_BOX(audio_box), audio_entry);
    gtk_box_append(GTK_BOX(audio_box), audio_button);
    gtk_box_append(GTK_BOX(container), audio_box);
    
    // Engine selection
    GtkWidget* engine_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* engine_label = gtk_label_new("엔진:");
    GtkWidget* engine_combo = gtk_combo_box_text_new();
    
    // Add available engines
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(engine_combo), NULL, "Whisper (OpenAI)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(engine_combo), NULL, "Google Speech-to-Text");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(engine_combo), NULL, "Azure Speech Service");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(engine_combo), NULL, "Vosk (오프라인)");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(engine_combo), 0);
    gtk_widget_set_hexpand(engine_combo, TRUE);
    gtk_box_append(GTK_BOX(engine_box), engine_label);
    gtk_box_append(GTK_BOX(engine_box), engine_combo);
    gtk_box_append(GTK_BOX(container), engine_box);
    
    // Language selection
    GtkWidget* lang_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* lang_label = gtk_label_new("언어:");
    GtkWidget* lang_combo = gtk_combo_box_text_new();
    
    // Add common languages
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "자동 감지");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "한국어 (ko-KR)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "영어 (en-US)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "일본어 (ja-JP)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(lang_combo), NULL, "중국어 (zh-CN)");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(lang_combo), 0);
    gtk_widget_set_hexpand(lang_combo, TRUE);
    gtk_box_append(GTK_BOX(lang_box), lang_label);
    gtk_box_append(GTK_BOX(lang_box), lang_combo);
    gtk_box_append(GTK_BOX(container), lang_box);
    
    // Options
    GtkWidget* options_frame = gtk_frame_new("옵션");
    GtkWidget* options_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    
    // Add timestamps checkbox
    GtkWidget* timestamps_check = gtk_check_button_new_with_label("타임스탬프 추가");
    gtk_box_append(GTK_BOX(options_box), timestamps_check);
    
    // Add punctuation checkbox
    GtkWidget* punctuation_check = gtk_check_button_new_with_label("구두점 추가");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(punctuation_check), TRUE);
    gtk_box_append(GTK_BOX(options_box), punctuation_check);
    
    gtk_frame_set_child(GTK_FRAME(options_frame), options_box);
    gtk_box_append(GTK_BOX(container), options_frame);
    
    // Transcribe button
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* transcribe_button = gtk_button_new_with_label("음성 인식 시작");
    gtk_widget_set_hexpand(transcribe_button, TRUE);
    gtk_box_append(GTK_BOX(button_box), transcribe_button);
    gtk_box_append(GTK_BOX(container), button_box);
    
    // Result text area
    GtkWidget* result_frame = gtk_frame_new("인식 결과");
    GtkWidget* result_scroll = gtk_scrolled_window_new();
    GtkWidget* result_text = gtk_text_view_new();
    
    gtk_text_view_set_editable(GTK_TEXT_VIEW(result_text), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(result_text), GTK_WRAP_WORD);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(result_scroll), result_text);
    gtk_widget_set_size_request(result_scroll, -1, 200);
    gtk_frame_set_child(GTK_FRAME(result_frame), result_scroll);
    gtk_box_append(GTK_BOX(container), result_frame);
    
    // Export button
    GtkWidget* export_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* export_button = gtk_button_new_with_label("텍스트로 내보내기");
    GtkWidget* export_subtitle_button = gtk_button_new_with_label("자막으로 내보내기");
    gtk_widget_set_hexpand(export_button, TRUE);
    gtk_widget_set_hexpand(export_subtitle_button, TRUE);
    gtk_box_append(GTK_BOX(export_box), export_button);
    gtk_box_append(GTK_BOX(export_box), export_subtitle_button);
    gtk_box_append(GTK_BOX(container), export_box);
    
    // TODO: Connect signals to callbacks
    
    return container;
}

} // namespace AI
} // namespace BlouEdit 