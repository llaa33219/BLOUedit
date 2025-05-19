#include "text_to_speech.h"
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
    PyObject* pTtsEngine;
};

TextToSpeech::TextToSpeech() : python_state(nullptr), is_initialized(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

TextToSpeech::~TextToSpeech() {
    if (python_state) {
        PythonState* state = static_cast<PythonState*>(python_state);
        Py_XDECREF(state->pModule);
        Py_XDECREF(state->pTtsEngine);
        delete state;
        
        // Don't finalize Python here as it might be used by other modules
        // Py_Finalize();
    }
}

bool TextToSpeech::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool TextToSpeech::initialize_python_environment() {
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
    PyObject* pName = PyUnicode_DecodeFSDefault("tts_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python TTS module" << std::endl;
        return false;
    }
    
    // Get the TTS engine class
    PyObject* pTtsClass = PyObject_GetAttrString(pModule, "TextToSpeechEngine");
    if (pTtsClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get TTS engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the TTS engine
    PyObject* pTtsEngine = PyObject_CallObject(pTtsClass, nullptr);
    Py_DECREF(pTtsClass);
    
    if (pTtsEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create TTS engine instance" << std::endl;
        return false;
    }
    
    // Store state
    PythonState* state = new PythonState();
    state->pModule = pModule;
    state->pTtsEngine = pTtsEngine;
    python_state = state;
    
    return true;
}

std::vector<TtsVoice> TextToSpeech::get_available_voices(const std::string& language_code) {
    std::vector<TtsVoice> voices;
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        return voices;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Call get_available_voices method
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(language_code.c_str()));
    
    PyObject* pResult = PyObject_CallMethod(state->pTtsEngine, "get_available_voices", "O", pArgs);
    Py_DECREF(pArgs);
    
    if (pResult == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to get available voices" << std::endl;
        return voices;
    }
    
    // Parse the result (list of voice dictionaries)
    if (PyList_Check(pResult)) {
        Py_ssize_t size = PyList_Size(pResult);
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* pVoice = PyList_GetItem(pResult, i);
            if (PyDict_Check(pVoice)) {
                TtsVoice voice;
                
                // Get id
                PyObject* pId = PyDict_GetItemString(pVoice, "id");
                if (pId && PyUnicode_Check(pId)) {
                    voice.id = PyUnicode_AsUTF8(pId);
                }
                
                // Get name
                PyObject* pName = PyDict_GetItemString(pVoice, "name");
                if (pName && PyUnicode_Check(pName)) {
                    voice.name = PyUnicode_AsUTF8(pName);
                }
                
                // Get gender
                PyObject* pGender = PyDict_GetItemString(pVoice, "gender");
                if (pGender && PyUnicode_Check(pGender)) {
                    std::string gender = PyUnicode_AsUTF8(pGender);
                    if (gender == "MALE") {
                        voice.gender = TtsVoiceGender::MALE;
                    } else if (gender == "FEMALE") {
                        voice.gender = TtsVoiceGender::FEMALE;
                    } else {
                        voice.gender = TtsVoiceGender::NEUTRAL;
                    }
                }
                
                // Get model
                PyObject* pModel = PyDict_GetItemString(pVoice, "model");
                if (pModel && PyUnicode_Check(pModel)) {
                    std::string model = PyUnicode_AsUTF8(pModel);
                    if (model == "NEURAL") {
                        voice.model = TtsVoiceModel::NEURAL;
                    } else if (model == "CUSTOM") {
                        voice.model = TtsVoiceModel::CUSTOM;
                    } else {
                        voice.model = TtsVoiceModel::STANDARD;
                    }
                }
                
                // Get language code
                PyObject* pLangCode = PyDict_GetItemString(pVoice, "language_code");
                if (pLangCode && PyUnicode_Check(pLangCode)) {
                    voice.language_code = PyUnicode_AsUTF8(pLangCode);
                }
                
                voices.push_back(voice);
            }
        }
    }
    
    Py_DECREF(pResult);
    
    return voices;
}

bool TextToSpeech::synthesize_speech(
    const TtsParameters& params,
    const std::string& output_path,
    std::function<void(bool, const std::string&)> callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (callback) callback(false, "Python environment not initialized");
        return false;
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::path output_dir = std::filesystem::path(output_path).parent_path();
    if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    
    // Create dictionary of parameters
    PyObject* pDict = PyDict_New();
    PyDict_SetItemString(pDict, "text", PyUnicode_FromString(params.text.c_str()));
    PyDict_SetItemString(pDict, "voice_id", PyUnicode_FromString(params.voice.id.c_str()));
    PyDict_SetItemString(pDict, "speaking_rate", PyFloat_FromDouble(params.speaking_rate));
    PyDict_SetItemString(pDict, "pitch", PyFloat_FromDouble(params.pitch));
    PyDict_SetItemString(pDict, "volume_gain_db", PyFloat_FromDouble(params.volume_gain_db));
    PyDict_SetItemString(pDict, "add_timestamps", PyBool_FromLong(params.add_timestamps ? 1 : 0));
    
    // Call synthesize_speech method
    PyObject* pResult = PyObject_CallMethod(
        state->pTtsEngine, "synthesize_speech", "Os",
        pDict, output_path.c_str()
    );
    
    Py_DECREF(pDict);
    
    if (pResult == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to synthesize speech" << std::endl;
        if (callback) callback(false, "Failed to synthesize speech");
        return false;
    }
    
    bool success = PyObject_IsTrue(pResult);
    Py_DECREF(pResult);
    
    if (callback) {
        if (success) {
            callback(true, output_path);
        } else {
            callback(false, "Speech synthesis failed");
        }
    }
    
    return success;
}

GtkWidget* TextToSpeech::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 텍스트 → 음성 변환");
    PangoAttrList* attr_list = pango_attr_list_new();
    pango_attr_list_insert(attr_list, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attr_list, pango_attr_scale_new(1.2));
    gtk_label_set_attributes(GTK_LABEL(title), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_append(GTK_BOX(container), title);
    
    // Text input
    GtkWidget* text_frame = gtk_frame_new("텍스트 입력");
    GtkWidget* text_view = gtk_text_view_new();
    GtkWidget* text_scrolled = gtk_scrolled_window_new();
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(text_scrolled), text_view);
    gtk_widget_set_size_request(text_scrolled, -1, 150);
    gtk_frame_set_child(GTK_FRAME(text_frame), text_scrolled);
    gtk_box_append(GTK_BOX(container), text_frame);
    
    // Voice selection
    GtkWidget* voice_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* voice_label = gtk_label_new("음성:");
    GtkWidget* voice_combo = gtk_combo_box_text_new();
    
    // Add some default voices for UI preview
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(voice_combo), NULL, "Korean Male (ko-KR)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(voice_combo), NULL, "Korean Female (ko-KR)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(voice_combo), NULL, "English US Male (en-US)");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(voice_combo), NULL, "English US Female (en-US)");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(voice_combo), 0);
    gtk_widget_set_hexpand(voice_combo, TRUE);
    gtk_box_append(GTK_BOX(voice_box), voice_label);
    gtk_box_append(GTK_BOX(voice_box), voice_combo);
    gtk_box_append(GTK_BOX(container), voice_box);
    
    // Settings section
    GtkWidget* settings_frame = gtk_frame_new("음성 설정");
    GtkWidget* settings_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_frame_set_child(GTK_FRAME(settings_frame), settings_box);
    gtk_box_append(GTK_BOX(container), settings_frame);
    
    // Speaking rate
    GtkWidget* rate_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* rate_label = gtk_label_new("말하기 속도:");
    GtkWidget* rate_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.5, 2.0, 0.1);
    gtk_range_set_value(GTK_RANGE(rate_scale), 1.0);
    gtk_widget_set_hexpand(rate_scale, TRUE);
    gtk_box_append(GTK_BOX(rate_box), rate_label);
    gtk_box_append(GTK_BOX(rate_box), rate_scale);
    gtk_box_append(GTK_BOX(settings_box), rate_box);
    
    // Pitch
    GtkWidget* pitch_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* pitch_label = gtk_label_new("음높이:");
    GtkWidget* pitch_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, -10.0, 10.0, 0.5);
    gtk_range_set_value(GTK_RANGE(pitch_scale), 0.0);
    gtk_widget_set_hexpand(pitch_scale, TRUE);
    gtk_box_append(GTK_BOX(pitch_box), pitch_label);
    gtk_box_append(GTK_BOX(pitch_box), pitch_scale);
    gtk_box_append(GTK_BOX(settings_box), pitch_box);
    
    // Volume
    GtkWidget* volume_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* volume_label = gtk_label_new("볼륨:");
    GtkWidget* volume_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, -6.0, 6.0, 0.5);
    gtk_range_set_value(GTK_RANGE(volume_scale), 0.0);
    gtk_widget_set_hexpand(volume_scale, TRUE);
    gtk_box_append(GTK_BOX(volume_box), volume_label);
    gtk_box_append(GTK_BOX(volume_box), volume_scale);
    gtk_box_append(GTK_BOX(settings_box), volume_box);
    
    // Add timestamps checkbox
    GtkWidget* timestamps_check = gtk_check_button_new_with_label("타임스탬프 추가");
    gtk_box_append(GTK_BOX(settings_box), timestamps_check);
    
    // Output file
    GtkWidget* output_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* output_label = gtk_label_new("출력 파일:");
    GtkWidget* output_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(output_entry), "audio.wav");
    GtkWidget* output_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(output_entry, TRUE);
    gtk_box_append(GTK_BOX(output_box), output_label);
    gtk_box_append(GTK_BOX(output_box), output_entry);
    gtk_box_append(GTK_BOX(output_box), output_button);
    gtk_box_append(GTK_BOX(container), output_box);
    
    // Generate button
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* generate_button = gtk_button_new_with_label("음성 생성");
    gtk_widget_set_hexpand(generate_button, TRUE);
    gtk_box_append(GTK_BOX(button_box), generate_button);
    gtk_box_append(GTK_BOX(container), button_box);
    
    // Audio player
    GtkWidget* player_frame = gtk_frame_new("미리 듣기");
    GtkWidget* player_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    
    GtkWidget* play_controls = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* play_button = gtk_button_new_from_icon_name("media-playback-start");
    GtkWidget* stop_button = gtk_button_new_from_icon_name("media-playback-stop");
    GtkWidget* progress_bar = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 100.0, 1.0);
    gtk_widget_set_hexpand(progress_bar, TRUE);
    
    gtk_box_append(GTK_BOX(play_controls), play_button);
    gtk_box_append(GTK_BOX(play_controls), stop_button);
    gtk_box_append(GTK_BOX(play_controls), progress_bar);
    
    gtk_box_append(GTK_BOX(player_box), play_controls);
    gtk_frame_set_child(GTK_FRAME(player_frame), player_box);
    gtk_box_append(GTK_BOX(container), player_frame);
    
    // TODO: Connect signals to callbacks
    
    return container;
}

} // namespace AI
} // namespace BlouEdit 