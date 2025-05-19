#include "music_generator.h"
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

MusicGenerator::MusicGenerator() : python_state(nullptr), is_initialized(false), is_processing_active(false) {
    // Initialize Python environment on construction
    is_initialized = initialize_python_environment();
}

MusicGenerator::~MusicGenerator() {
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

bool MusicGenerator::initialize() {
    if (!is_initialized || !python_state) {
        return initialize_python_environment();
    }
    return true;
}

bool MusicGenerator::initialize_python_environment() {
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
    PyObject* pName = PyUnicode_DecodeFSDefault("music_generator_module");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python music generator module" << std::endl;
        return false;
    }
    
    // Get the music generator engine class
    PyObject* pClass = PyObject_GetAttrString(pModule, "MusicGeneratorEngine");
    if (pClass == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to get music generator engine class" << std::endl;
        return false;
    }
    
    // Create an instance of the music generator engine
    PyObject* pEngine = PyObject_CallObject(pClass, nullptr);
    Py_DECREF(pClass);
    
    if (pEngine == nullptr) {
        PyErr_Print();
        Py_DECREF(pModule);
        std::cerr << "Failed to create music generator engine instance" << std::endl;
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

bool MusicGenerator::generate_music(
    const MusicGeneratorParameters& params,
    std::function<void(float)> progress_callback,
    std::function<void(bool, const std::string&)> completion_callback) {
    
    if (!is_initialized || !python_state) {
        std::cerr << "Python environment not initialized" << std::endl;
        if (completion_callback) completion_callback(false, "Python environment not initialized");
        return false;
    }
    
    if (is_processing_active) {
        std::cerr << "Music generation already in progress" << std::endl;
        if (completion_callback) completion_callback(false, "Music generation already in progress");
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
            
            // Set basic parameters
            PyDict_SetItemString(pDict, "output_path", PyUnicode_FromString(params.output_path.c_str()));
            PyDict_SetItemString(pDict, "duration_seconds", PyLong_FromLong(params.duration_seconds));
            
            // Set genre
            const char* genre_str;
            switch (params.genre) {
                case MusicGenre::AMBIENT: genre_str = "ambient"; break;
                case MusicGenre::CLASSICAL: genre_str = "classical"; break;
                case MusicGenre::ELECTRONIC: genre_str = "electronic"; break;
                case MusicGenre::HIP_HOP: genre_str = "hip_hop"; break;
                case MusicGenre::JAZZ: genre_str = "jazz"; break;
                case MusicGenre::LO_FI: genre_str = "lo_fi"; break;
                case MusicGenre::POP: genre_str = "pop"; break;
                case MusicGenre::ROCK: genre_str = "rock"; break;
                case MusicGenre::SOUNDTRACK: genre_str = "soundtrack"; break;
                case MusicGenre::CUSTOM: genre_str = "custom"; break;
                default: genre_str = "ambient";
            }
            PyDict_SetItemString(pDict, "genre", PyUnicode_FromString(genre_str));
            
            // Set mood
            const char* mood_str;
            switch (params.mood) {
                case MusicMood::HAPPY: mood_str = "happy"; break;
                case MusicMood::SAD: mood_str = "sad"; break;
                case MusicMood::CALM: mood_str = "calm"; break;
                case MusicMood::ENERGETIC: mood_str = "energetic"; break;
                case MusicMood::ROMANTIC: mood_str = "romantic"; break;
                case MusicMood::SUSPENSEFUL: mood_str = "suspenseful"; break;
                case MusicMood::EPIC: mood_str = "epic"; break;
                case MusicMood::PLAYFUL: mood_str = "playful"; break;
                case MusicMood::MYSTERIOUS: mood_str = "mysterious"; break;
                case MusicMood::DARK: mood_str = "dark"; break;
                default: mood_str = "calm";
            }
            PyDict_SetItemString(pDict, "mood", PyUnicode_FromString(mood_str));
            
            // Set mode
            const char* mode_str;
            switch (params.mode) {
                case GenerationMode::FULL_TRACK: mode_str = "full_track"; break;
                case GenerationMode::CONTINUATION: mode_str = "continuation"; break;
                case GenerationMode::ACCOMPANIMENT: mode_str = "accompaniment"; break;
                case GenerationMode::STEM_SEPARATION: mode_str = "stem_separation"; break;
                default: mode_str = "full_track";
            }
            PyDict_SetItemString(pDict, "mode", PyUnicode_FromString(mode_str));
            
            // Set other numerical parameters
            PyDict_SetItemString(pDict, "tempo_bpm", PyLong_FromLong(params.tempo_bpm));
            PyDict_SetItemString(pDict, "volume", PyFloat_FromDouble(params.volume));
            PyDict_SetItemString(pDict, "normalize_audio", PyBool_FromLong(params.normalize_audio ? 1 : 0));
            PyDict_SetItemString(pDict, "loop_friendly", PyBool_FromLong(params.loop_friendly ? 1 : 0));
            
            // Set path parameters
            PyDict_SetItemString(pDict, "input_audio_path", PyUnicode_FromString(params.input_audio_path.c_str()));
            PyDict_SetItemString(pDict, "text_prompt", PyUnicode_FromString(params.text_prompt.c_str()));
            PyDict_SetItemString(pDict, "reference_track_path", PyUnicode_FromString(params.reference_track_path.c_str()));
            PyDict_SetItemString(pDict, "custom_genre", PyUnicode_FromString(params.custom_genre.c_str()));
            
            // Set structure parameters
            PyDict_SetItemString(pDict, "intro_seconds", PyLong_FromLong(params.intro_seconds));
            PyDict_SetItemString(pDict, "outro_seconds", PyLong_FromLong(params.outro_seconds));
            PyDict_SetItemString(pDict, "include_drums", PyBool_FromLong(params.include_drums ? 1 : 0));
            PyDict_SetItemString(pDict, "include_bass", PyBool_FromLong(params.include_bass ? 1 : 0));
            PyDict_SetItemString(pDict, "include_melody", PyBool_FromLong(params.include_melody ? 1 : 0));
            PyDict_SetItemString(pDict, "include_chords", PyBool_FromLong(params.include_chords ? 1 : 0));
            
            // Set export parameters
            PyDict_SetItemString(pDict, "file_format", PyUnicode_FromString(params.file_format.c_str()));
            PyDict_SetItemString(pDict, "sample_rate", PyLong_FromLong(params.sample_rate));
            PyDict_SetItemString(pDict, "bit_depth", PyLong_FromLong(params.bit_depth));
            
            // Set keypoints if any
            if (!params.keypoints.empty()) {
                PyObject* pKeypoints = PyList_New(params.keypoints.size());
                for (size_t i = 0; i < params.keypoints.size(); i++) {
                    PyObject* pKeypoint = PyTuple_New(2);
                    PyTuple_SetItem(pKeypoint, 0, PyFloat_FromDouble(params.keypoints[i].first));
                    PyTuple_SetItem(pKeypoint, 1, PyUnicode_FromString(params.keypoints[i].second.c_str()));
                    PyList_SetItem(pKeypoints, i, pKeypoint);
                }
                PyDict_SetItemString(pDict, "keypoints", pKeypoints);
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
            
            // Call the generate_music method on the Python engine
            PyObject* pEngine = state->pEngine;
            PyObject* pGenerateMethod = PyObject_GetAttrString(pEngine, "generate_music");
            if (!pGenerateMethod) {
                PyErr_Print();
                throw std::runtime_error("Failed to get generate_music method");
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
            PyObject* pResult = PyObject_CallObject(pGenerateMethod, pArgs);
            Py_DECREF(pGenerateMethod);
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
                result_message = "Python error occurred during music generation";
            }
            
        } catch (const std::exception& e) {
            success = false;
            result_message = std::string("Exception during music generation: ") + e.what();
        } catch (...) {
            success = false;
            result_message = "Unknown exception during music generation";
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
}

void MusicGenerator::cancel() {
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

bool MusicGenerator::is_processing() const {
    return is_processing_active;
}

std::vector<std::string> MusicGenerator::get_available_models() const {
    std::vector<std::string> models;
    
    if (!is_initialized || !python_state) {
        return models;
    }
    
    PythonState* state = static_cast<PythonState*>(python_state);
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    // Call the get_available_models method on the Python engine
    PyObject* pResult = PyObject_CallMethod(state->pEngine, "get_available_models", "");
    if (pResult && PyList_Check(pResult)) {
        Py_ssize_t size = PyList_Size(pResult);
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* pItem = PyList_GetItem(pResult, i);
            if (PyUnicode_Check(pItem)) {
                models.push_back(PyUnicode_AsUTF8(pItem));
            }
        }
        Py_DECREF(pResult);
    }
    
    PyGILState_Release(gstate);
    return models;
}

GtkWidget* MusicGenerator::create_widget() {
    // Create a container for the UI
    GtkWidget* container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(container, 10);
    gtk_widget_set_margin_end(container, 10);
    gtk_widget_set_margin_top(container, 10);
    gtk_widget_set_margin_bottom(container, 10);
    
    // Title
    GtkWidget* title = gtk_label_new("AI 음악 생성기");
    PangoAttrList* attr_list = pango_attr_list_new();
    pango_attr_list_insert(attr_list, pango_attr_weight_new(PANGO_WEIGHT_BOLD));
    pango_attr_list_insert(attr_list, pango_attr_scale_new(1.2));
    gtk_label_set_attributes(GTK_LABEL(title), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_append(GTK_BOX(container), title);
    
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
    
    // Basic parameters frame
    GtkWidget* basic_frame = gtk_frame_new("기본 설정");
    GtkWidget* basic_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(basic_box, 5);
    gtk_widget_set_margin_end(basic_box, 5);
    gtk_widget_set_margin_top(basic_box, 5);
    gtk_widget_set_margin_bottom(basic_box, 5);
    
    // Duration
    GtkWidget* duration_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* duration_label = gtk_label_new("길이 (초):");
    GtkWidget* duration_spin = gtk_spin_button_new_with_range(5, 300, 5);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(duration_spin), 30);
    gtk_widget_set_hexpand(duration_spin, TRUE);
    gtk_box_append(GTK_BOX(duration_box), duration_label);
    gtk_box_append(GTK_BOX(duration_box), duration_spin);
    gtk_box_append(GTK_BOX(basic_box), duration_box);
    
    // Genre selection
    GtkWidget* genre_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* genre_label = gtk_label_new("장르:");
    GtkWidget* genre_combo = gtk_combo_box_text_new();
    
    // Add genres
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Ambient");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Classical");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Electronic");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Hip Hop");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Jazz");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Lo-Fi");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Pop");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Rock");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Soundtrack");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(genre_combo), NULL, "Custom");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(genre_combo), 0);
    gtk_widget_set_hexpand(genre_combo, TRUE);
    gtk_box_append(GTK_BOX(genre_box), genre_label);
    gtk_box_append(GTK_BOX(genre_box), genre_combo);
    gtk_box_append(GTK_BOX(basic_box), genre_box);
    
    // Custom genre entry (initially hidden)
    GtkWidget* custom_genre_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* custom_genre_label = gtk_label_new("사용자 정의 장르:");
    GtkWidget* custom_genre_entry = gtk_entry_new();
    gtk_widget_set_hexpand(custom_genre_entry, TRUE);
    gtk_box_append(GTK_BOX(custom_genre_box), custom_genre_label);
    gtk_box_append(GTK_BOX(custom_genre_box), custom_genre_entry);
    gtk_box_append(GTK_BOX(basic_box), custom_genre_box);
    gtk_widget_set_visible(custom_genre_box, FALSE);
    
    // Mood selection
    GtkWidget* mood_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* mood_label = gtk_label_new("분위기:");
    GtkWidget* mood_combo = gtk_combo_box_text_new();
    
    // Add moods
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Happy");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Sad");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Calm");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Energetic");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Romantic");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Suspenseful");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Epic");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Playful");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Mysterious");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mood_combo), NULL, "Dark");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(mood_combo), 2); // Calm as default
    gtk_widget_set_hexpand(mood_combo, TRUE);
    gtk_box_append(GTK_BOX(mood_box), mood_label);
    gtk_box_append(GTK_BOX(mood_box), mood_combo);
    gtk_box_append(GTK_BOX(basic_box), mood_box);
    
    // Tempo slider
    GtkWidget* tempo_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* tempo_label = gtk_label_new("템포 (BPM):");
    GtkWidget* tempo_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 60, 200, 1);
    gtk_range_set_value(GTK_RANGE(tempo_scale), 120);
    gtk_widget_set_hexpand(tempo_scale, TRUE);
    gtk_box_append(GTK_BOX(tempo_box), tempo_label);
    gtk_box_append(GTK_BOX(tempo_box), tempo_scale);
    gtk_box_append(GTK_BOX(basic_box), tempo_box);
    
    // Complete the basic frame
    gtk_frame_set_child(GTK_FRAME(basic_frame), basic_box);
    gtk_box_append(GTK_BOX(container), basic_frame);
    
    // Advanced parameters frame
    GtkWidget* advanced_frame = gtk_frame_new("고급 설정");
    GtkWidget* advanced_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(advanced_box, 5);
    gtk_widget_set_margin_end(advanced_box, 5);
    gtk_widget_set_margin_top(advanced_box, 5);
    gtk_widget_set_margin_bottom(advanced_box, 5);
    
    // Generation mode
    GtkWidget* mode_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* mode_label = gtk_label_new("생성 모드:");
    GtkWidget* mode_combo = gtk_combo_box_text_new();
    
    // Add modes
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mode_combo), NULL, "Full Track");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mode_combo), NULL, "Continuation");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mode_combo), NULL, "Accompaniment");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(mode_combo), NULL, "Stem Separation");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(mode_combo), 0);
    gtk_widget_set_hexpand(mode_combo, TRUE);
    gtk_box_append(GTK_BOX(mode_box), mode_label);
    gtk_box_append(GTK_BOX(mode_box), mode_combo);
    gtk_box_append(GTK_BOX(advanced_box), mode_box);
    
    // Text prompt
    GtkWidget* prompt_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    GtkWidget* prompt_label = gtk_label_new("텍스트 프롬프트:");
    gtk_widget_set_halign(prompt_label, GTK_ALIGN_START);
    GtkWidget* prompt_entry = gtk_text_view_new();
    gtk_widget_set_size_request(prompt_entry, -1, 60);
    
    GtkWidget* prompt_scroll = gtk_scrolled_window_new();
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(prompt_scroll), prompt_entry);
    
    gtk_box_append(GTK_BOX(prompt_box), prompt_label);
    gtk_box_append(GTK_BOX(prompt_box), prompt_scroll);
    gtk_box_append(GTK_BOX(advanced_box), prompt_box);
    
    // Input audio path (for continuation/accompaniment)
    GtkWidget* input_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* input_label = gtk_label_new("입력 오디오:");
    GtkWidget* input_entry = gtk_entry_new();
    GtkWidget* input_button = gtk_button_new_with_label("찾아보기");
    gtk_widget_set_hexpand(input_entry, TRUE);
    gtk_box_append(GTK_BOX(input_box), input_label);
    gtk_box_append(GTK_BOX(input_box), input_entry);
    gtk_box_append(GTK_BOX(input_box), input_button);
    gtk_box_append(GTK_BOX(advanced_box), input_box);
    gtk_widget_set_sensitive(input_box, FALSE); // Initially disabled
    
    // Loop-friendly checkbox
    GtkWidget* loop_check = gtk_check_button_new_with_label("루프 가능한 음악 만들기");
    gtk_box_append(GTK_BOX(advanced_box), loop_check);
    
    // Normalize audio checkbox
    GtkWidget* normalize_check = gtk_check_button_new_with_label("오디오 정규화");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(normalize_check), TRUE);
    gtk_box_append(GTK_BOX(advanced_box), normalize_check);
    
    // Volume setting
    GtkWidget* volume_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* volume_label = gtk_label_new("볼륨:");
    GtkWidget* volume_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 2.0, 0.1);
    gtk_range_set_value(GTK_RANGE(volume_scale), 1.0);
    gtk_widget_set_hexpand(volume_scale, TRUE);
    gtk_box_append(GTK_BOX(volume_box), volume_label);
    gtk_box_append(GTK_BOX(volume_box), volume_scale);
    gtk_box_append(GTK_BOX(advanced_box), volume_box);
    
    // Complete the advanced frame
    gtk_frame_set_child(GTK_FRAME(advanced_frame), advanced_box);
    gtk_box_append(GTK_BOX(container), advanced_frame);
    
    // Structure parameters frame
    GtkWidget* structure_frame = gtk_frame_new("음악 구조");
    GtkWidget* structure_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(structure_box, 5);
    gtk_widget_set_margin_end(structure_box, 5);
    gtk_widget_set_margin_top(structure_box, 5);
    gtk_widget_set_margin_bottom(structure_box, 5);
    
    // Intro and outro duration
    GtkWidget* intro_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* intro_label = gtk_label_new("인트로 길이 (초):");
    GtkWidget* intro_spin = gtk_spin_button_new_with_range(0, 30, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(intro_spin), 4);
    gtk_widget_set_hexpand(intro_spin, TRUE);
    gtk_box_append(GTK_BOX(intro_box), intro_label);
    gtk_box_append(GTK_BOX(intro_box), intro_spin);
    gtk_box_append(GTK_BOX(structure_box), intro_box);
    
    GtkWidget* outro_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* outro_label = gtk_label_new("아웃트로 길이 (초):");
    GtkWidget* outro_spin = gtk_spin_button_new_with_range(0, 30, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(outro_spin), 4);
    gtk_widget_set_hexpand(outro_spin, TRUE);
    gtk_box_append(GTK_BOX(outro_box), outro_label);
    gtk_box_append(GTK_BOX(outro_box), outro_spin);
    gtk_box_append(GTK_BOX(structure_box), outro_box);
    
    // Instrument checkboxes
    GtkWidget* instruments_label = gtk_label_new("포함할 악기:");
    gtk_widget_set_halign(instruments_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(structure_box), instruments_label);
    
    GtkWidget* drums_check = gtk_check_button_new_with_label("드럼");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(drums_check), TRUE);
    gtk_box_append(GTK_BOX(structure_box), drums_check);
    
    GtkWidget* bass_check = gtk_check_button_new_with_label("베이스");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(bass_check), TRUE);
    gtk_box_append(GTK_BOX(structure_box), bass_check);
    
    GtkWidget* melody_check = gtk_check_button_new_with_label("멜로디");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(melody_check), TRUE);
    gtk_box_append(GTK_BOX(structure_box), melody_check);
    
    GtkWidget* chords_check = gtk_check_button_new_with_label("코드");
    gtk_check_button_set_active(GTK_CHECK_BUTTON(chords_check), TRUE);
    gtk_box_append(GTK_BOX(structure_box), chords_check);
    
    // Complete the structure frame
    gtk_frame_set_child(GTK_FRAME(structure_frame), structure_box);
    gtk_box_append(GTK_BOX(container), structure_frame);
    
    // Export parameters frame
    GtkWidget* export_frame = gtk_frame_new("내보내기 설정");
    GtkWidget* export_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(export_box, 5);
    gtk_widget_set_margin_end(export_box, 5);
    gtk_widget_set_margin_top(export_box, 5);
    gtk_widget_set_margin_bottom(export_box, 5);
    
    // File format
    GtkWidget* format_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* format_label = gtk_label_new("파일 형식:");
    GtkWidget* format_combo = gtk_combo_box_text_new();
    
    // Add formats
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "WAV");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "MP3");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(format_combo), NULL, "OGG");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(format_combo), 0);
    gtk_widget_set_hexpand(format_combo, TRUE);
    gtk_box_append(GTK_BOX(format_box), format_label);
    gtk_box_append(GTK_BOX(format_box), format_combo);
    gtk_box_append(GTK_BOX(export_box), format_box);
    
    // Sample rate
    GtkWidget* sample_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* sample_label = gtk_label_new("샘플링 레이트:");
    GtkWidget* sample_combo = gtk_combo_box_text_new();
    
    // Add sample rates
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(sample_combo), NULL, "44100 Hz");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(sample_combo), NULL, "48000 Hz");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(sample_combo), NULL, "96000 Hz");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(sample_combo), 0);
    gtk_widget_set_hexpand(sample_combo, TRUE);
    gtk_box_append(GTK_BOX(sample_box), sample_label);
    gtk_box_append(GTK_BOX(sample_box), sample_combo);
    gtk_box_append(GTK_BOX(export_box), sample_box);
    
    // Bit depth
    GtkWidget* bit_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* bit_label = gtk_label_new("비트 심도:");
    GtkWidget* bit_combo = gtk_combo_box_text_new();
    
    // Add bit depths
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(bit_combo), NULL, "16 bit");
    gtk_combo_box_text_append(GTK_COMBO_BOX_TEXT(bit_combo), NULL, "24 bit");
    
    gtk_combo_box_set_active(GTK_COMBO_BOX(bit_combo), 0);
    gtk_widget_set_hexpand(bit_combo, TRUE);
    gtk_box_append(GTK_BOX(bit_box), bit_label);
    gtk_box_append(GTK_BOX(bit_box), bit_combo);
    gtk_box_append(GTK_BOX(export_box), bit_box);
    
    // Complete the export frame
    gtk_frame_set_child(GTK_FRAME(export_frame), export_box);
    gtk_box_append(GTK_BOX(container), export_frame);
    
    return container;
}
}
} 