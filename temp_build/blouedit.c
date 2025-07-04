#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>
#include <gst/pbutils/pbutils.h>
#include <locale.h>

// 타임라인 클립 구조체
typedef struct {
    char *filename;
    char *display_name;
    int track_index;
    int start_position;
    int duration;
    gboolean selected;
    GtkWidget *widget;
    
    // 클립 편집 속성들
    double volume;
    double speed;
    gboolean muted;
} TimelineClip;

typedef struct {
    GtkWidget *main_window;
    
    // CapCut 스타일 UI 컴포넌트
    GtkWidget *toolbar;      // 상단 툴바
    GtkWidget *left_panel;   // 좌측 패널 (미디어/효과/스티커/텍스트)
    GtkWidget *center_panel; // 중앙 패널 (미리보기)
    GtkWidget *right_panel;  // 우측 패널 (속성 편집)
    GtkWidget *bottom_panel; // 하단 패널 (타임라인)
    
    // 미디어 패널 컴포넌트
    GtkWidget *media_stack;
    GtkWidget *media_library;
    GtkWidget *effects_library;
    GtkWidget *text_library;
    GtkWidget *stickers_library;
    
    // AI 기능 컴포넌트
    GtkWidget *ai_stack;
    GtkWidget *ai_tools;
    
    // 고급 비디오 기능 컴포넌트
    GtkWidget *advanced_video_stack;
    GtkWidget *advanced_video_tools;
    
    // 오디오 편집 기능 컴포넌트
    GtkWidget *audio_stack;
    GtkWidget *audio_tools;
    
    // 타임라인 컴포넌트
    GtkWidget *timeline;
    GtkWidget **timeline_tracks; // 동적 트랙 배열
    int track_count;          // 트랙 수
    GtkAdjustment *timeline_adj;
    GtkWidget *timeline_scale;
    GtkWidget *add_track_button; // 트랙 추가 버튼
    
    // 미리보기 컴포넌트
    GtkWidget *preview;
    GtkWidget *preview_controls;
    
    // 속성 편집 컴포넌트
    GtkWidget *properties_panel;
    
    // 버튼들
    GtkWidget *open_button;
    GtkWidget *new_button;
    GtkWidget *play_button;
    GtkWidget *export_button;
    GtkWidget *split_button;
    GtkWidget *delete_button;
    GtkWidget *proxy_button;       // 프록시 워크플로우 버튼
    GtkWidget *performance_button; // 성능 모드 버튼
    
    // 기존 컴포넌트들
    GtkFileChooserDialog *file_chooser;
    GtkListStore *media_store;
    GtkTreeView *media_view;
    
    // GStreamer 관련
    GstElement *pipeline;
    GstElement *source;
    GstElement *sink;
    gulong video_window_handle;
    gboolean playing;
    
    // 타임라인 클립 관리
    TimelineClip *clips[50];
    int clip_count;
    TimelineClip *selected_clip;
    
    // 미디어 파일 관리
    char *loaded_files[50];
    int file_count;
    
    // 프로젝트 정보
    int project_duration; // 밀리초 단위
    int timeline_scale_value; // 타임라인 확대/축소 값
    char *project_name;
} BlouEditApp;

// Global app instance for callbacks
BlouEditApp *global_app = NULL;

// 함수 선언들
static void add_media_file(BlouEditApp *app, const char *filename);
static void on_clip_clicked(GtkGestureClick *gesture, int n_press, double x, double y, gpointer user_data);
static void on_clip_drag_begin(GtkGestureDrag *gesture, double start_x, double start_y, gpointer user_data);
static void on_clip_drag_update(GtkGestureDrag *gesture, double offset_x, double offset_y, gpointer user_data);
static void on_clip_drag_end(GtkGestureDrag *gesture, double offset_x, double offset_y, gpointer user_data);
static void create_media_library(BlouEditApp *app);
static void on_file_chooser_response(GtkDialog *dialog, int response, gpointer user_data);
static void on_open_clicked(GtkButton *button, gpointer user_data);
static void on_play_clicked(GtkButton *button, gpointer user_data);
static void on_export_clicked(GtkButton *button, gpointer user_data);
static void on_split_clicked(GtkButton *button, gpointer user_data);
static void on_delete_clicked(GtkButton *button, gpointer user_data);
static void on_proxy_clicked(GtkButton *button, gpointer user_data);
static void on_performance_clicked(GtkButton *button, gpointer user_data);
static void on_dialog_response(GtkDialog *dialog, int response, gpointer user_data);
static void on_new_project_clicked(GtkButton *button, gpointer user_data);
static void update_preview(BlouEditApp *app, TimelineClip *clip);
static void select_clip(BlouEditApp *app, TimelineClip *clip);
static void deselect_all_clips(BlouEditApp *app);
static TimelineClip* create_timeline_clip(BlouEditApp *app, const char *filename, int track);
static void delete_selected_clip(BlouEditApp *app);
static void create_export_pipeline(BlouEditApp *app, const char *output_file);
static void on_pad_added(GstElement *element, GstPad *pad, gpointer data);
static void add_timeline_track(BlouEditApp *app);
static void on_add_track_clicked(GtkButton *button, gpointer user_data);
static void on_manage_tracks_clicked(GtkButton *button, gpointer user_data);
static void show_track_properties_dialog(BlouEditApp *app);

// 클립 생성자 함수
static TimelineClip* create_timeline_clip(BlouEditApp *app, const char *filename, int track) {
    TimelineClip *clip = g_new0(TimelineClip, 1);
    clip->filename = g_strdup(filename);
    clip->display_name = g_strdup(g_path_get_basename(filename));
    clip->track_index = track;
    clip->start_position = 0; // 트랙 내 시작 위치는 나중에 계산
    clip->duration = 5000; // 기본값 5초 (나중에 미디어 파일에서 추출)
    clip->selected = FALSE;
    clip->volume = 1.0;
    clip->speed = 1.0;
    clip->muted = FALSE;
    
    // 클립의 UI 위젯 생성
    clip->widget = gtk_button_new_with_label(clip->display_name);
    gtk_widget_set_size_request(clip->widget, clip->duration / 50, 30); // 시간에 비례한 너비
    
    // 클립 스타일 설정
    const char *track_colors[] = {
        "button { background: #3498db; color: white; }",
        "button { background: #e74c3c; color: white; }",
        "button { background: #2ecc71; color: white; }"
    };
    
    GtkCssProvider *provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(provider, track_colors[track % 3], -1);
    gtk_style_context_add_provider(
        gtk_widget_get_style_context(clip->widget),
        GTK_STYLE_PROVIDER(provider),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
    );
    g_object_unref(provider);
    
    // 클릭 제스처 추가
    GtkGestureClick *click_gesture = gtk_gesture_click_new();
    g_signal_connect(click_gesture, "pressed", G_CALLBACK(on_clip_clicked), clip);
    gtk_widget_add_controller(clip->widget, GTK_EVENT_CONTROLLER(click_gesture));
    
    // 드래그 제스처 추가
    GtkGestureDrag *drag_gesture = gtk_gesture_drag_new();
    g_signal_connect(drag_gesture, "drag-begin", G_CALLBACK(on_clip_drag_begin), clip);
    g_signal_connect(drag_gesture, "drag-update", G_CALLBACK(on_clip_drag_update), clip);
    g_signal_connect(drag_gesture, "drag-end", G_CALLBACK(on_clip_drag_end), clip);
    gtk_widget_add_controller(clip->widget, GTK_EVENT_CONTROLLER(drag_gesture));
    
    return clip;
}

// 클립 선택 핸들러
static void on_clip_clicked(GtkGestureClick *gesture, int n_press, double x, double y, gpointer user_data) {
    if (!user_data) {
        g_print("Error: Null clip data in on_clip_clicked\n");
        return;
    }

    TimelineClip *clip = (TimelineClip *)user_data;
    BlouEditApp *app = global_app; // 전역 앱 인스턴스 사용
    
    if (!app) {
        g_print("Error: Global app instance is null\n");
        return;
    }
    
    // 기존 선택 해제
    deselect_all_clips(app);
    
    // 새 클립 선택
    select_clip(app, clip);
    
    // 미리보기 업데이트
    update_preview(app, clip);
    
    g_print("클립 선택: %s\n", clip->display_name);
}

// 클립 드래그 시작 핸들러
static void on_clip_drag_begin(GtkGestureDrag *gesture, double start_x, double start_y, gpointer user_data) {
    if (!user_data) {
        g_print("Error: Null clip data in on_clip_drag_begin\n");
        return;
    }

    TimelineClip *clip = (TimelineClip *)user_data;
    BlouEditApp *app = global_app;

    if (!app) {
        g_print("Error: Global app instance is null\n");
        return;
    }
    
    // 선택되지 않았다면 먼저 선택
    if (!clip->selected) {
        deselect_all_clips(app);
        select_clip(app, clip);
    }
    
    g_print("클립 드래그 시작: %s\n", clip->display_name);
}

// 클립 드래그 업데이트 핸들러
static void on_clip_drag_update(GtkGestureDrag *gesture, double offset_x, double offset_y, gpointer user_data) {
    if (!user_data) {
        g_print("Error: Null clip data in on_clip_drag_update\n");
        return;
    }

    TimelineClip *clip = (TimelineClip *)user_data;
    BlouEditApp *app = global_app;

    if (!app) {
        g_print("Error: Global app instance is null\n");
        return;
    }
    
    // 타임라인 스케일 값 계산 (픽셀당 시간)
    int scale = app->timeline_scale_value > 0 ? app->timeline_scale_value : 50;
    
    // 마진 값 계산 (타임라인에서의 위치), 직접 오프셋 적용
    int new_margin = MAX(0, clip->start_position / scale + (int)offset_x);
    
    // UI 업데이트
    gtk_widget_set_margin_start(clip->widget, new_margin);
    
    g_print("클립 드래그 중: %s, 오프셋: %.0f\n", clip->display_name, offset_x);
}

// 클립 드래그 종료 핸들러
static void on_clip_drag_end(GtkGestureDrag *gesture, double offset_x, double offset_y, gpointer user_data) {
    if (!user_data) {
        g_print("Error: Null clip data in on_clip_drag_end\n");
        return;
    }

    TimelineClip *clip = (TimelineClip *)user_data;
    BlouEditApp *app = global_app;

    if (!app) {
        g_print("Error: Global app instance is null\n");
        return;
    }
    
    // 타임라인 스케일 값 계산 (픽셀당 시간)
    int scale = app->timeline_scale_value > 0 ? app->timeline_scale_value : 50;
    
    // 새 위치 계산 및 저장 - 드래그 업데이트와 동일한 방식 사용
    int new_position = MAX(0, clip->start_position + (int)(offset_x * scale));
    clip->start_position = new_position;
    
    g_print("클립 드래그 종료: %s, 새 위치: %d\n", clip->display_name, new_position);
}

// 모든 클립 선택 해제
static void deselect_all_clips(BlouEditApp *app) {
    if (!app) {
        g_print("Error: Null app in deselect_all_clips\n");
        return;
    }
    
    if (app->selected_clip) {
        app->selected_clip->selected = FALSE;
        
        // 선택 해제 시 스타일 변경
        const char *track_colors[] = {
            "button { background: #3498db; color: white; }",
            "button { background: #e74c3c; color: white; }",
            "button { background: #2ecc71; color: white; }",
            "button { background: #f1c40f; color: white; }",
            "button { background: #9b59b6; color: white; }"
        };
        
        if (!app->selected_clip->widget) {
            g_print("Warning: Selected clip has no widget\n");
            app->selected_clip = NULL;
            return;
        }
        
        int track_index = app->selected_clip->track_index;
        if (track_index < 0) {
            track_index = 0;
        }
        
        GtkCssProvider *provider = gtk_css_provider_new();
        gtk_css_provider_load_from_data(provider, track_colors[track_index % 5], -1);
        gtk_style_context_add_provider(
            gtk_widget_get_style_context(app->selected_clip->widget),
            GTK_STYLE_PROVIDER(provider),
            GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
        );
        g_object_unref(provider);
        
        app->selected_clip = NULL;
    }
}

// 클립 선택
static void select_clip(BlouEditApp *app, TimelineClip *clip) {
    if (!app) {
        g_print("Error: Null app in select_clip\n");
        return;
    }
    
    if (!clip) {
        g_print("Error: Null clip in select_clip\n");
        return;
    }
    
    if (!clip->widget) {
        g_print("Error: Clip has no widget in select_clip\n");
        return;
    }
    
    app->selected_clip = clip;
    clip->selected = TRUE;
    
    // 선택 스타일
    GtkCssProvider *provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(provider, 
        "button { background: #f39c12; color: white; border: 2px solid white; }", -1);
    gtk_style_context_add_provider(
        gtk_widget_get_style_context(clip->widget),
        GTK_STYLE_PROVIDER(provider),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
    );
    g_object_unref(provider);
    
    // 속성 패널 업데이트 (추후 구현)
}

// 미리보기 업데이트
static void update_preview(BlouEditApp *app, TimelineClip *clip) {
    if (!app) {
        g_print("Error: Null app in update_preview\n");
        return;
    }
    
    if (!clip || !clip->filename) {
        g_print("Warning: Invalid clip or filename in update_preview\n");
        return;
    }
    
    // 기존 파이프라인 정리
    if (app->pipeline) {
        gst_element_set_state(app->pipeline, GST_STATE_NULL);
        g_object_unref(app->pipeline);
    }
    
    // 새 파이프라인 생성
    app->pipeline = gst_pipeline_new("video-player");
    app->source = gst_element_factory_make("filesrc", "file-source");
    GstElement *decode = gst_element_factory_make("decodebin", "decoder");
    app->sink = gst_element_factory_make("gtksink", "video-output");
    
    if (!app->pipeline || !app->source || !decode || !app->sink) {
        g_print("Failed to create elements\n");
        return;
    }
    
    g_object_set(G_OBJECT(app->source), "location", clip->filename, NULL);
    
    gst_bin_add_many(GST_BIN(app->pipeline), app->source, decode, app->sink, NULL);
    gst_element_link(app->source, decode);
    
    // Get the video widget from gtksink and add it to preview
    GtkWidget *video_widget;
    g_object_get(app->sink, "widget", &video_widget, NULL);
    
    // Remove any existing widget in preview
    GtkWidget *child = gtk_widget_get_first_child(app->preview);
    if (child)
        gtk_widget_unparent(child);
        
    gtk_box_append(GTK_BOX(app->preview), video_widget);
    
    // Start playing
    gst_element_set_state(app->pipeline, GST_STATE_PLAYING);
    app->playing = TRUE;
    
    // 재생 버튼 아이콘 업데이트
    gtk_button_set_icon_name(GTK_BUTTON(app->play_button), "media-playback-pause");
}

// 선택된 클립 삭제
static void delete_selected_clip(BlouEditApp *app) {
    if (!app->selected_clip) {
        return;
    }
    
    TimelineClip *clip = app->selected_clip;
    
    // 위젯 제거
    gtk_widget_unparent(clip->widget);
    
    // 클립 배열에서 제거
    for (int i = 0; i < app->clip_count; i++) {
        if (app->clips[i] == clip) {
            // 배열 앞으로 이동
            for (int j = i; j < app->clip_count - 1; j++) {
                app->clips[j] = app->clips[j + 1];
            }
            app->clip_count--;
            break;
        }
    }
    
    // 메모리 정리
    g_free(clip->filename);
    g_free(clip->display_name);
    g_free(clip);
    
    app->selected_clip = NULL;
    
    g_print("클립 삭제됨\n");
}

// 클립 분할 (향후 구현)
static void split_clip(BlouEditApp *app, TimelineClip *clip, int position) {
    g_print("클립 분할 기능은 아직 구현되지 않았습니다.\n");
}

// 내보내기 파이프라인 생성
static void create_export_pipeline(BlouEditApp *app, const char *output_file) {
    if (app->clip_count == 0) {
        g_print("내보낼 클립이 없습니다.\n");
        return;
    }
    
    // 간단한 내보내기: 첫 번째 클립만 처리
    GstElement *pipeline = gst_pipeline_new("export-pipeline");
    GstElement *source = gst_element_factory_make("filesrc", "file-source");
    GstElement *decode = gst_element_factory_make("decodebin", "decoder");
    GstElement *convert = gst_element_factory_make("videoconvert", "converter");
    GstElement *encode = gst_element_factory_make("x264enc", "encoder");
    GstElement *mux = gst_element_factory_make("mp4mux", "muxer");
    GstElement *sink = gst_element_factory_make("filesink", "file-sink");
    
    if (!pipeline || !source || !decode || !convert || !encode || !mux || !sink) {
        g_print("Failed to create export elements\n");
        return;
    }
    
    g_object_set(G_OBJECT(source), "location", app->clips[0]->filename, NULL);
    g_object_set(G_OBJECT(sink), "location", output_file, NULL);
    
    gst_bin_add_many(GST_BIN(pipeline), source, decode, convert, encode, mux, sink, NULL);
    gst_element_link(source, decode);
    gst_element_link_many(convert, encode, mux, sink, NULL);
    
    // On pad-added from decodebin
    g_signal_connect(decode, "pad-added", G_CALLBACK(on_pad_added), convert);
    
    // 내보내기 시작
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    g_print("내보내기 시작: %s\n", output_file);
    
    // 진행 상황 모니터링 (간단한 구현)
    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, 
        GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
    
    if (msg != NULL) {
        gst_message_unref(msg);
    }
    
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    
    g_print("내보내기 완료: %s\n", output_file);
}

// GStreamer 디코더 패드 추가 콜백
static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstElement *target = (GstElement *)data;
    GstPad *sink_pad = gst_element_get_static_pad(target, "sink");
    
    if (gst_pad_link(pad, sink_pad) != GST_PAD_LINK_OK) {
        g_print("Failed to link decoder with converter\n");
    }
    
    gst_object_unref(sink_pad);
}

// 내보내기 대화상자 응답 처리
static void on_export_dialog_response(GtkDialog *dialog, int response, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp *)user_data;
    
    if (response == GTK_RESPONSE_ACCEPT) {
        GFile *file = gtk_file_chooser_get_file(GTK_FILE_CHOOSER(dialog));
        char *path = g_file_get_path(file);
        
        // 내보내기 파이프라인 생성 및 시작
        create_export_pipeline(app, path);
        
        g_free(path);
        g_object_unref(file);
    }
    
    gtk_window_destroy(GTK_WINDOW(dialog));
}

static void add_media_file(BlouEditApp *app, const char *filename) {
    if (!app || !filename) {
        g_print("Error: Invalid arguments to add_media_file\n");
        return;
    }
    
    // 미디어 파일 추가
    app->loaded_files[app->file_count] = g_strdup(filename);
    app->file_count++;
    
    // 미디어 라이브러리에 추가
    GtkTreeIter iter;
    gtk_list_store_append(app->media_store, &iter);
    gtk_list_store_set(app->media_store, &iter, 0, g_path_get_basename(filename), 1, filename, -1);
    
    // 타임라인 클립 생성
    int track = 0;
    if (app->track_count > 0) {
        // 클립을 생성할 트랙을 선택 (순환식으로)
        track = app->clip_count % app->track_count;
    } else {
        g_print("Warning: No tracks available, cannot add clip\n");
        return;
    }
    
    // 타임라인 클립 생성
    TimelineClip *clip = create_timeline_clip(app, filename, track);
    
    // 겹치지 않게 위치 설정
    int position = 0;
    for (int i = 0; i < app->clip_count; i++) {
        TimelineClip *other = app->clips[i];
        if (other->track_index == track) {
            position = MAX(position, other->start_position + other->duration);
        }
    }
    clip->start_position = position;
    
    // UI 반영
    gtk_widget_set_margin_start(clip->widget, position / 50);
    
    // 타임라인에 추가
    app->clips[app->clip_count++] = clip;
    
    // UI에 추가
    if (track < app->track_count) {
        gtk_box_append(GTK_BOX(app->timeline_tracks[track]), clip->widget);
    } else {
        g_print("Error: Invalid track index %d (max: %d)\n", track, app->track_count - 1);
    }
}

// 미디어 아이템 더블클릭 핸들러
static void media_item_activated(GtkTreeView *tree_view, GtkTreePath *path, 
                                GtkTreeViewColumn *column, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp *)user_data;
    
    GtkTreeIter iter;
    gchar *filename;
    GtkTreeModel *model = gtk_tree_view_get_model(tree_view);
    
    if (gtk_tree_model_get_iter(model, &iter, path)) {
        gtk_tree_model_get(model, &iter, 1, &filename, -1);
        
        // Check if we have any tracks
        if (app->track_count <= 0) {
            g_print("Error: Cannot add clip - no tracks available\n");
            g_free(filename);
            return;
        }
        
        // 클립 생성하고 타임라인에 추가
        int track = app->clip_count % app->track_count; // 트랙 순환식으로 추가
        TimelineClip *clip = create_timeline_clip(app, filename, track);
        
        // 겹치지 않게 위치 설정
        int position = 0;
        for (int i = 0; i < app->clip_count; i++) {
            TimelineClip *other = app->clips[i];
            if (other->track_index == track) {
                position = MAX(position, other->start_position + other->duration);
            }
        }
        clip->start_position = position;
        
        // UI 반영
        gtk_widget_set_margin_start(clip->widget, position / 50);
        
        // 클립 배열에 추가
        app->clips[app->clip_count++] = clip;
        
        // UI에 추가
        gtk_box_append(GTK_BOX(app->timeline_tracks[track]), clip->widget);
        
        g_free(filename);
    }
}

static void create_media_library(BlouEditApp *app) {
    // 스크롤 가능한 영역 생성
    GtkWidget *scrolled = gtk_scrolled_window_new();
    gtk_widget_set_vexpand(scrolled, TRUE);
    
    // 미디어 파일 목록 저장소
    app->media_store = gtk_list_store_new(2, G_TYPE_STRING, G_TYPE_STRING);
    app->media_view = GTK_TREE_VIEW(gtk_tree_view_new_with_model(GTK_TREE_MODEL(app->media_store)));
    
    // 컬럼 설정
    GtkTreeViewColumn *column = gtk_tree_view_column_new_with_attributes(
        "미디어 파일", 
        gtk_cell_renderer_text_new(),
        "text", 0,
        NULL);
    
    gtk_tree_view_append_column(app->media_view, column);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled), GTK_WIDGET(app->media_view));
    
    // 미디어 라이브러리 컨테이너에 추가
    gtk_box_append(GTK_BOX(app->media_library), scrolled);
    
    // 더블클릭 시 파일 타임라인에 추가
    g_signal_connect(app->media_view, "row-activated", 
                    G_CALLBACK(media_item_activated), app);
}

// 파일 선택 대화상자 응답 핸들러
static void on_file_chooser_response(GtkDialog *dialog, int response, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    if (response == GTK_RESPONSE_ACCEPT) {
        GFile *file = gtk_file_chooser_get_file(GTK_FILE_CHOOSER(dialog));
        char *filename = g_file_get_path(file);
        
        add_media_file(app, filename);
        
        g_free(filename);
        g_object_unref(file);
    }
    
    gtk_window_destroy(GTK_WINDOW(dialog));
}

static void on_open_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    app->file_chooser = GTK_FILE_CHOOSER_DIALOG(
        gtk_file_chooser_dialog_new("미디어 파일 열기",
                                  GTK_WINDOW(app->main_window),
                                  GTK_FILE_CHOOSER_ACTION_OPEN,
                                  "취소", GTK_RESPONSE_CANCEL,
                                  "열기", GTK_RESPONSE_ACCEPT,
                                  NULL));
    
    // 비디오 파일 필터
    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "비디오 파일");
    gtk_file_filter_add_mime_type(filter, "video/*");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(app->file_chooser), filter);
    
    // 오디오 파일 필터
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "오디오 파일");
    gtk_file_filter_add_mime_type(filter, "audio/*");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(app->file_chooser), filter);
    
    // 이미지 파일 필터
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "이미지 파일");
    gtk_file_filter_add_mime_type(filter, "image/*");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(app->file_chooser), filter);
    
    // 모든 파일 필터
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "모든 파일");
    gtk_file_filter_add_pattern(filter, "*");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(app->file_chooser), filter);
    
    // 대화상자 표시
    gtk_window_present(GTK_WINDOW(app->file_chooser));
    
    // 응답 핸들러 연결
    g_signal_connect(app->file_chooser, "response", 
                    G_CALLBACK(on_file_chooser_response), app);
}

static void on_play_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    if (app->pipeline) {
        if (app->playing) {
            // 일시정지
            gst_element_set_state(app->pipeline, GST_STATE_PAUSED);
            gtk_button_set_icon_name(button, "media-playback-start");
            app->playing = FALSE;
        } else {
            // 재생
            gst_element_set_state(app->pipeline, GST_STATE_PLAYING);
            gtk_button_set_icon_name(button, "media-playback-pause");
            app->playing = TRUE;
        }
    }
}

// 내보내기 버튼 클릭
static void on_export_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    // 저장 대화상자 생성
    GtkFileChooserDialog *chooser = GTK_FILE_CHOOSER_DIALOG(
        gtk_file_chooser_dialog_new("내보내기",
                                  GTK_WINDOW(app->main_window),
                                  GTK_FILE_CHOOSER_ACTION_SAVE,
                                  "취소", GTK_RESPONSE_CANCEL,
                                  "저장", GTK_RESPONSE_ACCEPT,
                                  NULL));
    
    // 기본 파일 이름 설정
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(chooser), "내보내기.mp4");
    
    // 대화상자 표시
    gtk_window_present(GTK_WINDOW(chooser));
    
    // 응답 핸들러
    g_signal_connect(chooser, "response", G_CALLBACK(on_export_dialog_response), app);
}

// 분할 버튼
static void on_split_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    if (app->selected_clip) {
        // 현재 선택된 클립의 중간 지점에서 분할
        int position = app->selected_clip->duration / 2;
        split_clip(app, app->selected_clip, position);
    } else {
        // 알림 메시지
        GtkWidget *dialog = gtk_message_dialog_new(
            GTK_WINDOW(app->main_window),
            GTK_DIALOG_MODAL,
            GTK_MESSAGE_INFO,
            GTK_BUTTONS_OK,
            "분할할 클립을 선택해 주세요."
        );
        gtk_window_present(GTK_WINDOW(dialog));
        g_signal_connect(dialog, "response", G_CALLBACK(on_dialog_response), NULL);
    }
}

// 삭제 버튼
static void on_delete_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    if (app->selected_clip) {
        delete_selected_clip(app);
    } else {
        // 알림 메시지
        GtkWidget *dialog = gtk_message_dialog_new(
            GTK_WINDOW(app->main_window),
            GTK_DIALOG_MODAL,
            GTK_MESSAGE_INFO,
            GTK_BUTTONS_OK,
            "삭제할 클립을 선택해 주세요."
        );
        gtk_window_present(GTK_WINDOW(dialog));
        g_signal_connect(dialog, "response", G_CALLBACK(on_dialog_response), NULL);
    }
}

// 프록시 워크플로우 버튼 핸들러
static void on_proxy_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    // 프록시 설정 대화상자 생성
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "프록시 워크플로우 설정",
        GTK_WINDOW(app->main_window),
        GTK_DIALOG_MODAL | GTK_DIALOG_USE_HEADER_BAR,
        "취소", GTK_RESPONSE_CANCEL,
        "확인", GTK_RESPONSE_ACCEPT,
        NULL
    );
    
    // 대화상자 내용 영역 가져오기
    GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_widget_set_margin_start(content_area, 12);
    gtk_widget_set_margin_end(content_area, 12);
    gtk_widget_set_margin_top(content_area, 12);
    gtk_widget_set_margin_bottom(content_area, 12);
    
    // 프록시 설정 위젯 추가
    GtkWidget *proxy_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    
    // 프록시 활성화 스위치
    GtkWidget *enable_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *enable_label = gtk_label_new("프록시 워크플로우 활성화");
    gtk_widget_set_halign(enable_label, GTK_ALIGN_START);
    gtk_widget_set_hexpand(enable_label, TRUE);
    
    GtkWidget *enable_switch = gtk_switch_new();
    gtk_widget_set_halign(enable_switch, GTK_ALIGN_END);
    
    gtk_box_append(GTK_BOX(enable_box), enable_label);
    gtk_box_append(GTK_BOX(enable_box), enable_switch);
    gtk_box_append(GTK_BOX(proxy_box), enable_box);
    
    // 자동 프록시 생성 스위치
    GtkWidget *auto_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *auto_label = gtk_label_new("고해상도 파일 자동 프록시 생성");
    gtk_widget_set_halign(auto_label, GTK_ALIGN_START);
    gtk_widget_set_hexpand(auto_label, TRUE);
    
    GtkWidget *auto_switch = gtk_switch_new();
    gtk_widget_set_halign(auto_switch, GTK_ALIGN_END);
    
    gtk_box_append(GTK_BOX(auto_box), auto_label);
    gtk_box_append(GTK_BOX(auto_box), auto_switch);
    gtk_box_append(GTK_BOX(proxy_box), auto_box);
    
    // 프록시 해상도 설정
    GtkWidget *res_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *res_label = gtk_label_new("프록시 해상도");
    gtk_widget_set_halign(res_label, GTK_ALIGN_START);
    
    GtkWidget *res_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "원본 크기의 1/2");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "원본 크기의 1/4");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "720p");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "480p");
    gtk_combo_box_set_active(GTK_COMBO_BOX(res_combo), 2); // 기본값: 720p
    
    gtk_box_append(GTK_BOX(res_box), res_label);
    gtk_box_append(GTK_BOX(res_box), res_combo);
    gtk_box_append(GTK_BOX(proxy_box), res_box);
    
    // 프록시 형식 설정
    GtkWidget *format_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *format_label = gtk_label_new("프록시 형식");
    gtk_widget_set_halign(format_label, GTK_ALIGN_START);
    
    GtkWidget *format_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(format_combo), "H.264");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(format_combo), "ProRes");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(format_combo), "DNxHD");
    gtk_combo_box_set_active(GTK_COMBO_BOX(format_combo), 0); // 기본값: H.264
    
    gtk_box_append(GTK_BOX(format_box), format_label);
    gtk_box_append(GTK_BOX(format_box), format_combo);
    gtk_box_append(GTK_BOX(proxy_box), format_box);
    
    // 프록시 저장 위치
    GtkWidget *path_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *path_label = gtk_label_new("저장 위치");
    gtk_widget_set_halign(path_label, GTK_ALIGN_START);
    
    GtkWidget *path_entry = gtk_entry_new();
    gtk_editable_set_text(GTK_EDITABLE(path_entry), "./proxies");
    
    GtkWidget *path_button = gtk_button_new_with_label("찾아보기");
    
    gtk_box_append(GTK_BOX(path_box), path_label);
    gtk_box_append(GTK_BOX(path_box), path_entry);
    gtk_box_append(GTK_BOX(path_box), path_button);
    gtk_box_append(GTK_BOX(proxy_box), path_box);
    
    // 프록시 생성 버튼
    GtkWidget *gen_button = gtk_button_new_with_label("선택한 클립에 대한 프록시 생성");
    gtk_box_append(GTK_BOX(proxy_box), gen_button);
    
    // 설명 레이블
    GtkWidget *info_label = gtk_label_new("프록시 워크플로우는 편집 중 성능을 향상시키기 위해 고화질 원본 대신 저해상도 미디어 파일을 사용합니다.");
    gtk_label_set_wrap(GTK_LABEL(info_label), TRUE);
    gtk_box_append(GTK_BOX(proxy_box), info_label);
    
    // 모든 위젯을 대화상자에 추가
    gtk_box_append(GTK_BOX(content_area), proxy_box);
    
    // 대화상자 표시 및 응답 처리
    gtk_widget_set_size_request(dialog, 500, -1);
    gtk_window_present(GTK_WINDOW(dialog));
    g_signal_connect(dialog, "response", G_CALLBACK(on_dialog_response), NULL);
}

// 성능 모드 버튼 핸들러
static void on_performance_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    // 성능 모드 설정 대화상자 생성
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "성능 모드 설정",
        GTK_WINDOW(app->main_window),
        GTK_DIALOG_MODAL | GTK_DIALOG_USE_HEADER_BAR,
        "취소", GTK_RESPONSE_CANCEL,
        "확인", GTK_RESPONSE_ACCEPT,
        NULL
    );
    
    // 대화상자 내용 영역 가져오기
    GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_widget_set_margin_start(content_area, 12);
    gtk_widget_set_margin_end(content_area, 12);
    gtk_widget_set_margin_top(content_area, 12);
    gtk_widget_set_margin_bottom(content_area, 12);
    
    // 성능 모드 설정 위젯 추가
    GtkWidget *perf_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    
    // 성능 모드 선택 옵션
    GtkWidget *mode_label = gtk_label_new("성능 모드 선택");
    gtk_widget_set_halign(mode_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(perf_box), mode_label);
    
    // 라디오 버튼 그룹
    GtkWidget *quality_radio = gtk_check_button_new_with_label("품질 우선 (고화질 재생)");
    GtkWidget *balanced_radio = gtk_check_button_new_with_label("균형 모드 (기본값)");
    GtkWidget *performance_radio = gtk_check_button_new_with_label("성능 우선 (부드러운 재생)");
    
    gtk_check_button_set_group(GTK_CHECK_BUTTON(balanced_radio), GTK_CHECK_BUTTON(quality_radio));
    gtk_check_button_set_group(GTK_CHECK_BUTTON(performance_radio), GTK_CHECK_BUTTON(quality_radio));
    
    // 기본값 설정
    gtk_check_button_set_active(GTK_CHECK_BUTTON(balanced_radio), TRUE);
    
    gtk_box_append(GTK_BOX(perf_box), quality_radio);
    gtk_box_append(GTK_BOX(perf_box), balanced_radio);
    gtk_box_append(GTK_BOX(perf_box), performance_radio);
    
    // 재생 해상도 설정
    GtkWidget *res_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *res_label = gtk_label_new("재생 해상도");
    gtk_widget_set_halign(res_label, GTK_ALIGN_START);
    gtk_widget_set_hexpand(res_label, TRUE);
    
    GtkWidget *res_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "최대 해상도");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "원본 크기의 1/2");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "원본 크기의 1/4");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(res_combo), "원본 크기의 1/8");
    gtk_combo_box_set_active(GTK_COMBO_BOX(res_combo), 0); // 기본값: 최대 해상도
    
    gtk_box_append(GTK_BOX(res_box), res_label);
    gtk_box_append(GTK_BOX(res_box), res_combo);
    gtk_box_append(GTK_BOX(perf_box), res_box);
    
    // 캐시 설정
    GtkWidget *cache_frame = gtk_frame_new("타임라인 렌더링 캐시");
    GtkWidget *cache_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_margin_start(cache_box, 12);
    gtk_widget_set_margin_end(cache_box, 12);
    gtk_widget_set_margin_top(cache_box, 12);
    gtk_widget_set_margin_bottom(cache_box, 12);
    
    // 캐시 활성화 스위치
    GtkWidget *cache_enable_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *cache_enable_label = gtk_label_new("타임라인 렌더링 캐시 활성화");
    gtk_widget_set_halign(cache_enable_label, GTK_ALIGN_START);
    gtk_widget_set_hexpand(cache_enable_label, TRUE);
    
    GtkWidget *cache_enable_switch = gtk_switch_new();
    gtk_widget_set_halign(cache_enable_switch, GTK_ALIGN_END);
    gtk_switch_set_active(GTK_SWITCH(cache_enable_switch), TRUE); // 기본값: 활성화
    
    gtk_box_append(GTK_BOX(cache_enable_box), cache_enable_label);
    gtk_box_append(GTK_BOX(cache_enable_box), cache_enable_switch);
    gtk_box_append(GTK_BOX(cache_box), cache_enable_box);
    
    // 캐시 크기 설정
    GtkWidget *cache_size_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *cache_size_label = gtk_label_new("최대 캐시 크기 (GB)");
    gtk_widget_set_halign(cache_size_label, GTK_ALIGN_START);
    gtk_widget_set_hexpand(cache_size_label, TRUE);
    
    GtkAdjustment *cache_adj = gtk_adjustment_new(
        10.0,  // 값
        1.0,   // 최소
        100.0, // 최대
        1.0,   // 증가량
        5.0,   // 페이지 증가량
        0.0    // 페이지 크기
    );
    
    GtkWidget *cache_spin = gtk_spin_button_new(cache_adj, 1.0, 0);
    gtk_widget_set_halign(cache_spin, GTK_ALIGN_END);
    
    gtk_box_append(GTK_BOX(cache_size_box), cache_size_label);
    gtk_box_append(GTK_BOX(cache_size_box), cache_spin);
    gtk_box_append(GTK_BOX(cache_box), cache_size_box);
    
    // 캐시 위치 설정
    GtkWidget *cache_path_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *cache_path_label = gtk_label_new("캐시 저장 위치");
    gtk_widget_set_halign(cache_path_label, GTK_ALIGN_START);
    
    GtkWidget *cache_path_entry = gtk_entry_new();
    gtk_editable_set_text(GTK_EDITABLE(cache_path_entry), "./cache");
    
    GtkWidget *cache_path_button = gtk_button_new_with_label("찾아보기");
    
    gtk_box_append(GTK_BOX(cache_path_box), cache_path_label);
    gtk_box_append(GTK_BOX(cache_path_box), cache_path_entry);
    gtk_box_append(GTK_BOX(cache_path_box), cache_path_button);
    gtk_box_append(GTK_BOX(cache_box), cache_path_box);
    
    // 프레임 프리렌더 버튼
    GtkWidget *prerender_button = gtk_button_new_with_label("선택한 구간 프리렌더");
    gtk_box_append(GTK_BOX(cache_box), prerender_button);
    
    gtk_frame_set_child(GTK_FRAME(cache_frame), cache_box);
    gtk_box_append(GTK_BOX(perf_box), cache_frame);
    
    // 설명 레이블
    GtkWidget *info_label = gtk_label_new("성능 모드 설정은 편집 중 재생 품질과 시스템 반응성 간의 균형을 조절합니다.");
    gtk_label_set_wrap(GTK_LABEL(info_label), TRUE);
    gtk_box_append(GTK_BOX(perf_box), info_label);
    
    // 모든 위젯을 대화상자에 추가
    gtk_box_append(GTK_BOX(content_area), perf_box);
    
    // 대화상자 표시 및 응답 처리
    gtk_widget_set_size_request(dialog, 500, -1);
    gtk_window_present(GTK_WINDOW(dialog));
    g_signal_connect(dialog, "response", G_CALLBACK(on_dialog_response), NULL);
}

// 대화상자 응답 핸들러
static void on_dialog_response(GtkDialog *dialog, int response, gpointer user_data) {
    gtk_window_destroy(GTK_WINDOW(dialog));
}

// 새 프로젝트 버튼
static void on_new_project_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp *)user_data;
    
    // 기존 클립 제거
    for (int i = 0; i < app->clip_count; i++) {
        if (app->clips[i]) {
            TimelineClip *clip = app->clips[i];
            
            // 위젯 제거
            if (clip->widget) {
                gtk_widget_unparent(clip->widget);
            }
            
            // 메모리 해제
            g_free(clip->filename);
            g_free(clip->display_name);
            g_free(clip);
            app->clips[i] = NULL;
        }
    }
    app->clip_count = 0;
    app->selected_clip = NULL;
    
    // 미디어 파일 목록 비우기
    gtk_list_store_clear(app->media_store);
    
    // 로드된 파일 정리
    for (int i = 0; i < app->file_count; i++) {
        g_free(app->loaded_files[i]);
    }
    app->file_count = 0;
    
    // 비디오 중지
    if (app->pipeline) {
        gst_element_set_state(app->pipeline, GST_STATE_NULL);
        app->playing = FALSE;
        gtk_button_set_icon_name(GTK_BUTTON(app->play_button), "media-playback-start");
    }
    
    // 대화상자 표시
    GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(app->main_window),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_INFO,
        GTK_BUTTONS_OK,
        "새 프로젝트가 생성되었습니다.\n미디어 파일을 불러오려면 '열기' 버튼을 클릭하세요."
    );
    gtk_window_present(GTK_WINDOW(dialog));
    g_signal_connect(dialog, "response", G_CALLBACK(on_dialog_response), NULL);
}

// 타임라인 트랙 추가 함수
static void add_timeline_track(BlouEditApp *app) {
    if (!app) {
        g_print("Error: App instance is null in add_timeline_track\n");
        return;
    }

    // 트랙 개수 증가
    int new_track_index = app->track_count;
    app->track_count++;
    
    // 트랙 배열 크기 조정
    app->timeline_tracks = g_realloc(app->timeline_tracks, sizeof(GtkWidget*) * app->track_count);
    if (!app->timeline_tracks) {
        g_print("Error: Failed to allocate memory for timeline tracks\n");
        app->track_count--;
        return;
    }
    
    // 타임라인 트랙 컨테이너 가져오기
    GtkWidget *tracks_box = gtk_widget_get_parent(app->add_track_button);
    if (!tracks_box) {
        g_print("Error: Could not find tracks container\n");
        app->track_count--;
        return;
    }
    
    // 새 트랙 생성
    GtkWidget *track_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    
    // 트랙 이름 생성
    char track_name[50];
    g_snprintf(track_name, sizeof(track_name), "트랙 %d", new_track_index + 1);
    
    GtkWidget *track_label = gtk_label_new(track_name);
    gtk_widget_set_size_request(track_label, 100, -1);
    gtk_widget_set_halign(track_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(track_row), track_label);
    
    // 트랙 생성
    app->timeline_tracks[new_track_index] = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 3);
    gtk_widget_set_hexpand(app->timeline_tracks[new_track_index], TRUE);
    
    // 각 트랙별 고유 클래스 이름 생성
    char class_name[20];
    g_snprintf(class_name, sizeof(class_name), "track-%d", new_track_index);
    gtk_widget_add_css_class(app->timeline_tracks[new_track_index], class_name);
    
    // 스타일 설정 - 색상을 인덱스에 따라 순환
    const char *track_colors[] = {
        "background-color: rgba(52, 152, 219, 0.2);", // 파란색
        "background-color: rgba(231, 76, 60, 0.2);",  // 빨간색
        "background-color: rgba(46, 204, 113, 0.2);", // 녹색
        "background-color: rgba(241, 196, 15, 0.2);", // 노란색
        "background-color: rgba(155, 89, 182, 0.2);"  // 보라색
    };
    
    GtkCssProvider *track_provider = gtk_css_provider_new();
    char css[100];
    g_snprintf(css, sizeof(css), ".track-%d { %s border-radius: 3px; min-height: 40px; }", 
              new_track_index, track_colors[new_track_index % 5]);
    gtk_css_provider_load_from_data(track_provider, css, -1);
    gtk_style_context_add_provider_for_display(
        gtk_widget_get_display(app->timeline_tracks[new_track_index]),
        GTK_STYLE_PROVIDER(track_provider),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
    );
    g_object_unref(track_provider);
    
    gtk_box_append(GTK_BOX(track_row), app->timeline_tracks[new_track_index]);
    
    // 트랙 row를 트랙 컨테이너에 추가 (add_track_button 바로 앞에)
    gtk_box_remove(GTK_BOX(tracks_box), app->add_track_button);
    gtk_box_append(GTK_BOX(tracks_box), track_row);
    gtk_box_append(GTK_BOX(tracks_box), app->add_track_button);
    
    g_print("트랙 %d 추가됨\n", new_track_index + 1);
}

// 트랙 추가 버튼 클릭 핸들러
static void on_add_track_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp *)user_data;
    if (!app) {
        g_print("Error: App instance is null in on_add_track_clicked\n");
        return;
    }
    
    add_timeline_track(app);
}

// 트랙 관리 함수 구현 (on_add_track_clicked 함수 아래, 약 1206줄 부근)
static void on_manage_tracks_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp *)user_data;
    if (!app) {
        g_print("Error: App instance is null in on_manage_tracks_clicked\n");
        return;
    }
    
    show_track_properties_dialog(app);
}

// 트랙 속성 대화상자 표시
static void show_track_properties_dialog(BlouEditApp *app) {
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "트랙 관리",
        GTK_WINDOW(app->main_window),
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "닫기", GTK_RESPONSE_CLOSE,
        NULL
    );
    
    // 대화상자 컨텐츠 영역 가져오기
    GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_widget_set_margin_start(content_area, 12);
    gtk_widget_set_margin_end(content_area, 12);
    gtk_widget_set_margin_top(content_area, 12);
    gtk_widget_set_margin_bottom(content_area, 12);
    
    // 트랙 리스트
    GtkWidget *tracks_list = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    
    // 트랙 정보 및 컨트롤
    for (int i = 0; i < app->track_count; i++) {
        GtkWidget *track_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        
        // 트랙 이름
        char track_name[50];
        g_snprintf(track_name, sizeof(track_name), "트랙 %d", i + 1);
        
        GtkWidget *label = gtk_label_new(track_name);
        gtk_widget_set_hexpand(label, TRUE);
        gtk_widget_set_halign(label, GTK_ALIGN_START);
        gtk_box_append(GTK_BOX(track_row), label);
        
        // 트랙 삭제 버튼 (첫 번째 트랙은 삭제 불가)
        if (i > 0) {
            GtkWidget *delete_btn = gtk_button_new_from_icon_name("edit-delete");
            gtk_widget_set_tooltip_text(delete_btn, "트랙 삭제");
            g_object_set_data(G_OBJECT(delete_btn), "track-index", GINT_TO_POINTER(i));
            // 실제 삭제 기능은 구현하지 않았음
            gtk_box_append(GTK_BOX(track_row), delete_btn);
        }
        
        gtk_box_append(GTK_BOX(tracks_list), track_row);
    }
    
    gtk_box_append(GTK_BOX(content_area), tracks_list);
    
    // 대화상자 크기 설정
    gtk_window_set_default_size(GTK_WINDOW(dialog), 400, 300);
    
    // 대화상자 표시
    gtk_widget_show(dialog);
    
    // 응답 시그널 연결
    g_signal_connect(dialog, "response", G_CALLBACK(on_dialog_response), NULL);
}

// 애플리케이션 활성화 함수
static void activate(GtkApplication *app, gpointer user_data) {
    g_print("Activating application...\n");
    
    BlouEditApp *blouedit_app = g_new0(BlouEditApp, 1);
    if (!blouedit_app) {
        g_printerr("Failed to allocate application structure\n");
        return;
    }
    
    global_app = blouedit_app;  // 전역 앱 인스턴스 설정
    
    // 초기화
    blouedit_app->playing = FALSE;
    blouedit_app->clip_count = 0;
    blouedit_app->file_count = 0;
    blouedit_app->selected_clip = NULL;
    blouedit_app->pipeline = NULL;
    blouedit_app->project_name = g_strdup("새 프로젝트");
    blouedit_app->project_duration = 0;
    blouedit_app->timeline_scale_value = 1;
    blouedit_app->track_count = 0;
    blouedit_app->timeline_tracks = NULL; // 동적 할당 배열
    
    // 메인 윈도우
    blouedit_app->main_window = gtk_application_window_new(app);
    if (!blouedit_app->main_window) {
        g_printerr("Failed to create main window\n");
        g_free(blouedit_app);
        return;
    }
    
    gtk_window_set_title(GTK_WINDOW(blouedit_app->main_window), "BLOUedit");
    gtk_window_set_default_size(GTK_WINDOW(blouedit_app->main_window), 1280, 720);
    
    // GStreamer 초기화
    GError *error = NULL;
    if (!gst_init_check(NULL, NULL, &error)) {
        g_printerr("Failed to initialize GStreamer: %s\n", error ? error->message : "unknown error");
        if (error) g_error_free(error);
    }
    
    // 메인 레이아웃
    GtkWidget *main_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    
    // 툴바
    blouedit_app->toolbar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_widget_set_margin_start(blouedit_app->toolbar, 6);
    gtk_widget_set_margin_end(blouedit_app->toolbar, 6);
    gtk_widget_set_margin_top(blouedit_app->toolbar, 6);
    gtk_widget_set_margin_bottom(blouedit_app->toolbar, 6);
    
    // 새 프로젝트 버튼
    blouedit_app->new_button = gtk_button_new_from_icon_name("document-new");
    gtk_widget_set_tooltip_text(blouedit_app->new_button, "새 프로젝트");
    g_signal_connect(blouedit_app->new_button, "clicked", G_CALLBACK(on_new_project_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->new_button);
    
    // 열기 버튼
    blouedit_app->open_button = gtk_button_new_from_icon_name("document-open");
    gtk_widget_set_tooltip_text(blouedit_app->open_button, "미디어 파일 열기");
    g_signal_connect(blouedit_app->open_button, "clicked", G_CALLBACK(on_open_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->open_button);
    
    // 구분선
    GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_VERTICAL);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), separator);
    
    // 재생 버튼
    blouedit_app->play_button = gtk_button_new_from_icon_name("media-playback-start");
    gtk_widget_set_tooltip_text(blouedit_app->play_button, "재생/일시정지");
    g_signal_connect(blouedit_app->play_button, "clicked", G_CALLBACK(on_play_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->play_button);
    
    // 내보내기 버튼
    blouedit_app->export_button = gtk_button_new_from_icon_name("document-save-as");
    gtk_widget_set_tooltip_text(blouedit_app->export_button, "내보내기");
    g_signal_connect(blouedit_app->export_button, "clicked", G_CALLBACK(on_export_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->export_button);
    
    // 분할 버튼
    blouedit_app->split_button = gtk_button_new_from_icon_name("edit-cut");
    gtk_widget_set_tooltip_text(blouedit_app->split_button, "클립 분할");
    g_signal_connect(blouedit_app->split_button, "clicked", G_CALLBACK(on_split_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->split_button);
    
    // 삭제 버튼
    blouedit_app->delete_button = gtk_button_new_from_icon_name("edit-delete");
    gtk_widget_set_tooltip_text(blouedit_app->delete_button, "클립 삭제");
    g_signal_connect(blouedit_app->delete_button, "clicked", G_CALLBACK(on_delete_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->delete_button);
    
    // 구분선
    GtkWidget *separator2 = gtk_separator_new(GTK_ORIENTATION_VERTICAL);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), separator2);
    
    // 프록시 워크플로우 버튼
    blouedit_app->proxy_button = gtk_button_new_from_icon_name("document-properties");
    gtk_widget_set_tooltip_text(blouedit_app->proxy_button, "프록시 워크플로우 설정");
    g_signal_connect(blouedit_app->proxy_button, "clicked", G_CALLBACK(on_proxy_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->proxy_button);
    
    // 성능 모드 버튼
    blouedit_app->performance_button = gtk_button_new_from_icon_name("preferences-system-performance");
    gtk_widget_set_tooltip_text(blouedit_app->performance_button, "성능 모드 설정");
    g_signal_connect(blouedit_app->performance_button, "clicked", G_CALLBACK(on_performance_clicked), blouedit_app);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), blouedit_app->performance_button);
    
    // 제목 레이블 (여백 확장)
    GtkWidget *title_label = gtk_label_new("BLOUedit - 프로페셔널 비디오 편집기");
    gtk_widget_set_hexpand(title_label, TRUE);
    gtk_box_append(GTK_BOX(blouedit_app->toolbar), title_label);
    
    // 툴바를 메인 레이아웃에 추가
    gtk_box_append(GTK_BOX(main_box), blouedit_app->toolbar);
    
    // 메인 콘텐츠 영역 (미리보기, 타임라인, 미디어 라이브러리)
    GtkWidget *content_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_vexpand(content_box, TRUE);
    
    // 좌측 패널 (미디어 라이브러리)
    blouedit_app->left_panel = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_size_request(blouedit_app->left_panel, 250, -1);
    
    // 미디어 패널 설정
    blouedit_app->media_stack = gtk_stack_new();
    
    // 미디어 라이브러리 생성
    blouedit_app->media_library = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget *media_header = gtk_label_new("미디어 라이브러리");
    gtk_widget_add_css_class(media_header, "title-4");
    gtk_widget_set_margin_top(media_header, 6);
    gtk_widget_set_margin_bottom(media_header, 6);
    gtk_box_append(GTK_BOX(blouedit_app->media_library), media_header);
    
    create_media_library(blouedit_app);
    
    // AI 도구 생성
    blouedit_app->ai_tools = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget *ai_header = gtk_label_new("AI 기능");
    gtk_widget_add_css_class(ai_header, "title-4");
    gtk_widget_set_margin_top(ai_header, 6);
    gtk_widget_set_margin_bottom(ai_header, 6);
    gtk_box_append(GTK_BOX(blouedit_app->ai_tools), ai_header);
    
    // AI 버튼 컨테이너
    GtkWidget *ai_buttons = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_margin_start(ai_buttons, 12);
    gtk_widget_set_margin_end(ai_buttons, 12);
    
    // AI 기능 버튼 생성
    const char *ai_functions[] = {
        "텍스트 → 영상 변환", "오디오 → 영상 변환", "이미지 → 영상 변환",
        "스토리보드 생성기", "썸네일 생성기", "음악 생성기",
        "음성 제거기", "음성 클로닝", "얼굴 모자이크",
        "자동 자막 생성", "스마트 컷아웃", "음성 → 텍스트 변환",
        "텍스트 → 음성 변환", "스티커 생성기", "이미지 생성기",
        "AI 카피라이팅", "프레임 보간", "씬 감지",
        "스타일 트랜스퍼", "화질 향상"
    };
    
    for (int i = 0; i < 20; i++) {
        GtkWidget *button = gtk_button_new_with_label(ai_functions[i]);
        gtk_box_append(GTK_BOX(ai_buttons), button);
    }
    
    gtk_box_append(GTK_BOX(blouedit_app->ai_tools), ai_buttons);
    
    // 고급 비디오 도구 생성
    blouedit_app->advanced_video_tools = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget *adv_header = gtk_label_new("고급 비디오 편집");
    gtk_widget_add_css_class(adv_header, "title-4");
    gtk_widget_set_margin_top(adv_header, 6);
    gtk_widget_set_margin_bottom(adv_header, 6);
    gtk_box_append(GTK_BOX(blouedit_app->advanced_video_tools), adv_header);
    
    // 고급 비디오 버튼 컨테이너
    GtkWidget *adv_buttons = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_margin_start(adv_buttons, 12);
    gtk_widget_set_margin_end(adv_buttons, 12);
    
    // 고급 비디오 기능 버튼 생성
    const char *adv_functions[] = {
        "플래너 트래킹", "멀티 카메라 편집", "이미지 시퀀스 → 영상 변환",
        "고급 영상 압축기", "키프레임 경로 곡선", "색상 보정 및 LUTs",
        "속도 램핑", "모션 트래킹", "그린 스크린(크로마 키)",
        "자동 리프레임", "조정 레이어", "빠른 분할 모드",
        "키보드 단축키 프리셋"
    };
    
    for (int i = 0; i < 13; i++) {
        GtkWidget *button = gtk_button_new_with_label(adv_functions[i]);
        gtk_box_append(GTK_BOX(adv_buttons), button);
    }
    
    gtk_box_append(GTK_BOX(blouedit_app->advanced_video_tools), adv_buttons);
    
    // 오디오 도구 생성
    blouedit_app->audio_tools = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget *audio_header = gtk_label_new("오디오 편집");
    gtk_widget_add_css_class(audio_header, "title-4");
    gtk_widget_set_margin_top(audio_header, 6);
    gtk_widget_set_margin_bottom(audio_header, 6);
    gtk_box_append(GTK_BOX(blouedit_app->audio_tools), audio_header);
    
    // 오디오 버튼 컨테이너
    GtkWidget *audio_buttons = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_margin_start(audio_buttons, 12);
    gtk_widget_set_margin_end(audio_buttons, 12);
    
    // 오디오 기능 버튼 생성
    const char *audio_functions[] = {
        "AI 음성 클로닝", "음성 변조기", "자동 비트 동기화",
        "오디오 시각화", "자동 동기화", "AI 오디오 스트레치",
        "AI 오디오 노이즈 제거"
    };
    
    for (int i = 0; i < 7; i++) {
        GtkWidget *button = gtk_button_new_with_label(audio_functions[i]);
        gtk_box_append(GTK_BOX(audio_buttons), button);
    }
    
    gtk_box_append(GTK_BOX(blouedit_app->audio_tools), audio_buttons);
    
    // 스택에 모든 도구 추가
    gtk_stack_add_titled(GTK_STACK(blouedit_app->media_stack), blouedit_app->media_library, "media", "미디어");
    gtk_stack_add_titled(GTK_STACK(blouedit_app->media_stack), blouedit_app->ai_tools, "ai", "AI");
    gtk_stack_add_titled(GTK_STACK(blouedit_app->media_stack), blouedit_app->advanced_video_tools, "advanced", "고급 편집");
    gtk_stack_add_titled(GTK_STACK(blouedit_app->media_stack), blouedit_app->audio_tools, "audio", "오디오");
    
    // 스택 스위처 생성
    GtkWidget *stack_switcher = gtk_stack_switcher_new();
    gtk_stack_switcher_set_stack(GTK_STACK_SWITCHER(stack_switcher), GTK_STACK(blouedit_app->media_stack));
    gtk_widget_set_halign(stack_switcher, GTK_ALIGN_CENTER);
    
    // 좌측 패널에 스위처와 스택 추가
    gtk_box_append(GTK_BOX(blouedit_app->left_panel), stack_switcher);
    gtk_box_append(GTK_BOX(blouedit_app->left_panel), blouedit_app->media_stack);
    
    // 우측 영역 (미리보기 및 타임라인)
    GtkWidget *right_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_hexpand(right_box, TRUE);
    
    // 미리보기 영역
    blouedit_app->preview = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_vexpand(blouedit_app->preview, TRUE);
    gtk_widget_set_size_request(blouedit_app->preview, -1, 300);
    gtk_widget_add_css_class(blouedit_app->preview, "preview-area");
    
    GtkCssProvider *preview_provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(preview_provider, 
        ".preview-area { background-color: #1e1e1e; border-radius: 3px; }", -1);
    gtk_style_context_add_provider_for_display(
        gtk_widget_get_display(blouedit_app->preview),
        GTK_STYLE_PROVIDER(preview_provider),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
    );
    g_object_unref(preview_provider);
    
    // 타임라인 영역
    blouedit_app->timeline = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_vexpand(blouedit_app->timeline, TRUE);
    gtk_widget_set_margin_top(blouedit_app->timeline, 6);
    gtk_widget_set_margin_bottom(blouedit_app->timeline, 6);
    
    GtkWidget *timeline_label = gtk_label_new("타임라인");
    gtk_widget_set_halign(timeline_label, GTK_ALIGN_START);
    gtk_widget_set_margin_start(timeline_label, 6);
    gtk_widget_set_margin_bottom(timeline_label, 6);
    gtk_box_append(GTK_BOX(blouedit_app->timeline), timeline_label);
    
    // 타임라인 트랙 컨테이너
    GtkWidget *tracks_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 3);
    gtk_widget_set_margin_start(tracks_box, 6);
    gtk_widget_set_margin_end(tracks_box, 6);
    
    // 트랙 기본 정의
    const int DEFAULT_TRACKS = 3;
    const char *default_track_names[] = {"비디오 트랙", "오디오 트랙", "효과 트랙"};
    const char *track_colors[] = {
        "background-color: rgba(52, 152, 219, 0.2);",
        "background-color: rgba(231, 76, 60, 0.2);",
        "background-color: rgba(46, 204, 113, 0.2);",
        "background-color: rgba(241, 196, 15, 0.2);",
        "background-color: rgba(155, 89, 182, 0.2);"
    };
    
    // 동적 트랙 배열 초기화
    blouedit_app->track_count = DEFAULT_TRACKS;
    blouedit_app->timeline_tracks = g_new0(GtkWidget*, blouedit_app->track_count);
    
    // 기본 트랙 생성
    for (int i = 0; i < blouedit_app->track_count; i++) {
        GtkWidget *track_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        
        GtkWidget *track_label = gtk_label_new(default_track_names[i]);
        gtk_widget_set_size_request(track_label, 100, -1);
        gtk_widget_set_halign(track_label, GTK_ALIGN_START);
        gtk_box_append(GTK_BOX(track_row), track_label);
        
        blouedit_app->timeline_tracks[i] = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 3);
        gtk_widget_set_hexpand(blouedit_app->timeline_tracks[i], TRUE);
        
        // 각 트랙별 고유 클래스 이름 생성
        char class_name[20];
        g_snprintf(class_name, sizeof(class_name), "track-%d", i);
        gtk_widget_add_css_class(blouedit_app->timeline_tracks[i], class_name);
        
        GtkCssProvider *track_provider = gtk_css_provider_new();
        char css[100];
        g_snprintf(css, sizeof(css), ".track-%d { %s border-radius: 3px; min-height: 40px; }", 
                  i, track_colors[i % 5]);
        gtk_css_provider_load_from_data(track_provider, css, -1);
        gtk_style_context_add_provider_for_display(
            gtk_widget_get_display(blouedit_app->timeline_tracks[i]),
            GTK_STYLE_PROVIDER(track_provider),
            GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
        );
        g_object_unref(track_provider);
        
        gtk_box_append(GTK_BOX(track_row), blouedit_app->timeline_tracks[i]);
        gtk_box_append(GTK_BOX(tracks_box), track_row);
    }
    
    // 트랙 추가 버튼
    blouedit_app->add_track_button = gtk_button_new_with_label("트랙 추가");
    gtk_widget_set_margin_top(blouedit_app->add_track_button, 6);
    g_signal_connect(blouedit_app->add_track_button, "clicked", 
                     G_CALLBACK(on_add_track_clicked), blouedit_app);
                     
    // 트랙 관리 버튼
    GtkWidget *manage_tracks_button = gtk_button_new_with_label("트랙 관리");
    gtk_widget_set_margin_top(manage_tracks_button, 6);
    gtk_widget_set_margin_start(manage_tracks_button, 6);
    g_signal_connect(manage_tracks_button, "clicked", 
                     G_CALLBACK(on_manage_tracks_clicked), blouedit_app);
    
    // 버튼을 수평 상자에 배치
    GtkWidget *track_buttons_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_append(GTK_BOX(track_buttons_box), blouedit_app->add_track_button);
    gtk_box_append(GTK_BOX(track_buttons_box), manage_tracks_button);
    gtk_box_append(GTK_BOX(tracks_box), track_buttons_box);
    
    gtk_box_append(GTK_BOX(blouedit_app->timeline), tracks_box);
    
    // 타임라인 스케일
    blouedit_app->timeline_adj = gtk_adjustment_new(0, 0, 100, 1, 10, 0);
    blouedit_app->timeline_scale = gtk_scale_new(GTK_ORIENTATION_HORIZONTAL, blouedit_app->timeline_adj);
    gtk_scale_set_draw_value(GTK_SCALE(blouedit_app->timeline_scale), FALSE);
    gtk_widget_set_margin_start(blouedit_app->timeline_scale, 106);
    gtk_widget_set_margin_end(blouedit_app->timeline_scale, 6);
    gtk_box_append(GTK_BOX(blouedit_app->timeline), blouedit_app->timeline_scale);
    
    // 우측 영역에 요소 추가
    gtk_box_append(GTK_BOX(right_box), blouedit_app->preview);
    gtk_box_append(GTK_BOX(right_box), blouedit_app->timeline);
    
    // 모든 박스를 콘텐츠 영역에 추가
    gtk_box_append(GTK_BOX(content_box), blouedit_app->left_panel);
    gtk_box_append(GTK_BOX(content_box), right_box);
    
    // 콘텐츠를 메인 레이아웃에 추가
    gtk_box_append(GTK_BOX(main_box), content_box);
    
    // 상태 표시줄
    GtkWidget *statusbar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    GtkWidget *status_label = gtk_label_new("준비");
    gtk_widget_set_margin_start(status_label, 6);
    gtk_widget_set_margin_top(status_label, 3);
    gtk_widget_set_margin_bottom(status_label, 3);
    gtk_box_append(GTK_BOX(statusbar), status_label);
    gtk_box_append(GTK_BOX(main_box), statusbar);
    
    // 윈도우에 메인 콘텐츠 설정
    gtk_window_set_child(GTK_WINDOW(blouedit_app->main_window), main_box);
    
    // 윈도우 표시
    gtk_window_present(GTK_WINDOW(blouedit_app->main_window));
    
    // 환영 대화상자 초기화
    GtkWidget *welcome = gtk_message_dialog_new(
        GTK_WINDOW(blouedit_app->main_window),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_INFO,
        GTK_BUTTONS_OK,
        "BLOUedit 비디오 편집기에 오신 것을 환영합니다!\n\n"
        "새 프로젝트를 시작하려면 '새 프로젝트' 버튼을 클릭하시고,\n"
        "미디어 파일을 불러오려면 '미디어 파일 열기' 버튼을 클릭하세요."
    );
    gtk_window_present(GTK_WINDOW(welcome));
    
    g_signal_connect(welcome, "response", G_CALLBACK(on_dialog_response), NULL);
}

int main(int argc, char *argv[]) {
    // GTK 초기화 전에 로케일 설정 추가
    setlocale(LC_ALL, "C.UTF-8");
    
    // 디버그 정보 출력
    g_print("Current locale: %s\n", setlocale(LC_ALL, NULL));
    
    // Set program name explicitly for better D-Bus registration
    g_set_prgname("com.blouedit.BLOUedit");
    
    // Use NON_UNIQUE flag to avoid D-Bus ownership conflicts 
    GtkApplication *app = gtk_application_new("com.blouedit.BLOUedit", 
                                              G_APPLICATION_NON_UNIQUE);
    
    if (app == NULL) {
        g_printerr("Failed to create application\n");
        return 1;
    }
    
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    
    // Add error handling for application run
    GError *error = NULL;
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    
    if (error != NULL) {
        g_printerr("Application error: %s\n", error->message);
        g_error_free(error);
    }
    
    // 전역 앱 인스턴스 정리
    if (global_app) {
        // 클립 메모리 정리
        for (int i = 0; i < global_app->clip_count; i++) {
            if (global_app->clips[i]) {
                g_free(global_app->clips[i]->filename);
                g_free(global_app->clips[i]->display_name);
                g_free(global_app->clips[i]);
            }
        }
        
        // 타임라인 트랙 배열 정리
        if (global_app->timeline_tracks) {
            g_free(global_app->timeline_tracks);
        }
        
        // 로드된 파일 정리
        for (int i = 0; i < global_app->file_count; i++) {
            g_free(global_app->loaded_files[i]);
        }
        
        // 프로젝트 이름 정리
        g_free(global_app->project_name);
        
        // 앱 인스턴스 정리
        g_free(global_app);
        global_app = NULL;
    }
    
    g_object_unref(app);
    return status;
} 