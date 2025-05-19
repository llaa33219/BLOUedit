#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>
#include <gst/pbutils/pbutils.h>

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
    GtkWidget *timeline_tracks[3]; // 비디오, 오디오, 효과 트랙
    GtkAdjustment *timeline_adj;
    GtkWidget *timeline_scale;
    
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
static void on_dialog_response(GtkDialog *dialog, int response, gpointer user_data);
static void on_new_project_clicked(GtkButton *button, gpointer user_data);
static void update_preview(BlouEditApp *app, TimelineClip *clip);
static void select_clip(BlouEditApp *app, TimelineClip *clip);
static void deselect_all_clips(BlouEditApp *app);
static TimelineClip* create_timeline_clip(BlouEditApp *app, const char *filename, int track);
static void delete_selected_clip(BlouEditApp *app);
static void create_export_pipeline(BlouEditApp *app, const char *output_file);
static void on_pad_added(GstElement *element, GstPad *pad, gpointer data);

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
    TimelineClip *clip = (TimelineClip *)user_data;
    BlouEditApp *app = global_app; // 전역 앱 인스턴스 사용
    
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
    TimelineClip *clip = (TimelineClip *)user_data;
    
    // 선택되지 않았다면 먼저 선택
    if (!clip->selected) {
        BlouEditApp *app = global_app;
        deselect_all_clips(app);
        select_clip(app, clip);
    }
    
    g_print("클립 드래그 시작: %s\n", clip->display_name);
}

// 클립 드래그 업데이트 핸들러
static void on_clip_drag_update(GtkGestureDrag *gesture, double offset_x, double offset_y, gpointer user_data) {
    TimelineClip *clip = (TimelineClip *)user_data;
    BlouEditApp *app = global_app;
    
    // 마진 값 계산 (타임라인에서의 위치)
    int new_margin = MAX(0, clip->start_position / 50 + (int)offset_x);
    
    // UI 업데이트
    gtk_widget_set_margin_start(clip->widget, new_margin);
    
    g_print("클립 드래그 중: %s, 오프셋: %.0f\n", clip->display_name, offset_x);
}

// 클립 드래그 종료 핸들러
static void on_clip_drag_end(GtkGestureDrag *gesture, double offset_x, double offset_y, gpointer user_data) {
    TimelineClip *clip = (TimelineClip *)user_data;
    BlouEditApp *app = global_app;
    
    // 새 위치 계산 및 저장
    int new_position = MAX(0, clip->start_position + (int)(offset_x * 50));
    clip->start_position = new_position;
    
    g_print("클립 드래그 종료: %s, 새 위치: %d\n", clip->display_name, new_position);
}

// 모든 클립 선택 해제
static void deselect_all_clips(BlouEditApp *app) {
    if (app->selected_clip) {
        app->selected_clip->selected = FALSE;
        
        // 선택 해제 시 스타일 변경
        const char *track_colors[] = {
            "button { background: #3498db; color: white; }",
            "button { background: #e74c3c; color: white; }",
            "button { background: #2ecc71; color: white; }"
        };
        
        GtkCssProvider *provider = gtk_css_provider_new();
        gtk_css_provider_load_from_data(provider, track_colors[app->selected_clip->track_index % 3], -1);
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
    if (!clip || !clip->filename) {
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
    GtkTreeIter iter;
    const char *basename = g_path_get_basename(filename);
    
    // 미디어 라이브러리에 추가
    gtk_list_store_append(app->media_store, &iter);
    gtk_list_store_set(app->media_store, &iter, 
                      0, basename, 
                      1, filename, -1);
    
    // 파일 목록에 추가
    if (app->file_count < 50) {
        app->loaded_files[app->file_count++] = g_strdup(filename);
    }
    
    g_free((gpointer)basename);
    
    g_print("미디어 추가됨: %s\n", filename);
}

static void media_item_activated(GtkTreeView *tree_view, GtkTreePath *path, 
                                GtkTreeViewColumn *column, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    GtkTreeIter iter;
    GtkTreeModel *model = gtk_tree_view_get_model(tree_view);
    
    if (gtk_tree_model_get_iter(model, &iter, path)) {
        gchar *filename;
        gtk_tree_model_get(model, &iter, 1, &filename, -1);
        
        // 새 클립 생성
        TimelineClip *clip = create_timeline_clip(app, filename, app->clip_count % 3);
        
        // 타임라인에 추가
        int track = clip->track_index;
        
        // 현재 트랙의 끝에 배치
        int track_end = 0;
        for (int i = 0; i < app->clip_count; i++) {
            TimelineClip *other = app->clips[i];
            if (other->track_index == track) {
                int other_end = other->start_position + other->duration;
                if (other_end > track_end) {
                    track_end = other_end;
                }
            }
        }
        
        clip->start_position = track_end;
        gtk_widget_set_margin_start(clip->widget, track_end / 50);
        
        // 트랙에 추가
        gtk_box_append(GTK_BOX(app->timeline_tracks[track]), clip->widget);
        
        // 클립 배열에 추가
        app->clips[app->clip_count++] = clip;
        
        // 선택 및 미리보기
        deselect_all_clips(app);
        select_clip(app, clip);
        update_preview(app, clip);
        
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

// 대화상자 응답 핸들러
static void on_dialog_response(GtkDialog *dialog, int response, gpointer user_data) {
    gtk_window_destroy(GTK_WINDOW(dialog));
}

// 새 프로젝트 버튼
static void on_new_project_clicked(GtkButton *button, gpointer user_data) {
    BlouEditApp *app = (BlouEditApp*)user_data;
    
    // 기존 클립 정리
    for (int i = 0; i < app->clip_count; i++) {
        TimelineClip *clip = app->clips[i];
        gtk_widget_unparent(clip->widget);
        g_free(clip->filename);
        g_free(clip->display_name);
        g_free(clip);
    }
    app->clip_count = 0;
    app->selected_clip = NULL;
    
    // 미디어 라이브러리 정리
    gtk_list_store_clear(app->media_store);
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

// 애플리케이션 활성화 함수
static void activate(GtkApplication *app, gpointer user_data) {
    BlouEditApp *blouedit_app = g_new0(BlouEditApp, 1);
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
    
    // 메인 윈도우
    blouedit_app->main_window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(blouedit_app->main_window), "BLOUedit");
    gtk_window_set_default_size(GTK_WINDOW(blouedit_app->main_window), 1280, 720);
    
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
    
    // 타임라인 트랙
    GtkWidget *tracks_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 3);
    gtk_widget_set_margin_start(tracks_box, 6);
    gtk_widget_set_margin_end(tracks_box, 6);
    
    const char *track_names[] = {"비디오 트랙", "오디오 트랙", "효과 트랙"};
    const char *track_colors[] = {
        "background-color: rgba(52, 152, 219, 0.2);",
        "background-color: rgba(231, 76, 60, 0.2);",
        "background-color: rgba(46, 204, 113, 0.2);"
    };
    
    for (int i = 0; i < 3; i++) {
        GtkWidget *track_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        
        GtkWidget *track_label = gtk_label_new(track_names[i]);
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
                  i, track_colors[i]);
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
    
    // GStreamer 초기화
    gst_init(NULL, NULL);
    
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
    GtkApplication *app = gtk_application_new("com.blouedit.BLOUedit", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return status;
} 