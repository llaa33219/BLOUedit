cmake_minimum_required(VERSION 3.10)

set(TIMELINE_SOURCES
    # Core files
    core/timeline.cpp
    core/types.h
    core/timeline.h
    
    # Tracks module
    tracks/tracks.cpp
    tracks/tracks.h
    
    # Keyframes module
    keyframes/keyframes.h
    keyframes/keyframe_curve_editor.c
    
    # Markers module
    markers/markers.h
    markers/marker_color.c
    markers/marker_color.h
    markers/marker_search.c
    markers/marker_search.h
    
    # Groups module (not yet implemented)
    # groups/groups.cpp
    groups/groups.h
    
    # UI modules
    ui/timeline_view_mode.c
    ui/timeline_view_mode.h
    ui/timeline_hand_tool.c
    ui/timeline_hand_tool.h
    
    # Clip alignment module
    clip_alignment.cpp
    
    # Advanced edit operations
    slip_slide_edits.c
    
    # Main include file
    timeline_all.h
    
    timeline.cpp
    timeline.h
    unlimited_tracks.c
    unlimited_tracks.h
)

add_library(timeline STATIC ${TIMELINE_SOURCES})

target_include_directories(timeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${GTK_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GES_INCLUDE_DIRS}
)

target_link_libraries(timeline
    ${GTK_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${GES_LIBRARIES}
) 