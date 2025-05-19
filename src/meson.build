blouedit_sources = [
  'application.cpp',
  'window.cpp',
  # Commented out non-existent files
  # 'main.cpp',
  # 'timeline/timeline.cpp',
  # 'timeline/track.cpp',
  # 'timeline/clip.cpp',
  # 'player/player.cpp',
  # 'editor/editor.cpp',
  # 'editor/effects.cpp',
  'editor/transitions.cpp',
  'editor/filters.cpp',
  # 'editor/text.cpp',
  # 'editor/audio.cpp',
  'ai/scene_detection.cpp',
  'ai/image_generator.cpp',
  'ai/thumbnail_generator.cpp',
  'ai/text_to_speech.cpp',
  'ai/speech_to_text.cpp',
  'ai/style_transfer.cpp',
  'ai/enhancement.cpp',
  'ai/face_mosaic.cpp',
  'ai/smart_cutout.cpp',
  'ai/voice_modulator.cpp',
  'ai/audio_noise_remover.cpp',
  # 'utils/media_library.cpp',
  # 'utils/file_utils.cpp',
]

blouedit_headers = [
  'application.h',
  'window.h',
  'timeline/timeline.h',
  # Commented out non-existent files
  # 'timeline/track.h',
  # 'timeline/clip.h',
  # 'player/player.h',
  # 'editor/editor.h',
  # 'editor/effects.h',
  'editor/transitions.h',
  'editor/filters.h',
  # 'editor/text.h',
  # 'editor/audio.h',
  'ai/scene_detection.h',
  'ai/image_generator.h',
  'ai/thumbnail_generator.h',
  'ai/text_to_speech.h',
  'ai/speech_to_text.h',
  'ai/style_transfer.h',
  'ai/enhancement.h',
  'ai/face_mosaic.h',
  'ai/smart_cutout.h',
  'ai/voice_modulator.h',
  'ai/audio_noise_remover.h',
  # 'utils/media_library.h',
  # 'utils/file_utils.h',
]

blouedit_deps = [
  gtk_dep,
  gstreamer_dep,
  gst_video_dep,
  gst_audio_dep,
  gst_editing_dep,
  adwaita_dep,
  python3_dep,
]

# Let's create a simple main.cpp file first
main_content = '''
#include "application.h"

int main(int argc, char *argv[]) {
  auto app = blouedit::Application::create();
  return app->run(argc, argv);
}
'''

# Write main.cpp to the source directory
run_command('sh', '-c', 'echo "' + main_content + '" > @0@/main.cpp'.format(meson.current_source_dir()))

# Add main.cpp to sources
blouedit_sources += ['main.cpp']

# Check if gresource file exists, if not create a minimal one
gresource_path = join_paths(meson.project_source_root(), 'resources/blouedit.gresource.xml')
if run_command('test', '-f', gresource_path).returncode() != 0
  # Create a minimal resources directory and gresource file
  run_command('mkdir', '-p', join_paths(meson.project_source_root(), 'resources'))
  minimal_gresource = '''<?xml version="1.0" encoding="UTF-8"?>
<gresources>
  <gresource prefix="/com/blouedit/BLOUedit">
    <!-- Add resources here when available -->
  </gresource>
</gresources>
'''
  run_command('sh', '-c', 'echo "' + minimal_gresource + '" > ' + gresource_path)
endif

# Generate resources
blouedit_resources = gnome.compile_resources(
  'blouedit-resources',
  '../resources/blouedit.gresource.xml',
  c_name: 'blouedit'
)

executable('blouedit', 
  blouedit_sources,
  blouedit_resources,
  dependencies: blouedit_deps,
  install: true,
)

# Python modules for AI features
install_subdir('ai/python', install_dir: pkgdatadir) 