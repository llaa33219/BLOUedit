project('blouedit', 'c', 'cpp',
  version: '0.1.0',
  license: 'GPL-3.0-or-later',
  meson_version: '>= 0.59.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
  ],
)

gnome = import('gnome')
i18n = import('i18n')
python = import('python')

prefix = get_option('prefix')
bindir = join_paths(prefix, get_option('bindir'))
datadir = join_paths(prefix, get_option('datadir'))
libdir = join_paths(prefix, get_option('libdir'))
pkgdatadir = join_paths(datadir, meson.project_name())

# Dependencies
gtk_dep = dependency('gtk4', version: '>= 4.8.0')
gstreamer_dep = dependency('gstreamer-1.0', version: '>= 1.20.0')
gst_video_dep = dependency('gstreamer-video-1.0', version: '>= 1.20.0')
gst_audio_dep = dependency('gstreamer-audio-1.0', version: '>= 1.20.0')
gst_editing_dep = dependency('gst-editing-services-1.0', version: '>= 1.20.0')
adwaita_dep = dependency('libadwaita-1', version: '>= 1.2.0')
python3_dep = dependency('python3', version: '>=3.8')
torch_dep = dependency('torch', required: false)

subdir('src')
subdir('data')
subdir('resources')

meson.add_install_script('build-aux/meson/postinstall.py') 