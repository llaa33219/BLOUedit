# Process UI files (commented out until UI files are created)
# blouedit_ui_files = files(
#   'ui/window.ui',
#   'ui/timeline.ui',
#   'ui/player.ui',
#   'ui/media_library.ui',
#   'ui/effects_panel.ui',
#   'ui/text_panel.ui',
#   'ui/preferences.ui',
#   'ui/export_dialog.ui',
#   'ui/ai_panel.ui',
# )

# # Install UI files
# install_data(blouedit_ui_files,
#   install_dir: join_paths(pkgdatadir, 'ui')
# )

# Create empty ui directory to satisfy structure
run_command('mkdir', '-p', join_paths(meson.current_source_dir(), 'ui'))

# Install icons if they exist
icons_dir = join_paths(meson.current_source_dir(), 'icons')
if run_command('test', '-d', icons_dir).returncode() == 0
  install_subdir('icons',
    install_dir: join_paths(datadir, 'icons')
  )
endif 