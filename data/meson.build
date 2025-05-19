# Desktop file
desktop_file = i18n.merge_file(
  input: 'com.blouedit.BLOUedit.desktop.in',
  output: 'com.blouedit.BLOUedit.desktop',
  type: 'desktop',
  po_dir: '../po',
  install: true,
  install_dir: join_paths(datadir, 'applications')
)

# Validate desktop file
desktop_utils = find_program('desktop-file-validate', required: false)
if desktop_utils.found()
  test('Validate desktop file', desktop_utils,
    args: [desktop_file]
  )
endif

# Appdata file
appstream_file = i18n.merge_file(
  input: 'com.blouedit.BLOUedit.appdata.xml.in',
  output: 'com.blouedit.BLOUedit.appdata.xml',
  po_dir: '../po',
  install: true,
  install_dir: join_paths(datadir, 'appdata')
)

# Validate appdata file
appstream_util = find_program('appstream-util', required: false)
if appstream_util.found()
  test('Validate appstream file', appstream_util,
    args: ['validate', appstream_file]
  )
endif

# Install icons
install_data('icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg',
  install_dir: join_paths(datadir, 'icons/hicolor/scalable/apps')
)

install_data('icons/hicolor/symbolic/apps/com.blouedit.BLOUedit-symbolic.svg',
  install_dir: join_paths(datadir, 'icons/hicolor/symbolic/apps')
) 