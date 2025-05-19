#!/usr/bin/env python3

import os
import subprocess
import sys

prefix = os.environ.get('MESON_INSTALL_PREFIX', '/usr/local')
datadir = os.path.join(prefix, 'share')
destdir = os.environ.get('DESTDIR', '')

# Package managers set this so we don't need to run
if not destdir and not os.environ.get('BLOUEDIT_SKIP_SCHEMA_COMPILE'):
    print('Updating icon cache...')
    subprocess.call(['gtk-update-icon-cache', '-qtf', 
                    os.path.join(datadir, 'icons', 'hicolor')])

    print('Updating desktop database...')
    subprocess.call(['update-desktop-database', '-q', 
                    os.path.join(datadir, 'applications')])

    print('Compiling GSettings schemas...')
    subprocess.call(['glib-compile-schemas', 
                    os.path.join(datadir, 'glib-2.0', 'schemas')])
    
    print('Updating MIME database...')
    subprocess.call(['update-mime-database',
                    os.path.join(datadir, 'mime')])

# Make sure Python modules have correct permissions
python_dir = os.path.join(destdir + datadir, 'blouedit', 'ai', 'python')
if os.path.exists(python_dir):
    print(f'Setting permissions for Python modules in {python_dir}...')
    for root, dirs, files in os.walk(python_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f'  Making executable: {file_path}')
                mode = os.stat(file_path).st_mode
                os.chmod(file_path, mode | 0o111)  # Add executable bit

print('BLOUedit installation completed successfully.')
print('Execute "blouedit" to launch the application.') 