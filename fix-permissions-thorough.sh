#!/bin/bash
# Fix permissions on all files
find . -type f -name "*.c" -exec chmod 644 {} \;
find . -type f -name "*.h" -exec chmod 644 {} \;
find . -type f -name "*.yml" -exec chmod 644 {} \;
find . -type f -name "*.md" -exec chmod 644 {} \;
find . -type f -name "*.sh" -exec chmod 755 {} \;
find . -type d -exec chmod 755 {} \;

# Fix specific files
chmod 644 blouedit.c
chmod 644 com.blouedit.BLOUedit.yml
chmod 755 generate-icon.sh
chmod 755 fix-blouedit.sh

echo "All permissions fixed thoroughly!" 