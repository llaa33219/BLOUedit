#!/bin/bash

# Fix permissions for BLOUedit binary
BINARY_PATH=$(find ~/.local/share/flatpak/app/com.blouedit.BLOUedit -name blouedit 2>/dev/null)
if [ -n "$BINARY_PATH" ]; then
  echo "Found BLOUedit binary at: $BINARY_PATH"
  chmod 755 "$BINARY_PATH" 2>/dev/null || echo "Failed to set permissions (expected on NTFS)"
  
  # Create a file with execution bit in the same directory
  SCRIPT_PATH=$(dirname "$BINARY_PATH")
  echo '#!/bin/bash
exec /app/bin/blouedit.real "$@"' > "$SCRIPT_PATH/blouedit.wrapper"
  chmod 755 "$SCRIPT_PATH/blouedit.wrapper" 2>/dev/null || echo "Failed to set permissions on wrapper"
  
  # Rename original file
  mv "$BINARY_PATH" "$SCRIPT_PATH/blouedit.real" 2>/dev/null
  
  # Set wrapper as the executable
  mv "$SCRIPT_PATH/blouedit.wrapper" "$BINARY_PATH" 2>/dev/null
  
  echo "BLOUedit fixed!"
else
  echo "BLOUedit binary not found"
fi 