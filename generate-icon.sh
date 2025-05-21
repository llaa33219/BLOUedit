echo 'Creating app icon'
mkdir -p /app/share/icons/hicolor/scalable/apps
echo '<?xml version="1.0" encoding="UTF-8"?>' > /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg
echo '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64" fill="none">' >> /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg
echo '  <rect width="64" height="64" rx="8" fill="#3584e4"/>' >> /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg
echo '  <path d="M16 16h32v32H16z" fill="#ffffff"/>' >> /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg
echo '  <rect x="20" y="20" width="24" height="4" fill="#e01b24"/>' >> /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg
echo '  <rect x="20" y="28" width="24" height="4" fill="#26a269"/>' >> /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg
echo '  <rect x="20" y="36" width="24" height="4" fill="#1c71d8"/>' >> /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg
echo '</svg>' >> /app/share/icons/hicolor/scalable/apps/com.blouedit.BLOUedit.svg 