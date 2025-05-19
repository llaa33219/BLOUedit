# Building and Installing BLOUedit

BLOUedit is a professional video editing application with advanced features and AI capabilities, built for Linux systems using GTK4 and GStreamer.

## Requirements

### Build Dependencies

- Meson (>= 0.59.0)
- Ninja
- C/C++ compiler with C++17 support
- GTK4 (>= 4.8.0)
- Libadwaita (>= 1.2.0)
- GStreamer (>= 1.20.0) and related plugins (gst-plugins-base, gst-plugins-good, gst-plugins-bad)
- GStreamer Editing Services (>= 1.20.0)
- Python 3 (>= 3.8)
- JSON-GLib
- FFmpeg

### Runtime Dependencies for AI Features

- Python 3 (>= 3.8)
- PyTorch
- torchvision
- NumPy
- OpenCV (cv2)
- Pillow (PIL)

## Building from Source

### Step 1: Install Build Dependencies

On Fedora:

```bash
sudo dnf install meson ninja-build gcc g++ \
  gtk4-devel libadwaita-devel json-glib-devel \
  gstreamer1-devel gstreamer1-plugins-base-devel \
  gstreamer1-plugins-good gstreamer1-plugins-bad-free \
  gst-editing-services-devel python3-devel ffmpeg-devel
```

On Ubuntu/Debian:

```bash
sudo apt install meson ninja-build gcc g++ \
  libgtk-4-dev libadwaita-1-dev libjson-glib-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  libges-1.0-dev python3-dev ffmpeg
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/blouedit/blouedit.git
cd blouedit
```

### Step 3: Configure and Build

```bash
meson setup build
ninja -C build
```

### Step 4: Install

```bash
sudo ninja -C build install
```

This will install BLOUedit to your system.

## Installing via Flatpak

BLOUedit is also available as a Flatpak package, which bundles all dependencies including AI libraries.

### Step 1: Install Flatpak

If you don't have Flatpak installed, follow the instructions for your distribution at https://flatpak.org/setup/

### Step 2: Add Flathub Repository

```bash
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
```

### Step 3: Install BLOUedit

```bash
flatpak install flathub com.blouedit.BLOUedit
```

### Step 4: Run BLOUedit

```bash
flatpak run com.blouedit.BLOUedit
```

## Building the Flatpak Package

If you want to build the Flatpak package yourself:

### Step 1: Install Flatpak and Flatpak Builder

```bash
# Fedora
sudo dnf install flatpak flatpak-builder

# Ubuntu/Debian
sudo apt install flatpak flatpak-builder
```

### Step 2: Add Flathub Repository

```bash
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
```

### Step 3: Install GNOME SDK

```bash
flatpak install flathub org.gnome.Platform//44 org.gnome.Sdk//44
```

### Step 4: Build and Install

```bash
flatpak-builder --user --install build-flatpak com.blouedit.BLOUedit.yml
```

### Step 5: Run

```bash
flatpak run com.blouedit.BLOUedit
```

## Installing Python Dependencies for Development

If you're developing the AI features and want to test them outside the Flatpak:

```bash
pip3 install torch torchvision torchaudio numpy opencv-python pillow
```

## Additional Notes

- For hardware-accelerated video processing, make sure you have the appropriate GPU drivers installed.
- For NVIDIA GPUs, install the CUDA toolkit for better AI performance.
- For AMD GPUs, install ROCm for PyTorch acceleration.

## Troubleshooting

### Missing GStreamer Plugins

If you encounter issues with media playback, you might need additional GStreamer plugins:

```bash
# Fedora
sudo dnf install gstreamer1-plugins-ugly gstreamer1-libav

# Ubuntu/Debian
sudo apt install gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

### Python Module Not Found

If you installed BLOUedit from source and encounter Python module import errors:

```bash
export PYTHONPATH=/usr/local/share/blouedit/ai/python:$PYTHONPATH
```

### GPU Acceleration Not Working

Make sure your GPU drivers are properly installed and that PyTorch can access your GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Should return `True` if CUDA is properly configured.

## Contact and Support

For issues and support, please file an issue on our GitHub repository or contact us at support@blouedit.com. 