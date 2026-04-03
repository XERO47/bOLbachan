#!/usr/bin/env bash
# Set up a v4l2loopback virtual camera on the SERVER so the lip-synced stream
# can appear as /dev/video10 — useful if you want the server itself to
# present a virtual webcam to other processes.
#
# On the CLIENT side (Windows/Mac), use OBS Virtual Camera instead.

set -euo pipefail

VCAM_DEV="${1:-10}"   # creates /dev/video10 by default

echo "Loading v4l2loopback module as /dev/video${VCAM_DEV} ..."
sudo modprobe v4l2loopback \
    video_nr=$VCAM_DEV \
    card_label="LipSync-VCam" \
    exclusive_caps=1

echo "Virtual camera ready at /dev/video${VCAM_DEV}"
echo ""
echo "To push the server's output stream into it:"
echo "  ffmpeg -re -i http://localhost:8000/stream.mjpeg \\"
echo "    -vf scale=1280:720 \\"
echo "    -f v4l2 /dev/video${VCAM_DEV}"
echo ""
echo "Or from inside the container:"
echo "  docker exec lipsync-server python3 /app/server/vcam_writer.py"
