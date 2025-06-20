import cv2
import subprocess
import json
import os
import tempfile

def get_rotation_from_ffprobe(path):
    try:
        result = subprocess.run([
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(result.stdout.decode())
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                side_data_list = stream.get('side_data_list', [])
                for side_data in side_data_list:
                    if side_data.get('side_data_type') == 'Display Matrix':
                        rotation = side_data.get('rotation', 0)
                        return int(rotation)
        return 0
    except Exception as e:
        print("ffprobe failed:", e)
        return 0

def rotated_frame_generator(input_path):
    """
    저장하지 않고, 회전된 프레임을 메모리에서 바로 yield로 전달합니다.
    """
    rotation = get_rotation_from_ffprobe(input_path)
    cap = cv2.VideoCapture(input_path)
    def rotate(frame):
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == -90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180 or rotation == -180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        else:
            return frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield rotate(frame)
    cap.release()

def rotated_frame_generator_from_bytes(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    yield from rotated_frame_generator(tmp_path)