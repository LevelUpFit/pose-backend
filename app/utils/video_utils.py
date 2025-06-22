import subprocess
import json
import cv2
import ffmpeg

def get_video_info(path):
    """
    영상의 orientation, width, height, fps 등 분석에 필요한 정보를 반환
    """
    try:
        probe = ffmpeg.probe(path)
        stream = probe['streams'][0]
        tags = stream.get('tags', {})
        rotate = int(tags.get('rotate', 0)) if 'rotate' in tags else 0
        width = int(stream['width'])
        height = int(stream['height'])
        # avg_frame_rate가 더 정확한 경우가 많음
        fps_str = stream.get('avg_frame_rate', '0/1')
        try:
            num, denom = map(int, fps_str.split('/'))
            fps = num / denom if denom != 0 else 0
        except Exception:
            fps = 30  # fallback
        print(f"[video_utils] rotate: {rotate}, width: {width}, height: {height}, fps: {fps}", flush=True)
        return {
            "rotate": rotate,
            "width": width,
            "height": height,
            "fps": fps
        }
    except Exception as e:
        print("[video_utils] ffmpeg probe error:", e, flush=True)
        return {
            "rotate": 0,
            "width": None,
            "height": None,
            "fps": 30
        }

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
                        print(f"[video_utils] ffprobe rotation: {rotation}", flush=True)
                        return int(rotation)
        return 0
    except Exception as e:
        print("ffprobe failed:", e)
        return 0

def rotate_frame(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    else:
        return frame

def correct_landmark_rotation(landmark, rotation, rot_width, rot_height):
    x, y = landmark.x, landmark.y
    if rotation == -90:
        corrected_x = 1 - y
        corrected_y = x
    elif rotation == 90:
        corrected_x = y
        corrected_y = 1 - x
    elif abs(rotation) == 180:
        corrected_x = 1 - x
        corrected_y = 1 - y
    else:
        corrected_x = x
        corrected_y = y
    return corrected_x * rot_width, corrected_y * rot_height