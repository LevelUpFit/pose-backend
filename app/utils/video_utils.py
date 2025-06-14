import cv2
import ffmpeg

def get_video_info(path):
    """
    영상의 orientation, width, height, fps 등 분석에 필요한 정보를 반환
    """
    try:
        probe = ffmpeg.probe(path)
        tags = probe['streams'][0].get('tags', {})
        rotate = int(tags.get('rotate', 0)) if 'rotate' in tags else 0
        width = int(probe['streams'][0]['width'])
        height = int(probe['streams'][0]['height'])
        fps = eval(probe['streams'][0]['r_frame_rate'])
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

# def correct_video_orientation(frame, rotate):
#     if rotate == 90:
#         return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     elif rotate == 180:
#         return cv2.rotate(frame, cv2.ROTATE_180)
#     elif rotate == 270:
#         return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     return frame

def correct_video_orientation(frame):
    # 무조건 오른쪽(시계방향)으로 90도 회전
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)