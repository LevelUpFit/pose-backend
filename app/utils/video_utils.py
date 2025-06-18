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