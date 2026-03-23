import subprocess
import json
import sys

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
        return 0  # 기본값: 회전 없음
    except Exception as e:
        print("ffprobe failed:", e)
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python ffprobe_test.py <영상파일경로>")
        sys.exit(1)
    video_path = sys.argv[1]
    rotation = get_rotation_from_ffprobe(video_path)
    print(f"회전 정보: {rotation}도")