#!/usr/bin/env python3
"""
테스트용 비디오 생성 스크립트

간단한 얼굴 + 음성 비디오를 생성합니다.
"""

import numpy as np
import cv2
import soundfile as sf
import subprocess
import tempfile
import os
from pathlib import Path


def create_test_video(
    output_path: str,
    duration_sec: int = 10,
    fps: int = 30,
    sample_rate: int = 16000,
    width: int = 640,
    height: int = 480
):
    """
    테스트용 비디오 생성 (얼굴 이미지 + 사인파 오디오).
    
    실제 사용 시에는 실제 비디오 파일을 사용하세요.
    이 스크립트는 파이프라인 테스트용입니다.
    """
    print(f"테스트 비디오 생성 중: {output_path}")
    
    # 임시 파일들
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        video_temp = tmp_video.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_temp = tmp_audio.name
    
    try:
        # 1. 비디오 생성 (간단한 얼굴 애니메이션)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_temp, fourcc, fps, (width, height))
        
        total_frames = duration_sec * fps
        
        for frame_idx in range(total_frames):
            # 배경
            frame = np.ones((height, width, 3), dtype=np.uint8) * 200
            
            # 간단한 얼굴 그리기
            center_x, center_y = width // 2, height // 2
            
            # 얼굴 (원)
            cv2.circle(frame, (center_x, center_y), 80, (255, 220, 180), -1)
            
            # 눈 (시간에 따라 변화)
            eye_offset = int(10 * np.sin(frame_idx / fps * 2))
            cv2.circle(frame, (center_x - 30, center_y - 20 + eye_offset), 10, (0, 0, 0), -1)
            cv2.circle(frame, (center_x + 30, center_y - 20 + eye_offset), 10, (0, 0, 0), -1)
            
            # 입 (시간에 따라 변화 - 말하는 듯)
            mouth_y = center_y + 20 + int(5 * np.sin(frame_idx / fps * 4))
            cv2.ellipse(frame, (center_x, mouth_y), (30, 15), 0, 0, 180, (0, 0, 0), 2)
            
            # 텍스트
            time_sec = frame_idx / fps
            cv2.putText(frame, f"Time: {time_sec:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            out.write(frame)
        
        out.release()
        print(f"  ✓ 비디오 생성 완료 ({total_frames} 프레임)")
        
        # 2. 오디오 생성 (사인파 + 휴지기)
        # 발화 1: 0-3초 (440Hz)
        # 휴지기: 3-4초
        # 발화 2: 4-7초 (523Hz)
        # 휴지기: 7-8초
        # 발화 3: 8-10초 (659Hz)
        
        audio = np.zeros(duration_sec * sample_rate, dtype=np.float32)
        
        def add_tone(start_sec, end_sec, freq_hz, amplitude=0.3):
            start_idx = int(start_sec * sample_rate)
            end_idx = int(end_sec * sample_rate)
            t = np.arange(end_idx - start_idx) / sample_rate
            tone = amplitude * np.sin(2 * np.pi * freq_hz * t)
            audio[start_idx:end_idx] = tone.astype(np.float32)
        
        # 3개 발화 추가
        add_tone(0, 3, 440)   # A4
        add_tone(4, 7, 523)   # C5
        add_tone(8, 10, 659)  # E5
        
        sf.write(audio_temp, audio, sample_rate)
        print(f"  ✓ 오디오 생성 완료 ({duration_sec}초)")
        
        # 3. 비디오 + 오디오 합치기 (ffmpeg)
        cmd = [
            'ffmpeg',
            '-i', video_temp,
            '-i', audio_temp,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            '-y',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✓ 최종 비디오 생성: {output_path}")
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(video_temp):
            os.remove(video_temp)
        if os.path.exists(audio_temp):
            os.remove(audio_temp)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="테스트용 비디오 생성")
    parser.add_argument(
        "--output",
        type=str,
        default="test_video/sample.mp4",
        help="출력 비디오 경로"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="비디오 길이 (초)"
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_test_video(str(output_path), args.duration)
    
    print("\n" + "="*60)
    print("✅ 테스트 비디오 생성 완료!")
    print("="*60)
    print(f"파일: {output_path}")
    print(f"\n다음 명령으로 테스트하세요:")
    print(f"  python scripts/demo_video_stream.py --video {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
