#!/bin/bash

echo "========================================"
echo "비디오 스트림 감정 분석 빠른 테스트"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# 1. 필수 패키지 확인
echo "📦 1단계: 필수 패키지 확인 중..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg가 설치되어 있지 않습니다."
    echo "설치 명령: sudo apt-get install ffmpeg"
    exit 1
fi
echo "✅ FFmpeg 설치 확인"

# 2. Python 패키지 확인
echo ""
echo "📦 2단계: Python 패키지 설치 중..."
echo "   (처음 설치 시 2-5분 정도 걸릴 수 있습니다...)"
pip install openai-whisper ffmpeg-python silero-vad
echo "✅ Python 패키지 설치 완료"

# 3. 테스트 비디오 생성
echo ""
echo "🎬 3단계: 테스트 비디오 생성 중..."
if [ ! -f "test_video/sample.mp4" ]; then
    python scripts/create_test_video.py --output test_video/sample.mp4 --duration 10
else
    echo "✅ 테스트 비디오가 이미 존재합니다: test_video/sample.mp4"
fi

# 4. 비디오 처리 실행
echo ""
echo "🚀 4단계: 비디오 처리 시작..."
echo "========================================"
echo ""

python scripts/demo_video_stream.py --video test_video/sample.mp4

echo ""
echo "========================================"
echo "✅ 테스트 완료!"
echo "========================================"
