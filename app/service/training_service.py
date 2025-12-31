"""
학습 관련 서비스
training_service.py

QLoRA/PEFT 기반 학습 전체 파이프라인 오케스트레이션:

학습 데이터 로드 (repository 사용),

모델/어댑터 로드,

Trainer(SFTTrainer 등) 설정,

학습 실행,

체크포인트/최종 모델 저장 (repository 사용).

“클라이언트/라우터” 입장에서는 start_training(request)만 부르면 되게 감싸는 역할.
"""
