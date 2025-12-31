"""채팅 서비스: PEFT QLoRA를 활용한 대화 및 학습.

QLoRA (Quantized LoRA): 4-bit 양자화된 모델에 LoRA 어댑터를 적용하는 방식.
단순 채팅/대화형 LLM 인터페이스.
세션별 히스토리 관리, 요약, 토큰 절약 전략 등.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

try:
    from app.service.midm_loader import DEFAULT_MIDM_PATH, load_midm  # EC2
except ImportError:
    from backend.app.service.midm_loader import DEFAULT_MIDM_PATH, load_midm  # Local

# QLoRA 어댑터 저장 경로
DEFAULT_QLORA_ADAPTER_PATH = "backend/app/model/midm_qlora"


class ChatService:
    """PEFT QLoRA를 사용한 대화 및 학습 서비스.

    QLoRA는 4-bit 양자화된 베이스 모델에 LoRA 어댑터를 적용하여
    메모리 효율적으로 파인튜닝하는 방식입니다.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        qlora_adapter_path: Optional[str] = None,
        use_quantization: bool = True,
        load_in_4bit: bool = True,
    ):
        """ChatService 초기화 (QLoRA 방식).

        Args:
            model_path: 기본 모델 경로. None이면 DEFAULT_MIDM_PATH 사용.
            qlora_adapter_path: QLoRA 어댑터 경로. None이면 DEFAULT_QLORA_ADAPTER_PATH 사용.
            use_quantization: 양자화 사용 여부 (QLoRA는 기본적으로 True).
            load_in_4bit: 4-bit 양자화 사용 여부 (QLoRA 필수).
        """
        self.model_path = model_path or DEFAULT_MIDM_PATH
        self.qlora_adapter_path = qlora_adapter_path or DEFAULT_QLORA_ADAPTER_PATH
        # QLoRA는 기본적으로 4-bit 양자화를 사용
        self.use_quantization = use_quantization
        self.load_in_4bit = load_in_4bit if use_quantization else False

        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model: Optional[PreTrainedModel] = None
        self.peft_model: Optional[PeftModel] = None

    def _load_base_model(self) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """QLoRA용 4-bit 양자화된 베이스 모델과 토크나이저 로드."""
        if self.tokenizer is None or self.model is None:
            if self.use_quantization and self.load_in_4bit:
                # QLoRA: 4-bit 양자화 설정 (NF4 양자화 + Double Quantization)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",  # NormalFloat4 양자화
                    bnb_4bit_use_double_quant=True,  # Double Quantization으로 추가 메모리 절약
                )
                print("📦 Loading model with QLoRA (4-bit quantization)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            else:
                # 양자화 없이 로드 (일반 LoRA 모드)
                print("⚠ Loading model without quantization (standard LoRA mode)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer, self.model

    def _setup_qlora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05) -> None:
        """QLoRA 설정 및 모델 준비.

        QLoRA는 4-bit 양자화된 모델에 LoRA 어댑터를 추가하는 방식입니다.
        """
        if self.model is None:
            self._load_base_model()

        # QLoRA: LoRA 설정
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,  # LoRA rank
            lora_alpha=lora_alpha,  # LoRA alpha
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Mi:dm 모델 구조에 맞게 조정 필요
            bias="none",
        )

        # QLoRA: 4-bit 양자화된 모델을 학습 가능하도록 준비
        if self.use_quantization and self.load_in_4bit:
            print("🔧 Preparing 4-bit quantized model for QLoRA training...")
            self.model = prepare_model_for_kbit_training(self.model)

        # QLoRA: PEFT 모델 생성 (양자화된 베이스 모델 + LoRA 어댑터)
        self.peft_model = get_peft_model(self.model, lora_config)
        print("📊 Trainable parameters:")
        self.peft_model.print_trainable_parameters()

    def load_qlora_adapter(self, adapter_path: Optional[str] = None) -> None:
        """저장된 QLoRA 어댑터 로드."""
        adapter_path = adapter_path or self.qlora_adapter_path
        if not os.path.exists(adapter_path):
            print(f"⚠ QLoRA adapter not found at {adapter_path}. Using base model.")
            self._load_base_model()
            return

        self._load_base_model()
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"✓ QLoRA adapter loaded from {adapter_path}")

    def chat(
        self,
        message: str,
        max_new_tokens: int = 2048,  # 기본값 증가: 더 긴 응답 허용
        temperature: float = 0.7,
        top_p: float = 0.9,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """대화 생성.

        Args:
            message: 사용자 메시지.
            max_new_tokens: 최대 생성 토큰 수.
            temperature: 생성 온도.
            top_p: nucleus sampling 파라미터.
            conversation_history: 대화 히스토리 [(user, assistant), ...].

        Returns:
            생성된 응답 텍스트.
        """
        if self.peft_model is None:
            # QLoRA 어댑터가 있으면 로드, 없으면 기본 모델 사용
            if os.path.exists(self.qlora_adapter_path):
                self.load_qlora_adapter()
            else:
                self._load_base_model()
                self._setup_qlora()

        # 모델 확인
        if self.peft_model is None and self.model is None:
            raise ValueError("Model not loaded. Please initialize the model first.")

        model = self.peft_model if self.peft_model else self.model
        if model is None:
            raise ValueError("Model is None")

        # 대화 히스토리 포맷팅
        if conversation_history:
            prompt = self._format_conversation(conversation_history, message)
        else:
            prompt = f"사용자: {message}\n어시스턴트:"

        # 토크나이징 (max_length 지정으로 경고 해결)
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        if model is None:
            raise ValueError("Model not initialized")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # 입력 길이 제한 증가: 더 긴 대화 히스토리 허용
        )
        # token_type_ids 제거 (Mi:dm 모델이 사용하지 않음)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 응답 부분만 추출 (더 안전한 방법)
        # 입력 프롬프트 이후의 생성된 부분만 추출
        if "어시스턴트:" in generated_text:
            # 마지막 "어시스턴트:" 이후의 텍스트만 추출
            parts = generated_text.split("어시스턴트:")
            # 입력 프롬프트의 "어시스턴트:" 이후의 생성된 부분만 가져옴
            response = parts[-1].strip()
        else:
            # "어시스턴트:"가 없으면 전체 생성된 텍스트 반환
            # (입력 프롬프트 제거를 위해 prompt 길이만큼 제거)
            if prompt in generated_text:
                response = generated_text.split(prompt, 1)[-1].strip()
            else:
                response = generated_text.strip()

        return response

    def _format_conversation(self, history: List[Tuple[str, str]], current_message: str) -> str:
        """대화 히스토리를 프롬프트 형식으로 변환."""
        formatted = ""
        for user_msg, assistant_msg in history:
            formatted += f"사용자: {user_msg}\n어시스턴트: {assistant_msg}\n"
        formatted += f"사용자: {current_message}\n어시스턴트:"
        return formatted

    def train(
        self,
        conversations: List[List[Tuple[str, str]]],
        output_dir: Optional[str] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        """대화 데이터로 QLoRA 파인튜닝.

        QLoRA는 4-bit 양자화된 베이스 모델에 LoRA 어댑터를 학습하는 방식입니다.

        Args:
            conversations: 대화 리스트. 각 대화는 [(user, assistant), ...] 형식.
            output_dir: 학습된 어댑터 저장 경로.
            num_epochs: 학습 에포크 수.
            batch_size: 배치 크기.
            learning_rate: 학습률.
            r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
        """
        output_dir = output_dir or self.qlora_adapter_path

        # QLoRA: 모델 및 LoRA 설정
        self._setup_qlora(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        # 데이터셋 준비
        def format_prompt(conv: List[Tuple[str, str]]) -> str:
            """대화를 프롬프트 형식으로 변환."""
            text = ""
            for user, assistant in conv:
                text += f"사용자: {user}\n어시스턴트: {assistant}\n"
            return text.strip()

        texts = [format_prompt(conv) for conv in conversations]

        # 토크나이징
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=2048,  # 학습 시 입력 길이 제한 증가
                return_tensors="pt",
            )

        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM이므로 MLM 사용 안 함
        )

        # 학습 인자
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            remove_unused_columns=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # QLoRA 학습 실행
        print("🚀 Starting QLoRA fine-tuning (4-bit quantized model + LoRA adapter)...")
        trainer.train()

        # QLoRA 어댑터 저장
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✓ QLoRA adapter saved to {output_dir}")

    def save_adapter(self, adapter_path: Optional[str] = None) -> None:
        """현재 QLoRA 어댑터 저장."""
        if self.peft_model is None:
            raise ValueError("No PEFT model loaded. Train or load an adapter first.")

        adapter_path = adapter_path or self.qlora_adapter_path
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(adapter_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(adapter_path)
        print(f"✓ QLoRA adapter saved to {adapter_path}")


# 편의 함수
def create_chat_service(
    model_path: Optional[str] = None,
    qlora_adapter_path: Optional[str] = None,
    use_quantization: bool = True,
) -> ChatService:
    """ChatService 인스턴스 생성 (QLoRA 방식).

    Args:
        model_path: 기본 모델 경로.
        qlora_adapter_path: QLoRA 어댑터 경로.
        use_quantization: 4-bit 양자화 사용 여부 (QLoRA는 기본적으로 True).
    """
    return ChatService(
        model_path=model_path,
        qlora_adapter_path=qlora_adapter_path,
        use_quantization=use_quantization,
    )
