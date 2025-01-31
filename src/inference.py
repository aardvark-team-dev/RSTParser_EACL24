from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from parse.parse import parse_doc

# 모델과 토크나이저 로드
base_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    load_in_4bit=True
)

# LoRA 모델 로드
model = PeftModel.from_pretrained(
    model,
    "arumaekawa/rstdt-7b-span",  # 또는 다른 학습된 모델 경로
    device_map="auto"
)


# @functions-framework
def parse(doc):
    parsed_tree = parse_doc(
        doc,
        model,
        tokenizer,
        parse_type="bottom_up",  # or "top_down"
        rel_type="rel_with_nuc",
        corpus="rstdt"
    )
    print(f"Document ID: {doc['doc_id']}")
    print(f"Parsed Tree: {parsed_tree}")
    return parsed_tree