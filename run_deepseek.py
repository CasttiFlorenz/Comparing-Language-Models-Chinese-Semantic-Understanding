from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

# 根据你下载的模型修改路径
model_name = "/mnt/data/DeepSeek-R1-Distill-Qwen-1.5B"  # 或 7B 版本

questions = [
    '请说出以下两句话区别在哪里？1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少',
    '请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上',
    '他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道',
    '明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？',
    '领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上"意思"分别是什么意思。'
]

print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # CPU 模式使用 float32
    device_map="cpu",           # 强制使用 CPU
    low_cpu_mem_usage=True      # 降低内存使用
).eval()

print("模型加载完成！\n")

streamer = TextStreamer(tokenizer, skip_prompt=True)

for i, q in enumerate(questions, 1):
    print(f"{'='*60}")
    print(f"问题 {i}: {q}")
    print(f"{'='*60}")
    
    # DeepSeek 使用 chat_template，需要按对话格式输入
    messages = [{"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )
    print("\n")
