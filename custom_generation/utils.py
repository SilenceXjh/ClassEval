from openai import OpenAI
import torch


def model_generate(prompt: str, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print("model input text:", text)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096, 
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in generated_text:
        generated_text = generated_text.split(prompt)[-1].strip()
    
    # print("model generated text:", generated_text)
    return generated_text


def ds_api_generate(prompt: str, client: OpenAI, max_new_tokens=4096):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_new_tokens,
        n=1,
        stream=False
    )
    generated_text = response.choices[0].message.content
    return generated_text


def extract_python_code(generated_text: str):
    if "```python" in generated_text:
        code = generated_text.split("```python")[1].split("```")[0].strip()
    elif "```" in generated_text:
        code = generated_text.split("```")[1].split("```")[0].strip()
    else:
        code = generated_text.strip()

    return code