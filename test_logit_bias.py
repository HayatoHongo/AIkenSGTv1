from openai import OpenAI
import tiktoken

client = OpenAI()

model = "gpt-4o-mini"
enc = tiktoken.encoding_for_model(model)

prompt = (
    "What is 2 + 2?\n"
    "A. 1\n"
    "B. 2\n"
    "C. 3\n"
    "D. 4\n"
    "Answer:"
)

d_id = enc.encode(" D")[0]

print("' D' token id:", d_id)
print("decoded:", repr(enc.decode([d_id])))

response = client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=1,
    temperature=0,
    logprobs=5,
    logit_bias={
        str(d_id): -100
    },
)

print(
    response.choices[0]
    .logprobs
    .top_logprobs[-1]
)