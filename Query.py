
from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text);
