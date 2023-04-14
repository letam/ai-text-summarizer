import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_PROMPT = "Tl;dr"
DEFAULT_MODEL = "text-davinci-003"

MAX_TOKEN_LENGTH = 4097


def summarize_text(text, prompt=DEFAULT_PROMPT, model=DEFAULT_MODEL):
  try:
    if model == "gpt-3.5-turbo":
      messages = [{"role": "user", "content": f"{text}\n\n{prompt}"}]
      response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
      )
      # print(response)
      # global r
      # r = response
      return response["choices"][0]["message"]["content"]
    else:
      response = openai.Completion.create(model=model,
                                          prompt=f"{text}\n\n{prompt}",
                                          temperature=0.7,
                                          max_tokens=2000,
                                          top_p=1.0,
                                          frequency_penalty=0.0,
                                          presence_penalty=1)
      return response["choices"][0]["text"]
  except openai.error.InvalidRequestError as error:
    if error.user_message.startswith("This model's maximum context length is"):
      text_part_1 = text[:int(len(text) / 2)]
      text_part_2 = text[int(len(text) / 2 + 1):]
      summary_part_1 = summarize_text(text_part_1, model=model)
      summary_part_2 = summarize_text(text_part_2, model=model)
      return summarize_text(summary_part_1 + summary_part_2, model=model)


def get_file_contents(path):
  with open(path, "r") as f:
    return f.read()


filename = input("Enter name of file to summarize: ")
prompt = get_file_contents(filename)

print("\nProcessing...\n")
print(summarize_text(prompt, model="gpt-3.5-turbo"))
