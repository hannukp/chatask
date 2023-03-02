#!/usr/bin/env python3
import json
import os
import sys
import urllib.request
from typing import Any

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def send_post_request(url: str, data: dict) -> str:
    json_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, json_data)
    req.add_header('Content-Type', 'application/json')
    req.add_header("Authorization", f"Bearer {OPENAI_API_KEY}")
    response = urllib.request.urlopen(req)
    return response.read().decode('utf-8')


def format_response(resp):
    return json.loads(resp)["choices"][0]["message"]["content"]


def query_chatgpt(messages):
    return format_response(send_post_request("https://api.openai.com/v1/chat/completions", {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }))


class ChatAsk:
    def __init__(self, context=None):
        self.messages = [{"role": "system", "content": context}] if context else []

    def ask(self, query: str):
        self.messages.append({"role": "user", "content": query})
        answer = query_chatgpt(self.messages)
        self.messages.append({"role": "assistant", "content": answer})
        return answer


templates = {
    "test": "Write a unit test for the following code:\n\n*BODY*"
}

if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
    print("No questions?")
    print("Available templates:")
    for cmd, expansion in sorted(templates.items()):
        print('  ', cmd.ljust(10), repr(expansion))
    sys.exit(1)

# first parameter may be a template invocation
template = templates.get(sys.argv[1])
if template:
    args = sys.argv[2:]
else:
    args = sys.argv[1:]

q = ' '.join(args).strip()
if not q:
    print("No questions?")
    sys.exit(1)
if len(q) > 1000:
    print("Too long questions")
    sys.exit(1)

if template:
    q = template.replace('*BODY*', q)


print('>>>', q)
chatask = ChatAsk()
print()
print(chatask.ask(q))
while True:
    print()
    try:
        text = input('>>> ')
    except EOFError:
        break
    text = text.strip()
    if not text:
        break
    print()
    print(chatask.ask(text))
