#!/usr/bin/env python3
import json
import os
import sys
import urllib.request

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def send_post_request(url: str, data: dict) -> str:
    json_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, json_data)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {OPENAI_API_KEY}")
    response = urllib.request.urlopen(req)
    return response.read().decode("utf-8")


def query_chatgpt(messages, temperature):
    resp = send_post_request(
        "https://api.openai.com/v1/chat/completions",
        {"model": "gpt-3.5-turbo", "messages": messages, "temperature": temperature}
    )
    return json.loads(resp)["choices"][0]["message"]["content"]


class ChatAsk:
    def __init__(self, context=None, temperature=1):
        self.temperature = temperature
        self.messages = [{"role": "system", "content": context}] if context else []

    def ask(self, query: str):
        self.messages.append({"role": "user", "content": query})
        answer = query_chatgpt(self.messages, self.temperature)
        # answer = "Hello!"
        self.messages.append({"role": "assistant", "content": answer})
        return answer


TEMPLATES = {
    "test": "Write a unit test for the following code:\n\n*BODY*",
    "doc": "Write documentation for the following code:\n\n*BODY*",
    "explain": "What does the following code do:\n\n*BODY*",
}

configfile = os.path.expanduser("~/.ask")
if os.path.exists(configfile):
    with open(configfile, "rb") as f:
        config = json.load(f)
    TEMPLATES.update(config["templates"])


def help_and_exit():
    print("No questions?")
    print()
    print("Switches:")
    print("    -t0.1  -- set temperature to 0.1 (valid range 0-2)")
    print()
    print("Example usage:")
    print("    ask what is the meaning of life")
    print("    ask test 'const adder = (a: number, b: number) => a + b'")
    print("    ask explain 'const adder = (a: number, b: number) => a + b'")
    print("    ask test <example.py")
    print("    ask -t0 convert to typescript <example.py")
    print()
    print("Available templates:")
    for cmd, expansion in sorted(TEMPLATES.items()):
        print(f"    {cmd:<10} {expansion!r}")
    sys.exit(1)


def main():
    if len(sys.argv) < 2 or "-h" in sys.argv or "--help" in sys.argv:
        help_and_exit()

    temperature = 1
    default_temperature = True
    for a in sys.argv[1:]:
        if a.startswith('-t'):
            temperature = float(a[2:])
            default_temperature = False

    # ignore args that look like switches
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    # first parameter may be a template invocation
    template = TEMPLATES.get(args[0])
    if template:
        args = args[1:]
        if default_temperature:
            # default to 0 temp for coding tasks
            temperature = 0

    q = " ".join(args).strip()
    if not sys.stdin.isatty():
        q += "\n\n" + sys.stdin.read()

    if not q:
        help_and_exit()
    if len(q) > 2000:
        print(f"Too long question ({len(q)})")
        sys.exit(1)

    if template:
        q = template.replace("*BODY*", q)

    chatask = ChatAsk(temperature=temperature)
    print(">>>", q)
    print('-' * 79)
    print(chatask.ask(q))
    print()
    while True:
        try:
            q = input(">>> ")
        except EOFError:
            break
        q = q.strip()
        if not q:
            break
        print()
        print(chatask.ask(q))
        print()


main()
