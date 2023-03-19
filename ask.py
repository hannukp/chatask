#!/usr/bin/env python3
import json
import os
import platform
import sys
import tempfile
import time
import urllib.request

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def send_post_request(url: str, data: dict) -> str:
    json_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, json_data)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {OPENAI_API_KEY}")
    response = urllib.request.urlopen(req)
    return response.read().decode("utf-8")


def write_log(logfile, content):
    with open(logfile, 'a') as f:
        json.dump(content, f)
        f.write('\n')


def estimate_cost(data):
    usage = data.get("usage")
    if not usage:
        return None
    model = data.get("model")
    if model.startswith("gpt-3.5"):
        prompt_cost = completion_cost = 0.002
    elif model.startswith("gpt-4"):
        prompt_cost = 0.03
        completion_cost = 0.06
    else:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (prompt_tokens * prompt_cost + completion_tokens * completion_cost) / 1e3


def query_chatgpt(messages, temperature: float, model: str, logfile: str):
    request_data = {"model": model, "messages": messages, "temperature": temperature}
    url = "https://api.openai.com/v1/chat/completions"
    start_time = time.time()
    write_log(logfile, {
        "time": start_time,
        "type": "request",
        "url": url,
        "data": request_data,
    })
    resp = send_post_request(
        url,
        request_data
    )
    resp_data = json.loads(resp)
    end_time = time.time()
    write_log(logfile, {
        "time": end_time,
        "seconds": end_time - start_time,
        "cost": estimate_cost(resp_data),
        "type": "response",
        "data": resp_data,
    })
    return resp_data["choices"][0]["message"]["content"]


class ChatAsk:
    def __init__(self, context: str, temperature: float, model: str, logfile: str):
        self.temperature = temperature
        self.messages = [{"role": "system", "content": context}] if context else []
        self.model = model
        self.logfile = logfile

    def ask(self, query: str):
        self.messages.append({"role": "user", "content": query})
        answer = query_chatgpt(
            messages=self.messages,
            temperature=self.temperature,
            model=self.model,
            logfile=self.logfile,
        )
        # answer = "Hello!"
        self.messages.append({"role": "assistant", "content": answer})
        return answer


TEMPLATES = {
    "test": "Write a unit test for the following code:\n\n*BODY*",
    "doc": "Write documentation for the following code:\n\n*BODY*",
    "explaincode": "What does the following code do:\n\n*BODY*",
}

configfile = os.path.expanduser("~/.ask")
config = {
    "temperature": 0.7,
    "model": "gpt-3.5-turbo",
    "logfile": os.path.join(tempfile.gettempdir(), 'ask.log'),
    "templates": {}
}
if os.path.exists(configfile):
    with open(configfile, "rb") as f:
        config.update(json.load(f))
    TEMPLATES.update(config["templates"])


def help_and_exit():
    print("No questions?")
    print()
    print("Switches:")
    print("    -t0.1  -- set temperature to 0.1 (valid range 0-2)")
    print("    -4     -- use gpt-4 model")
    print()
    print("Example usage:")
    print("    ask what is the meaning of life")
    print("    ask test 'const adder = (a: number, b: number) => a + b'")
    print("    ask explaincode 'const adder = (a: number, b: number) => a + b'")
    print("    ask test <example.py")
    print("    ask -t0 convert to typescript <example.py")
    print()
    print("Available templates:")
    for cmd, expansion in sorted(TEMPLATES.items()):
        print(f"    {cmd:<10} {expansion!r}")
    print()
    print("Config:")
    print(json.dumps(config, indent=2))
    sys.exit(1)


def main():
    if len(sys.argv) < 2 or "-h" in sys.argv or "--help" in sys.argv:
        help_and_exit()

    temperature = config["temperature"]
    model = config["model"]
    logfile = config["logfile"]
    default_temperature = True
    for a in sys.argv[1:]:
        if a.startswith('-t'):
            temperature = float(a[2:])
            default_temperature = False
        if a == '-4':
            model = 'gpt-4'

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
        if platform.system() != 'Windows':
            sys.stdin = open('/dev/tty', 'r')

    if not q:
        help_and_exit()
    if len(q) > 16000:
        print(f"Too long question ({len(q)})")
        sys.exit(1)

    if template:
        q = template.replace("*BODY*", q)

    # This context seems to help with answers that start with 'As an AI language model...':
    context = "You are a helpful assistant."

    chatask = ChatAsk(temperature=temperature, context=context, model=model, logfile=logfile)
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
