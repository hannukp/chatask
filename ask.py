#!/usr/bin/env python3
import base64
import io
import json
import os
import platform
import sys
import tempfile
import time
import urllib.request
import urllib.error


def send_post_request(url: str, data: dict, api_key: str):
    """Make an HTTP POST request to OpenAI API"""
    req = urllib.request.Request(
        url,
        json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "chatask",
            "Authorization": f"Bearer {api_key}",
        },
    )
    return urllib.request.urlopen(req)


def receive_full_json(response):
    """Receive full response body and parse as JSON."""
    return json.loads(response.read().decode("utf-8"))


def receive_streaming(response):
    """Generator that receives server-sent events and yields contents as they arrive."""
    buffer = io.BytesIO()
    running = True
    message_separator = b"\n\n"
    while not response.closed and running:
        chunk = response.read(256)
        if not chunk:
            # no more data, stop
            break
        buffer.write(chunk)
        # double line break = one message
        if message_separator in chunk:
            received = buffer.getvalue()
            # find where last full message ends in the buffer
            message_stop = received.rfind(message_separator)
            messages = received[:message_stop]
            for message in messages.split(message_separator):
                if message.startswith(b"data:"):
                    message = message[5:].strip()
                    if message == b" [DONE]" or message == b"[DONE]":
                        running = False
                        break
                    part = json.loads(message)
                    content = part["choices"][0]["delta"].get("content")
                    if content is not None:
                        yield content
                    if part["choices"][0]["finish_reason"] == "stop":
                        running = False
                        break
            # replace buffer with what was remaining of the old buffer:
            remaining = received[message_stop + 2 :]
            buffer = io.BytesIO(remaining)
            buffer.seek(len(remaining))


def write_log(logfile, content):
    with open(logfile, "a") as f:
        json.dump(content, f)
        f.write("\n")


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


def query_chatgpt(messages, temperature: float, model: str, logfile: str, api_key: str, streaming=True):
    request_data = {"model": model, "messages": messages, "temperature": temperature, "stream": streaming}
    url = "https://api.openai.com/v1/chat/completions"
    start_time = time.time()
    write_log(
        logfile,
        {
            "time": start_time,
            "type": "request",
            "url": url,
            "data": request_data,
        },
    )

    resp = send_post_request(
        url,
        request_data,
        api_key=api_key,
    )

    if streaming:
        resp_data = {}
        content_buffer = io.StringIO()
        for c in receive_streaming(resp):
            print(c, end="", flush=True)
            content_buffer.write(c)
        print()
        content = content_buffer.getvalue()
        # It seems that the streaming endpoint doesn't respond with token counts
        cost = 0
    else:
        resp_data = receive_full_json(resp)
        cost = estimate_cost(resp_data)
        content = resp_data["choices"][0]["message"]["content"]

    end_time = time.time()
    time_taken = end_time - start_time
    write_log(
        logfile,
        {
            "time": end_time,
            "seconds": time_taken,
            "cost": cost,
            "type": "response",
            "data": resp_data,
        },
    )
    return content


def query_dall_e(prompt: str, logfile: str, api_key: str, dalle3=False) -> bytes:
    if dalle3:
        request_data = {"model": "dall-e-3", "prompt": prompt, "n": 1, "size": "1024x1024", "quality": "standard"}
    else:
        request_data = {"model": "dall-e-2", "prompt": prompt, "n": 1, "size": "512x512"}
    start_time = time.time()
    url = "https://api.openai.com/v1/images/generations"
    write_log(
        logfile,
        {
            "time": start_time,
            "type": "request",
            "url": url,
            "data": request_data,
        },
    )
    resp = send_post_request(
        url,
        request_data,
        api_key=api_key,
    )
    image_url = receive_full_json(resp)["data"][0]["url"]
    return urllib.request.urlopen(image_url).read()


def query_stability_text_to_image(prompt: str, engine_id: str, logfile: str, api_key: str) -> bytes:
    url = f"https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
    request_data = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE",
        "height": 512,
        "width": 512,
        "samples": 1,
        "steps": 30,
    }
    start_time = time.time()
    write_log(
        logfile,
        {
            "time": start_time,
            "type": "request",
            "url": url,
            "data": request_data,
        },
    )

    response = send_post_request(url, request_data, api_key=api_key)
    response_data = receive_full_json(response)
    img_base64 = response_data["artifacts"][0]["base64"]
    return base64.b64decode(img_base64)


class ChatAsk:
    def __init__(self, context: str, temperature: float, model: str, logfile: str, streaming: bool, api_key: str):
        self.context = context
        self.temperature = temperature
        self.messages = [{"role": "system", "content": context}] if context else []
        self.model = model
        self.logfile = logfile
        self.streaming = streaming
        self.history_length = 5
        self.api_key = api_key

    def ask(self, query: str):
        self.messages.append({"role": "user", "content": query})
        context_messages = [{"role": "system", "content": self.context}] if self.context else []
        try:
            answer = query_chatgpt(
                messages=context_messages + self.messages[-self.history_length :],
                temperature=self.temperature,
                model=self.model,
                logfile=self.logfile,
                streaming=self.streaming,
                api_key=self.api_key,
            )
            if not self.streaming:
                print(answer)
        except KeyboardInterrupt:
            print()
            print("Interrupted query... You can retry with -r")
            answer = ""
        except urllib.error.HTTPError as e:
            print(f"Query failed with code={e.code}, reason={e.reason}, body={e.read()}")
            answer = ""
        # answer = "Hello!"
        self.messages.append({"role": "assistant", "content": answer})

    def ask_again(self):
        while len(self.messages) and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        self.ask(self.messages.pop()["content"])


TEMPLATES = {
    "test": "Write a unit test for the following code:\n\n*BODY*",
    "doc": "Write documentation for the following code:\n\n*BODY*",
    "explaincode": "What does the following code do:\n\n*BODY*",
}

configfile = os.path.expanduser("~/.ask")
config = {
    "temperature": 0.7,
    "model": "gpt-3.5-turbo",
    "streaming": True,
    "logfile": os.path.join(tempfile.gettempdir(), "ask.log"),
    "templates": {},
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
    print("    -3     -- use gpt-3.5-turbo model (default)")
    print("    -4     -- use gpt-4 model")
    print("    -4t    -- use gpt-4-turbo model")
    print("    -i     -- generate image using dall-e 2 (must be streamed to output)")
    print("    -id3   -- generate image using dall-e 3 (must be streamed to output)")
    print("    -isd   -- generate image using stable diffusion (must be streamed to output)")
    print("    -v     -- verbose output")
    print()
    print("Example usage:")
    print("    ask what is the meaning of life")
    print("    ask -4 what is the meaning of life")
    print("    ask test 'const adder = (a: number, b: number) => a + b'")
    print("    ask explaincode 'const adder = (a: number, b: number) => a + b'")
    print("    ask test <example.py")
    print("    ask -t0 convert to typescript <example.py")
    print("    ask -i 'beautiful banana' >banana.png")
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
    streaming = config["streaming"]
    openai_api_key = os.environ.get("OPENAI_API_KEY", config.get("OPENAI_API_KEY"))
    stability_api_key = os.environ.get("STABILITY_API_KEY", config.get("STABILITY_API_KEY"))
    default_temperature = True
    image = None
    verbose = False
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            continue
        if a.startswith("-t"):
            temperature = float(a[2:])
            default_temperature = False
        elif a == "-3":
            model = "gpt-3.5-turbo"
        elif a == "-4":
            model = "gpt-4"
        elif a == "-4t":
            model = "gpt-4-1106-preview"
        elif a == "-v":
            verbose = True
        elif a == "-s":
            streaming = True
        elif a.startswith("-i"):
            key = a[2:]
            if key == "sd":
                image = "sd"
            elif key == "d3" or key == "dalle3":
                image = "dalle3"
            elif key == "" or key == "d2" or key == "dalle2":
                image = "dalle"
            else:
                print(f"ERROR: Unknown image model {key!r}", file=sys.stderr)
                sys.exit(1)

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
        if platform.system() != "Windows":
            sys.stdin = open("/dev/tty", "r")

    if not q:
        help_and_exit()
    if len(q) > 20000:
        print(f"ERROR: Too long question ({len(q)})", file=sys.stderr)
        sys.exit(1)

    if template:
        q = template.replace("*BODY*", q)

    if image:
        if sys.stdout.isatty():
            print("ERROR: The output is an PNG file; You must pipe it to a file or another process", file=sys.stderr)
            sys.exit(1)

        if image == "sd":
            engine_id = "stable-diffusion-xl-beta-v2-2-2"
            if verbose:
                print(f"# generating image using {engine_id}", file=sys.stderr)
            image_bytes = query_stability_text_to_image(
                prompt=q, engine_id=engine_id, logfile=logfile, api_key=stability_api_key
            )
        else:
            dalle3=image == "dalle3"
            if verbose:
                print("# generating image using", "dall-e-3" if dalle3 else "dall-e-2", file=sys.stderr)
            image_bytes = query_dall_e(prompt=q, logfile=logfile, api_key=openai_api_key, dalle3=dalle3)
        sys.stdout.buffer.write(image_bytes)
        return

    # This context seems to help with answers that start with 'As an AI language model...':
    context = "You are a helpful assistant."

    chatask = ChatAsk(
        temperature=temperature,
        context=context,
        model=model,
        logfile=logfile,
        streaming=streaming,
        api_key=openai_api_key,
    )
    if verbose:
        print(f"# {temperature=}, {model=}", file=sys.stderr)
    print(">>>", q)
    print("-" * 79)
    chatask.ask(q)
    if not sys.stdout.isatty():
        return

    print()
    while True:
        try:
            q = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            break
        q = q.strip()
        if not q:
            break
        print()
        if q == "-r":
            chatask.ask_again()
        elif q.startswith("-"):
            print("Use -r to repeat your previous query")
        else:
            chatask.ask(q)
        print()


main()
