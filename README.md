# ChatAsk

Query ChatGPT or Claude (Sonnet) from command line. Implemented in Python, using only standard libraries.

First get your own API key from https://platform.openai.com/ or https://www.anthropic.com/api

## Installation

Create file called ~/.ask for the settings:

    {
        "OPENAI_API_KEY": "sk-xxx",
        "STABILITY_API_KEY": "sk-xxx",
        "ANTHROPIC_API_KEY": "sk-xxx",
        "model": "gpt-4o"
    }

Create an alias that points to the main python file:

    alias ask="$(pwd)/ask.py"

## Usage

    # let's start with an easy task
    ask what is the meaning of life
    
    # maybe gpt-4 knows?
    ask -m4 what is the meaning of life
    
    # write unit tests for code
    ask test 'const adder = (a: number, b: number) => a + b'

    # write unit tests for a file:
    ask test <example.py

    # explain code
    ask explaincode 'const adder = (a: number, b: number) => a + b'

    # use temperature=0 for coding tasks
    ask -t0 convert to typescript <example.py

During an interactive session, up to two of your previous prompts and ChatGPT answers are sent as context.

## Alternatives

The purpose of this project is to be a minimal starting point for working with ChatGPT (or GPT-4).
You can fork it and extend it to suit your own workflow.

If you're looking for a more feature-complete CLI tool, consider using one of the following:

* https://github.com/sigoden/aichat -- Rust
* https://github.com/npiv/chatblade -- Python
* https://github.com/marcolardera/chatgpt-cli -- Python
* https://github.com/clevercli/clevercli -- Node & Typescript
* https://github.com/Nemoden/gogpt -- Go
