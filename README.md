# ChatAsk

Query ChatGPT from command line. Implemented in Python, using only standard libraries.

First get your own API key from https://platform.openai.com/

"Installation":

    alias ask="OPENAI_API_KEY=*key* $(pwd)/ask.py"

Usage:

    # let's start with an easy task
    ask what is the meaning of life
    
    # maybe gpt-4 knows?
    ask -4 what is the meaning of life
    
    # write unit tests for code
    ask test 'const adder = (a: number, b: number) => a + b'

    # write unit tests for a file:
    ask test <example.py

    # explain code
    ask explaincode 'const adder = (a: number, b: number) => a + b'

    # use temperature=0 for coding tasks
    ask -t0 convert to typescript <example.py

Note that currently your full chat history is fed back to ChatGPT so long chats become slower and more expensive.

## Alternatives

The purpose of this project is to be a minimal starting point for working with ChatGPT (or GPT-4).
You can fork it and extend it to suit your own workflow.

If you're looking for a more feature-complete CLI tool, consider using one of the following:

* https://github.com/sigoden/aichat -- Rust
* https://github.com/npiv/chatblade -- Python
* https://github.com/marcolardera/chatgpt-cli -- Python
* https://github.com/clevercli/clevercli -- Node & Typescript
* https://github.com/Nemoden/gogpt -- Go
