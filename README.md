# ChatAsk

Query ChatGPT from command line.

First get your own API key from https://platform.openai.com/

"Installation":

    alias ask="OPENAI_API_KEY=*key* $(pwd)/ask.py"

Usage:

    ask how to do a somersault

Note that currently your full chat history is fed back to ChatGPT so long chats become slower and more expensive.

