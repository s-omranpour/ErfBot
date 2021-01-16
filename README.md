# ErfBot
This is a simple project to train a chatbot and deploy it on telegram. You can export your own private telegram chats as json files and then train your model on your own data.

## Model
The model used in this chatbot is DialoGPT (basically GPT2 with special training). We used huggingface transformers library for tokenizing and building the model and pytorch lightning to simplify training/evaluating/deploying/saving process.

## Telegram Bot
The code used for deploying the telegram bot is mostly borrowed from [gpt2bot](https://github.com/polakowo/gpt2bot)
