import argparse

from src.bot import TelegramBot
from src.utils import parse_config

if __name__ == '__main__':
    # Script arguments can include path of the config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default="chatbot.cfg")
    args = arg_parser.parse_args()
    config_path = args.config

    config = parse_config(config_path)
    telegram_bot = TelegramBot(**config)
    telegram_bot.run_bot()