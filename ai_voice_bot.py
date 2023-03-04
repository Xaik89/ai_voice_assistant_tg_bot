import os

import telebot
from dotenv import load_dotenv

from assistant_ai import VOICE_INPUT_FILE, AssistantAI

load_dotenv()
BOT_TOKEN = os.environ.get("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)
model = AssistantAI(language="ru")


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(
        message,
        "how are you doing, how can I help you?\nThe default language is Russian,"
        " you can change it to English by typing /use_en \n"
        "The default assistant is for Child, if you want to change it to adult, please type /adult",
    )


@bot.message_handler(commands=["use_en"])
def change_language_to_en(message):
    global model
    model = AssistantAI(language="en")
    bot.send_message(message.chat.id, "Done")


@bot.message_handler(commands=["use_ru"])
def change_language_to_ru(message):
    global model
    model = AssistantAI(language="ru")
    bot.send_message(message.chat.id, "Done")


@bot.message_handler(commands=["adult"])
def change_gpt_system_setup_to_adult(message):
    model.change_gpt_system_prompt(message.from_user.id, is_adult=True)
    bot.send_message(message.chat.id, "Done")


@bot.message_handler(commands=["child"])
def change_gpt_system_setup_to_child(message):
    model.change_gpt_system_prompt(message.from_user.id, is_adult=False)
    bot.send_message(message.chat.id, "Done")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    text_response = model.create_response_from_text(message.text, message.from_user.id)
    bot.send_message(message.chat.id, text_response)


@bot.message_handler(content_types=["voice"])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(VOICE_INPUT_FILE, "wb") as new_file:
        new_file.write(downloaded_file)

    voice_response = model.create_response_from_voice(message.from_user.id)
    bot.send_voice(message.chat.id, voice_response)


bot.infinity_polling()
