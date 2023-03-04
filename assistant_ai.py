import logging
import os
from typing import Any

import openai
import torch
import whisper
from dotenv import load_dotenv
from TTS.api import TTS

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

VOICE_OUTPUT_FILE = "output.wav"
VOICE_INPUT_FILE = "input.ogg"
MAXIMUM_WINDOW_SIZE = 10


GPT_ASSISTANT_PROMPT_FOR_CHILD = {
    "role": "system",
    "content": "You are a helpful assistant to the child."
    " You need to prevent any harmless information"
    " to a child under 10 years. "
    " Try to understand text with a lot of mistakes in it."
    " You are highly intelligent little girl.",
}


GPT_ASSISTANT_PROMPT_FOR_ADULT = {
    "role": "system",
    "content": "You are a helpful assistant." " You are highly intelligent teacher",
}


class AssistantAI:
    def __init__(self, language: str = "en") -> None:
        self._voice_recognition_model = whisper.load_model("base")

        if language == "en":
            self._text_to_speech_model = TTS(TTS.list_models()[0])
        elif language == "ru":
            self._text_to_speech_model = self._init_tts_model_ru_model()
        else:
            raise Exception("Not supported language")
        self._text_generator_model = "gpt-3.5-turbo"

        self._current_talk_per_user = {}
        self._system_msg = GPT_ASSISTANT_PROMPT_FOR_CHILD
        self._lang = language
        self._use_previous_history_per_user = None  # disabled for privacy reasons

    @staticmethod
    def _init_tts_model_ru_model() -> torch.nn.Module:
        language = "ru"
        model_id = "v3_1_ru"
        device = torch.device("cuda")

        model, example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=language,
            speaker=model_id,
        )
        model.to(device)
        return model

    def change_gpt_system_prompt(self, user_id: int, is_adult: bool = True) -> None:
        if is_adult is True:
            new_system_prompt = GPT_ASSISTANT_PROMPT_FOR_ADULT
        else:
            new_system_prompt = GPT_ASSISTANT_PROMPT_FOR_CHILD

        self._current_talk_per_user[user_id] = [new_system_prompt]

    def create_response_from_text(self, message: str, user_id: int) -> str:
        self._update_user_current_history(user_id, message)

        response = self.make_request_to_open_ai_chat_gpt(
            self._current_talk_per_user[user_id]
        )

        self._current_talk_per_user[user_id].append(
            {"role": "assistant", "content": response}
        )
        return response

    def create_response_from_voice(self, user_id: int) -> Any:
        # generate text from user voice with Whisper Open AI model
        text_from_user_voice = self._voice_recognition_model.transcribe(
            VOICE_INPUT_FILE
        )["text"]

        logging.info(f"response from Whisper AI: {text_from_user_voice}")

        self._update_user_current_history(user_id, text_from_user_voice)

        # run chatGPT to get response with history
        text_response = self.make_request_to_open_ai_chat_gpt(
            self._current_talk_per_user[user_id]
        )
        self._current_talk_per_user[user_id].append(
            {"role": "assistant", "content": text_response}
        )

        logging.info(f"response from GPT: {text_response}")

        # generate Voice from text by using YourTTs model
        if self._lang == "en":
            self._text_to_speech_model.tts_to_file(
                text=text_response,
                speaker=self._text_to_speech_model.speakers[1],
                language=self._text_to_speech_model.languages[0],
                file_path=VOICE_OUTPUT_FILE,
            )
            # self._text_to_speech_model.tts_to_file(text_response,
            #                                        speaker_wav="custom_voice.wav",
            #                                        language="en",
            #                                        file_path=VOICE_OUTPUT_FILE)
            voice_output = VOICE_OUTPUT_FILE

        else:
            speaker = "xenia"
            put_accent = True
            put_yo = True
            sample_rate = 48000
            voice_output = self._text_to_speech_model.save_wav(
                text=text_response,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo,
            )
        return open(voice_output, "rb")

    def make_request_to_open_ai_chat_gpt(self, prompt: str) -> str:
        # Generate a response
        completion = openai.ChatCompletion.create(
            model=self._text_generator_model, messages=prompt
        )
        return completion["choices"][0]["message"]["content"]

    def _update_user_current_history(
        self, user_id: int, text_from_user_voice: str
    ) -> None:
        if user_id not in self._current_talk_per_user:
            self._current_talk_per_user[user_id] = [self._system_msg]

        # we want to move it like sliding window
        # in purpose to limit input tokes (minimize our payments)
        elif len(self._current_talk_per_user[user_id]) >= MAXIMUM_WINDOW_SIZE:
            del self._current_talk_per_user[user_id][1]
        self._current_talk_per_user[user_id].append(
            {"role": "user", "content": text_from_user_voice}
        )
        logging.info(
            f"queue size of messages per user for GPT {len(self._current_talk_per_user[user_id])}"
        )
