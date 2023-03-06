AI Assistant for Telegram Bot:<br />

This bot provides two input options: text and voice.<br />
The bot retrieves responses from ChatGPT for both types of input.<br />

For voice input, the pipeline includes: <br />
     &ensp;1. Receive voice and process it on the server (ai_voice_bot.py) <br />
     &ensp;2. Use Whisper AI (from OpenAI) model locally that converts Speech to Text (STT model) <br />
     &ensp;3. The text generated is then corrected using ChatGPT API, which can be especially useful for children <br />
     &ensp;4. Use ChatGPT API with prompt and limited history for each particular user <br />
     &ensp;5. Convert response from ChatGPT to voice by using Text to Speech models ( TTS by coqui-AI and silero_tts for RU language) <br />
     

This is a pet project created for my daughter :)


link to medium article - https://medium.com/@andreyhaykin/make-your-own-telegram-voice-assistant-with-chatgpt-20d79c042a4f?source=friends_link&sk=27c8477a5209807be01fb054351bb0f0
