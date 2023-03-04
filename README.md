AI Assistant in Telegram Bot:<br />

It has two options for input:<br />
    a. text input (regular message)<br />
    b. voice input<br />
    
In both cases, the bot gets a response from ChatGPT. <br />
The pipeline for voice flow is: <br />
     1. Receive voice and process it on the server (ai_voice_bot.py) <br />
     2. Use Whisper AI (from OpenAI) model locally that converts Speech to Text (STT model) <br />
     3. Use ChatGPT API with prompt and limited history for each particular user <br />
     4. Convert response from ChatGPT to voice by using Text to Speech models ( TTS by coqui-AI and silero_tts for RU language) <br />
     

*-> It's my pet project for my daughter :)
