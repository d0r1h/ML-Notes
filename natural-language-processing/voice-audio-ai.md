---
description: Sub-field in AI which handles text to audio and vice versa ....
icon: microphone
---

# Voice (Audio) AI

There are two types of architecture for Voice AI agents :-<br>

1. **Speech to Speech** :- Native Audio handling using model in real time
2. **Chained** :- Transforming audio to text and back using models

#### Chained Model Architecture ....

Core Pipeline Component for Any Chained Voice AI Agent :-

<figure><img src="../.gitbook/assets/unknown (2) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

A chained architecture processes audio sequentially, converting audio to text, generating intelligent responses using large language models (LLMs), and synthesizing audio from text.<br>

1. **Speech to Text (ASR)** → The foundation of any voice agent begins with accurate speech recognition. Modern ASR systems like Deepgram, Whisper, and AssemblyAI provide the critical first step of converting audio input into processable text. The choice of ASR provider significantly impacts accuracy, especially in specialized domains like healthcare where medical vocabulary precision is paramount.
2. **Large Language Model (LLM)** → At the heart of the system lies the reasoning engine. Popular choices include OpenAI’s GPT models, Anthropic’s Claude, Meta’s Llama variants, and specialized models from providers like Deepseek and Gemini. For production applications requiring low latency, fast inference providers like Groq, Cerebras, and TogetherAI have become increasingly popular.
3. **Text to Speech (TTS)** → The final output layer converts generated responses back into natural-sounding speech. ElevenLabs has emerged as a leader in this space, offering high-quality voice synthesis including custom voice cloning capabilities. Other notable providers include Microsoft Azure Speech Services and Google’s WaveNet technology.<br>

Advance Processing Layers

1. **Voice Activity Detection (VAD)** → Critical for natural conversation flow, VAD systems detect when users are speaking versus when they’ve finished their turn. This component is essential for managing interruptions and maintaining conversational rhythm.
2. **End-of-Turn Detection** → Working alongside VAD, this system determines when a speaker has completed their thought, enabling the agent to respond at appropriate moments without awkward pauses or interruptions.
3. **Emotional Intelligence Engine** → Modern voice agents incorporate emotional processing capabilities through services like Hume AI and Affectiva, enabling them to detect and respond to emotional cues in speech patterns and tone

**END to END flow of Chained Model**

1. Input Layer → Person interacting with Voice AI agent, and asking questions
2. Speech Processing Pipeline
3. Pre Processing → audio signal is cleaned and normalized for to enhance clarity
4. Feature Extraction → Techniques such as the Mel Spectrogram are used to convert the audio into a visual representation. This representation highlights frequency changes over time, making the system's analysis easier.
5. ASR → The next step is Automatic Speech Recognition (ASR) or STT, transforming the audio signal into text.
6. Natural Language Understanding → Once text is generated it is sent to NLU system
7. Dialogue management and State handling → The AI voice agent must maintain context throughout the conversation. This is achieved through dialog management and state handling, which allows the system to track the conversation's flow and manage different states over time.
8. Processing and decision making → In this step, the AI voice agent determines the appropriate action based on the analysis of the input data. This can be enhanced by using RAG.
9. Response Generation → After processing the request, the system generates a response using an LLM to ensure the reply is clear and professional.
10. TTS → The text-based reply is then converted into speech through a TTS system, which synthesizes the response to sound natural.
11. Voice Output → Finally, the synthesized speech is played back to the user through the device’s speaker, completing the interaction.

**TTS (Text to Speech)**

Speech synthesis is the task of generating speech from some other modality like text, lip movements, etc. In most applications, text is chosen as the preliminary form because of the rapid advance of natural language systems. A Text To Speech (TTS) system aims to convert natural language into speech.

**ASR (Automatic Speech Recognition)**

Processing Human speech into readable text, popular example are youtube, tiktok captions,

* Traditional Hidden Markov models (HHM) and Gaussian Mixture Model (GMM)
* Large Langaue Model (LLM)

#### S2S Model Architecture

The emergence of direct voice-to-voice models represents a significant architectural shift. These systems, including OpenAI’s real-time API and Hume AI’s EVI 2, bypass the traditional STT-LLM-TTS pipeline, offering potentially lower latency and more natural conversational flow.

<figure><img src="../.gitbook/assets/unknown (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

Resource for code and Datasets :-

**`ASR Dataset (Indian)`**

* [ai4bharat/Lahaja · Datasets at Hugging Face](https://huggingface.co/datasets/ai4bharat/Lahaja)
* [ai4bharat/Shrutilipi · Datasets at Hugging Face](https://huggingface.co/datasets/ai4bharat/Shrutilipi)
* [ai4bharat/Svarah · Datasets at Hugging Face](https://huggingface.co/datasets/ai4bharat/Svarah)

**`Fine Tuning LLM for ASR data :-`**

* [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)
* [What you’ll learn and what you’ll build - Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter5/introduction)
* [Speech LLMs: Models that listen and talk back](https://www.youtube.com/watch?v=MyxgEx4_Moo)

**`YouTube Videos & Blogs for Hands-on`**

* [https://www.youtube.com/watch?v=anplUNnkM68](https://www.youtube.com/watch?v=anplUNnkM68) \[Fine tuning Whisper for Speech Transcription]
* [https://www.youtube.com/watch?v=5-Dk3ooxn2Q](https://www.youtube.com/watch?v=5-Dk3ooxn2Q) \[Text to Speech Fine-tuning Tutorial]
* [https://www.youtube.com/watch?v=TTx6M4CCbXk](https://www.youtube.com/watch?v=TTx6M4CCbXk) \[Streaming Speech to Text Models]
* [https://www.youtube.com/watch?v=MyxgEx4\_Moo](https://www.youtube.com/watch?v=MyxgEx4_Moo) \[Speech LLMs: Models that listen and talk back]
* [https://www.youtube.com/watch?v=oZoy79S-edw](https://www.youtube.com/watch?v=oZoy79S-edw) \[GPT OSS Release, Inference and Fine tuning]
* [https://trelis.com/ADVANCED-audio/](https://trelis.com/ADVANCED-audio/)&#x20;
