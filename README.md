# Orpheus-TTS-Quick-Python-Implementation
I first tried running the Orpheus TTS system on Google Colab because there was a direct link on the Orpheus site (https://canopylabs.ai/model-releases), where you can also find a voice sample. However, Colab on the free tier crashes due to a RAM overflow.

I then ran it on my laptop (16GB RAM, 8GB VRAM, RTX 3080 Laptop GPU), and it worked great. I had to slightly modify the Colab script to make it work in Visual Studio Code.

It only requires one file, orpheus_TTS.py, to run, along with the following dependencies:

pip install torch transformers huggingface_hub numpy soundfile snac ipywebrtc

Make sure to replace the placeholder with your Hugging Face token, which you can find under Settings → Tokens in your free Hugging Face account.

The model download takes a while—around 30–60 minutes for the roughly 15GB model. Generation time is not real-time, taking about 30 seconds on my low-spec laptop (Looking at my task-manager I think it ignores my 8GB GPU and runs on CPU and the 16GB RAM instead), but the output sounds incredible good.

Be aware the model halucinates meaning it might say additional words beyond the provided text. If that's not acceptable for your usecase you might need to switch to a regular non transformer- based TTS system.
