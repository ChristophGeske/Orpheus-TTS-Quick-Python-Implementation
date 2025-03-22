# Orpheus-TTS-Quick-Python-Implementation
I first tried to run it on Google Colabs because there was a direct link on their site https://canopylabs.ai/model-releases but Colab crashed on my free accound because of ram overflow.

I than simply run it on my laptop 16GB Ram and 8 GB VRAM 3080 Laptop GPU and it worked great. I had to slightly modefy the colab script but now it works and the generated voices sound great.

Only needs one file to run and the following installs:

pip install torch transformers huggingface_hub numpy soundfile snac ipywebrtc


Make sure to replace the placeholder with your huggingface token which you can get under settings - tokens in your free hugging face account.

Model download will take a wile 30-60 minutes for the roughly 15 GB model.
Generation time is also not realtime and takes half a minute or so on my low spec laptop, but it sounds incredible.





