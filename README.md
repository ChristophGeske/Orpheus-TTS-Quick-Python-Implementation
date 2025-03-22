# Orpheus-TTS-Quick-Python-Implementation
I first tried to run it on Colabs https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing but it crashed on my free colab accound because of ram overflow.

I than simply run it on my laptop 16GB Ram and 8 GB VRAM 3080 Laptop GPU and it worked great.

Only needs one file to run and the following installs:

pip install torch transformers huggingface_hub numpy soundfile snac ipywebrtc


Make sure to replace the placeholder with your huggingface token which you can get under settings - tokens in your free hugging face account.

Download will take a wile 30-60 minutes for the roughly 15 GB model.
Generation time is also not realtime and takes half a minute or so, but it sounds incredible.



More about the original project here:

https://canopylabs.ai/model-releases
