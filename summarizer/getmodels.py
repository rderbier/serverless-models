# Load model from HF and save locally
from transformers import AutoTokenizer, T5ForConditionalGeneration 
# can use the generic class AutoModelForSeq2SeqLM instead of T5ForConditionalGeneration

# https://huggingface.co/docs/transformers/main_classes/text_generation

modelPath = "./model"
text = """ 
summarize: Hugging Face: Revolutionizing Natural Language Processing
Introduction
In the rapidly evolving field of Natural Language Processing (NLP), Hugging Face has emerged as a prominent and innovative force. This article will explore the story and significance of Hugging Face, a company that has made remarkable contributions to NLP and AI as a whole. From its inception to its role in democratizing AI, Hugging Face has left an indelible mark on the industry.
The Birth of Hugging Face
Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf. The name "Hugging Face" was chosen to reflect the company's mission of making AI models more accessible and friendly to humans, much like a comforting hug. Initially, they began as a chatbot company but later shifted their focus to NLP, driven by their belief in the transformative potential of this technology.
Transformative Innovations
Hugging Face is best known for its open-source contributions, particularly the "Transformers" library. This library has become the de facto standard for NLP and enables researchers, developers, and organizations to easily access and utilize state-of-the-art pre-trained language models, such as BERT, GPT-3, and more. These models have countless applications, from chatbots and virtual assistants to language translation and sentiment analysis.
"""

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = T5ForConditionalGeneration.from_pretrained("Falconsai/text_summarization")

model.save_pretrained(modelPath)

input_ids = tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt")
outputs = model.generate(input_ids, max_new_tokens=100, temperature=0.8, do_sample=True)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
