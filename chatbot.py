import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from flask import Flask, render_template, request, jsonify
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline
from translate import Translator
import logging
from transformers import AutoModelForSequenceClassification
from gtts import gTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


nltk.download('punkt')
nltk.download('stopwords')


model_name_gpt_neo = "EleutherAI/gpt-neo-125M"
model_name_distilbert = "distilbert-base-uncased-finetuned-sst-2-english"

try:
    intent_recognition_model = AutoModelForSequenceClassification.from_pretrained(model_name_distilbert)
except Exception as e:
    logger.error(f"Error loading intent recognition model: {e}")
    raise


sentiment_analysis_gpt_neo = pipeline("sentiment-analysis", model=model_name_gpt_neo)


sentiment_analysis_distilbert = pipeline("sentiment-analysis", model=model_name_distilbert)


fine_tuned_model_path = r'C:\Users\Brock\bot\fine-tuned-gpt-neo-125M'
fine_tuned_model = GPTNeoForCausalLM.from_pretrained(fine_tuned_model_path)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_path)


intent_recognition_pipeline = pipeline("sentiment-analysis", model=model_name_distilbert, device=device)


app = Flask(__name__)

class VirtualAssistant:
    def __init__(self, fine_tuned_model, fine_tuned_tokenizer):
        self.purpose = "I am a general-purpose virtual assistant that can answer questions and perform basic tasks."
        self.fine_tuned_model = fine_tuned_model
        self.fine_tuned_tokenizer = fine_tuned_tokenizer
        self.context = ""  
        self.user_intent = ""  
        self.previous_questions = {}
        self.conversation_memory = []  


    def translate_text(self, text, target_language='en'):
        """
        Translate text to the target language.

        :param text: The text to be translated.
        :param target_language: The target language code (e.g., 'es' for Spanish).
        :return: Translated text.
        """
        translator = Translator(to_lang=target_language)
        translated_text = translator.translate(text)
        return translated_text

    def generate_response(self, user_input_ids, max_length=200, num_options=3):
    
        user_input_ids = user_input_ids.to(device)

        
        self.conversation_memory.append({'user': user_input_ids, 'response': None})  # Initialize response as None

        input_text = f"{self.context} {self.fine_tuned_tokenizer.decode(user_input_ids[0], skip_special_tokens=True)}"

        
        generated_responses = []

        for _ in range(num_options):
            with torch.no_grad():
                output = self.fine_tuned_model.generate(
                    self.fine_tuned_tokenizer.encode(input_text, return_tensors="pt"),
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.5,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.9,
                    pad_token_id=self.fine_tuned_tokenizer.eos_token_id,
                    eos_token_id=self.fine_tuned_tokenizer.eos_token_id,
                    do_sample=True
                )

            decoded_response = self.fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)
            generated_responses.append(decoded_response)

            
            input_text = f"{self.context} {decoded_response}"

        
            print("Generated Responses:", generated_responses)

        
        unique_responses = list(set(generated_responses))
        print("Unique Responses:", unique_responses)

        
        if unique_responses:
            response = random.choice(unique_responses)
            print("Selected Response:", response)

            
            user_input_strings = [self.fine_tuned_tokenizer.decode(user_input_id, skip_special_tokens=True) for user_input_id in self.conversation_memory[-1]['user']]
            print("User Input Strings:", user_input_strings)

            while any(response_part in input_text for response_part in user_input_strings[:-1]) and len(user_input_strings[0]) > 1:
                print("Repetition detected. Regenerating response.")
                print("User Input Strings:", user_input_strings)
                print("Generated Response:", response)
                response = random.choice(unique_responses)


            return [response]
        else:
            return ["I'm sorry, I couldn't generate a response at the moment."]

    def determine_intent(self, user_input_ids):
        user_input = self.fine_tuned_tokenizer.decode(user_input_ids[0], skip_special_tokens=True).lower()

        try:
            
            intent_result = intent_recognition_model(user_input)
            intent_label = intent_result[0]['label']

            
            if intent_label == 'POSITIVE':
                return "positive"
            elif intent_label == 'NEGATIVE':
                return "negative"
            else:
                return "unknown"
        except Exception as e:
            logger.error(f"Error determining intent: {e}")
            return "unknown"

    def respond_to_user(self, user_input):
        
        preprocessed_input = self.preprocess_text(user_input)

        
        context_specific_response = self.handle_follow_up(preprocessed_input, self.conversation_memory[-1]['response'])
        if context_specific_response:
            return context_specific_response

        
        self.user_intent = self.determine_intent(self.conversation_memory[-1]['user'])

        
        generated_response = self.generate_response(self.conversation_memory[-1]['user'])

        
        self.conversation_memory[-1]['response'] = generated_response[0]

        return generated_response[0]

    def recall_memory(self):
        
        if self.conversation_memory:
            return self.conversation_memory[-1]['response']
        else:
            return "I don't have any information in my memory."


    def answer_question(self, question):
        
        preprocessed_question = self.preprocess_text(question)

        
        if preprocessed_question in self.previous_questions:
            return "I already answered that question: " + self.previous_questions[preprocessed_question]

        
        
        dynamic_responses = self.get_dynamic_responses()

        
        for keyword, response in dynamic_responses.items():
            if keyword.lower() in preprocessed_question:
                
                self.previous_questions[preprocessed_question] = response
                return response

        
        
        dataset = self.get_dataset()

        
        for conv in dataset["conversations"]:
            if preprocessed_question == self.preprocess_text(conv["user"]):
                
                self.previous_questions[preprocessed_question] = conv["assistant"]
                return conv["assistant"]

       
        responses = ["I'm not sure, could you please rephrase your question.", "I don't have that information.",
                     "I'm here to help with general queries."]
        response = random.choice(responses)
        
        self.previous_questions[preprocessed_question] = response
        return response
    
    
def convert_text_to_speech(text, save_path):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(save_path)


assistant = VirtualAssistant(fine_tuned_model, fine_tuned_tokenizer)

conversation_memory = []

@app.route('/')
def index():
    global conversation_memory
    return render_template('index.html', conversation=conversation_memory)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    global conversation_memory

    if request.method == 'POST':
        
        user_question = request.form['user_input']

        
        input_ids = fine_tuned_tokenizer.encode(user_question, return_tensors="pt").to(device)

       
        generated_response = assistant.generate_response(input_ids)

       
        audio_save_path = 'static/audio/response.mp3'
        convert_text_to_speech(generated_response[0], audio_save_path)

        
        conversation_memory.append({'user': user_question, 'response': generated_response[0]})

        return jsonify({'user': user_question, 'response': generated_response[0], 'audio_url': '/' + audio_save_path})

    
    return jsonify({'message': 'Welcome to the Virtual Assistant!'})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

