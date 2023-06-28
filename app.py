import numpy as np
import pandas as pd
import tensorflow as tf
import os, PyPDF2, re, pickle

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from KGQnA._exportPairs import exportToJSON
from KGQnA._getentitypair import GetEntity
from KGQnA._graph import GraphEnt
from KGQnA._qna import QuestionAnswer

# import secure_filename
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

tokenizer_facts_path_cases = 'weights/TOKENIZER_FACTS_MODEL_CASES.pkl'
tokenizer_facts_path_facts = 'weights/TOKENIZER_FACTS_MODEL_FACTS.pkl'
summarization_model_facts_path = 'weights/FACTS_SUMMARIZATION_MODEL.h5'

tokenizer_judgements_path_cases = 'weights/TOKENIZER_JUDGEMENTS_MODEL_CASES.pkl'
tokenizer_judgements_path_facts = 'weights/TOKENIZER_JUDGEMENTS_MODEL_FACTS.pkl'
summarization_model_judgements_path = 'weights/JUDGEMENTS_SUMMARIZATION_MODEL.h5'

with open(tokenizer_facts_path_cases, 'rb') as handle:
    tokenizer_facts_cases = pickle.load(handle)

with open(tokenizer_facts_path_facts, 'rb') as handle:
    tokenizer_facts_summarize = pickle.load(handle)

with open(tokenizer_judgements_path_cases, 'rb') as handle:
    tokenizer_judgements_cases = pickle.load(handle)

with open(tokenizer_judgements_path_facts, 'rb') as handle:
    tokenizer_judgements_summarize = pickle.load(handle)

def encoder(max_x_len, x_voc_size):
    encoder_inputs = tf.keras.layers.Input(shape=(max_x_len,))
    enc_emb = tf.keras.layers.Embedding(x_voc_size, 300, mask_zero=True)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(300, return_sequences=True, return_state=True)
    _, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]
    return encoder_inputs, encoder_states

def decoder(y_voc_size, encoder_states):
    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    dec_emb_layer = tf.keras.layers.Embedding(y_voc_size, 300, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(300, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(y_voc_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)
    return decoder_inputs, decoder_outputs

encoder_inputs, encoder_states = encoder(5500, len(tokenizer_facts_cases.word_index) + 1)
decoder_inputs, decoder_outputs = decoder(len(tokenizer_facts_summarize.word_index) + 1, encoder_states)
inference_model_facts = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
inference_model_facts.load_weights(summarization_model_facts_path)

encoder_inputs, encoder_states = encoder(5500, len(tokenizer_judgements_cases.word_index) + 1)
decoder_inputs, decoder_outputs = decoder(len(tokenizer_judgements_summarize.word_index) + 1, encoder_states)
inference_model_judgements = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
inference_model_judgements.load_weights(summarization_model_judgements_path)

def decontracted(phrase): 
    phrase = re.sub(r"won't", "will not", phrase) 
    phrase = re.sub(r"can\'t", "can not", phrase)  
    phrase = re.sub(r"n\'t", " not", phrase)  
    phrase = re.sub(r"\'re", " are", phrase)  
    phrase = re.sub(r"\'s", " is", phrase)  
    phrase = re.sub(r"\'d", " would", phrase)  
    phrase = re.sub(r"\'ll", " will", phrase)  
    phrase = re.sub(r"\'t", " not", phrase)  
    phrase = re.sub(r"\'ve", " have", phrase)  
    phrase = re.sub(r"\'m", " am", phrase)  
    return phrase

def clean_case(case):
    case=re.sub(r'\s+',' ', case)
    case=re.sub(r'\n',' ', case)
    case=re.sub(r"([?!¿])", r" \1 ", case)
    case=decontracted(case)
    case = re.sub('[^A-Za-z0-9.,]+', ' ', case)
    case = case.lower()

    return case
    
def inference_summarization(
                        input_text,
                        tokenizer_cases,
                        tokenizer_summarize,
                        inference_model,
                        max_x_len = 5500,
                        max_y_len = 600
                        ):

    input_text = clean_case(input_text)
    input_text = tokenizer_cases.texts_to_sequences([input_text])
    input_text = tf.keras.preprocessing.sequence.pad_sequences(input_text, maxlen=max_x_len, padding='post')

    summary = np.zeros((1, max_y_len))
    summary[0,0] = tokenizer_summarize.word_index['sostok']
    stop_condition = False
    i = 1
    while not stop_condition:
        preds = inference_model.predict([input_text, summary], verbose=0)
        pred = np.argmax(preds[0,i-1])
        summary[0,i] = pred
        i += 1
        if pred == tokenizer_summarize.word_index['eostok'] or i >= max_y_len:
            stop_condition = True

    summary = summary[0]
    new_summary = []
    for i in summary:
        if i != 0:
            new_summary.append(i)
    summary = ' '.join([tokenizer_summarize.index_word[i] for i in new_summary])
    summary = summary.replace('eostok', '').replace('sostok', '').strip()
    return summary

class QnA(object):
    def __init__(self):
        super(QnA, self).__init__()
        self.qna = QuestionAnswer()
        self.getEntity = GetEntity()
        self.export = exportToJSON()
        self.graph = GraphEnt()
        self.pdf_dir = 'data/references/'
        
    def read_pdf_data(self, pdf_file):
        pdf_path = self.pdf_dir + pdf_file + '.pdf' if pdf_file[-1] != '.' else self.pdf_dir + pdf_file + 'pdf' 
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        num_pages = pdf_reader.getNumPages()

        whole_text = ''
        for page in range(num_pages):
            page_obj = pdf_reader.getPage(page)
            text = page_obj.extractText()
            whole_text += f" {text}"
        pdf_file.close()

        whole_text = whole_text.replace('\n', ' ')
        whole_text = re.sub(' +', ' ', whole_text)
        whole_text = whole_text.strip().lower()
        return whole_text

    def extract_answers(self, question):
        all_files = os.listdir(self.pdf_dir)
        all_files = [file[:-3] for file in all_files if file[-3:] == 'pdf']  
        all_outputs = []
        for idx, file in enumerate(all_files):
            context = self.read_pdf_data(file)
            refined_context = self.getEntity.preprocess_text(context)
            try:
                outputs = self.qna.findanswer(question, con=context)
            except:
                _, numberOfPairs = self.getEntity.get_entity(refined_context)
                outputs = self.qna.findanswer(question, numberOfPairs)
            all_outputs.append(outputs)

            print("Processing file {} of {}".format(idx + 1, len(all_files)))

        answers = [output['answer'] for output in all_outputs]
        scores = [output['score'] for output in all_outputs]

        # get the best answer
        best_answer = answers[scores.index(max(scores))]
        reference = all_files[scores.index(max(scores))]
        return best_answer, reference
    

lemmatizer = WordNetLemmatizer()
re_tokenizer = RegexpTokenizer(r'\w+')
stopwords_list = stopwords.words('english')

tokenizer_pvd_path = 'weights/TOKENIZER_PVD.pkl'
model_pvd_weights = 'weights/MODEL_PVD.h5'

data_path = 'data/judgments/public-stories.xlsx'
class_dict_violation_flag = {
                            'yes': 1, 
                            'no': 0
                            }
class_dict_violation_type = {
                            'article 11. of the constitution' : 4,
                            'article 12. (1) of the constitution' : 3,
                            'article 13. (1) of the constitution' : 2,
                            'article 17. of the constitution' : 1,
                            'no-violation': 0
                            }

class_dict_violation_flag_rev = {v: k for k, v in class_dict_violation_flag.items()}
class_dict_violation_type_rev = {v: k for k, v in class_dict_violation_type.items()}

with open(tokenizer_pvd_path, 'rb') as fp:
    tokenizer_pvd = pickle.load(fp)

model_pvd = tf.keras.models.load_model(model_pvd_weights)

def extract_violation_data(violationType):
    df_ = pd.read_excel(data_path)
    df_.ViolationType = df_.ViolationType.str.lower().str.strip()
    df_ = df_[df_.ViolationType == violationType]
    df_ = df_.iloc[0]

    Lawyers = df_.Lawyers.replace('\n', ' ')
    Court = df_.Court.replace('\n', ' ')
    DocumentShouldBring = df_.DocumentShouldBring.replace('\n', ' ')
    Suggetion = df_.Suggetion.replace('\n', ' ')
    
    return {
        "Lawyers" : f"{Lawyers}",
        "Court" : f"{Court}",
        "DocumentShouldBring" : f"{DocumentShouldBring}",
        "Suggetion" : f"{Suggetion}"
        }


def read_pdf_data(
                pdf_file
                ):
    pdf_file = open(pdf_file, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    num_pages = pdf_reader.getNumPages()

    whole_text = ''
    for page in range(num_pages):
        page_obj = pdf_reader.getPage(page)
        text = page_obj.extractText()
        whole_text += f" {text}"
    pdf_file.close()

    whole_text = whole_text.replace('\n', ' ')
    whole_text = re.sub(' +', ' ', whole_text)
    whole_text = whole_text.strip().lower()
    return whole_text

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    return [k for k in lem if k]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(description):
    description = description.lower()
    remove_punc = re_tokenizer.tokenize(description) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_description = ' '.join(remove_stop)
    return updated_description

def inference_pvd(description):
    description = preprocess_one(description)
    description = tokenizer_pvd.texts_to_sequences([description])
    description = tf.keras.preprocessing.sequence.pad_sequences(
                                                                description, 
                                                                maxlen=500, 
                                                                padding='pre'
                                                                )
    prediction = model_pvd.predict(description)
    p1, p2 = prediction

    p1 = np.argmax(p1.squeeze())
    p2 = np.argmax(p2.squeeze())

    violationFlag, violationType =  class_dict_violation_flag_rev[p1], class_dict_violation_type_rev[p2]
    if (violationFlag == 'no') or (violationType == 'no-violation'):
        violationType, violationData =  'no-violation', None
    else:
        violationData =  extract_violation_data(violationType)    

    return {
        "violationType" : f"{violationType}",
        "violationData" : violationData
        }

app = Flask(__name__)
CORS(app)
qna_ = QnA()

app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/pvd', methods=['POST'])
def pvd():
    # data = request.files
    # file = data['file']
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # file.save(file_path)
    data = request.get_json()
    story = data['story']
    return jsonify(inference_pvd(story))

@app.route('/qna', methods=['POST'])
def qna():
    data = request.get_json()
    question = data['question']
    answer, reference = qna_.extract_answers(question)
    return jsonify({
                    "answer" : f"{answer}",
                    "reference" : f"{reference}"
                    })

@app .route('/summary', methods=['POST'])
def summary():
    data = request.files
    file = data['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    text = read_pdf_data(file_path)
    summary_facts = inference_summarization( 
                                            text,
                                            tokenizer_facts_cases,
                                            tokenizer_facts_summarize,
                                            inference_model_facts,
                                            )

    summary_judgements = inference_summarization(
                                                text,   
                                                tokenizer_judgements_cases,
                                                tokenizer_judgements_summarize,
                                                inference_model_judgements,
                                                )
    
    return jsonify({
                    "summary_facts" : f"{summary_facts}",
                    "summary_judgements" : f"{summary_judgements}"
                    })

if __name__ == '__main__':
    app.run(
            debug=True,
            host='0.0.0.0',
            port=5003
            )