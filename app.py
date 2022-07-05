from flask import Flask,render_template, render_template_string,url_for,request,redirect
import os
import re
import PyPDF2
from werkzeug.utils import secure_filename
import random as ran

import nltk
en_stop = set(nltk.corpus.stopwords.words('english'))
 
from gensim.models.fasttext import FastText

# Lemmatization
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        # creating a pdf file object1
        pdfFileObj = open('instance/uploads/pdf1.pdf', 'rb')

        # creating a pdf file object2
        pdfFileObj1 = open('instance/uploads/pdf2.pdf', 'rb')

        # creating a pdf reader object1
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        # creating a pdf reader object2
        pdfReader1 = PyPDF2.PdfFileReader(pdfFileObj1)
        # Pdf 1
        corpus = ''
        pages = pdfReader.numPages
        for i in range(pages):
            pageObj = pdfReader.getPage(i)
            paragraph = pageObj.extractText()
            corpus = corpus + paragraph

        sentence_tokens = nltk.sent_tokenize(corpus)
        clean_corpus = process_text(corpus)
        sen_tokens = nltk.sent_tokenize(clean_corpus)
        word_tokens = [nltk.word_tokenize(word) for word in sen_tokens]
        # Pdf 2
        corpus1 = ''
        pages1 = pdfReader1.numPages
        for i in range(pages1):
            pageObj1 = pdfReader1.getPage(i)
            paragraph1 = pageObj1.extractText()
            corpus1 = corpus1 + paragraph1

            clean_corpus1 = process_text(corpus1)
            word_tokens1 = nltk.sent_tokenize(clean_corpus1)
            word_tokens1 = [nltk.word_tokenize(word) for word in word_tokens1]

    # merging tokens from both corpus
    merged = word_tokens + word_tokens1

    # Defining values for parameters
    embedding_size = 300
    window_size = 6
    min_word = 3
    down_sampling = 1e-2
    # For merged corpus
    global fast_Text_model
    fast_Text_model = FastText(merged, vector_size = embedding_size, window = window_size,
                      min_count = min_word, sample = down_sampling, workers = 4, sg = 1,
                      epochs = 100)
    # # For Pdf 1
    # global fast_Text_model1
    # fast_Text_model1 = FastText(word_tokens, vector_size = embedding_size, window = window_size,
    #                   min_count = min_word, sample = down_sampling, workers = 4, sg = 1,
    #                   epochs = 100)
    # # For Pdf 2
    # global fast_Text_model2
    # fast_Text_model2 = FastText(word_tokens1, vector_size = embedding_size, window = window_size,
    #                   min_count = min_word, sample = down_sampling, workers = 4, sg = 1,
    #                   epochs = 100)

    # for saving and reloading the model
    # from gensim.models import Word2Vec
    # # Save fastText gensim model
    # fast_Text_model.save("model/ft_model_exp")
    # # Load saved gensim fastText model
    # fast_Text_model = Word2Vec.load("model/ft_model_exp")

    # Pdf 1
    nouns = NounExtractor(clean_corpus)
    unique_nouns = list(dict.fromkeys(nouns))
    # number_of_nouns = len(unique_nouns)
    global verbs
    # verbs = verbExtractor(clean_corpus)
    # unique_verbs = list(dict.fromkeys(verbs))

    #Pdf 2
    nouns1 = NounExtractor(clean_corpus1)
    unique_nouns1 = list(dict.fromkeys(nouns1))
    # number_of_nouns1 = len(unique_nouns1)
    global verbs1
    # verbs1 = verbExtractor(clean_corpus1)
    # unique_verbs1 = list(dict.fromkeys(verbs1))

    # Common Nouns
    common_nouns = set(unique_nouns).intersection(set(unique_nouns1))
    # convert common_nouns to list
    common_nouns_list = []
    for i in common_nouns:
        common_nouns_list.append(i)
    number_of_common_nouns = len(common_nouns_list)

    # Common Verbs
    # common_verbs = set(unique_verbs).intersection(set(unique_verbs1))
    # convert common_nouns to list
    global common_verbs_list
    common_verbs_list = []
    # for i in common_verbs:
        # common_verbs_list.append(i)
    # number_of_common_verbs = len(common_verbs_list)

    # top_7 = []
    # for noun in common_nouns_list:
    #     top_7.append(noun)
    #     top_7.append(str(fast_Text_model.wv.most_similar(noun, topn=7)))
    
    # string_to_return = str("The number of common nouns extracted : " + str(number_of_common_nouns) 
    #                                  + ". They are : \n" + 
    #                                 str(common_nouns_list) + ".\n" +
    #                                 "The top 10 most similar words for the nouns extracted: \n" + 
    #                                 str(top_10) + ".")
    
    string1 = str("The number of common nouns extracted : " + str(number_of_common_nouns) + ".")
    string2 = str("They are : " + str(common_nouns_list) + ".")
    # string3 = str("The number of verbs extracted : " + str(len(verbs) + len(verbs1)) + ".")
    # string4 = str("They are : " + str(verbs + verbs1) + ".")
    # string5 = str("Random sentence generated from common verbs and nouns : ")
    # string6 = str(sentence(np1, vp1) + ".")
    # string3 = str("The top 7 most similar words for the nouns extracted are :")
    # string4 = str(top_7)

    # return render_template_string()
    return render_template('result.html', output = [string1, string2])

@app.route('/result', methods =["GET", "POST"])
def result():
    if request.method == "POST":
        # getting input with name = fname in HTML form
        search = request.form.get("fname")
        # search1 = request.form.get("fname1")
        # search2 = request.form.get("fname2")
        string_1 = str(fast_Text_model.wv.most_similar(search, topn=7))
        # string_2 = str(fast_Text_model1.wv.most_similar(search1, topn=7))
        # string_3 = str(fast_Text_model2.wv.most_similar(search2, topn=7))
        # string_4 = str("Sentence generated : ")
    # return render_template("result1.html", output1 = [search1, search2, string_2, string_3, string_4, generateSentence([search1]),
                                                                                    # generateSentence([search2])])
    return render_template("result1.html", output1 = [search, string_1])

# Function to generate sentences [using common_nouns_list, verbs + verbs1, t]

# def assemble(*args):
#     return " ".join(args)
# def NP(t, n):
#     return assemble(t, n)
# def VP(verb, np):
#     return assemble(verb, np)
# def sentence(np, vp):
#     return assemble(np, vp)

# def generateSentence(l1):    
#     t = ['The']
#     # for i in range(1):
#     n1, n2 = ran.choice(l1), ran.choice(l1)
#     t1, t2 = ran.choice(t), ran.choice(t)
#     verb1 = ran.choice((verbs + verbs1))
#     np1 = NP(t1, n1)
#     np2 = NP(t2, n2)
#     vp1 = VP(verb1, np2)
#     return sentence(np1, vp1)


# Function to extract all the nouns
def NounExtractor(clean_corpus):    
    sentences = nltk.sent_tokenize(clean_corpus)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in en_stop]
        tagged = nltk.pos_tag(words)
        ans = []
        for (word, tag) in tagged:
            if tag == 'NN': # If the word is a proper noun
                ans.append(word)
    return ans


# Function to extract all verbs
# def verbExtractor(clean_corpus):
#     sentences = nltk.sent_tokenize(clean_corpus)
#     for sentence in sentences:
#         words = nltk.word_tokenize(sentence)
#         words = [word for word in words if word not in en_stop]
#         tagged = nltk.pos_tag(words)
#         ans = []
#         for (word, tag) in tagged:
#             if tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ': # If it is verb
#                 ans.append(word)
#     return ans


# Text cleaning function for gensim fastText word embeddings in python
def process_text(document):
    # Remove extra white space from text
    document = re.sub(r'\s+', ' ', document, flags=re.I)
        
    # Remove nos. from 0-9
    document = re.sub(r'\[[0-9]*\]',' ', document)
        
    # Remove all the special characters from text
    document = re.sub(r'\W', ' ', str(document))

    # Remove all single characters from text
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Converting to Lowercase
    document = document.lower()

    # Word tokenization       
    tokens = document.split()
    # Drop words 
    tokens = [word for word in tokens if len(word) > 3]
    
    # Lemmatization using NLTK
    lemma_txt = [stemmer.lemmatize(word) for word in tokens]
    # Remove stop words
    lemma_no_stop_txt = [word for word in lemma_txt if word not in en_stop]
                
    clean_txt = ' '.join(lemma_no_stop_txt)

    return clean_txt


# Create a directory in a known location to save files to.
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # save the single "profile" file
        profile = request.files['profile']
        profile1 = request.files['profile1']
        profile.save(os.path.join(uploads_dir, secure_filename('pdf1.pdf')))
        profile1.save(os.path.join(uploads_dir, secure_filename('pdf2.pdf')))

        # save each "charts" file
        for file in request.files.getlist('charts'):
            file.save(os.path.join(uploads_dir, secure_filename(file.name)))
            # filename = file.filename
        return redirect(url_for('upload'))
        
    return render_template('upload.html')



if __name__ == '__main__':
	app.run(debug=True)