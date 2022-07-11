from flask import Flask,render_template, render_template_string,url_for,request,redirect
import os
import re
import PyPDF2
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

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
        global corpus
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
        global corpus1
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
        # merged = word_tokens + word_tokens1

        # Defining values for parameters
        embedding_size = 300
        window_size = 6
        min_word = 4
        down_sampling = 1e-2
        # For merged corpus
        # global fast_Text_model
        # fast_Text_model = FastText(merged, vector_size = embedding_size, window = window_size,
        #                   min_count = min_word, sample = down_sampling, workers = 4, sg = 1,
        #                   epochs = 100)
        # For Pdf 1
        global fast_Text_model1
        fast_Text_model1 = FastText(word_tokens, vector_size = embedding_size, window = window_size,
                        min_count = min_word, sample = down_sampling, workers = 4, sg = 1,
                        epochs = 100)
        # For Pdf 2
        global fast_Text_model2
        fast_Text_model2 = FastText(word_tokens1, vector_size = embedding_size, window = window_size,
                        min_count = min_word, sample = down_sampling, workers = 4, sg = 1,
                        epochs = 100)

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
        # global verbs
        # verbs = verbExtractor(clean_corpus)
        # unique_verbs = list(dict.fromkeys(verbs))

        #Pdf 2
        nouns1 = NounExtractor(clean_corpus1)
        unique_nouns1 = list(dict.fromkeys(nouns1))
        # number_of_nouns1 = len(unique_nouns1)
        # global verbs1
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
        # global common_verbs_list
        # common_verbs_list = []
        # for i in common_verbs:
            # common_verbs_list.append(i)
        # number_of_common_verbs = len(common_verbs_list)
        
        # top_7 = []
        # for noun in common_nouns_list:
        #     top_7.append(noun)
        #     top_7.append(str(fast_Text_model.wv.most_similar(noun, topn=7)))
        
        string1 = str("The number of common nouns extracted : " + str(number_of_common_nouns) + ".")
        string2 = str("They are : " + str(common_nouns_list) + ".")
        # string3 = str("The number of verbs extracted : " + str(len(verbs) + len(verbs1)) + ".")
        # string4 = str("They are : " + str(verbs + verbs1) + ".")

    # return render_template_string()
    return render_template('result.html', output = [string1, string2])

@app.route('/result', methods =["GET", "POST"])
def result():
    if request.method == "POST":
        # getting input with name = fname in HTML form
        search = request.form.get("fname")
        # search1 = request.form.get("fname1")
        # search2 = request.form.get("fname2")

        # cosine similarity between common words of two different corpus
        v1 = fast_Text_model1.wv[search]
        v2 = fast_Text_model2.wv[search]
        string_0 = str(get_cosine_similarity(v1, v2))

        # string_11 = str("With ref to PDF 1 :")
        string_1 = str(fast_Text_model1.wv.most_similar(search, topn=7))
        # string_12 = str("With ref to PDF 2 :")
        string_2 = str(fast_Text_model2.wv.most_similar(search, topn=7))
        # string_3 = str(fast_Text_model2.wv.most_similar(search2, topn=7))
        # string_3 = str("Sentence generated (with ref to PDF1): ")
        extracted_sentences = sentence_finder(corpus, [search])
        sentence_list = []
        for sent in extracted_sentences:
            sentence_list.append(sent)
        if len(sentence_list) != 0:
            if len(sentence_list) > 1:
                t = sentence_list[1]
            else:
                t = sentence_list[0]
            words_list = ""
            for word in t:
                if word == " ":
                    words_list += " "
                elif word != "\n":
                    words_list += word
            string_3 = str(words_list)
        else:
            string_3 = "Error1 : word not found"
        # string_5 = str("Sentence generated (with ref to PDF2): ")
        extracted_sentences1 = sentence_finder(corpus1, [search])
        sentence_list1 = []
        for sent in extracted_sentences1:
            sentence_list1.append(sent)
        if len(sentence_list1) != 0:
            if len(sentence_list1) > 1:
                t = sentence_list1[1]
            else:
                t = sentence_list1[0]
            words_list1 = ""
            for word in t:
                if word == " ":
                    words_list1 += " "
                elif word != "\n":
                    words_list1 += word
            string_4 = str(words_list1)
        else:
            string_4 = "Error2 : word not found"
    # return render_template("result1.html", output1 = [search1, search2, string_2, string_3, string_4, generateSentence([search1]), generateSentence([search2])])
    return render_template("result1.html", output1 = [search, string_0, string_1, string_2, string_3,
                                                        string_4])

# function to get the cosine similarity between two common words in different domains
def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

# Function to generate sentences [extracting sentences from the respective pdfs itself]

def sentence_finder(corpus: str, words: list) -> list:
    sentences = sent_tokenize(corpus)
    return [sentence for sentence in sentences if any(word.lower() in word_tokenize(sentence.lower()) for word in words)]

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


# # Function to extract all verbs
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