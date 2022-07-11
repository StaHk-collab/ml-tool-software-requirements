# ml-tool-for-software-requirements

__Requirements__ are descriptions of how a system should behave. The quality of the requirements determines the overall quality of the software product. The various phases involved in software development are requirements, design, coding, testing, deploying. According to past research, around 60% of all errors in software development projects initiate during the requirements engineering phase. A major cause of poor quality requirements is that the stakeholders involved in the development phase have different interpretations of technical terms, and that is due to the ambiguity present in the __natural language__ text.

For example, the word __cookie__ in context of computer science (CS) means a small amount of data stored on user’s machine by the web browser while it is a sweet biscuit for a food engineer. Similarly, the word __table__ in context of computer science (CS) means a grid of rows and columns while it may also mean a furniture in different context.

This project focuses on extraction of text from requirements documents. Since 99% of all the relevant terms are noun phrases, we only focus on extracting them. We will be using neural __word embedding__ technique for detecting the domain-specific technical terms from two different corpora (in PDF format).

### Libraries or Frameworks

Flask, PyPDF2, Werkzeug, nltk, scikit-learn, gensim, re

#### Executable file link : https://drive.google.com/drive/u/1/folders/1TOGHUuozTLD8O1nEObhodLiAsdM3cfFu

The file will take 5-6 mins to load in the system. No pre-requisites.

### Prototype Screenshots:

#### 1. __HomePage__:

![homepage](https://github.com/StaHk-collab/ml-tool-software-requirements/blob/main/ss-of-prototype/home.jpg)

#### 2. After uploading the files, it will ask to analyze them.

![analyze](https://github.com/StaHk-collab/ml-tool-software-requirements/blob/main/ss-of-prototype/analyze.jpg)

#### 3. Now it will show the common nouns extracted from the two corpora (here, research papers of computer science and civil engineering domains are used as corpus). Enter the word to get the cosine similarity of the word in two different domains along with that both top 7 similar context words and the sentences in their given context.

![result](https://github.com/StaHk-collab/ml-tool-software-requirements/blob/main/ss-of-prototype/result.jpg)

#### 4. The results of the entered word is displayed after clicking on search button.

![result1](https://github.com/StaHk-collab/ml-tool-software-requirements/blob/main/ss-of-prototype/result1.jpg)
