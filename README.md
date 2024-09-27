# Similarity Measurement of Scholarly Articles
This project aims to create a system that helps researchers find similar research papers using advanced computer techniques. The system can show which papers are related by analyzing the words and ideas in the documents. This project focuses on developing a system that can automatically identify similarities between research papers by leveraging natural language processing (NLP) and information retrieval (IR) techniques. By analyzing the content of research papers, the system will provide insights into the relatedness of topics, methodologies, and findings across a large corpus of academic literature. The system will be easy to use, with a simple interface that anyone can understand. It will help researchers quickly find relevant papers for their work without spending plenty of time searching. Our project aims to make it easier for researchers to discover new information and learn from existing research. This project can potentially improve how research is done and help scientists make discoveries faster.

## Data Collection
We have collected 24 research papers from Google Scholar based on 6 different topics.
- Large Language Model(CS) - 6 Papers
- Mitochondria Research(BIO) - 4 Papers
- Clustering Technique(CS) - 4 Papers
- Cyber Security(CS) - 3 Papers
- Reinforcement Learning(CS) - 4 Papers
- Cancer Research(BIO) - 3 Papers

## Data Preprocessing
After Collecting the research papers, we started preprocessing them. The papers are in pdf format. We extracted text from the pdf files using the PyPDF2 python library {https://pypi.org/project/PyPDF2/}. Next, we have removed the header and footer information from each page of the papers. Moreover, we followed standard text processing steps for extracting interesting and non-trivial knowledge from unstructured text data. Our preprocess steps are as follows:
- Tokenize a document
- Convert tokens to lowercase
- Remove punctuations and stopwords
- Stem the tokens
We used Natural Language Toolkit (NLTK - https://www.nltk.org/) to preprocess the data.

## Information Retrieval Model
To identify similarities in research papers, we need to count the term frequency of each term in the vocabulary. Because of this crucial reason, we have considered the Vector Space Model (VSM) as a relevant data representation way for our project. The VSM is a widely used information retrieval model representing documents as vectors in a high-dimensional space, where each dimension corresponds to a term in the vocabulary. The VSM is based on the assumption that the meaning of a document can be inferred from the distribution of its terms and that documents with similar content will have similar term distributions. We process our text data before constructing this model. Then, A term-document matrix is implemented, where each row represents a term and each column represents a document. The matrix contains the frequency of each term in each document, or some variant of it (e.g., term frequency-inverse document frequency, TF-IDF).

## Similarity Computation
After constructing our model, we aim to reveal the closest documents as accurately as possible. To calculate this distance measurement, we experiment with two types of well-known measurements: Euclidian and Cosine similarity. We apply cosine similarity for all three variants of term frequency: Term frequency (TF), Term/Inverse document frequency (TF-IDF), and Weight/Inverse document frequency (WF-IDF). By applying similarity measurements, we get the closest documents in pairs in sorted order. We also want to uncover the clusters of documents, to be more specific, the documents are closest to each other in a cluster and the documents in different clusters are dissimilar. We apply both Euclidian and cosine similarity measures to find the relevant clusters in this case.

## Project Outcome
In this project, we have calculated the similarity among research papers using different Information retrieval and machine learning techniques. From our experimental results, we can say that TF-IDF measurement with clustering algorithm using cosine similarity measurement shows great performance in identifying similar research papers. In the future, we can add a large number of research papers on diverse topics to our dataset which will create unique challenges to the given problem. Moreover, we can add various feature selection techniques like dimensionality reduction which will result in more accurate results.
