---
description: Understanding and exploring Practical NLP.
---

# Natural language processing

**Natural language processing(NLP)** is an area of research in computer science and artificial intelligence (AI) concerned with processing natural languages such as English or Hindi. This processing generally involves translating natural language into data (numbers) that a computer can use to learn about the world.&#x20;

There are different task to be done in NLP and some of them are following :&#x20;

* Question Answering | Paraphrase | Summarization | Dialog
* Information Extraction (IE) | Sentiment Analysis | Parsing
* Machine Translation(MT) -- Fully Automatic, Helping Human Translator
* Spam detection | Pos Tagging | Named Entity recognition(NER)

**There are two approaches for NLP :-**&#x20;

1. Rule based system (Uni-gram, bigram, N-gram models)
2. Neural Based system  (RNN, LSTM, Transformer)&#x20;

These days mostly we work on the 2nd one and we'll focus on that only.&#x20;

**Language Modeling**

In NLP Language Modeling is the task of predicting what word comes next. Or say, given a sequence of words x(1), x(2) … x(t) it computes the probability of occurrence of the next word x(t+1).  Where x(t+1) can be any word in the vocabulary.  Ex.  the student opened their \_\_\_

Examples in daily life—

* Next word popup in typing keyboard on mobile devices
* Suggesting/completing query on google search engine

