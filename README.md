# A Linguistic Evaluation of NLP Models on Pragmatic Ambiguity

## Overview
This project presents a linguistically grounded evaluation of contemporary Natural Language Processing (NLP) models on **pragmatic ambiguity**, with a focus on implicit meaning beyond literal interpretation. While modern NLP systems achieve strong performance on syntax- and semantics-driven tasks, human communication relies heavily on contextual inference, shared knowledge, and speaker intention.

The study examines how transformer-based models handle pragmatic phenomena such as **sarcasm and irony** in informal social media text.

---

## Abstract
Natural Language Processing (NLP) systems have achieved remarkable success in tasks centered on literal meaning, including syntactic parsing and semantic classification. However, human communication relies heavily on pragmatic inference—implicit meaning derived from context, shared knowledge, and speaker intent.

This study examines the capacity of contemporary transformer-based NLP models to handle pragmatic ambiguity, with a specific focus on sarcasm and irony in informal social media texts. Using a pre-trained transformer model fine-tuned for utterance classification, we combine quantitative evaluation with detailed qualitative linguistic error analysis.

While the model achieves high overall accuracy, results reveal systematic failures when interpretation depends on contextual or cultural cues rather than explicit lexical markers. These findings demonstrate that standard evaluation metrics may obscure significant pragmatic limitations in current NLP systems and highlight the need for linguistically informed evaluation frameworks.

---

## Problem Statement
Despite strong performance on syntax- and semantics-driven tasks, modern NLP models frequently struggle with pragmatics. Context, speaker intention, and world knowledge often determine meaning, yet models tend to process utterances in isolation.

Phenomena such as sarcasm, irony, and metaphor create a gap between what is said and what is meant. When NLP systems fail to bridge this gap, errors arise in real-world applications such as sentiment analysis, content moderation, and dialogue systems.

---

## Objective
The objective of this project is to evaluate standard NLP architectures through a **linguistic lens**, moving beyond accuracy scores to understand **where and why models fail pragmatically**.

---

## Methodology
Our approach follows a standard NLP pipeline adapted for linguistic analysis:

- **Data & Annotation**: A corpus of informal social media texts was compiled and manually annotated by linguists for pragmatic markers (e.g., sarcasm and irony), establishing a gold-standard dataset.
- **Preprocessing**: Raw text was cleaned, tokenized into linguistic units, and converted into numerical embeddings.
- **Model Training**: A pre-trained transformer model (BERT) was fine-tuned to classify utterances based on context using an 80/20 train-test split.
- **Evaluation**: Performance was assessed using standard quantitative metrics (Accuracy, F1-score) alongside critical qualitative analysis of linguistic error patterns.

---

## Results & Linguistic Analysis
The fine-tuned transformer achieved **84% accuracy** and an **F1-score of 0.81**, indicating strong performance under standard evaluation metrics. However, linguistic error analysis reveals systematic pragmatic limitations.

Errors are dominated by **false negatives**, where sarcastic utterances are misclassified as literal when interpretation depends on contextual or cultural knowledge. The model also overgeneralizes positive lexical cues (e.g., *great*, *love*), leading to false positives. These patterns indicate reliance on surface-level lexical signals rather than genuine pragmatic inference.

---

## Conclusion
High aggregate accuracy in NLP models contrasts with reduced performance on tasks requiring contextual inference, non-literal meaning, and world knowledge. This discrepancy reveals pragmatic limitations that are obscured by standard evaluation metrics.

The findings emphasize the importance of integrating linguistic theory into NLP evaluation and model design to better align machine understanding with human communication.

---

## Future Work
- **Linguistic Diversity & Low-Resource Languages**: Extending the evaluation framework to African languages and dialects (e.g., Swahili, Yoruba, Amharic).
- **Culturally-Aware Sentiment Analysis**: Developing benchmarks that incorporate African socio-cultural contexts.
- **Explainable Pragmatics**: Moving toward neuro-symbolic AI models that can explain why an utterance was classified as sarcastic or ironic.

---

## Technologies Used
- Python
- Jupyter Notebook
- Transformer-based NLP models (BERT)
- Linguistic qualitative analysis

---

## Author
**Asmaa Moawwad**  
Concordia University  
Linguistics & Natural Language Processing  
Deep Learning Indaba X Egypt
