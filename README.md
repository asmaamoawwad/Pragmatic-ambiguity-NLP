# A Linguistic Evaluation of NLP Models on Pragmatic Ambiguity

## Overview
This project presents a linguistically grounded evaluation of contemporary Natural Language Processing (NLP) models on **pragmatic ambiguity**. While modern NLP systems excel at syntactic and semantic tasks, human communication relies heavily on pragmatic inference—the implicit meaning derived from context, shared knowledge, and speaker intent.This study investigates the capacity of transformer-based models to handle phenomena like sarcasm and irony in informal social media texts.

---

## Abstract
Natural Language Processing (NLP) systems have achieved remarkable success in tasks centered on literal meaning. However, there remains a persistent **Pragmatic Gap** where models fail to bridge the mathematical **Semantic Displacement Vector** between literal output and pragmatic intent.Using a pre-trained **BERT transformer** fine-tuned for utterance classification, we combine quantitative metrics (Accuracy=84%, F1=0.81) with detailed qualitative linguistic error analysis. Our findings reveal systematic failures when interpretation depends on contextual or cultural cues rather than explicit lexical markers.

---

## Problem Statement
Despite strong performance on syntax and semantics, modern NLP models consistently struggle with pragmatics.

- **Contextual Isolation:** Models tend to process utterances in isolation, missing the speaker intention and world knowledge that determine meaning.
- **The Training Paradox:** Increased training data alone does not resolve the lack of pragmatic understanding.
- **Real-world Impact:** Failure to bridge this gap leads to errors in sentiment analysis, content moderation, and dialogue systems.

---

## Objective
The objective of this project is to evaluate standard NLP architectures through a **linguistic lens**, moving beyond accuracy scores to understand **where and why models fail pragmatically**.

---

## Methodology
Our approach follows a standard NLP pipeline adapted for linguistic analysis:

- **Data & Annotation**: Curated an informal text corpus manually annotated by trained linguists to establish a **gold-standard dataset** for sarcasm and irony.
- **Neural Preprocessing**: Included noise reduction, tokenization into linguistic units, and transformation into numerical embeddings.
- **Model Training**: A pre-trained BERT transformer was fine-tuned via an 80/20 train-test split.
- **Hybrid Evaluation**: Paired quantitative metrics with a qualitative analysis of **systematic linguistic error patterns** to identify the Pragmatic Gap.

---

## Results & Linguistic Analysis
The fine-tuned transformer achieved **84% accuracy**. However, an **Accuracy Cascade** analysis reveals a significant performance drop-off at high complexity.

- **The "Literal Cluster" Error:** Errors are dominated by false negatives in sarcasm detection.Surface-Level 
- **Reliance:** The model maps sarcastic utterances into literal clusters based on surface-level signals (e.g., the word "Great"), failing to calculate the **Semantic Displacement Vector**.
- **Contextual Degradation:** Performance declines sharply when interpretation requires contextual and cultural inference.

---

## Conclusion
High overall accuracy masks significant pragmatic weaknesses in NLP models. This study reveals a persistent gap between model and human understanding that is not resolved by increased data or standard metrics. We highlight the urgent need for **linguistically informed evaluation frameworks**, particularly for culturally-nuanced and low-resource contexts.

---

## Future Work
- **Linguistic Diversity & Low-Resource Languages**: Extending the evaluation framework to African languages and dialects (e.g., Swahili, Yoruba, Amharic).
- **Culturally-Aware Sentiment Analysis**: Developing benchmarks that incorporate African socio-cultural contexts.
- **Explainable Pragmatics**: Moving toward neuro-symbolic AI models that can explain why an utterance was classified as sarcastic or ironic.

---

## Technologies Used
- Python
- Transformer-based NLP models (BERT)
- Linguistic qualitative analysis

---

