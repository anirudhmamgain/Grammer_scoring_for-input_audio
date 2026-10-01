# Grammer_scoring_for-input_audio

# Grammar Scoring Engine — Technical Report

## 1. Objective

The objective of this assessment is to predict a continuous **grammar quality score between 0 and 5** from spoken English audio recordings.

The dataset contains:

- **769 training recordings**
- **216 test recordings**
- Audio duration of approximately 45–60 seconds
- A continuous grammar-quality label for the training set
- Evaluation based on **RMSE** and **Pearson correlation**

The primary model-selection criterion used during experimentation was **validation RMSE**, with Pearson correlation additionally considered when evaluating competition submissions.

---

## 2. Data Preprocessing and Quality Analysis

The audio recordings were initially inspected using `librosa` to analyze:

- Sampling rate
- Number of channels
- Duration
- RMS amplitude
- Peak amplitude
- Waveform characteristics
- Mel-spectrograms

This provided an overview of the acoustic characteristics of the dataset and helped identify potentially problematic recordings.

### Noise-only recordings

During ASR preprocessing, seven training recordings were identified as containing noise without usable speech content. Whisper generated empty or unusable transcripts for these recordings, and manual inspection confirmed the absence of meaningful speech.

The affected files were:

- `audio_5069.wav`
- `audio_5056.wav`
- `audio_5043.wav`
- `audio_5058.wav`
- `audio_5049.wav`
- `audio_5042.wav`
- `audio_5060.wav`

These recordings were excluded because grammar prediction fundamentally depends on linguistic content.

Therefore:

**769 original training samples → 762 usable training samples**

---

## 3. Speech-to-Text Conversion

Because the target is explicitly related to grammar quality, the linguistic content of the speech is more directly relevant than purely acoustic characteristics.

OpenAI Whisper was therefore used for automatic speech recognition:

**Audio → Whisper → Transcript**

The `small` Whisper model was used with English-language transcription.

The resulting transcripts were cached locally to avoid repeatedly performing computationally expensive ASR inference.

The same preprocessing pipeline was applied to both training and test audio.

---

## 4. Exploratory Data Analysis

The transcripts were analyzed using linguistic statistics including:

- Word count
- Unique word count
- Vocabulary diversity
- Average sentence length
- Sentence-length variability
- Function-word usage
- Repeated-word ratio
- Filler-word ratio
- Questions and punctuation
- Words per minute

Some variables showed meaningful relationships with the grammar score. In particular, **unique word count** and **word count** provided stronger marginal signal than most individual handcrafted features.

However, individual linguistic statistics were insufficient to fully capture grammatical quality, so they were used as complementary features rather than as the primary representation.

---

## 5. Modeling Approaches

Several modeling strategies were evaluated.

### 5.1 TF-IDF + Ridge Regression

A traditional text baseline was constructed using word-level TF-IDF features with unigram and bigram representations.

Pipeline:

**Transcript → TF-IDF → Ridge Regression → Grammar Score**

The best validation RMSE was approximately:

**RMSE ≈ 1.21**

This provided a useful baseline and demonstrated that lexical patterns contain predictive information.

---

### 5.2 MPNet Sentence Embeddings + Ridge

To capture semantic information beyond sparse lexical features, transcripts were encoded using:

`all-mpnet-base-v2`

The resulting 768-dimensional sentence embeddings were passed to Ridge regression.

Pipeline:

**Transcript → MPNet Embedding → Ridge → Grammar Score**

Best validation result:

**RMSE ≈ 1.05**

This was a substantial improvement over the TF-IDF baseline, indicating that contextual sentence representations capture useful information about the quality of spoken language.

---

### 5.3 Explicit Linguistic Features

A compact set of linguistically motivated features was also evaluated:

- Word count
- Unique word count
- Vocabulary diversity
- Words per minute
- Unique words per minute

These features were standardized before regression.

The model provided additional complementary information, although its standalone performance was weaker than the transformer-based models.

---

### 5.4 Word + Character TF-IDF + MPNet

A richer traditional representation was constructed by combining:

- Word-level TF-IDF
- Character-level TF-IDF
- MPNet sentence embeddings

The motivation was to combine:

- Word n-grams for lexical and grammatical patterns
- Character n-grams for morphological/spelling patterns and robustness to ASR variations
- MPNet embeddings for contextual semantic information

The representations were concatenated and modeled using Ridge regression.

This model provided a complementary prediction source for ensemble experiments.

---

## 6. DeBERTa Fine-Tuning

The strongest individual modeling direction was supervised fine-tuning of:

`microsoft/deberta-v3-base`

The transcripts were tokenized with a maximum sequence length of **256 tokens**.

A token-length analysis showed that essentially all usable transcripts fit within this limit, so transcript chunking was unnecessary.

Pipeline:

**Audio → Whisper → Transcript → DeBERTa-v3 → Regression Head → Grammar Score**

The model was configured for single-output regression.

The main training configuration included:

- Maximum sequence length: 256
- Batch size: 8 for training
- Batch size: 16 for evaluation
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Maximum training epochs: 8
- Best checkpoint selected using validation RMSE
- Predictions clipped to the valid `[0, 5]` score range

### DeBERTa validation results

| Epoch | Validation RMSE |
|---:|---:|
| 1 | 1.2356 |
| **2** | **0.9329** |
| 3 | 1.2071 |
| 4 | 0.9338 |

The best validation RMSE was approximately:

**0.9328**

The strong performance and subsequent competition submission improvement made DeBERTa the strongest individual model tested.

The validation curve also demonstrated that simply training for more epochs does not necessarily improve generalization; the validation error fluctuated substantially after the best checkpoint. Therefore, checkpoint selection based on validation RMSE was retained.

---

## 7. Ensemble / Prediction Fusion

Since the DeBERTa and traditional NLP models learn different representations, prediction-level fusion was investigated.

The final ensemble combines:

1. **DeBERTa-v3 predictions**
2. **Word + Character TF-IDF + MPNet predictions**
3. **Explicit linguistic-feature predictions**

The ensemble prediction is:

\[
\hat{y}
=
0.48\hat{y}_{DeBERTa}
+
0.46\hat{y}_{WC+MPNet}
+
0.06\hat{y}_{Ling}
\]

All component predictions are clipped to the valid grammar-score range `[0, 5]`.

The ensemble achieved a validation RMSE of approximately:

**0.9071**

This was better than the standalone DeBERTa validation RMSE of approximately 0.9328, indicating that the complementary models provided additional information.

---

## 8. Final Pipeline Architecture

The final system can be summarized as:

```text
                    ┌─────────────────────┐
                    │     Audio (.wav)    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Whisper       │
                    │    Speech → Text    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Transcript     │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
      ┌─────────────┐  ┌────────────────┐  ┌─────────────────┐
      │  DeBERTa    │  │ Word + Char    │  │ Linguistic      │
      │  v3-base    │  │ TF-IDF + MPNet  │  │ Features        │
      └──────┬──────┘  └───────┬────────┘  └────────┬────────┘
             │                 │                    │
             ▼                 ▼                    ▼
       Prediction 1      Prediction 2         Prediction 3
             │                 │                    │
             └─────────────────┼────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Weighted Ensemble   │
                    │ 0.48 / 0.46 / 0.06  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Clip score to 0–5   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   submission.csv    │
                    └─────────────────────┘
```

---

## 9. Model Comparison

| Approach | Validation RMSE | Role |
|---|---:|---|
| TF-IDF + Ridge | ~1.21 | Baseline |
| MPNet + Ridge | ~1.05 | Contextual embedding model |
| Linguistic features + regression | ~1.10 | Interpretable complementary model |
| Word + Char TF-IDF + MPNet | <1.0 | Complementary NLP model |
| DeBERTa-v3-base | **0.9328** | Strongest individual model |
| **Final ensemble** | **0.9071** | Combined model |

The experiments show a progression from sparse lexical representations toward contextual transformer-based representations and finally toward model fusion.

---

## 10. Final Submission

The final test predictions are generated using the trained models and combined using the validated ensemble weights.

Predictions are constrained to the competition's valid score range:

\[
0 \leq \hat{y} \leq 5
\]

The final file is saved as:

```text
submission.csv
```

with the required columns:

```text
filename
label
```

The row order follows the original `test.csv` ordering to ensure correct alignment between predictions and test recordings.

---

## 11. Key Conclusions

The experiments indicate that grammar scoring from spoken audio benefits strongly from converting the audio into text and modeling the resulting linguistic representation.

The main findings were:

1. **Whisper transcription provided a strong bridge from audio to linguistic modeling.**
2. **TF-IDF established a useful lexical baseline but was limited in capturing contextual information.**
3. **MPNet embeddings substantially improved local validation performance.**
4. **DeBERTa-v3 fine-tuning provided the strongest individual model and produced a significant improvement in competition performance over the initial baseline.**
5. **Explicit linguistic features were weaker individually but provided complementary information.**
6. **Combining heterogeneous model predictions reduced validation RMSE further, reaching approximately 0.9071.**
7. **Validation-based checkpoint selection was important because additional training epochs did not consistently improve generalization.**

The resulting solution therefore uses a hybrid architecture combining **ASR, contextual transformer fine-tuning, sparse NLP representations, sentence embeddings, and interpretable linguistic features**.
