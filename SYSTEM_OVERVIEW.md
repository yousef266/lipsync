**System Architecture/System Design**

The Arabic lip sync system would be a **rule-based hybrid system**. For instance, the first component is **Whisper** (a pre-trained Transformer encoder-decoder ASR model trained on 680k hours of speech) trained enough to work on multilingual speech and specifically Arabic for the source audio, whereas the remaining components are **deterministic phonetic rules** which create time-aligned mouth shapes (visemes) that require no additional training but instead phonological evolution and auditory guidance. Whisper provides multilingual ASR in addition to very efficient zero-shot generalization in Arabic, but the subsequent steps (G2P, viseme alignment, VAD and timeline adjustment) are rule-based and exclude any training. 

---

**Input to output**

audio (wav) → whisper ASR (speech to text + word timestamps) → arabic G2P (text-to-phonemes) → Phoneme-to-Viseme Mapping (articulatory rules) → Timeline Optimization (tweening + merging) → Export JSON 

---

**Pipeline Step Breakdown**

1. **Input**

• WAV audio file (mono/stereo/16kHz+ for best results)

• System will resample and check compatibility if necessary. 

2. **ASR**

**· Name of Model**: Whisper (Transformer encoder-decoder)

**· Model Size**: Base size (74M parameters, 140MB, can adjust size to tiny/small/medium/large)

**· Language Trained For**: Arabic

**· Output**: Transcribed text + timestamps per word (via return_timestamps="word") 

**· Performance**: 16× real-time on CPU for base model

3. **Grapheme-to-Phoneme (G2P)**

• The first step converts the Arabic text into a linear sequence of phonemes via a dictionary rule-based lookup 

**· Phoneme Set Used**: 28 consonants per Modern Standard Arabic + 6 vowels (3 short + 3 long) + 3 special

**· Examples**: 

• "ب" = /b/, "ع" = /ʕ/ (ain), "ش" = /ʃ/ (sh), "غ" = /gh/, Diacritics: "َ" = /a/, "ِ" = /i/, "ُ" = /u/

The phonemes follow Modern Standard Arabic with an exception lexicon in case of rare/loaned words.

The phonemes are apportioned evenly based upon word timestamps acquired from Whisper (therefore each phoneme has a start/end time within the duration of the entire word).

### 4. Phoneme-to-Viseme Mapping 

- **Viseme mapping** to 9 different mouth shapes based on **articulatory phonetics** - Phonemes form clusters of static mouth states and then consecutively through the process: - **A**. Closed (م، ب) - bilabials - **B**. Slightly open (ت، د، ك، ن، ر، س) - dentals/alveolars - **C**. Open (ع، ح، ه، ء) - pharyngeals/glottals - **D**. Wide Open (آ، ا) - open vowel - **G**. Labiodental (ف، ث، ذ) - **H**. L-sound (ل) - The viseme classes relevant to Arabic are those most widely accepted through phonetic testing (Damien & Wakim, 2009) and thus it's crucial that as the right phonemes play, the respective mouth shapes or viseme are on par for proper lip synchronization (Huang et al., 2025)  

### 5. Timeline Optimization - **Optimization** - **Merging**: Redundant entries by virtue of the same viseme being played back to back will be merged to fill duration with one frame vs two. - **Tweening**: When similar visemes or visemes that look completely different happen, a quick duration of the middle mouth shape will be injected to more naturally come to the next viseme shape as opposed to a curt switching motion. 

- This allows for natural motion as frame interpolation occurs into visemes - just like in any other animated project. 

- Final output is a JSON entry of {start, end, viseme, phoneme} in centiseconds.     

### 6. Export - **JSON** - With metadata about the voicing and time-stamped mouth movements of phonemes through phonemes.  

--- 

## System Parts  

### Major Parts 

- **Whisper ASR** - Pre-trained audio-to-text transformer for Arabic word timestamps (no student trained) - At CPU 16x realtime with base model, at GPU 50x realtime  

- **ArabicG2P** - A dictionary-style text-to-phoneme engine 

- **ArabicShapeMapper** - A phoneme to viseme mapper based on articulatory phonetics  

- **Timeline Engine** - Merging tweening and optimization

---

## Representation

### Input
- **Audio**: WAV file

### Output (JSON Format)
```json
{
  "mouth": [
    {
      "start": 0.0,      
      "end": 0.12,       
      "value": "B",      
      "metadata": {
        "phone": "y"     
      }
    }
    // ... 
  ]
}
```

## Learning Type

**Hybrid System**:
- **Deep Learning**: Whisper ASR 
- **Rule-Based**: G2P, viseme mapping, timeline optimization

**Training Data**: None required for end users 

**Inference Requirements**:
- CPU: Sufficient for real-time processing
- GPU: Optional (faster processing)

---

## Fine-Tuning Whisper

### Objective

To enhance the system's performance on Egyptian Arabic speech, fine-tuning the Whisper model enables improved recognition of local dialectal words, pronunciation nuances, and timestamp accuracy.

This section outlines how to fine-tune the Whisper Base model using open-source Egyptian Arabic datasets for better alignment with regional accent and phonetic patterns.

### Datasets

The following datasets provide clean Egyptian Arabic speech and transcriptions suitable for ASR fine-tuning:

**HuggingFace: MAdel121/arabic-egy-cleaned**
- 72 hours of aligned Egyptian Arabic audio-transcript pairs
- Cleaned, segmented recordings of Egyptian speakers with text transcripts
- Mix of broadcast and read speech
- Link: https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned

**Kaggle: EgySpeech**
- Egyptian dialect audio samples paired with transcripts
- Collected dialect utterances for ASR training
- Link: https://www.kaggle.com/datasets/iraqyomar/egyspeech

### Fine-Tuning Process

1. **Prepare Dataset**: Load and preprocess audio-text pairs from chosen dataset
2. **Configure Training**: Set low learning rate, appropriate batch size, and train for a few epochs on GPU
3. **Evaluate**: Test on held-out Egyptian Arabic samples for WER (Word Error Rate) improvement
4. **Deploy**: Replace base model with fine-tuned checkpoint in the pipeline

**Expected Improvements**:
- Better recognition of Egyptian colloquial vocabulary
- Improved timestamp accuracy for dialectal speech patterns
- Significantly reduced WER on Egyptian-specific phonetic variations
- **Research shows**: Even fine-tuning on ~2 hours of dialect-specific data "drastically improves" ASR accuracy, nearly closing the gap to fully supervised systems
- Fine-tuning on hundreds of hours of Egyptian Arabic can yield substantial WER reductions

---

## Processing Modes

### Mode 1: Automatic (Whisper Enabled)
1. Whisper transcribes audio → Arabic text + word timestamps
2. G2P converts words → phonemes
3. Distribute phonemes across word durations
4. Map phonemes → visemes
5. Optimize timeline → export

---

## Deployment

### Current Implementation
- **Platform**: Cross-platform
- **Output**: JSON file for offline animation

### Integration Path
1. **Export**: JSON with per-frame mouth shapes
2. **Frontend**: Load JSON in 3D renderer 

### Potential Enhancements
- **ONNX Export**: Export Whisper + rules as unified ONNX model
- **API Server**: FastAPI endpoint returning JSON

---

## Technical Stack

### Dependencies
```
whisper==20250625          # ASR model
torch==2.9.0               # Backend
torchaudio==2.9.0          # Audio processing
numpy==2.2.6               # Numerical computing
sounddevice==0.5.3         # Real-time audio I/O
```

### Core Modules
- `SpeechRecognizer`: Whisper wrapper for Arabic ASR
- `ArabicG2P`: Text-to-phoneme converter
- `ArabicShapeMapper`: Phoneme-to-viseme rules
- `AudioClip`: WAV loading 
- `Timeline`: Animation timeline with tweening/optimization

---

## References

### Academic & Technical Sources

1. **Whisper ASR Model**
   - Radford et al. (2022): "Robust Speech Recognition via Large-Scale Weak Supervision"
   - Paper: https://arxiv.org/abs/2212.04356
   - Model: https://huggingface.co/openai/whisper-base
   - Transformer encoder-decoder trained on 680k+ hours of multilingual speech

2. **Whisper Implementation**
   - HuggingFace Whisper documentation for word-level timestamps
   - https://huggingface.co/openai/whisper-large-v3
   - Usage: `return_timestamps="word"` for time-aligned output

3. **Arabic Phonetics & G2P**
   - "Automatic Grapheme-to-Phoneme Conversion of Arabic Text"
   - https://www.scribd.com/document/320807531/Automatic-Grapheme-to-Phoneme-Conversion-of-Arabic-Text
   - Standard Arabic: 28 consonants + 6 vowels (3 short + 3 long)

4. **Viseme Theory & Mapping**
   - Damien & Wakim (2009): "Viseme classes for Modern Arabic"
   - Paper: https://new.eurasip.org/Proceedings/Eusipco/Eusipco2006/papers/1568982208.pdf
   - Established viseme classes for Arabic via phonetic analysis


5. **Phoneme-Viseme Alignment**
   - Huang et al (2025): "PASE: Phoneme-Aware Speech Encoder to Improve Lip Sync Accuracy"
   - Paper: https://arxiv.org/abs/2504.05803
   - Emphasizes critical importance of precise phoneme-viseme alignment

6. **Audio-Driven Viseme Dynamics**
   - "Learning Audio-Driven Viseme Dynamics for 3D Face Animation"
   - Paper: https://arxiv.org/abs/2301.06059
   - Interpolation between viseme keyframes for natural motion

7. **Egyptian Arabic Dataset**
   - MAdel121/arabic-egy-cleaned on HuggingFace & EgySpeech
   - https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned
   - https://www.kaggle.com/datasets/iraqyomar/egyspeech
   - ~72 hours of aligned Egyptian Arabic audio-transcript pairs



