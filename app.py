import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import torch
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import random
import string
import datetime
from datetime import datetime

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Custom CSS injection
def local_css():
    st.markdown("""
    <style>
    .stTextArea textarea { min-height: 150px; }
    .history-entry { border-left: 3px solid #4a90e2; padding: 0.5rem; margin: 0.5rem 0; }
    .stButton button { width: 100%; }
    .stSuccess { background-color: #e8f5e9!important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load FLAN-T5 model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
    
    # Load semantic similarity model
    sbert = SentenceTransformer('all-mpnet-base-v2').to(device)
    
    return model, tokenizer, sbert, device

class HybridParaphraser:
    def __init__(self, ai_model, ai_tokenizer, ai_sbert):
        self.ai_model = ai_model
        self.ai_tokenizer = ai_tokenizer
        self.ai_sbert = ai_sbert
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.pos_mapping = {
            'NN': wordnet.NOUN, 'NNS': wordnet.NOUN,
            'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
            'VB': wordnet.VERB, 'VBD': wordnet.VERB, 'VBG': wordnet.VERB, 
            'VBN': wordnet.VERB, 'VBP': wordnet.VERB, 'VBZ': wordnet.VERB,
            'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV
        }
        self.phrase_dict = {
            "how are you": ["how's it going", "how do you do","what's up"],
            "what is your name": ["may I know your name", "what should I call you","what are you called"]
        }
    
    def _calculate_similarity(self, original, paraphrase):
        """Calculate semantic similarity between original and paraphrase"""
        orig_embed = self.ai_sbert.encode(original, convert_to_tensor=True)
        para_embed = self.ai_sbert.encode(paraphrase, convert_to_tensor=True)
        return torch.nn.functional.cosine_similarity(orig_embed, para_embed, dim=0).item()

    
    def _get_phrases(self, sentence):
        """Check for predefined phrase replacements"""
        sentence_lower = sentence.lower()
        for phrase, alternatives in self.phrase_dict.items():
            if f' {phrase} ' in f' {sentence_lower} ':
                return random.choice(alternatives)
        return None

    def _get_synonyms(self, word, pos_tag):
        """Get synonyms based on POS tag"""
        lemma = self.lemmatizer.lemmatize(
            word, 
            pos=self.pos_mapping.get(pos_tag[:2], wordnet.NOUN)
        )
        synonyms = set()
        for syn in wordnet.synsets(lemma, pos=self.pos_mapping.get(pos_tag[:2])):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        return sorted(synonyms) if synonyms else []

    def _should_replace(self, word, pos_tag):
        """Determine if a word should be replaced"""
        return (
            word.lower() not in self.stop_words and
            word not in string.punctuation and
            pos_tag[:2] in self.pos_mapping and
            random.random() < 0.6  # 60% replacement probability
        )

    def _ai_paraphrase(self, text, style, temp, beams, max_length, similarity_threshold):
        input_prompt = f"{style} paraphrase: {text}"
        inputs = self.ai_tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.ai_model.device)
        
        outputs = self.ai_model.generate(
            inputs.input_ids,
            temperature=temp,
            max_length=max_length,
            num_beams=beams,
            num_return_sequences=5,
            early_stopping=True,
            repetition_penalty=2.5
        )
        
        paraphrases = [self.ai_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        return self._semantic_filter(text, paraphrases, similarity_threshold)

    def _semantic_filter(self, original, candidates, threshold):
        orig_embed = self.ai_sbert.encode(original, convert_to_tensor=True)
        para_embeds = self.ai_sbert.encode(candidates, convert_to_tensor=True)
        scores = torch.nn.functional.cosine_similarity(orig_embed, para_embeds)
        return [candidates[i] for i in torch.where(scores >= threshold)[0]]

    def _rule_based_paraphrase(self, text):
        return ' '.join([self._paraphrase_sentence(sent) for sent in sent_tokenize(text)])

    def _paraphrase_sentence(self, sentence):
        phrase_replacement = self._get_phrases(sentence)
        if phrase_replacement:
            return phrase_replacement.capitalize()
            
        words = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        paraphrased = []
        
        for word, tag in pos_tags:
            if not self._should_replace(word, tag):
                paraphrased.append(word)
                continue
                
            synonyms = self._get_synonyms(word, tag)
            paraphrased.append(random.choice(synonyms) if synonyms else word)
        
        if paraphrased:
            paraphrased[0] = paraphrased[0].capitalize()
            if sentence[-1] in string.punctuation:
                paraphrased[-1] += sentence[-1]
        
        return ' '.join(paraphrased)

    def generate_paraphrases(self, text, params):
        results = set()
        
        # Generate AI paraphrases
        if params['use_ai']:
            ai_paras = self._ai_paraphrase(
                text, 
                params['style'],
                params['temperature'],
                params['num_beams'],
                params['max_length'],
                params['similarity_threshold']
            )
            results.update(ai_paras)
        
        # Generate rule-based paraphrases
        if params['use_rule_based']:
            for _ in range(3):
                results.add(self._rule_based_paraphrase(text))
        
        return list(results)[:params['num_outputs']]

def main():
    st.set_page_config(
        page_title="Hybrid Paraphraser", 
        page_icon="✍️",
        layout="centered"
    )
    local_css()
    
    # Load models
    ai_model, ai_tokenizer, ai_sbert, device = load_models()
    
    # Initialize session state
    if 'paraphraser' not in st.session_state:
        st.session_state.paraphraser = HybridParaphraser(ai_model, ai_tokenizer, ai_sbert)
    
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ Settings")
        params = {
            'use_ai': st.checkbox("Use AI Model", True),
            'use_rule_based': st.checkbox("Use Rule-Based", True),
            'style': st.selectbox("Paraphrase Style", ["formal", "casual", "academic"]),
            'temperature': st.slider("Creativity", 0.1, 1.5, 0.7),
            'num_beams': st.slider("Search Beams", 1, 8, 5),
            'max_length': st.slider("Max Length", 50, 300, 150),
            'similarity_threshold': st.slider("Semantic Similarity", 0.5, 1.0, 0.85),
            'num_outputs': st.slider("Output Count", 1, 10, 3)
        }
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    # Main interface
    st.title("🤖 Hybrid AI Paraphraser")
    input_text = st.text_area("Input Text:", placeholder="Enter text to paraphrase...")
    
    if st.button("Generate Paraphrases"):
        if input_text.strip():
            with st.spinner("Generating optimal paraphrases..."):
                try:
                    paraphrases = st.session_state.paraphraser.generate_paraphrases(input_text, params)
                    
                    # Store in history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "original": input_text,
                        "paraphrases": paraphrases,
                        "method": "AI + Rules" if params['use_ai'] and params['use_rule_based'] else 
                                 "AI Only" if params['use_ai'] else "Rules Only"
                    })
                    
                    # Display results
                    st.success("Generated Paraphrases:")
                    for idx, para in enumerate(paraphrases, 1):
                        with st.expander(f"Version {idx}", expanded=True):
                            st.write(para)
                            # Calculate similarity using the paraphraser instance
                            similarity = st.session_state.paraphraser._calculate_similarity(input_text, para)
                            st.caption(f"Semantic Similarity Score: {similarity:.2f}")
                            
                except Exception as e:
                    st.error(f"Error generating paraphrases: {str(e)}")
        else:
            st.warning("Please enter text to paraphrase")

    # History section
    st.sidebar.title("📚 History")
    for entry in reversed(st.session_state.history[-5:]):
        with st.sidebar.container():
            st.markdown(f"""
            <div class='history-entry'>
                <small>{entry['timestamp']}</small>
                <p><b>Original:</b> {entry['original'][:50]}...</p>
                <small>Method: {entry['method']}</small>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()