import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import timm
from huggingface_hub import hf_hub_download
import requests
from pathlib import Path
import json

# Set page config
st.set_page_config(
    page_title="OceanEcho - Marine Species Classifier",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - BACKGROUND LEBIH TERANG
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #e1f5fe 0%, #81d4fa 100%);
    }
    h1 {
        color: #01579b;
        text-align: center;
        font-size: 3.5rem;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.5);
    }
    .upload-section {
        background: rgba(255,255,255,0.9);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 2px solid #0277bd;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0288d1 0%, #0277bd 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(2, 119, 189, 0.37);
    }
    .metric-card {
        background: rgba(255,255,255,0.85);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
        border: 1px solid #0288d1;
    }
    /* Text colors for better visibility */
    p, span, div {
        color: #01579b !important;
    }
    .stMarkdown {
        color: #01579b;
    }
    </style>
    """, unsafe_allow_html=True)

def download_from_huggingface():
    """Download models from Hugging Face Hub"""
    os.makedirs('models', exist_ok=True)
    
    repo_id = "HyacinthiaIca/OceanEcho"  # Replace with your repo
    
    model_files = [
        'marine_model_fold_0_best.pth',
        'marine_model_fold_1_best.pth',
        'marine_model_fold_2_best.pth',
        'marine_model_fold_3_best.pth',
        'marine_model_fold_4_best.pth',
    ]
    
    for filename in model_files:
        output_path = f'models/{filename}'
        
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir='models',
                    local_dir_use_symlinks=False
                )
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed: {e}")
        else:
            print(f"⏭️ {filename} already exists")

# Configuration class (same as in notebook)
class CFG:
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_channels = 1
    spec_height = 256
    spec_width = 256

cfg = CFG()

# Model architectures (copy from notebook)
class MarineSpeciesSmallModel(nn.Module):
    """Small model for marine species classification"""
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        
        # Smaller ViT for AST branch
        self.ast = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            img_size=(cfg.spec_height, cfg.spec_width),
            in_chans=cfg.in_channels,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=6,
            drop_path_rate=0.3,
            drop_rate=0.4,
            num_classes=0
        )
        
        # Smaller EfficientNet
        self.efficientnet = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            in_chans=cfg.in_channels,
            num_classes=0,
            drop_rate=0.4,
            drop_path_rate=0.3
        )
        
        self.ast_dim = 768
        
        # Get EfficientNet dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, cfg.in_channels, cfg.spec_height, cfg.spec_width)
            efficientnet_features = self.efficientnet(dummy_input)
            self.efficientnet_dim = efficientnet_features.shape[1]
        
        # Fusion layers
        self.fusion_dim = self.ast_dim + self.efficientnet_dim
        fusion_hidden = min(512, self.fusion_dim // 2)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(fusion_hidden // 2, num_classes)
        )
        
    def forward(self, x):
        ast_features = self.ast(x)
        efficientnet_features = self.efficientnet(x)
        combined_features = torch.cat([ast_features, efficientnet_features], dim=1)
        fused_features = self.fusion(combined_features)
        logits = self.classifier(fused_features)
        return logits

# Audio processing functions
def process_audio_file(audio_path, cfg):
    """Process audio file to mel spectrogram"""
    try:
        # Load audio
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)
        
        if len(audio_data) < target_samples:
            n_copy = int(np.ceil(target_samples / len(audio_data)))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)
        
        # Extract center portion
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]
        
        # Pad if necessary
        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, 
                                 (0, target_samples - len(center_audio)), 
                                 mode='constant')
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=center_audio, sr=cfg.FS,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=cfg.FMAX,
            power=2.0
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel_spec).astype(np.float32)
        
        # Resize to target shape
        if mel_db.shape != cfg.TARGET_SHAPE:
            mel_db = cv2.resize(mel_db, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
        return mel_db
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def normalize_melspec(mel_spec):
    """Normalize mel spectrogram"""
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    mel_spec = np.clip(mel_spec, -3, 3)
    return mel_spec

# Load species mapping
@st.cache_resource
def load_species_mapping():
    """Load species index mapping"""
    species_list = [
        "Atlantic Spotted Dolphin", "Bearded Seal", "Beluga White Whale",
        "Bottlenose Dolphin", "Bowhead Whale", "Clymene Dolphin",
        "Common Dolphin", "False Killer Whale", "Fin Finback Whale",
        "Frasers Dolphin", "Grampus Rissos Dolphin", "Harp Seal",
        "Humpback Whale", "Killer Whale", "Leopard Seal",
        "Long Finned Pilot Whale", "Melon Headed Whale", "Minke Whale",
        "Narwhal", "Northern Right Whale", "Pantropical Spotted Dolphin",
        "Ross Seal", "Rough Toothed Dolphin", "Short Finned Pacific Pilot Whale",
        "Southern Right Whale", "Sperm Whale", "Spinner Dolphin",
        "Striped Dolphin", "Walrus", "White beaked Dolphin",
        "White sided Dolphin"
    ]
    
    idx_to_species = {i: species for i, species in enumerate(species_list)}
    species_to_idx = {species: i for i, species in enumerate(species_list)}
    
    return idx_to_species, species_to_idx

# Load ensemble of models
@st.cache_resource
def load_ensemble_models(model_dir="models"):
    """Load all trained models for ensemble prediction"""
    models = []
    model_paths = []
    
    # Look for model files with pattern: marine_model_fold_X_best.pth
    for i in range(5):  # 5 folds
        model_path = os.path.join(model_dir, f"marine_model_fold_{i}_best.pth")
        if os.path.exists(model_path):
            model_paths.append(model_path)
    
    if not model_paths:
        st.warning("⚠️ No model files found! Using default path...")
        model_paths = [
            'marine_model_fold_0_best.pth',
            'marine_model_fold_1_best.pth',
            'marine_model_fold_2_best.pth',
            'marine_model_fold_3_best.pth',
            'marine_model_fold_4_best.pth'
        ]
    
    # Load models
    idx_to_species, _ = load_species_mapping()
    num_classes = len(idx_to_species)
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=cfg.device)
                
                # Load model
                model = MarineSpeciesSmallModel(num_classes, cfg)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model_state' in checkpoint:
                    model.load_state_dict(checkpoint['model_state'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(cfg.device)
                model.eval()
                models.append(model)
                
            except Exception as e:
                st.warning(f"Could not load model {model_path}: {e}")
    
    if len(models) == 0:
        st.error("❌ No models loaded successfully!")
        return None
    
    st.success(f"✅ Loaded {len(models)} models for ensemble prediction")
    return models

def ensemble_predict(models, mel_spec, cfg):
    """Make ensemble prediction with individual model tracking - EXACTLY like notebook"""
    # Normalize (EXACTLY like notebook)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    mel_spec = np.clip(mel_spec, -3, 3)
    
    # Convert to tensor (EXACTLY like notebook)
    mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mel_tensor = mel_tensor.to(cfg.device)
    
    # Get predictions from all models
    all_probs = []
    individual_predictions = []  # Track each model's prediction
    
    with torch.no_grad():
        for i, model in enumerate(models):
            # EXACTLY like notebook
            outputs = model(mel_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            # Store full probability distribution
            probs_np = probabilities.cpu().numpy()[0]
            all_probs.append(probs_np)
            
            # Store individual model prediction
            individual_predictions.append({
                'model_num': i + 1,
                'top_class_idx': predicted_idx,
                'top_prob': confidence,
                'all_probs': probs_np
            })
            
            # DEBUG: Print individual model confidence
            print(f"Model {i+1}: Class {predicted_idx}, Confidence {confidence*100:.1f}%")
    
    # Average probabilities (ensemble)
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get top predictions
    top_5_idx = np.argsort(avg_probs)[-5:][::-1]
    top_5_probs = avg_probs[top_5_idx]
    
    return top_5_idx, top_5_probs, avg_probs, individual_predictions

# Marine species info
SPECIES_INFO = {
    "Atlantic Spotted Dolphin": {
        "emoji": "🐬",
        "habitat": "Atlantic Ocean",
        "size": "2.3m length",
        "weight": "140kg",
        "facts": "Known for their distinctive spotted pattern"
    },
    "Humpback Whale": {
        "emoji": "🐋",
        "habitat": "All major oceans",
        "size": "12-16m length",
        "weight": "25-30 tons",
        "facts": "Famous for their complex songs"
    },
    "Killer Whale": {
        "emoji": "🐋",
        "habitat": "All oceans",
        "size": "6-8m length",
        "weight": "3-5 tons",
        "facts": "Apex predator, highly intelligent"
    },
    # Add more species info as needed
}

def get_species_info(species_name):
    """Get species information with default fallback"""
    return SPECIES_INFO.get(species_name, {
        "emoji": "🐋",
        "habitat": "Ocean",
        "size": "Varies",
        "weight": "Varies",
        "facts": "Amazing marine creature"
    })

# Main app
def main():
    # Header with ocean animation
    st.markdown("""
        <h1>🌊 OceanEcho 🐋</h1>
        <p style='text-align: center; color: #ffffff; font-size: 1.2rem; margin-top: -1rem;'>
            Marine Species Sound Classification powered by AI
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Model info
        st.markdown("#### 🤖 Model Information")
        st.info("""
        **Architecture**: Hybrid AST + EfficientNet  
        **Models**: 5-fold ensemble  
        **Classes**: 31 marine species  
        **Accuracy**: ~85-92% CV
        """)
        
        # About
        st.markdown("#### ℹ️ About")
        st.markdown("""
        OceanEcho uses deep learning to identify marine species from their sounds.
        
        **Features:**
        - 🎵 Audio analysis
        - 🔊 Mel spectrogram visualization
        - 🎯 Ensemble predictions
        - 📊 Confidence scores
        - 🐋 Species information
        """)
        
        # Upload settings
        st.markdown("#### ⚙️ Settings")
        show_spectrogram = st.checkbox("Show Mel Spectrogram", value=True)
        show_waveform = st.checkbox("Show Waveform", value=True)
        show_top_n = st.slider("Top predictions to show", 3, 10, 5)
        show_individual_models = st.checkbox("Show Individual Model Predictions", value=False)
    
    # Load models
    idx_to_species, species_to_idx = load_species_mapping()
    
    with st.spinner("🔄 Loading AI models..."):
        models = load_ensemble_models()
    
    if models is None:
        st.error("❌ Failed to load models. Please check model files.")
        st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎵 Upload Audio")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV format)",
            type=['wav'],
            help="Upload a marine mammal sound recording"
        )
        
        # Sample audio option
        use_sample = st.checkbox("🎲 Use sample audio")
        
        if use_sample:
            sample_dir = "sample_audio"
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.wav')]
                if sample_files:
                    selected_sample = st.selectbox("Select sample", sample_files)
                    audio_path = os.path.join(sample_dir, selected_sample)
                else:
                    st.warning("No sample files found")
                    audio_path = None
            else:
                st.warning("Sample directory not found")
                audio_path = None
        elif uploaded_file:
            # Save uploaded file
            audio_path = f"temp_{uploaded_file.name}"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.read())
        else:
            audio_path = None
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if audio_path:
            # Display audio player
            st.audio(audio_path, format='audio/wav')
            
            # Process and predict
            if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the sound..."):
                    # Process audio
                    mel_spec = process_audio_file(audio_path, cfg)
                    
                    if mel_spec is not None:
                        # Make prediction with individual tracking
                        top_idx, top_probs, all_probs, individual_preds = ensemble_predict(models, mel_spec, cfg)
                        
                        # Get predictions
                        predictions = []
                        for idx, prob in zip(top_idx[:show_top_n], top_probs[:show_top_n]):
                            species = idx_to_species[idx]
                            predictions.append({
                                'species': species,
                                'probability': float(prob),
                                'confidence': float(prob) * 100
                            })
                        
                        # Store in session state
                        st.session_state['predictions'] = predictions
                        st.session_state['mel_spec'] = mel_spec
                        st.session_state['audio_path'] = audio_path
                        st.session_state['all_probs'] = all_probs
                        st.session_state['individual_preds'] = individual_preds
                    else:
                        st.error("Failed to process audio file")
    
    with col2:
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            
            # Main prediction (ENSEMBLE)
            st.markdown("### 🎯 Ensemble Prediction")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            top_pred = predictions[0]
            species_info = get_species_info(top_pred['species'])
            
            st.markdown(f"""
                <h2 style='text-align: center; color: white;'>
                    {species_info['emoji']} {top_pred['species']}
                </h2>
                <h3 style='text-align: center; color: #90EE90;'>
                    Confidence: {top_pred['confidence']:.1f}%
                </h3>
                <p style='text-align: center; color: white; font-size: 0.9rem; opacity: 0.9;'>
                    📊 Average of {len(st.session_state.get('individual_preds', []))} models
                </p>
            """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(float(top_pred['probability']))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Individual model predictions (NEW)
            if show_individual_models and 'individual_preds' in st.session_state:
                st.markdown("---")
                st.markdown("### 🔬 Individual Model Predictions")
                
                individual_preds = st.session_state['individual_preds']
                
                # Create comparison table
                model_data = []
                for pred in individual_preds:
                    species = idx_to_species[pred['top_class_idx']]
                    model_data.append({
                        'Model': f"Fold {pred['model_num']}",
                        'Predicted Species': species,
                        'Confidence': f"{pred['top_prob']*100:.1f}%",
                        'Probability': pred['top_prob']
                    })
                
                df_models = pd.DataFrame(model_data)
                
                # Display as table
                st.dataframe(
                    df_models[['Model', 'Predicted Species', 'Confidence']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Agreement visualization
                st.markdown("#### 📊 Model Agreement Analysis")
                
                # Count agreements
                predictions_list = [idx_to_species[p['top_class_idx']] for p in individual_preds]
                from collections import Counter
                pred_counts = Counter(predictions_list)
                
                # Create agreement chart
                fig, ax = plt.subplots(figsize=(10, 4))
                
                species_names = list(pred_counts.keys())
                counts = list(pred_counts.values())
                colors = plt.cm.Set3(range(len(species_names)))
                
                bars = ax.bar(range(len(species_names)), counts, color=colors)
                ax.set_xticks(range(len(species_names)))
                ax.set_xticklabels([s[:25] for s in species_names], rotation=45, ha='right')
                ax.set_ylabel('Number of Models')
                ax.set_title('How Many Models Agree on Each Prediction?')
                ax.set_ylim(0, 5)
                ax.grid(axis='y', alpha=0.3)
                
                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}/5',
                           ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Agreement metrics
                max_agreement = max(counts)
                total_models = len(individual_preds)
                
                agreement_col1, agreement_col2, agreement_col3 = st.columns(3)
                
                with agreement_col1:
                    st.metric("🎯 Max Agreement", f"{max_agreement}/{total_models} models")
                
                with agreement_col2:
                    agreement_pct = (max_agreement / total_models) * 100
                    st.metric("📊 Agreement %", f"{agreement_pct:.0f}%")
                
                with agreement_col3:
                    unique_preds = len(pred_counts)
                    st.metric("🔀 Unique Predictions", f"{unique_preds}")
                
                # Detailed individual model predictions
                with st.expander("🔍 View Detailed Model Predictions"):
                    for i, pred in enumerate(individual_preds, 1):
                        st.markdown(f"**Model {i} (Fold {pred['model_num']}):**")
                        
                        # Top 3 predictions for this model
                        top_3_idx = np.argsort(pred['all_probs'])[-3:][::-1]
                        
                        for rank, idx in enumerate(top_3_idx, 1):
                            species = idx_to_species[idx]
                            prob = pred['all_probs'][idx]
                            
                            st.write(f"  {rank}. {species}: {prob*100:.1f}%")
                            st.progress(float(prob))
                        
                        if i < len(individual_preds):
                            st.markdown("---")
            
            # Species info card
            st.markdown("#### 📋 Species Information")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.metric("🌍 Habitat", species_info['habitat'])
                st.metric("📏 Size", species_info['size'])
            
            with info_col2:
                st.metric("⚖️ Weight", species_info['weight'])
            
            st.info(f"💡 **Fun Fact:** {species_info['facts']}")
            
            # Top ensemble predictions
            st.markdown("#### 📊 Top Ensemble Predictions")
            
            for i, pred in enumerate(predictions, 1):
                with st.expander(f"#{i} - {pred['species']} ({pred['confidence']:.1f}%)"):
                    st.progress(float(pred['probability']))
                    info = get_species_info(pred['species'])
                    st.write(f"{info['emoji']} **Habitat:** {info['habitat']}")
                    st.write(f"📏 **Size:** {info['size']}")
    
    # Visualizations
    if 'mel_spec' in st.session_state and (show_spectrogram or show_waveform):
        st.markdown("---")
        st.markdown("### 📈 Audio Analysis")
        
        viz_cols = st.columns(2 if show_spectrogram and show_waveform else 1)
        
        col_idx = 0
        
        if show_spectrogram:
            with viz_cols[col_idx]:
                st.markdown("#### 🎨 Mel Spectrogram")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                img = ax.imshow(st.session_state['mel_spec'], 
                               aspect='auto', origin='lower', 
                               cmap='viridis')
                ax.set_xlabel('Time Frames')
                ax.set_ylabel('Mel Frequency Bins')
                ax.set_title('Mel Spectrogram Visualization')
                plt.colorbar(img, ax=ax, label='dB')
                st.pyplot(fig)
                plt.close()
                
                col_idx += 1
        
        if show_waveform:
            with viz_cols[col_idx]:
                st.markdown("#### 🌊 Waveform")
                
                # Load audio for waveform
                y, sr = librosa.load(st.session_state['audio_path'], sr=cfg.FS)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                time = np.arange(len(y)) / sr
                ax.plot(time, y, linewidth=0.5, color='steelblue')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Audio Waveform')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        # Probability distribution
        if 'all_probs' in st.session_state:
            st.markdown("#### 📊 Probability Distribution (Top 15)")
            
            all_probs = st.session_state['all_probs']
            top_15_idx = np.argsort(all_probs)[-15:][::-1]
            top_15_probs = all_probs[top_15_idx]
            top_15_species = [idx_to_species[i] for i in top_15_idx]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_15_species)))
            bars = ax.barh(range(len(top_15_species)), top_15_probs, color=colors)
            ax.set_yticks(range(len(top_15_species)))
            ax.set_yticklabels([s[:30] for s in top_15_species])
            ax.set_xlabel('Probability')
            ax.set_title('Top 15 Species Probability Distribution')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, prob) in enumerate(zip(bars, top_15_probs)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{prob*100:.1f}%',
                       ha='left', va='center', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #ffffff; padding: 2rem;'>
            <h3>🌊 OceanEcho - Protecting Marine Life through AI 🐋</h3>
            <p>Powered by Deep Learning | AST + EfficientNet Hybrid Architecture</p>
            <p style='font-size: 0.9rem; opacity: 0.8;'>
                Made with ❤️ for ocean conservation
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    download_from_huggingface()
    main()