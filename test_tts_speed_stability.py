#!/usr/bin/env python3
"""
Test script to verify TTS speed stability during long audio generation.
This test generates audio for texts of increasing length and analyzes speed consistency.
"""

import torch
import numpy as np
import soundfile as sf
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def analyze_audio_speed(audio_data: np.ndarray, sample_rate: int, text: str) -> dict:
    """
    Analyze the speed characteristics of generated audio.
    
    Returns:
        dict with speed metrics including:
        - duration: total audio duration
        - words_per_minute: estimated speaking rate
        - tempo_variance: measure of speed consistency
    """
    # Calculate basic metrics
    duration = len(audio_data) / sample_rate
    word_count = len(text.split())
    words_per_minute = (word_count / duration) * 60 if duration > 0 else 0
    
    # Analyze tempo variance by looking at energy segments
    # Split audio into segments and measure energy per segment
    segment_length = int(0.5 * sample_rate)  # 0.5 second segments
    segments = []
    
    for i in range(0, len(audio_data) - segment_length, segment_length):
        segment = audio_data[i:i + segment_length]
        energy = np.mean(segment ** 2)
        segments.append(energy)
    
    # Calculate variance in energy (proxy for tempo changes)
    tempo_variance = np.var(segments) if len(segments) > 1 else 0
    
    return {
        'duration': duration,
        'words_per_minute': words_per_minute,
        'tempo_variance': tempo_variance,
        'segment_count': len(segments),
        'word_count': word_count
    }

def test_speed_stability():
    """Test TTS speed stability with texts of increasing length."""
    
    print("🎤 Testing TTS Speed Stability")
    print("="*50)
    
    # Initialize TTS
    print("📥 Loading TTS model...")
    tts = ChatterboxMultilingualTTS()
    
    # Test texts of increasing length with Italian accents
    test_texts = [
        # Short text
        "Ciao! Come stai? È una bella giornata.",
        
        # Medium text
        """Ciao! Come stai? È una bella giornata oggi. 
        Le parole italiane con accenti come città, università, così, più, perché 
        dovrebbero essere pronunciate correttamente senza accelerazione.""",
        
        # Long text
        """Benvenuti a questo test di stabilità della velocità per il sistema TTS italiano.
        Questo testo contiene molte parole con caratteri accentati come è, à, ì, ò, ù.
        Vogliamo verificare che la velocità di pronuncia rimanga costante durante 
        tutta la generazione dell'audio, senza accelerazioni progressive.
        
        Parole come città, università, così, più, perché, caffè dovrebbero essere
        pronunciate con una velocità costante e naturale. La qualità della sintesi
        vocale è molto importante per garantire una buona esperienza utente.""",
        
        # Very long text
        """La storia dell'Italia è lunga e complessa, caratterizzata da una ricchezza 
        culturale straordinaria che si è sviluppata nel corso di millenni. 
        Dall'antica Roma all'Impero Romano, dalle città-stato medievali al Rinascimento, 
        ogni epoca ha lasciato tracce indelebili nel patrimonio artistico e culturale del paese.
        
        Durante il periodo romano, l'Italia era il centro di un impero che si estendeva 
        dall'Atlantico all'Oceano Indiano. Le città come Roma, Milano, Napoli e Firenze 
        erano centri di commercio, arte e cultura. Gli antichi romani costruirono strade, 
        acquedotti, teatri e anfiteatri che ancora oggi testimoniano la loro grandezza.
        
        Il Rinascimento italiano, tra il XIV e il XVI secolo, fu un periodo di straordinaria 
        creatività artistica e intellettuale. Artisti come Leonardo da Vinci, Michelangelo, 
        Raffaello e Botticelli crearono opere d'arte che sono considerate tra le più belle 
        e importanti della storia dell'umanità. Città come Firenze, Roma e Venezia 
        divennero centri di innovazione artistica e scientifica."""
    ]
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {len(text)} characters, {len(text.split())} words")
        print(f"Text preview: {text[:100]}...")
        
        try:
            # Generate audio
            start_time = time.time()
            
            audio_data = tts.tts(
                text=text,
                language='it',
                speaker_id=1,
                speed=1.0
            )
            
            generation_time = time.time() - start_time
            
            if audio_data is not None:
                # Analyze speed characteristics
                metrics = analyze_audio_speed(audio_data, tts.sample_rate, text)
                metrics['generation_time'] = generation_time
                metrics['text_length'] = len(text)
                metrics['test_number'] = i
                
                results.append(metrics)
                
                # Save audio file for manual inspection
                filename = f"speed_test_{i}_{len(text.split())}_words.wav"
                sf.write(filename, audio_data, tts.sample_rate)
                
                print(f"✅ Generated in {generation_time:.1f}s")
                print(f"   Audio duration: {metrics['duration']:.1f}s")
                print(f"   Speaking rate: {metrics['words_per_minute']:.1f} WPM")
                print(f"   Tempo variance: {metrics['tempo_variance']:.6f}")
                print(f"   💾 Saved as: {filename}")
                
            else:
                print(f"❌ Failed to generate audio for test {i}")
                
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
    
    # Analyze results
    if len(results) >= 2:
        print(f"\n📊 Speed Stability Analysis")
        print("="*50)
        
        # Calculate speed consistency
        wpm_values = [r['words_per_minute'] for r in results]
        wpm_mean = np.mean(wpm_values)
        wpm_std = np.std(wpm_values)
        wpm_cv = (wpm_std / wpm_mean) * 100 if wpm_mean > 0 else 0
        
        print(f"Speaking Rate Analysis:")
        print(f"  Mean WPM: {wpm_mean:.1f}")
        print(f"  Std Dev: {wpm_std:.1f}")
        print(f"  Coefficient of Variation: {wpm_cv:.1f}%")
        
        # Check for acceleration trend
        if len(results) >= 3:
            # Linear regression to detect trend
            x = np.array([r['text_length'] for r in results])
            y = np.array([r['words_per_minute'] for r in results])
            
            # Simple linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            
            print(f"\nSpeed Trend Analysis:")
            print(f"  Slope: {slope:.4f} WPM per character")
            
            if abs(slope) < 0.001:
                print("  ✅ Speed appears stable (no significant trend)")
            elif slope > 0.001:
                print("  ⚠️  Speed increases with text length (acceleration detected)")
            else:
                print("  ⚠️  Speed decreases with text length (deceleration detected)")
        
        # Detailed results table
        print(f"\nDetailed Results:")
        print(f"{'Test':<4} {'Words':<6} {'Duration':<8} {'WPM':<6} {'Variance':<10} {'Gen Time':<8}")
        print("-" * 50)
        
        for r in results:
            print(f"{r['test_number']:<4} {r['word_count']:<6} {r['duration']:<8.1f} "
                  f"{r['words_per_minute']:<6.1f} {r['tempo_variance']:<10.6f} {r['generation_time']:<8.1f}")
        
        # Assessment
        print(f"\n🎯 Assessment:")
        if wpm_cv < 5:
            print("  ✅ Excellent speed stability (CV < 5%)")
        elif wpm_cv < 10:
            print("  ✅ Good speed stability (CV < 10%)")
        elif wpm_cv < 20:
            print("  ⚠️  Moderate speed variation (CV < 20%)")
        else:
            print("  ❌ High speed variation (CV >= 20%)")
    
    return results

def create_speed_plot(results: List[dict]):
    """Create a plot showing speed vs text length."""
    if len(results) < 2:
        return
    
    try:
        import matplotlib.pyplot as plt
        
        text_lengths = [r['text_length'] for r in results]
        wpm_values = [r['words_per_minute'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(text_lengths, wpm_values, s=100, alpha=0.7)
        plt.plot(text_lengths, wpm_values, 'b--', alpha=0.5)
        
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Speaking Rate (WPM)')
        plt.title('TTS Speed Stability Analysis')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(text_lengths, wpm_values, 1)
        p = np.poly1d(z)
        plt.plot(text_lengths, p(text_lengths), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('tts_speed_analysis.png', dpi=150, bbox_inches='tight')
        print(f"📈 Speed analysis plot saved as: tts_speed_analysis.png")
        
    except ImportError:
        print("📈 matplotlib not available - skipping plot generation")

if __name__ == "__main__":
    results = test_speed_stability()
    create_speed_plot(results)
    print("\n✅ Speed stability test completed!")
