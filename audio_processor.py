import os
import uuid
from dotenv import load_dotenv
from pydub import AudioSegment
import chromadb
from chromadb.utils import embedding_functions
from simple_diarizer.diarizer import Diarizer
from sarvamai import SarvamAI

load_dotenv()

# ==========================================
# 1. CHROMADB INITIALIZATION
# ==========================================
def init_rag_db(rag_store: str = "./rag_store"):
    client = chromadb.PersistentClient(path=rag_store)
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="google/embeddinggemma-300m"
    )
    
    collection_names = [
        "audios",
        "audio_segments",
        "transcribe_segments",
        "audio_attributes"
    ]
    
    collections = {}
    for name in collection_names:
        collections[name] = client.get_or_create_collection(
            name=name, 
            embedding_function=sentence_transformer_ef
        )
    return client, collections

# ==========================================
# 2. HELPER UTILITIES
# ==========================================
def merge_speaker_segments(segments):
    if not segments:
        return []

    merged = [dict(segments[0])]
    for current in segments[1:]:
        previous = merged[-1]
        if previous['label'] == current['label']:
            previous['end'] = max(float(previous['end']), float(current['end']))
            previous['end_sample'] = max(int(previous['end_sample']), int(current['end_sample']))
        else:
            merged.append(dict(current))
    return merged


def transcribe_audio(client: SarvamAI, file_path: str):
    try:
        with open(file_path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v4",
                language_code="en-IN",
                mode="transcribe",
            )
        return response.transcript if hasattr(response, 'transcript') else str(response)
    except Exception as e:
        print(f"Transcription error for {file_path}: {e}")
        return ""

# ==========================================
# 3. PIPELINE PROCESSOR
# ==========================================
def process_audio_file(
    audio_file_path: str,
    collections: dict,
    sarvam_client: SarvamAI,
    diarizer: Diarizer,
    num_speakers: int = 2,
    base_output_dir: str = "./data"
):
    segments_dir = os.path.join(base_output_dir, "segments")
    transcripts_dir = os.path.join(base_output_dir, "transcripts")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    audio_filename = os.path.basename(audio_file_path)
    audio_id = f"audio_{uuid.uuid4().hex[:8]}"
    
    print(f"--- Processing Root Audio: {audio_filename} (ID: {audio_id}) ---")
    
    # Load audio into memory once
    full_audio = AudioSegment.from_wav(audio_file_path)
    total_duration_sec = len(full_audio) / 1000.0

    # 1. Collection: 'audios' (Blob location for original file)
    collections["audios"].add(
        ids=[audio_id],
        documents=[f"Audio file reference for {audio_filename}"],
        metadatas=[{
            "audio_id": audio_id,
            "file_name": audio_filename,
            "blob_path": os.path.abspath(audio_file_path),
            "file_size_bytes": os.path.getsize(audio_file_path)
        }]
    )

    # 2. Diarization & Processing
    print("Running diarization...")
    raw_segments = diarizer.diarize(audio_file_path, num_speakers=num_speakers)
    merged_segments = merge_speaker_segments(raw_segments)

    print(f"Extracted {len(merged_segments)} merged segments. Processing cuts and transcriptions...")

    # Batch accumulators
    attr_ids, attr_docs, attr_metas = [], [], []
    seg_ids, seg_docs, seg_metas = [], [], []
    trans_ids, trans_docs, trans_metas = [], [], []

    # Store root audio attributes
    attr_ids.append(f"attr_{audio_id}")
    attr_docs.append(f"Global attributes for audio {audio_filename}")
    attr_metas.append({
        "audio_id": audio_id,
        "segment_id": "root",
        "type": "audio_global",
        "channels": full_audio.channels,
        "sample_width": full_audio.sample_width,
        "frame_rate": full_audio.frame_rate,
        "frame_width": full_audio.frame_width,
        "duration_sec": total_duration_sec
    })

    for i, seg in enumerate(merged_segments):
        start_sec = float(seg['start'])
        end_sec = float(seg['end'])
        start_sample = int(seg.get('start_sample', int(start_sec * full_audio.frame_rate)))
        end_sample = int(seg.get('end_sample', int(end_sec * full_audio.frame_rate)))
        speaker_label = int(seg['label'])
        
        segment_id = f"{audio_id}_seg_{i}"
        seg_file_path = os.path.join(segments_dir, f"{segment_id}.wav")
        txt_file_path = os.path.join(transcripts_dir, f"{segment_id}.txt")

        # Slice and export audio segment blob
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        clipped_audio = full_audio[start_ms:end_ms]
        clipped_audio.export(seg_file_path, format="wav")

        # Transcribe segment
        transcript_text = transcribe_audio(sarvam_client, seg_file_path)
        
        # Save transcript text blob to disk
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        # 3. Collection: 'audio_segments' (Segment Blob location)
        seg_ids.append(segment_id)
        seg_docs.append(f"Audio slice {i} from {start_sec:.2f}s to {end_sec:.2f}s")
        seg_metas.append({
            "audio_id": audio_id,
            "segment_id": segment_id,
            "blob_path": os.path.abspath(seg_file_path)
        })

        # 4. Collection: 'transcribe_segments' (Indexed Transcript Text)
        if transcript_text.strip():
            trans_ids.append(f"trans_{segment_id}")
            trans_docs.append(transcript_text)
            trans_metas.append({
                "audio_id": audio_id,
                "segment_id": segment_id,
                "blob_path": os.path.abspath(txt_file_path)
            })

        # 5. Collection: 'audio_attributes' (Time, Samples, Speaker Labels metadata)
        attr_ids.append(f"attr_{segment_id}")
        attr_docs.append(f"Attributes for segment {segment_id}")
        attr_metas.append({
            "audio_id": audio_id,
            "segment_id": segment_id,
            "type": "segment_attribute",
            "speaker_label": speaker_label,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "duration_sec": end_sec - start_sec
        })

    # Execute Batch Operations into ChromaDB
    collections["audio_attributes"].add(ids=attr_ids, documents=attr_docs, metadatas=attr_metas)
    collections["audio_segments"].add(ids=seg_ids, documents=seg_docs, metadatas=seg_metas)
    if trans_ids:
        collections["transcribe_segments"].add(ids=trans_ids, documents=trans_docs, metadatas=trans_metas)

    print(f"Finished processing {audio_filename}.\n")

# ==========================================
# 4. MAIN EXECUTION BLOCK FOR IPYNB
# ==========================================
def process_audio(audio_file: str):
    client, collections = init_rag_db()
    
    sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
    diarizer = Diarizer(embed_model='xvec', cluster_method='sc')

    process_audio_file(
        audio_file_path=audio_file,
        collections=collections,
        sarvam_client=sarvam_client,
        diarizer=diarizer,
        num_speakers=2
    )

    print("--- Collection Population Summary ---")
    for name, col in collections.items():
        print(f"Collection '{name}': {col.count()} documents")


def main():
    audio_file = "./data/Oral History Audio Interviews/AbigailBThomas_FINAL_1_.wav"
    process_audio(audio_file)

if __name__ == "__main__" or "__file__" not in globals():
    main()