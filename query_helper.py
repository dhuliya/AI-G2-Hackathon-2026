import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

class AudioRAGQuerier:
    def __init__(self, rag_store: str = "./rag_store"):
        self.client = chromadb.PersistentClient(path=rag_store)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="google/embeddinggemma-300m"
        )
        
        self.audios = self.client.get_collection("audios", embedding_function=self.sentence_transformer_ef)
        self.audio_segments = self.client.get_collection("audio_segments", embedding_function=self.sentence_transformer_ef)
        self.transcribe_segments = self.client.get_collection("transcribe_segments", embedding_function=self.sentence_transformer_ef)
        self.audio_attributes = self.client.get_collection("audio_attributes", embedding_function=self.sentence_transformer_ef)

    @staticmethod
    def _parse_to_seconds(val) -> float:
        """Converts timestamps (seconds, milliseconds, MM:SS, HH:MM:SS) into float seconds."""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val / 1000.0) if val > 10000 else float(val)
        if isinstance(val, str):
            val = val.strip().lower().replace("s", "")
            if ":" in val:
                parts = val.split(":")
                if len(parts) == 2:
                    return float(parts[0]) * 60.0 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            try:
                parsed = float(val)
                return float(parsed / 1000.0) if parsed > 10000 else parsed
            except ValueError:
                return 0.0
        return 0.0

    def _get_all_segments(self, audio_id: str = None):
        """Unified fetch merging transcribe_segments and audio_attributes metadata."""
        trans_data = self.transcribe_segments.get()
        segments_dict = {}

        # 1. Primary extraction from transcribe_segments
        if trans_data and trans_data.get('ids'):
            for i in range(len(trans_data['ids'])):
                doc = trans_data['documents'][i] if trans_data.get('documents') else ""
                meta = trans_data['metadatas'][i] if trans_data.get('metadatas') else {}
                
                seg_id = meta.get('segment_id') or trans_data['ids'][i].replace("trans_", "")
                segments_dict[seg_id] = {
                    "segment_id": seg_id,
                    "text": doc,
                    "metadata": dict(meta)
                }

        # 2. Enrich/fallback metadata from audio_attributes
        attr_data = self.audio_attributes.get()
        if attr_data and attr_data.get('metadatas'):
            for meta in attr_data['metadatas']:
                if not meta or meta.get("segment_id") == "root":
                    continue
                seg_id = meta.get('segment_id')
                if not seg_id:
                    continue
                
                if seg_id not in segments_dict:
                    segments_dict[seg_id] = {"segment_id": seg_id, "text": "", "metadata": {}}
                
                segments_dict[seg_id]["metadata"].update({k: v for k, v in meta.items() if v is not None})

        # 3. Standardize output
        results = []
        for seg_id, item in segments_dict.items():
            meta = item["metadata"]
            
            if audio_id and meta.get("audio_id") and meta.get("audio_id") != audio_id:
                continue

            start_raw = meta.get("start_sec") if meta.get("start_sec") is not None else meta.get("start_time", meta.get("start"))
            end_raw = meta.get("end_sec") if meta.get("end_sec") is not None else meta.get("end_time", meta.get("end"))

            start = self._parse_to_seconds(start_raw)
            end = self._parse_to_seconds(end_raw) if end_raw is not None else start

            speaker = meta.get("speaker_label") if meta.get("speaker_label") is not None else meta.get("speaker")

            results.append({
                "segment_id": seg_id,
                "speaker": speaker,
                "start_sec": start,
                "end_sec": end,
                "text": item["text"]
            })

        return results

    # 1. SEMANTIC SEARCH
    def semantic_search(self, query_text: str, n_results: int = 3, audio_id: str = None):
        total_docs = self.transcribe_segments.count()
        if total_docs == 0:
            return []

        where_clause = {"audio_id": audio_id} if audio_id else None

        results = self.transcribe_segments.query(
            query_texts=[query_text],
            n_results=min(n_results, total_docs),
            where=where_clause
        )
        
        output = []
        if not results.get('ids') or not results['ids'][0]:
            return output

        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i] if results.get('metadatas') else {}
            transcript = results['documents'][0][i] if results.get('documents') else ""
            distance = results['distances'][0][i] if results.get('distances') else None
            
            seg_id = meta.get('segment_id')
            
            # Lookup attributes
            attr_meta = {}
            if seg_id:
                attr_lookup = self.audio_attributes.get(where={"segment_id": seg_id})
                if attr_lookup.get('metadatas'):
                    attr_meta = attr_lookup['metadatas'][0]

            # Lookup blob path
            seg_blob_path = None
            if seg_id:
                seg_lookup = self.audio_segments.get(where={"segment_id": seg_id})
                if seg_lookup.get('metadatas'):
                    seg_blob_path = seg_lookup['metadatas'][0].get("blob_path")

            output.append({
                "segment_id": seg_id,
                "transcript": transcript,
                "distance": distance,
                "speaker_label": attr_meta.get("speaker_label", meta.get("speaker_label")),
                "start_sec": self._parse_to_seconds(attr_meta.get("start_sec", meta.get("start_sec"))),
                "end_sec": self._parse_to_seconds(attr_meta.get("end_sec", meta.get("end_sec"))),
                "audio_blob": seg_blob_path
            })
            
        return output

    # 2. ORDERED SEGMENTS
    def get_ordered_segments(self, audio_id: str = None):
        all_segments = self._get_all_segments(audio_id=audio_id)
        return sorted(all_segments, key=lambda x: x['start_sec'])

    # 3. TIME RANGE QUERY
    def get_segments_by_time_range(self, min_sec: float, max_sec: float, audio_id: str = None):
        all_segments = self._get_all_segments(audio_id=audio_id)
        
        filtered = [
            s for s in all_segments
            if s['start_sec'] <= max_sec and s['end_sec'] >= min_sec
        ]
        
        return sorted(filtered, key=lambda x: x['start_sec'])

    # 4. SPEAKER DIALOGUE QUERY
    def get_speaker_dialogue(self, speaker_label, audio_id: str = None):
        all_segments = self._get_all_segments(audio_id=audio_id)
        filtered = [s for s in all_segments if str(s['speaker']) == str(speaker_label)]
        return sorted(filtered, key=lambda x: x['start_sec'])
    
    # 5. QA CONTEXT WINDOW (Sequential N before & M after around an anchor)
    def get_qa_context_window(
        self, 
        query_text: str = None, 
        segment_id: str = None, 
        n_before: int = 2, 
        m_after: int = 2, 
        audio_id: str = None
    ):
        """
        Builds a QA context window containing N sequential segments before and M sequential 
        segments after an anchor segment (found either by segment_id or top query_text match).
        """
        ordered_segments = self.get_ordered_segments(audio_id=audio_id)
        if not ordered_segments:
            return {"anchor_segment": None, "context_segments": [], "formatted_context": ""}

        # Find target segment ID
        target_seg_id = segment_id
        if not target_seg_id and query_text:
            search_hits = self.semantic_search(query_text=query_text, n_results=1, audio_id=audio_id)
            if search_hits:
                target_seg_id = search_hits[0].get('segment_id')

        if not target_seg_id:
            return {"anchor_segment": None, "context_segments": [], "formatted_context": ""}

        # Find index in chronological sequence
        anchor_idx = None
        for idx, seg in enumerate(ordered_segments):
            if seg['segment_id'] == target_seg_id:
                anchor_idx = idx
                break

        if anchor_idx is None:
            return {"anchor_segment": None, "context_segments": [], "formatted_context": ""}

        # Extract sequence bounds
        start_idx = max(0, anchor_idx - n_before)
        end_idx = min(len(ordered_segments), anchor_idx + m_after + 1)

        context_segments = []
        for seg in ordered_segments[start_idx:end_idx]:
            seg_copy = dict(seg)
            seg_copy['is_anchor'] = (seg['segment_id'] == target_seg_id)
            context_segments.append(seg_copy)

        # Generate pre-formatted text block for LLM prompt context
        formatted_lines = []
        for seg in context_segments:
            spk = f"Speaker {seg['speaker']}" if seg['speaker'] is not None else "Speaker"
            anchor_tag = " [TARGET HIT]" if seg['is_anchor'] else ""
            formatted_lines.append(f"[{seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s] {spk}{anchor_tag}: {seg['text']}")
        
        return {
            "anchor_segment": ordered_segments[anchor_idx],
            "context_segments": context_segments,
            "formatted_context": "\n".join(formatted_lines)
        }
    

    def hybrid_search(self, query: str, top_k: int = 5, k: int = 60):
        """
        Simple hybrid search with guaranteed attribute resolution derived directly from trans_id.
        """

        trans_col = self.transcribe_segments
        attr_col = self.audio_attributes
        seg_col = self.audio_segments

        # 1. Vector Search (Semantic)
        query_res = trans_col.query(query_texts=[query], n_results=top_k)
        vector_ids = query_res["ids"][0] if query_res["ids"] else []

        # 2. Keyword Search (BM25)
        all_data = trans_col.get()
        if not all_data["documents"]:
            return []

        bm25 = BM25Okapi([doc.lower().split() for doc in all_data["documents"]])
        bm25_scores = bm25.get_scores(query.lower().split())

        keyword_ids = [
            all_data["ids"][i]
            for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        ]

        # 3. Simple RRF Scoring
        rrf_scores = {}

        for rank, doc_id in enumerate(vector_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        for rank, doc_id in enumerate(keyword_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # Sort top items by RRF score
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 4. Fetch Details & Format Output
        results = []
        for trans_id, score in fused_results:
            # Get transcript text
            trans_data = trans_col.get(ids=[trans_id])
            if not trans_data["ids"]:
                continue

            transcript_text = trans_data["documents"][0]
            trans_meta = trans_data["metadatas"][0] if trans_data["metadatas"] and trans_data["metadatas"][0] else {}

            # DIRECT ID DERIVATION (Guarantees matching across collections)
            # "trans_audio_123_seg_0" -> "attr_audio_123_seg_0"
            # "trans_audio_123_seg_0" -> "audio_123_seg_0"
            attr_id = trans_id.replace("trans_", "attr_", 1)
            seg_id = trans_id.replace("trans_", "", 1)

            # Query audio_attributes safely
            attr_data = attr_col.get(ids=[attr_id])
            attr = (
                attr_data["metadatas"][0]
                if attr_data["metadatas"] and attr_data["metadatas"][0]
                else {}
            )

            # Query audio_segments safely
            seg_data = seg_col.get(ids=[seg_id])
            seg = (
                seg_data["metadatas"][0]
                if seg_data["metadatas"] and seg_data["metadatas"][0]
                else {}
            )

            # Extract with multi-level fallbacks (attr_col -> trans_meta -> default)
            start_sec = attr.get("start_sec") if "start_sec" in attr else trans_meta.get("start_sec")
            end_sec = attr.get("end_sec") if "end_sec" in attr else trans_meta.get("end_sec")
            speaker = attr.get("speaker_label", trans_meta.get("speaker_label", "Unknown"))
            blob = seg.get("blob_path", trans_meta.get("blob_path", "N/A"))

            results.append({
                "transcript": transcript_text,
                "rrf_score": score,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "speaker_label": speaker,
                "audio_blob": blob
            })

        return results