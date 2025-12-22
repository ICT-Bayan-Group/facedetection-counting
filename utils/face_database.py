
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import time

class FaceDatabase:
    """
    Mengelola database wajah yang sudah terdeteksi
    Menggunakan JSON untuk menyimpan embeddings dan metadata
    """
    
    def __init__(self, db_path='data/face_database.json'):
        self.db_path = db_path
        self.faces = {}
        self.similarity_threshold = 0.6
        self._lock = threading.Lock()  # Thread safety
        self._dirty = False  # Flag untuk track perubahan
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.load_database()
        
        # Start auto-save thread
        self.auto_save_interval = 30  # Save setiap 30 detik
        self._running = True
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()
        print(f"✅ Auto-save enabled (every {self.auto_save_interval}s)")
    
    def _auto_save_loop(self):
        """Background thread untuk auto-save"""
        while self._running:
            time.sleep(self.auto_save_interval)
            if self._dirty:
                self.save_database()
                self._dirty = False
                print(f"💾 Auto-saved database: {len(self.faces)} faces")
    
    def load_database(self):
        """Load database dari JSON file"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    
                # Convert embeddings dari list ke numpy array
                with self._lock:
                    self.faces = {}
                    for face_id, face_data in data.items():
                        try:
                            self.faces[face_id] = {
                                'embedding': np.array(face_data['embedding'], dtype=np.float32),
                                'first_seen': face_data['first_seen'],
                                'last_seen': face_data['last_seen'],
                                'detection_count': face_data['detection_count']
                            }
                        except Exception as e:
                            print(f"⚠️  Skipping corrupted face {face_id}: {e}")
                            continue
                
                print(f"✅ Face database loaded: {len(self.faces)} unique faces")
                print(f"📁 Database path: {os.path.abspath(self.db_path)}")
                
            else:
                print(f"📁 Creating new face database at: {os.path.abspath(self.db_path)}")
                self.faces = {}
                self.save_database()  # Create empty file
                
        except Exception as e:
            print(f"❌ Error loading face database: {e}")
            print(f"📁 Attempted path: {os.path.abspath(self.db_path)}")
            self.faces = {}
    
    def save_database(self, force=False):
        """Save database ke JSON file"""
        try:
            with self._lock:
                if not self.faces and not force:
                    print("⚠️  No faces to save")
                    return False
                
                # Convert numpy arrays ke list untuk JSON serialization
                data = {}
                for face_id, face_data in self.faces.items():
                    try:
                        embedding_list = face_data['embedding'].tolist()
                        data[face_id] = {
                            'embedding': embedding_list,
                            'first_seen': face_data['first_seen'],
                            'last_seen': face_data['last_seen'],
                            'detection_count': face_data['detection_count']
                        }
                    except Exception as e:
                        print(f"⚠️  Error converting face {face_id}: {e}")
                        continue
                
                # Write to temporary file first
                temp_path = self.db_path + '.tmp'
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Atomic replace
                os.replace(temp_path, self.db_path)
                
                print(f"💾 Database saved: {len(data)} faces → {os.path.abspath(self.db_path)}")
                return True
            
        except Exception as e:
            print(f"❌ Error saving face database: {e}")
            print(f"📁 Attempted path: {os.path.abspath(self.db_path)}")
            return False
    
    def find_matching_face(self, embedding):
        """
        Cari wajah yang cocok di database
        Returns: (face_id, similarity) atau (None, 0) jika tidak ada yang cocok
        """
        if embedding is None:
            return None, 0
        
        if len(self.faces) == 0:
            return None, 0
        
        best_match_id = None
        best_similarity = 0
        
        # Normalize embedding
        try:
            embedding = np.array(embedding, dtype=np.float32)
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm == 0:
                return None, 0
            embedding = embedding / embedding_norm
        except Exception as e:
            print(f"⚠️  Error normalizing embedding: {e}")
            return None, 0
        
        # Cari wajah dengan similarity tertinggi
        with self._lock:
            for face_id, face_data in self.faces.items():
                try:
                    stored_embedding = face_data['embedding']
                    
                    # Normalize stored embedding
                    stored_norm = np.linalg.norm(stored_embedding)
                    if stored_norm == 0:
                        continue
                    stored_embedding = stored_embedding / stored_norm
                    
                    # Hitung cosine similarity
                    similarity = float(np.dot(embedding, stored_embedding))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = face_id
                
                except Exception as e:
                    print(f"⚠️  Error comparing with face {face_id}: {e}")
                    continue
        
        # Return jika similarity melebihi threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id, best_similarity
        
        return None, 0
    
    def add_or_update_face(self, face_id, embedding):
        """
        Tambahkan wajah baru atau update existing face
        Returns: (is_new_face, matched_id, similarity)
        """
        if embedding is None:
            print(f"⚠️  Cannot add face {face_id}: embedding is None")
            return False, None, 0
        
        try:
            # Ensure embedding is numpy array
            embedding = np.array(embedding, dtype=np.float32)
            
            # Validate embedding
            if embedding.size == 0 or np.isnan(embedding).any():
                print(f"⚠️  Invalid embedding for face {face_id}")
                return False, None, 0
            
        except Exception as e:
            print(f"⚠️  Error processing embedding for face {face_id}: {e}")
            return False, None, 0
        
        # Cek apakah wajah sudah ada di database
        matched_id, similarity = self.find_matching_face(embedding)
        
        current_time = datetime.now().isoformat()
        
        with self._lock:
            if matched_id is not None:
                # Wajah sudah ada di database - UPDATE
                self.faces[matched_id]['last_seen'] = current_time
                self.faces[matched_id]['detection_count'] += 1
                
                # Update embedding dengan weighted average
                old_embedding = self.faces[matched_id]['embedding']
                weight = 0.9  # 90% old, 10% new
                new_embedding = weight * old_embedding + (1 - weight) * embedding
                new_norm = np.linalg.norm(new_embedding)
                if new_norm > 0:
                    self.faces[matched_id]['embedding'] = new_embedding / new_norm
                
                self._dirty = True
                print(f"🔄 Updated existing face {matched_id} (similarity: {similarity:.2f}, count: {self.faces[matched_id]['detection_count']})")
                return False, matched_id, similarity
            
            else:
                # Wajah BARU - ADD
                embedding_norm = np.linalg.norm(embedding)
                if embedding_norm == 0:
                    print(f"⚠️  Cannot add face {face_id}: zero norm")
                    return False, None, 0
                
                self.faces[face_id] = {
                    'embedding': embedding / embedding_norm,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'detection_count': 1
                }
                
                self._dirty = True
                print(f"✨ NEW FACE added to database: {face_id} (Total: {len(self.faces)})")
                
                # Force save untuk face baru
                if len(self.faces) % 10 == 0:  # Save setiap 10 wajah baru
                    self.save_database()
                
                return True, face_id, 1.0
    
    def get_face_info(self, face_id):
        """Get informasi wajah dari database"""
        with self._lock:
            return self.faces.get(face_id, None)
    
    def remove_old_faces(self, days=30):
        """Hapus wajah yang sudah tidak terdeteksi lebih dari X hari"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        faces_to_remove = []
        
        with self._lock:
            for face_id, face_data in self.faces.items():
                try:
                    last_seen = datetime.fromisoformat(face_data['last_seen'])
                    if last_seen < cutoff_date:
                        faces_to_remove.append(face_id)
                except:
                    continue
            
            for face_id in faces_to_remove:
                del self.faces[face_id]
        
        if faces_to_remove:
            print(f"🗑️  Removed {len(faces_to_remove)} old faces from database")
            self.save_database(force=True)
        
        return len(faces_to_remove)
    
    def reset_database(self):
        """Reset seluruh database"""
        with self._lock:
            self.faces = {}
        self.save_database(force=True)
        print("🔄 Face database reset")
    
    def get_statistics(self):
        """Get statistik database"""
        with self._lock:
            if not self.faces:
                return {
                    'total_faces': 0,
                    'oldest_face': None,
                    'newest_face': None,
                    'most_detected': None
                }
            
            try:
                # Find oldest and newest
                oldest = min(self.faces.items(), key=lambda x: x[1]['first_seen'])
                newest = max(self.faces.items(), key=lambda x: x[1]['first_seen'])
                most_detected = max(self.faces.items(), key=lambda x: x[1]['detection_count'])
                
                return {
                    'total_faces': len(self.faces),
                    'oldest_face': {
                        'id': oldest[0],
                        'first_seen': oldest[1]['first_seen'],
                        'detection_count': oldest[1]['detection_count']
                    },
                    'newest_face': {
                        'id': newest[0],
                        'first_seen': newest[1]['first_seen']
                    },
                    'most_detected': {
                        'id': most_detected[0],
                        'detection_count': most_detected[1]['detection_count']
                    }
                }
            except Exception as e:
                print(f"⚠️  Error getting statistics: {e}")
                return {'total_faces': len(self.faces)}
    
    def shutdown(self):
        """Shutdown database dan save final"""
        self._running = False
        if self._dirty:
            self.save_database(force=True)
        print("💾 Face database shutdown complete")