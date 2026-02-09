import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class SessionManager:

    def __init__(self, sessions_dir: Path):
        
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions = {}
    
    def create_session(self, document_id: str, user_metadata: Optional[Dict] = None) -> str:
        
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'document_id': document_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'messages': [],
            'metadata': user_metadata or {},
            'context_summary': None
        }
        
        self.active_sessions[session_id] = session_data
        
        self._save_session(session_id, session_data)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
                self.active_sessions[session_id] = session_data
                return session_data
        
        return None
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        
        session = self.get_session(session_id)
        if not session:
            return False
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'citations': citations or [],
            'metadata': metadata or {}
        }
        
        session['messages'].append(message)
        session['updated_at'] = datetime.now().isoformat()
        
        self.active_sessions[session_id] = session
        self._save_session(session_id, session)
        
        return True
    
    def get_conversation_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        include_citations: bool = True
    ) -> List[Dict]:
        
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session['messages']
        
        if max_messages:
            messages = messages[-max_messages:]
        
        if not include_citations:
            messages = [
                {k: v for k, v in msg.items() if k != 'citations'}
                for msg in messages
            ]
        
        return messages
    
    def get_context_for_query(
        self,
        session_id: str,
        max_history: int = 5
    ) -> str:
        
        messages = self.get_conversation_history(
            session_id,
            max_messages=max_history * 2,
            include_citations=False
        )
        
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def update_context_summary(self, session_id: str, summary: str) -> bool:
        
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['context_summary'] = summary
        session['updated_at'] = datetime.now().isoformat()
        
        self.active_sessions[session_id] = session
        self._save_session(session_id, session)
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
       
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
    
    def list_sessions(self, document_id: Optional[str] = None) -> List[Dict]:
    
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            with open(session_file, 'r', encoding='utf-8') as f:
                session = json.load(f)
                
                if document_id and session.get('document_id') != document_id:
                    continue
                
                sessions.append({
                    'session_id': session['session_id'],
                    'document_id': session['document_id'],
                    'created_at': session['created_at'],
                    'updated_at': session['updated_at'],
                    'message_count': len(session['messages'])
                })
        
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)
        
        return sessions
    
    def _save_session(self, session_id: str, session_data: Dict):
        
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for session_file in self.sessions_dir.glob("*.json"):
            with open(session_file, 'r', encoding='utf-8') as f:
                session = json.load(f)
                updated_at = datetime.fromisoformat(session['updated_at'])
                
                if updated_at < cutoff:
                    session_file.unlink()
                    session_id = session['session_id']
                    if session_id in self.active_sessions:
                        del self.active_sessions[session_id]
                    deleted += 1
        
        return deleted
