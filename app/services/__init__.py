"""Service module initialization"""
from app.services.gait_processor import GaitProcessor, serialize_embedding, deserialize_embedding

__all__ = ['GaitProcessor', 'serialize_embedding', 'deserialize_embedding']
