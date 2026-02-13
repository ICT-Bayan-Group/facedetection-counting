"""
Core modules for Face Counter System
"""
from .config import Config
from .face_counter import OpenVINOFaceCounter, FaceCounter

__all__ = ['Config', 'OpenVINOFaceCounter', 'FaceCounter']