"""
HouseBrain v1.1.0 - AI-powered architectural design system

A comprehensive system for generating residential house designs using AI reasoning,
architectural principles, and building code compliance.
"""

from .schema import (
    HouseInput, HouseOutput, Level, Room, Stair, RoomType, Rectangle, Point2D,
    Door, Window, ArchitecturalStyle, Orientation, ValidationResult
)
from .layout import solve_house_layout, LayoutSolver
from .llm import generate_house_design, HouseBrainLLM

__version__ = "1.1.0"
__author__ = "HouseBrain Team"

__all__ = [
    # Schema classes
    "HouseInput", "HouseOutput", "Level", "Room", "Stair", "RoomType", 
    "Rectangle", "Point2D", "Door", "Window", "ArchitecturalStyle", 
    "Orientation", "ValidationResult",
    
    # Layout functionality
    "solve_house_layout", "LayoutSolver",
    
    # LLM functionality
    "generate_house_design", "HouseBrainLLM",
]