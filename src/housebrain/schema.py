from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum
import math


class RoomType(str, Enum):
    LIVING_ROOM = "living_room"
    DINING_ROOM = "dining_room"
    KITCHEN = "kitchen"
    MASTER_BEDROOM = "master_bedroom"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    HALF_BATH = "half_bath"
    FAMILY_ROOM = "family_room"
    STUDY = "study"
    GARAGE = "garage"
    UTILITY = "utility"
    STORAGE = "storage"
    STAIRWELL = "stairwell"
    CORRIDOR = "corridor"
    ENTRANCE = "entrance"


class ArchitecturalStyle(str, Enum):
    MODERN_CONTEMPORARY = "Modern Contemporary"
    TRADITIONAL = "Traditional"
    COLONIAL = "Colonial"
    MEDITERRANEAN = "Mediterranean"
    MINIMALIST = "Minimalist"
    INDUSTRIAL = "Industrial"
    SCANDINAVIAN = "Scandinavian"
    ASIAN_FUSION = "Asian Fusion"


class Orientation(str, Enum):
    NORTH = "N"
    NORTHEAST = "NE"
    EAST = "E"
    SOUTHEAST = "SE"
    SOUTH = "S"
    SOUTHWEST = "SW"
    WEST = "W"
    NORTHWEST = "NW"


class Point2D(BaseModel):
    x: float = Field(..., description="X coordinate in feet")
    y: float = Field(..., description="Y coordinate in feet")


class Rectangle(BaseModel):
    x: float = Field(..., description="X coordinate of bottom-left corner")
    y: float = Field(..., description="Y coordinate of bottom-left corner")
    width: float = Field(..., description="Width in feet")
    height: float = Field(..., description="Height in feet")

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D(x=self.x + self.width/2, y=self.y + self.height/2)


class Door(BaseModel):
    position: Point2D
    width: float = Field(default=3.0, description="Door width in feet")
    type: Literal["interior", "exterior", "sliding", "pocket"] = "interior"
    room1: str = Field(..., description="First room ID")
    room2: str = Field(..., description="Second room ID")


class Window(BaseModel):
    position: Point2D
    width: float = Field(..., description="Window width in feet")
    height: float = Field(default=4.0, description="Window height in feet")
    type: Literal["fixed", "casement", "sliding", "bay"] = "fixed"
    room_id: str = Field(..., description="Room ID")


class Room(BaseModel):
    id: str = Field(..., description="Unique room identifier")
    type: RoomType
    bounds: Rectangle
    doors: List[Door] = Field(default_factory=list)
    windows: List[Window] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list, description="Special features like fireplace, built-ins")
    
    @validator('bounds')
    def validate_room_size(cls, v, values):
        # Get the room type from the values dict
        room_type = values.get('type')
        if not room_type:
            return v  # Can't validate without room type
            
        min_area = {
            RoomType.LIVING_ROOM: 200,
            RoomType.DINING_ROOM: 120,
            RoomType.KITCHEN: 100,
            RoomType.MASTER_BEDROOM: 180,
            RoomType.BEDROOM: 120,
            RoomType.BATHROOM: 40,
            RoomType.HALF_BATH: 20,
            RoomType.FAMILY_ROOM: 200,
            RoomType.STUDY: 100,
            RoomType.GARAGE: 200,
            RoomType.UTILITY: 50,
            RoomType.STORAGE: 30,
            RoomType.STAIRWELL: 60,
            RoomType.CORRIDOR: 30,
            RoomType.ENTRANCE: 40
        }
        
        if v.area < min_area.get(room_type, 30):
            raise ValueError(f"Room {room_type} too small. Minimum area: {min_area.get(room_type, 30)} sqft")
        return v


class Stair(BaseModel):
    position: Point2D
    width: float = Field(default=3.5, description="Stair width in feet")
    length: float = Field(default=12.0, description="Stair length in feet")
    direction: Literal["up", "down"] = "up"
    type: Literal["straight", "L_shaped", "U_shaped", "spiral"] = "straight"
    floor_from: int = Field(..., description="Starting floor level")
    floor_to: int = Field(..., description="Ending floor level")


class Level(BaseModel):
    level_number: int = Field(..., description="Floor level (0 = ground floor)")
    rooms: List[Room] = Field(default_factory=list)
    stairs: List[Stair] = Field(default_factory=list)
    height: float = Field(default=10.0, description="Floor height in feet")
    
    @validator('rooms')
    def validate_room_overlap(cls, v):
        # Check for room overlaps
        for i, room1 in enumerate(v):
            for j, room2 in enumerate(v[i+1:], i+1):
                if cls._rectangles_overlap(room1.bounds, room2.bounds):
                    raise ValueError(f"Rooms {room1.id} and {room2.id} overlap")
        return v
    
    @staticmethod
    def _rectangles_overlap(r1: Rectangle, r2: Rectangle) -> bool:
        return not (r1.x + r1.width <= r2.x or r2.x + r2.width <= r1.x or
                   r1.y + r1.height <= r2.y or r2.y + r2.height <= r1.y)


class HouseInput(BaseModel):
    basicDetails: Dict[str, Union[int, float, str]] = Field(..., description="Basic house requirements")
    plot: Dict[str, Union[int, float, str, Dict]] = Field(..., description="Plot specifications")
    roomBreakdown: List[Dict[str, Union[str, int, List[str]]]] = Field(default_factory=list, description="Room requirements")
    
    @validator('basicDetails')
    def validate_basic_details(cls, v):
        required_fields = ['totalArea', 'unit', 'floors', 'bedrooms', 'bathrooms', 'style', 'budget']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v


class HouseOutput(BaseModel):
    input: HouseInput
    levels: List[Level] = Field(..., description="Multi-floor house design")
    total_area: float = Field(..., description="Total built area in sqft")
    construction_cost: float = Field(..., description="Estimated construction cost")
    materials: Dict[str, float] = Field(default_factory=dict, description="Material requirements")
    render_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to generated renders")
    
    @validator('levels')
    def validate_levels(cls, v):
        if not v:
            raise ValueError("At least one level must be defined")
        
        # Check level numbers are sequential
        level_numbers = [level.level_number for level in v]
        if level_numbers != list(range(min(level_numbers), max(level_numbers) + 1)):
            raise ValueError("Level numbers must be sequential")
        
        return v
    
    @validator('total_area')
    def validate_total_area(cls, v, values):
        if 'levels' in values:
            calculated_area = sum(sum(room.bounds.area for room in level.rooms) for level in values['levels'])
            if abs(v - calculated_area) > 50:  # Allow 50 sqft tolerance
                raise ValueError(f"Total area mismatch. Calculated: {calculated_area}, Provided: {v}")
        return v


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    compliance_score: float = Field(default=0.0, description="Building code compliance score (0-100)")


# Utility functions for schema validation
def validate_house_design(house: HouseOutput) -> ValidationResult:
    """Comprehensive validation of house design"""
    errors = []
    warnings = []
    compliance_score = 100.0
    
    # Validate each level
    for level in house.levels:
        # Check room sizes
        for room in level.rooms:
            if room.bounds.width < 8 or room.bounds.height < 8:
                errors.append(f"Room {room.id} too small: {room.bounds.width}' x {room.bounds.height}'")
                compliance_score -= 10
            
            # Check for windows in habitable rooms
            if room.type in [RoomType.LIVING_ROOM, RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.STUDY]:
                if not room.windows:
                    warnings.append(f"Habitable room {room.id} has no windows")
                    compliance_score -= 5
        
        # Check stair design
        for stair in level.stairs:
            if stair.width < 3.0:
                errors.append(f"Stair {stair.type} too narrow: {stair.width}'")
                compliance_score -= 15
            
            if stair.length < 10.0:
                warnings.append(f"Stair {stair.type} may be too short: {stair.length}'")
                compliance_score -= 5
    
    # Check multi-floor connectivity
    if len(house.levels) > 1:
        stair_connections = set()
        for level in house.levels:
            for stair in level.stairs:
                stair_connections.add((stair.floor_from, stair.floor_to))
        
        expected_connections = set()
        for i in range(len(house.levels) - 1):
            expected_connections.add((i, i + 1))
        
        if not expected_connections.issubset(stair_connections):
            errors.append("Missing stair connections between floors")
            compliance_score -= 20
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        compliance_score=max(0.0, compliance_score)
    )