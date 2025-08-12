from typing import List, Dict, Tuple, Optional
from .schema import HouseOutput, Level, Room, Stair, RoomType, Rectangle, Point2D, ValidationResult
import math


class HouseValidator:
    """Comprehensive validator for house designs"""
    
    def __init__(self):
        self.min_room_sizes = {
            RoomType.LIVING_ROOM: (16, 12),  # width, height in feet
            RoomType.DINING_ROOM: (12, 10),
            RoomType.KITCHEN: (12, 8),
            RoomType.MASTER_BEDROOM: (14, 12),
            RoomType.BEDROOM: (12, 10),
            RoomType.BATHROOM: (8, 6),
            RoomType.HALF_BATH: (6, 4),
            RoomType.FAMILY_ROOM: (16, 12),
            RoomType.STUDY: (10, 8),
            RoomType.GARAGE: (16, 12),
            RoomType.UTILITY: (8, 6),
            RoomType.STORAGE: (6, 5),
            RoomType.STAIRWELL: (8, 8),
            RoomType.CORRIDOR: (4, 8),
            RoomType.ENTRANCE: (8, 6)
        }
        
        self.min_corridor_width = 3.0  # feet
        self.min_stair_width = 3.0  # feet
        self.min_headroom = 6.5  # feet
        self.min_window_area_ratio = 0.1  # 10% of room area for habitable rooms
    
    def validate_house(self, house: HouseOutput) -> ValidationResult:
        """Main validation function"""
        errors = []
        warnings = []
        compliance_score = 100.0
        
        # Validate each level
        for level in house.levels:
            level_errors, level_warnings, level_score = self._validate_level(level)
            errors.extend(level_errors)
            warnings.extend(level_warnings)
            compliance_score = min(compliance_score, level_score)
        
        # Validate multi-floor connectivity
        if len(house.levels) > 1:
            connectivity_errors, connectivity_warnings, connectivity_score = self._validate_floor_connectivity(house.levels)
            errors.extend(connectivity_errors)
            warnings.extend(connectivity_warnings)
            compliance_score = min(compliance_score, connectivity_score)
        
        # Validate room adjacency and circulation
        adjacency_errors, adjacency_warnings, adjacency_score = self._validate_room_adjacency(house.levels)
        errors.extend(adjacency_errors)
        warnings.extend(adjacency_warnings)
        compliance_score = min(compliance_score, adjacency_score)
        
        # Validate daylight and ventilation
        daylight_errors, daylight_warnings, daylight_score = self._validate_daylight_ventilation(house.levels)
        errors.extend(daylight_errors)
        warnings.extend(daylight_warnings)
        compliance_score = min(compliance_score, daylight_score)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            compliance_score=max(0.0, compliance_score)
        )
    
    def _validate_level(self, level: Level) -> Tuple[List[str], List[str], float]:
        """Validate a single floor level"""
        errors = []
        warnings = []
        score = 100.0
        
        # Check room sizes
        for room in level.rooms:
            min_width, min_height = self.min_room_sizes.get(room.type, (8, 8))
            
            if room.bounds.width < min_width:
                errors.append(f"Room {room.id} ({room.type}) too narrow: {room.bounds.width}' < {min_width}'")
                score -= 10
            
            if room.bounds.height < min_height:
                errors.append(f"Room {room.id} ({room.type}) too short: {room.bounds.height}' < {min_height}'")
                score -= 10
            
            # Check room proportions (avoid very long narrow rooms)
            aspect_ratio = max(room.bounds.width, room.bounds.height) / min(room.bounds.width, room.bounds.height)
            if aspect_ratio > 3.0:
                warnings.append(f"Room {room.id} has poor proportions (ratio: {aspect_ratio:.1f})")
                score -= 5
        
        # Check stair design
        for stair in level.stairs:
            if stair.width < self.min_stair_width:
                errors.append(f"Stair too narrow: {stair.width}' < {self.min_stair_width}'")
                score -= 15
            
            # Check stair proportions
            if stair.length < stair.width * 2:
                warnings.append(f"Stair may be too short for comfortable use")
                score -= 5
        
        return errors, warnings, score
    
    def _validate_floor_connectivity(self, levels: List[Level]) -> Tuple[List[str], List[str], float]:
        """Validate stair connections between floors"""
        errors = []
        warnings = []
        score = 100.0
        
        stair_connections = set()
        for level in levels:
            for stair in level.stairs:
                stair_connections.add((stair.floor_from, stair.floor_to))
        
        # Check for missing connections between adjacent floors
        for i in range(len(levels) - 1):
            if (i, i + 1) not in stair_connections and (i + 1, i) not in stair_connections:
                errors.append(f"Missing stair connection between floors {i} and {i + 1}")
                score -= 20
        
        # Check for isolated floors
        connected_floors = set()
        for from_floor, to_floor in stair_connections:
            connected_floors.add(from_floor)
            connected_floors.add(to_floor)
        
        all_floors = set(range(len(levels)))
        isolated_floors = all_floors - connected_floors
        
        if isolated_floors:
            errors.append(f"Isolated floors: {isolated_floors}")
            score -= 25
        
        return errors, warnings, score
    
    def _validate_room_adjacency(self, levels: List[Level]) -> Tuple[List[str], List[str], float]:
        """Validate room adjacency and circulation patterns"""
        errors = []
        warnings = []
        score = 100.0
        
        for level in levels:
            # Check for room overlaps
            for i, room1 in enumerate(level.rooms):
                for j, room2 in enumerate(level.rooms[i+1:], i+1):
                    if self._rectangles_overlap(room1.bounds, room2.bounds):
                        errors.append(f"Rooms {room1.id} and {room2.id} overlap")
                        score -= 20
            
            # Check circulation (ensure rooms are accessible)
            accessible_rooms = self._find_accessible_rooms(level.rooms)
            inaccessible_rooms = [room.id for room in level.rooms if room.id not in accessible_rooms]
            
            if inaccessible_rooms:
                errors.append(f"Inaccessible rooms on level {level.level_number}: {inaccessible_rooms}")
                score -= 15
            
            # Check for proper room adjacencies
            adjacency_errors = self._check_room_adjacencies(level.rooms)
            errors.extend(adjacency_errors)
            if adjacency_errors:
                score -= len(adjacency_errors) * 5
        
        return errors, warnings, score
    
    def _validate_daylight_ventilation(self, levels: List[Level]) -> Tuple[List[str], List[str], float]:
        """Validate daylight and ventilation requirements"""
        errors = []
        warnings = []
        score = 100.0
        
        habitable_rooms = [
            RoomType.LIVING_ROOM, RoomType.DINING_ROOM, RoomType.MASTER_BEDROOM,
            RoomType.BEDROOM, RoomType.FAMILY_ROOM, RoomType.STUDY
        ]
        
        for level in levels:
            for room in level.rooms:
                if room.type in habitable_rooms:
                    # Check for windows
                    if not room.windows:
                        errors.append(f"Habitable room {room.id} has no windows")
                        score -= 10
                    else:
                        # Check window area ratio
                        window_area = sum(w.width * w.height for w in room.windows)
                        room_area = room.bounds.area
                        window_ratio = window_area / room_area
                        
                        if window_ratio < self.min_window_area_ratio:
                            warnings.append(f"Room {room.id} may have insufficient daylight (ratio: {window_ratio:.2f})")
                            score -= 5
        
        return errors, warnings, score
    
    def _rectangles_overlap(self, r1: Rectangle, r2: Rectangle) -> bool:
        """Check if two rectangles overlap"""
        return not (r1.x + r1.width <= r2.x or r2.x + r2.width <= r1.x or
                   r1.y + r1.height <= r2.y or r2.y + r2.height <= r1.y)
    
    def _find_accessible_rooms(self, rooms: List[Room]) -> List[str]:
        """Find all rooms accessible from the entrance"""
        if not rooms:
            return []
        
        # Find entrance room
        entrance_room = None
        for room in rooms:
            if room.type == RoomType.ENTRANCE:
                entrance_room = room
                break
        
        if not entrance_room:
            # If no entrance room, assume first room is accessible
            accessible = {rooms[0].id}
        else:
            accessible = {entrance_room.id}
        
        # Simple flood fill to find connected rooms
        changed = True
        while changed:
            changed = False
            for room in rooms:
                if room.id in accessible:
                    continue
                
                # Check if room is connected to any accessible room
                for door in room.doors:
                    if door.room1 in accessible or door.room2 in accessible:
                        accessible.add(room.id)
                        changed = True
                        break
        
        return list(accessible)
    
    def _check_room_adjacencies(self, rooms: List[Room]) -> List[str]:
        """Check for proper room adjacencies"""
        errors = []
        
        # Kitchen should be near dining room
        kitchen = next((r for r in rooms if r.type == RoomType.KITCHEN), None)
        dining = next((r for r in rooms if r.type == RoomType.DINING_ROOM), None)
        
        if kitchen and dining:
            distance = self._room_distance(kitchen, dining)
            if distance > 20:  # More than 20 feet apart
                errors.append("Kitchen and dining room should be closer together")
        
        # Bathrooms should be near bedrooms
        bathrooms = [r for r in rooms if r.type in [RoomType.BATHROOM, RoomType.HALF_BATH]]
        bedrooms = [r for r in rooms if r.type in [RoomType.BEDROOM, RoomType.MASTER_BEDROOM]]
        
        for bedroom in bedrooms:
            nearby_bathroom = False
            for bathroom in bathrooms:
                if self._room_distance(bedroom, bathroom) < 15:
                    nearby_bathroom = True
                    break
            
            if not nearby_bathroom:
                errors.append(f"Bedroom {bedroom.id} should be near a bathroom")
        
        return errors
    
    def _room_distance(self, room1: Room, room2: Room) -> float:
        """Calculate distance between room centers"""
        center1 = room1.bounds.center
        center2 = room2.bounds.center
        return math.sqrt((center1.x - center2.x)**2 + (center1.y - center2.y)**2)


# Convenience function for easy validation
def validate_house_design(house: HouseOutput) -> ValidationResult:
    """Main validation function for house designs"""
    validator = HouseValidator()
    return validator.validate_house(house)