from typing import List, Dict, Optional
from .schema import (
    HouseInput, HouseOutput, Level, Room, Stair, RoomType, Rectangle, Point2D, 
    Door, Window
)
import math
from dataclasses import dataclass


@dataclass
class RoomRequirement:
    type: RoomType
    count: int
    min_area: float
    preferred_area: float
    features: List[str]
    adjacency_preferences: List[RoomType]


class LayoutSolver:
    """Solves room layout problems using architectural principles"""
    
    def __init__(self):
        self.room_requirements = {
            RoomType.LIVING_ROOM: RoomRequirement(
                type=RoomType.LIVING_ROOM,
                count=1,
                min_area=200,
                preferred_area=300,
                features=["large_windows", "fireplace"],
                adjacency_preferences=[RoomType.DINING_ROOM, RoomType.ENTRANCE]
            ),
            RoomType.DINING_ROOM: RoomRequirement(
                type=RoomType.DINING_ROOM,
                count=1,
                min_area=120,
                preferred_area=180,
                features=["chandelier"],
                adjacency_preferences=[RoomType.LIVING_ROOM, RoomType.KITCHEN]
            ),
            RoomType.KITCHEN: RoomRequirement(
                type=RoomType.KITCHEN,
                count=1,
                min_area=100,
                preferred_area=150,
                features=["island", "pantry"],
                adjacency_preferences=[RoomType.DINING_ROOM, RoomType.UTILITY]
            ),
            RoomType.MASTER_BEDROOM: RoomRequirement(
                type=RoomType.MASTER_BEDROOM,
                count=1,
                min_area=180,
                preferred_area=250,
                features=["walk_in_closet", "en_suite"],
                adjacency_preferences=[RoomType.BATHROOM]
            ),
            RoomType.BEDROOM: RoomRequirement(
                type=RoomType.BEDROOM,
                count=0,  # Will be set based on input
                min_area=120,
                preferred_area=150,
                features=["closet"],
                adjacency_preferences=[RoomType.BATHROOM]
            ),
            RoomType.BATHROOM: RoomRequirement(
                type=RoomType.BATHROOM,
                count=0,  # Will be set based on input
                min_area=40,
                preferred_area=60,
                features=["shower", "toilet"],
                adjacency_preferences=[RoomType.BEDROOM, RoomType.MASTER_BEDROOM]
            ),
            RoomType.HALF_BATH: RoomRequirement(
                type=RoomType.HALF_BATH,
                count=0,
                min_area=20,
                preferred_area=30,
                features=["toilet"],
                adjacency_preferences=[RoomType.LIVING_ROOM, RoomType.ENTRANCE]
            ),
            RoomType.FAMILY_ROOM: RoomRequirement(
                type=RoomType.FAMILY_ROOM,
                count=0,
                min_area=200,
                preferred_area=250,
                features=["tv_wall", "comfortable_seating"],
                adjacency_preferences=[RoomType.KITCHEN]
            ),
            RoomType.STUDY: RoomRequirement(
                type=RoomType.STUDY,
                count=0,
                min_area=100,
                preferred_area=120,
                features=["built_in_shelves", "desk"],
                adjacency_preferences=[RoomType.LIVING_ROOM]
            ),
            RoomType.GARAGE: RoomRequirement(
                type=RoomType.GARAGE,
                count=0,
                min_area=200,
                preferred_area=300,
                features=["garage_door"],
                adjacency_preferences=[RoomType.ENTRANCE]
            ),
            RoomType.UTILITY: RoomRequirement(
                type=RoomType.UTILITY,
                count=0,
                min_area=50,
                preferred_area=80,
                features=["washer", "dryer"],
                adjacency_preferences=[RoomType.KITCHEN]
            ),
            RoomType.STORAGE: RoomRequirement(
                type=RoomType.STORAGE,
                count=0,
                min_area=30,
                preferred_area=50,
                features=["shelving"],
                adjacency_preferences=[]
            ),
            RoomType.ENTRANCE: RoomRequirement(
                type=RoomType.ENTRANCE,
                count=1,
                min_area=40,
                preferred_area=60,
                features=["coat_closet"],
                adjacency_preferences=[RoomType.LIVING_ROOM]
            )
        }
    
    def solve_layout(self, house_input: HouseInput) -> HouseOutput:
        """Main layout solving function"""
        # Parse input requirements
        plot_width = house_input.plot["width"]
        plot_length = house_input.plot["length"]
        floors = house_input.basicDetails["floors"]
        bedrooms = house_input.basicDetails["bedrooms"]
        bathrooms = house_input.basicDetails["bathrooms"]
        
        # Update room requirements based on input
        self.room_requirements[RoomType.BEDROOM].count = bedrooms - 1  # -1 for master bedroom
        self.room_requirements[RoomType.BATHROOM].count = int(bathrooms)
        self.room_requirements[RoomType.HALF_BATH].count = int((bathrooms % 1) * 2)  # Convert decimal to half baths
        
        # Generate levels
        levels = []
        for floor in range(floors):
            level = self._generate_level(floor, plot_width, plot_length, house_input)
            levels.append(level)
        
        # Add stairs for multi-floor houses
        if floors > 1:
            self._add_stairs(levels)
        
        # Calculate total area and cost
        total_area = sum(sum(room.bounds.area for room in level.rooms) for level in levels)
        construction_cost = self._estimate_construction_cost(total_area, house_input.basicDetails["budget"])
        
        return HouseOutput(
            input=house_input,
            levels=levels,
            total_area=total_area,
            construction_cost=construction_cost,
            materials=self._estimate_materials(total_area),
            render_paths={}
        )
    
    def _generate_level(self, level_number: int, plot_width: float, plot_length: float, house_input: HouseInput) -> Level:
        """Generate a single floor level"""
        rooms = []
        
        # Apply setbacks
        setbacks = house_input.plot.get("setbacks_ft", {"front": 5, "rear": 3, "left": 3, "right": 3})
        buildable_width = plot_width - setbacks["left"] - setbacks["right"]
        buildable_length = plot_length - setbacks["front"] - setbacks["rear"]
        
        # Start position (after setbacks)
        start_x = setbacks["left"]
        start_y = setbacks["rear"]
        
        # Generate room list based on requirements
        room_list = self._create_room_list()
        
        # Use grid-based layout algorithm
        rooms = self._grid_layout(room_list, start_x, start_y, buildable_width, buildable_length, level_number)
        
        # Add doors and windows
        self._add_doors_and_windows(rooms, level_number)
        
        return Level(
            level_number=level_number,
            rooms=rooms,
            stairs=[],
            height=10.0
        )
    
    def _create_room_list(self) -> List[RoomRequirement]:
        """Create list of rooms needed based on requirements"""
        room_list = []
        
        for room_type, requirement in self.room_requirements.items():
            for i in range(requirement.count):
                room_list.append(requirement)
        
        # Sort by priority (larger rooms first, then by adjacency preferences)
        room_list.sort(key=lambda r: (r.preferred_area, len(r.adjacency_preferences)), reverse=True)
        
        return room_list
    
    def _grid_layout(self, room_list: List[RoomRequirement], start_x: float, start_y: float, 
                    width: float, length: float, level_number: int) -> List[Room]:
        """Grid-based room layout algorithm"""
        rooms = []
        current_x = start_x
        current_y = start_y
        max_y = start_y + length
        
        # Calculate room dimensions based on area
        for i, requirement in enumerate(room_list):
            # Calculate room dimensions
            area = requirement.preferred_area
            aspect_ratio = self._get_optimal_aspect_ratio(requirement.type)
            
            room_width = math.sqrt(area * aspect_ratio)
            room_height = area / room_width
            
            # Check if room fits in current row
            if current_x + room_width > start_x + width:
                # Move to next row
                current_x = start_x
                current_y += max(r.bounds.height for r in rooms) if rooms else room_height
            
            # Check if we've exceeded the plot
            if current_y + room_height > max_y:
                # Try to fit in remaining space with smaller dimensions
                remaining_height = max_y - current_y
                if remaining_height > 8:  # Minimum room height
                    room_height = remaining_height
                    room_width = area / room_height
                else:
                    # Skip this room if no space
                    continue
            
            # Create room
            room = Room(
                id=f"{requirement.type.value}_{level_number}_{i}",
                type=requirement.type,
                bounds=Rectangle(
                    x=current_x,
                    y=current_y,
                    width=room_width,
                    height=room_height
                ),
                doors=[],
                windows=[],
                features=requirement.features.copy()
            )
            
            rooms.append(room)
            current_x += room_width
        
        return rooms
    
    def _get_optimal_aspect_ratio(self, room_type: RoomType) -> float:
        """Get optimal aspect ratio for room type"""
        ratios = {
            RoomType.LIVING_ROOM: 1.5,  # Wider than deep
            RoomType.DINING_ROOM: 1.2,
            RoomType.KITCHEN: 1.3,
            RoomType.MASTER_BEDROOM: 1.2,
            RoomType.BEDROOM: 1.1,
            RoomType.BATHROOM: 0.8,  # Deeper than wide
            RoomType.HALF_BATH: 0.7,
            RoomType.FAMILY_ROOM: 1.4,
            RoomType.STUDY: 1.0,  # Square
            RoomType.GARAGE: 1.5,
            RoomType.UTILITY: 1.0,
            RoomType.STORAGE: 0.8,
            RoomType.ENTRANCE: 1.0
        }
        return ratios.get(room_type, 1.0)
    
    def _add_doors_and_windows(self, rooms: List[Room], level_number: int):
        """Add doors and windows to rooms"""
        for room in rooms:
            # Add windows for habitable rooms
            if room.type in [RoomType.LIVING_ROOM, RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.STUDY]:
                self._add_windows_to_room(room)
            
            # Add doors between adjacent rooms
            for other_room in rooms:
                if room.id != other_room.id and self._rooms_adjacent(room, other_room):
                    self._add_door_between_rooms(room, other_room)
    
    def _add_windows_to_room(self, room: Room):
        """Add windows to a room"""
        # Add windows on longer walls
        if room.bounds.width > room.bounds.height:
            # Window on width wall
            window = Window(
                position=Point2D(
                    x=room.bounds.x + room.bounds.width * 0.3,
                    y=room.bounds.y + room.bounds.height
                ),
                width=min(6.0, room.bounds.width * 0.4),
                height=4.0,
                type="fixed",
                room_id=room.id
            )
            room.windows.append(window)
        else:
            # Window on height wall
            window = Window(
                position=Point2D(
                    x=room.bounds.x + room.bounds.width,
                    y=room.bounds.y + room.bounds.height * 0.3
                ),
                width=min(6.0, room.bounds.height * 0.4),
                height=4.0,
                type="fixed",
                room_id=room.id
            )
            room.windows.append(window)
    
    def _rooms_adjacent(self, room1: Room, room2: Room) -> bool:
        """Check if two rooms are adjacent"""
        # Simple adjacency check - rooms share a wall
        r1 = room1.bounds
        r2 = room2.bounds
        
        # Check if rooms share a vertical wall
        if (abs(r1.x + r1.width - r2.x) < 0.1 or abs(r2.x + r2.width - r1.x) < 0.1):
            # Check if they overlap in y direction
            return not (r1.y + r1.height <= r2.y or r2.y + r2.height <= r1.y)
        
        # Check if rooms share a horizontal wall
        if (abs(r1.y + r1.height - r2.y) < 0.1 or abs(r2.y + r2.height - r1.y) < 0.1):
            # Check if they overlap in x direction
            return not (r1.x + r1.width <= r2.x or r2.x + r2.width <= r1.x)
        
        return False
    
    def _add_door_between_rooms(self, room1: Room, room2: Room):
        """Add a door between two adjacent rooms"""
        # Find shared wall
        r1 = room1.bounds
        r2 = room2.bounds
        
        door_position = None
        
        # Vertical wall
        if abs(r1.x + r1.width - r2.x) < 0.1:
            # Room1 right wall meets Room2 left wall
            door_position = Point2D(
                x=r1.x + r1.width,
                y=max(r1.y, r2.y) + min(r1.height, r2.height) * 0.5
            )
        elif abs(r2.x + r2.width - r1.x) < 0.1:
            # Room2 right wall meets Room1 left wall
            door_position = Point2D(
                x=r2.x + r2.width,
                y=max(r1.y, r2.y) + min(r1.height, r2.height) * 0.5
            )
        # Horizontal wall
        elif abs(r1.y + r1.height - r2.y) < 0.1:
            # Room1 top wall meets Room2 bottom wall
            door_position = Point2D(
                x=max(r1.x, r2.x) + min(r1.width, r2.width) * 0.5,
                y=r1.y + r1.height
            )
        elif abs(r2.y + r2.height - r1.y) < 0.1:
            # Room2 top wall meets Room1 bottom wall
            door_position = Point2D(
                x=max(r1.x, r2.x) + min(r1.width, r2.width) * 0.5,
                y=r2.y + r2.height
            )
        
        if door_position:
            door = Door(
                position=door_position,
                width=3.0,
                type="interior",
                room1=room1.id,
                room2=room2.id
            )
            room1.doors.append(door)
            room2.doors.append(door)
    
    def _add_stairs(self, levels: List[Level]):
        """Add stairs between floors"""
        for i in range(len(levels) - 1):
            # Find a good location for stairs (near entrance or in a central location)
            stair_room = self._find_stair_location(levels[i].rooms)
            
            if stair_room:
                stair = Stair(
                    position=Point2D(
                        x=stair_room.bounds.x + stair_room.bounds.width * 0.2,
                        y=stair_room.bounds.y + stair_room.bounds.height * 0.2
                    ),
                    width=3.5,
                    length=12.0,
                    direction="up",
                    type="straight",
                    floor_from=i,
                    floor_to=i + 1
                )
                levels[i].stairs.append(stair)
    
    def _find_stair_location(self, rooms: List[Room]) -> Optional[Room]:
        """Find a good room for stairs"""
        # Prefer entrance or corridor
        for room in rooms:
            if room.type in [RoomType.ENTRANCE, RoomType.CORRIDOR]:
                return room
        
        # Otherwise, use the first room with enough space
        for room in rooms:
            if room.bounds.area >= 60:  # Minimum area for stairs
                return room
        
        return None
    
    def _estimate_construction_cost(self, total_area: float, budget: float) -> float:
        """Estimate construction cost"""
        # Rough cost per sqft (varies by region and quality)
        cost_per_sqft = 150.0  # USD per sqft
        return total_area * cost_per_sqft
    
    def _estimate_materials(self, total_area: float) -> Dict[str, float]:
        """Estimate material requirements"""
        return {
            "concrete_cubic_yards": total_area * 0.1,  # 0.1 cubic yards per sqft
            "steel_tons": total_area * 0.02,  # 0.02 tons per sqft
            "lumber_board_feet": total_area * 8,  # 8 board feet per sqft
            "drywall_sheets": total_area * 0.5,  # 0.5 sheets per sqft
            "roofing_sqft": total_area * 1.2,  # 1.2x for roof area
        }


# Convenience function
def solve_house_layout(house_input: HouseInput) -> HouseOutput:
    """Main function to solve house layout"""
    solver = LayoutSolver()
    return solver.solve_layout(house_input)