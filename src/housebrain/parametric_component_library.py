"""
Parametric Component Library for HouseBrain Professional

This module provides detailed parametric components:
- Advanced door types with hardware
- Comprehensive window systems
- Detailed fixtures and equipment
- Structural elements
- MEP components
"""

from __future__ import annotations

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ComponentDimensions:
    """Standard component dimensions"""
    width: float
    height: float
    depth: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"width": self.width, "height": self.height, "depth": self.depth}


@dataclass
class ComponentMaterials:
    """Component material specifications"""
    primary: str
    secondary: str = ""
    hardware: str = ""
    glazing: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if v}


class ParametricComponentLibrary:
    """Comprehensive parametric component library"""
    
    def __init__(self):
        self.door_types = self._initialize_door_types()
        self.window_types = self._initialize_window_types()
        self.fixture_types = self._initialize_fixture_types()
        self.furniture_types = self._initialize_furniture_types()
        self.equipment_types = self._initialize_equipment_types()
        self.structural_components = self._initialize_structural_components()
        self.mep_components = self._initialize_mep_components()
        self.accessibility_components = self._initialize_accessibility_components()
        
    def get_component(self, component_type: str, component_name: str, **kwargs) -> Dict[str, Any]:
        """Get parametric component with customization options"""
        
        component_library = getattr(self, f"{component_type}_types", {})
        base_component = component_library.get(component_name)
        
        if not base_component:
            return self._get_default_component(component_type)
        
        # Apply customizations
        customized_component = self._customize_component(base_component, **kwargs)
        
        return customized_component
    
    def get_components_by_type(self, component_type: str) -> Dict[str, Dict]:
        """Get all components of a specific type"""
        
        return getattr(self, f"{component_type}_types", {})
    
    def generate_component_3d_geometry(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D geometry for component"""
        
        component_type = component.get("type", "unknown")
        
        if component_type == "door":
            return self._generate_door_geometry(component)
        elif component_type == "window":
            return self._generate_window_geometry(component)
        elif component_type == "fixture":
            return self._generate_fixture_geometry(component)
        else:
            return self._generate_generic_geometry(component)
    
    def _initialize_door_types(self) -> Dict[str, Dict]:
        """Initialize comprehensive door types"""
        
        return {
            "hinged_single": {
                "name": "Single Hinged Door",
                "type": "door",
                "operation": "hinged",
                "dimensions": ComponentDimensions(900, 2100, 40).to_dict(),
                "materials": ComponentMaterials("timber_oak", hardware="steel_brushed").to_dict(),
                "hardware": {
                    "hinges": {"count": 3, "type": "ball_bearing", "material": "stainless_steel"},
                    "handle": {"type": "lever", "material": "stainless_steel", "height": 1050},
                    "lock": {"type": "cylinder", "security_grade": "grade_2"},
                    "closer": {"required": False, "type": "surface_mounted"}
                },
                "performance": {
                    "fire_rating": "30_minutes",
                    "acoustic_rating": "Rw35",
                    "thermal_rating": "U2.0",
                    "security_rating": "grade_2"
                },
                "variations": {
                    "width": [600, 700, 800, 900, 1000],
                    "height": [2000, 2100, 2400],
                    "materials": ["timber_oak", "timber_pine", "steel_painted", "aluminum"]
                }
            },
            
            "hinged_double": {
                "name": "Double Hinged Door",
                "type": "door",
                "operation": "hinged_double",
                "dimensions": ComponentDimensions(1800, 2100, 40).to_dict(),
                "materials": ComponentMaterials("timber_oak", hardware="steel_brushed").to_dict(),
                "hardware": {
                    "hinges": {"count": 6, "type": "ball_bearing", "material": "stainless_steel"},
                    "handles": {"count": 2, "type": "lever", "material": "stainless_steel"},
                    "locks": {"type": "multipoint", "security_grade": "grade_3"},
                    "coordinator": {"type": "automatic", "fire_rated": True}
                },
                "performance": {
                    "fire_rating": "60_minutes",
                    "acoustic_rating": "Rw40",
                    "thermal_rating": "U1.8",
                    "security_rating": "grade_3"
                }
            },
            
            "sliding_single": {
                "name": "Single Sliding Door",
                "type": "door",
                "operation": "sliding",
                "dimensions": ComponentDimensions(900, 2100, 40).to_dict(),
                "materials": ComponentMaterials("timber_oak", hardware="steel_brushed").to_dict(),
                "hardware": {
                    "track": {"type": "heavy_duty", "material": "stainless_steel", "load_capacity": 150},
                    "rollers": {"count": 4, "type": "ball_bearing", "material": "steel"},
                    "handle": {"type": "recessed", "material": "stainless_steel"},
                    "lock": {"type": "hook_bolt", "security_grade": "grade_2"}
                },
                "performance": {
                    "smooth_operation": True,
                    "acoustic_rating": "Rw32",
                    "thermal_rating": "U2.2"
                }
            },
            
            "sliding_stacking": {
                "name": "Sliding Stacking Door",
                "type": "door",
                "operation": "sliding_stacking",
                "dimensions": ComponentDimensions(4000, 2400, 40).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_low_e").to_dict(),
                "hardware": {
                    "track": {"type": "architectural", "material": "aluminum", "panels": 4},
                    "rollers": {"count": 16, "type": "precision_bearing", "material": "steel"},
                    "handles": {"count": 4, "type": "integrated", "material": "aluminum"},
                    "seals": {"type": "compression", "material": "epdm_rubber"}
                },
                "performance": {
                    "weather_sealing": "excellent",
                    "acoustic_rating": "Rw38",
                    "thermal_rating": "U1.6",
                    "wind_load": "high"
                }
            },
            
            "pivot": {
                "name": "Pivot Door",
                "type": "door",
                "operation": "pivot",
                "dimensions": ComponentDimensions(1200, 2700, 50).to_dict(),
                "materials": ComponentMaterials("steel_painted", glazing="glass_clear").to_dict(),
                "hardware": {
                    "pivot": {"type": "floor_ceiling", "material": "stainless_steel", "load_capacity": 200},
                    "handle": {"type": "continuous", "material": "stainless_steel", "length": 1500},
                    "lock": {"type": "magnetic", "security_grade": "grade_2"},
                    "damper": {"type": "hydraulic", "soft_close": True}
                },
                "performance": {
                    "architectural_feature": True,
                    "thermal_rating": "U1.4",
                    "security_rating": "grade_2"
                }
            },
            
            "folding_bifold": {
                "name": "Bi-fold Door",
                "type": "door", 
                "operation": "folding",
                "dimensions": ComponentDimensions(2400, 2100, 35).to_dict(),
                "materials": ComponentMaterials("timber_oak", glazing="glass_clear").to_dict(),
                "hardware": {
                    "hinges": {"count": 6, "type": "continuous", "material": "stainless_steel"},
                    "track": {"type": "top_hung", "material": "aluminum"},
                    "handles": {"count": 2, "type": "flush_pull", "material": "stainless_steel"},
                    "lock": {"type": "flush_bolt", "security_grade": "grade_1"}
                },
                "performance": {
                    "space_saving": True,
                    "acoustic_rating": "Rw30",
                    "thermal_rating": "U2.4"
                }
            },
            
            "revolving": {
                "name": "Revolving Door",
                "type": "door",
                "operation": "revolving",
                "dimensions": ComponentDimensions(2200, 2100, 2200).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_clear").to_dict(),
                "hardware": {
                    "mechanism": {"type": "manual_push", "material": "aluminum"},
                    "bearings": {"type": "precision", "material": "steel"},
                    "wings": {"count": 4, "material": "aluminum_glazed"},
                    "safety": {"sensors": True, "breakout": True}
                },
                "performance": {
                    "energy_efficiency": "excellent",
                    "traffic_capacity": "high",
                    "weather_sealing": "excellent"
                }
            },
            
            "automatic_sliding": {
                "name": "Automatic Sliding Door",
                "type": "door",
                "operation": "automatic_sliding",
                "dimensions": ComponentDimensions(1500, 2100, 40).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_clear").to_dict(),
                "hardware": {
                    "operator": {"type": "electric", "power": "24V_DC", "backup_battery": True},
                    "sensors": {"type": "microwave_infrared", "count": 2},
                    "track": {"type": "overhead", "material": "aluminum"},
                    "safety": {"light_curtain": True, "emergency_stop": True}
                },
                "performance": {
                    "accessibility": "ADA_compliant",
                    "opening_speed": "adjustable",
                    "hold_open_time": "adjustable"
                }
            }
        }
    
    def _initialize_window_types(self) -> Dict[str, Dict]:
        """Initialize comprehensive window types"""
        
        return {
            "casement_single": {
                "name": "Single Casement Window",
                "type": "window",
                "operation": "casement",
                "dimensions": ComponentDimensions(600, 1200, 100).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_low_e").to_dict(),
                "hardware": {
                    "hinges": {"count": 2, "type": "friction", "material": "stainless_steel"},
                    "handle": {"type": "espagnolette", "material": "aluminum", "locking_points": 3},
                    "stay": {"type": "friction", "material": "stainless_steel", "positions": 5},
                    "weatherstripping": {"type": "compression", "material": "epdm_rubber"}
                },
                "glazing": {
                    "type": "double_glazed",
                    "thickness": "6-12-6",
                    "gas_fill": "argon",
                    "coating": "low_e",
                    "spacer": "warm_edge"
                },
                "performance": {
                    "thermal_rating": "U1.4",
                    "acoustic_rating": "Rw32",
                    "air_infiltration": "A3",
                    "water_penetration": "9A",
                    "wind_load": "high"
                }
            },
            
            "casement_double": {
                "name": "Double Casement Window",
                "type": "window",
                "operation": "casement_double",
                "dimensions": ComponentDimensions(1200, 1200, 100).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_low_e").to_dict(),
                "hardware": {
                    "hinges": {"count": 4, "type": "friction", "material": "stainless_steel"},
                    "handles": {"count": 2, "type": "espagnolette", "material": "aluminum"},
                    "astragal": {"type": "removable", "material": "aluminum"},
                    "weatherstripping": {"type": "compression", "material": "epdm_rubber"}
                },
                "performance": {
                    "thermal_rating": "U1.2",
                    "acoustic_rating": "Rw35",
                    "ventilation": "excellent"
                }
            },
            
            "sliding_horizontal": {
                "name": "Horizontal Sliding Window",
                "type": "window",
                "operation": "sliding_horizontal",
                "dimensions": ComponentDimensions(1800, 1200, 80).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_low_e").to_dict(),
                "hardware": {
                    "track": {"type": "multi_chambered", "material": "aluminum"},
                    "rollers": {"count": 4, "type": "nylon", "material": "engineered_plastic"},
                    "locks": {"count": 2, "type": "cam", "material": "aluminum"},
                    "weatherstripping": {"type": "pile", "material": "polypropylene"}
                },
                "performance": {
                    "thermal_rating": "U1.8",
                    "acoustic_rating": "Rw30",
                    "ease_of_operation": "excellent"
                }
            },
            
            "sliding_vertical": {
                "name": "Single Hung Window",
                "type": "window",
                "operation": "sliding_vertical",
                "dimensions": ComponentDimensions(900, 1800, 100).to_dict(),
                "materials": ComponentMaterials("timber_pine", glazing="glass_low_e").to_dict(),
                "hardware": {
                    "sash_chain": {"type": "stainless_steel", "length": 3600},
                    "counterweight": {"type": "cast_iron", "weight": 15},
                    "pulleys": {"count": 2, "type": "bronze", "material": "bronze"},
                    "locks": {"count": 1, "type": "cam", "material": "brass"}
                },
                "performance": {
                    "traditional_appearance": True,
                    "thermal_rating": "U2.0",
                    "acoustic_rating": "Rw28"
                }
            },
            
            "awning": {
                "name": "Awning Window",
                "type": "window",
                "operation": "awning",
                "dimensions": ComponentDimensions(1200, 600, 100).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_low_e").to_dict(),
                "hardware": {
                    "hinges": {"count": 2, "type": "continuous", "material": "stainless_steel"},
                    "operator": {"type": "crank", "material": "aluminum", "gear_ratio": "6:1"},
                    "arms": {"count": 2, "type": "friction", "material": "stainless_steel"},
                    "weatherstripping": {"type": "compression", "material": "epdm_rubber"}
                },
                "performance": {
                    "rain_protection": "excellent",
                    "ventilation": "good",
                    "thermal_rating": "U1.6"
                }
            },
            
            "hopper": {
                "name": "Hopper Window",
                "type": "window",
                "operation": "hopper",
                "dimensions": ComponentDimensions(800, 400, 100).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_frosted").to_dict(),
                "hardware": {
                    "hinges": {"count": 2, "type": "continuous", "material": "stainless_steel"},
                    "operator": {"type": "push_out", "material": "aluminum"},
                    "safety_chain": {"type": "stainless_steel", "length": 300},
                    "screen": {"type": "removable", "material": "aluminum_mesh"}
                },
                "performance": {
                    "privacy": "good",
                    "ventilation": "controlled",
                    "security": "enhanced"
                }
            },
            
            "fixed": {
                "name": "Fixed Window",
                "type": "window",
                "operation": "fixed",
                "dimensions": ComponentDimensions(1500, 1000, 60).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_clear").to_dict(),
                "hardware": {
                    "glazing_beads": {"type": "structural", "material": "aluminum"},
                    "sealant": {"type": "structural", "material": "silicone"},
                    "gaskets": {"type": "epdm", "material": "epdm_rubber"}
                },
                "glazing": {
                    "type": "triple_glazed",
                    "thickness": "6-12-6-12-6",
                    "gas_fill": "krypton",
                    "coating": "triple_low_e",
                    "spacer": "super_spacer"
                },
                "performance": {
                    "thermal_rating": "U0.8",
                    "acoustic_rating": "Rw40",
                    "structural_glazing": True
                }
            },
            
            "bay": {
                "name": "Bay Window",
                "type": "window",
                "operation": "bay",
                "dimensions": ComponentDimensions(2400, 1500, 120).to_dict(),
                "materials": ComponentMaterials("timber_oak", glazing="glass_low_e").to_dict(),
                "hardware": {
                    "center_window": {"type": "fixed", "width": 1200},
                    "side_windows": {"type": "casement", "width": 600, "count": 2},
                    "corner_posts": {"material": "timber_oak", "size": "100x100"},
                    "roof": {"type": "hipped", "material": "metal_standing_seam"}
                },
                "performance": {
                    "architectural_feature": True,
                    "interior_space": "increased",
                    "natural_light": "maximum"
                }
            },
            
            "skylight_fixed": {
                "name": "Fixed Skylight",
                "type": "window",
                "operation": "fixed_skylight",
                "dimensions": ComponentDimensions(1200, 1200, 200).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_laminated").to_dict(),
                "hardware": {
                    "frame": {"type": "thermally_broken", "material": "aluminum"},
                    "flashing": {"type": "integrated", "material": "aluminum"},
                    "condensation_gutter": {"type": "built_in", "material": "aluminum"},
                    "shade": {"type": "motorized", "material": "fabric_blackout"}
                },
                "performance": {
                    "waterproofing": "guaranteed",
                    "thermal_rating": "U1.0",
                    "impact_resistance": "high",
                    "natural_light": "zenith"
                }
            },
            
            "skylight_venting": {
                "name": "Venting Skylight",
                "type": "window",
                "operation": "venting_skylight",
                "dimensions": ComponentDimensions(900, 1400, 200).to_dict(),
                "materials": ComponentMaterials("aluminum_anodized", glazing="glass_laminated").to_dict(),
                "hardware": {
                    "operator": {"type": "electric", "power": "24V_DC", "rain_sensor": True},
                    "hinges": {"type": "heavy_duty", "material": "stainless_steel"},
                    "weatherstripping": {"type": "compression", "material": "epdm_rubber"},
                    "remote_control": {"type": "wireless", "battery_backup": True}
                },
                "performance": {
                    "natural_ventilation": "stack_effect",
                    "automatic_operation": True,
                    "weather_protection": "automatic_closing"
                }
            }
        }
    
    def _initialize_fixture_types(self) -> Dict[str, Dict]:
        """Initialize bathroom and kitchen fixtures"""
        
        return {
            # Bathroom Fixtures
            "toilet_wall_hung": {
                "name": "Wall Hung Toilet",
                "type": "fixture",
                "category": "bathroom",
                "dimensions": ComponentDimensions(350, 350, 600).to_dict(),
                "materials": ComponentMaterials("ceramic_white").to_dict(),
                "features": {
                    "mounting": "concealed_carrier",
                    "flush_type": "dual_flush",
                    "water_efficiency": "4.5_3L",
                    "soft_close_seat": True,
                    "rimless_design": True
                },
                "installation": {
                    "carrier_frame": "steel_galvanized",
                    "wall_thickness": 100,
                    "access_panel": "required"
                }
            },
            
            "toilet_comfort_height": {
                "name": "Comfort Height Toilet",
                "type": "fixture",
                "category": "bathroom",
                "dimensions": ComponentDimensions(400, 650, 800).to_dict(),
                "materials": ComponentMaterials("ceramic_white").to_dict(),
                "features": {
                    "ada_compliant": True,
                    "seat_height": 480,
                    "flush_type": "dual_flush",
                    "water_efficiency": "4.5_3L",
                    "elongated_bowl": True
                }
            },
            
            "vanity_double": {
                "name": "Double Vanity",
                "type": "fixture",
                "category": "bathroom",
                "dimensions": ComponentDimensions(1500, 600, 850).to_dict(),
                "materials": ComponentMaterials("timber_oak", secondary="stone_quartz").to_dict(),
                "features": {
                    "basin_count": 2,
                    "basin_type": "undermount",
                    "storage": "soft_close_drawers",
                    "mirror": "integrated_lighting",
                    "electrical": "GFCI_outlets"
                }
            },
            
            "shower_walk_in": {
                "name": "Walk-in Shower",
                "type": "fixture",
                "category": "bathroom",
                "dimensions": ComponentDimensions(1200, 900, 2000).to_dict(),
                "materials": ComponentMaterials("tile_porcelain", secondary="glass_clear").to_dict(),
                "features": {
                    "barrier_free": True,
                    "drainage": "linear_drain",
                    "shower_head": "rain_and_handheld",
                    "controls": "thermostatic",
                    "niche": "built_in"
                }
            },
            
            "bathtub_freestanding": {
                "name": "Freestanding Bathtub",
                "type": "fixture",
                "category": "bathroom",
                "dimensions": ComponentDimensions(1700, 800, 600).to_dict(),
                "materials": ComponentMaterials("acrylic_white").to_dict(),
                "features": {
                    "style": "contemporary",
                    "capacity": 280,  # liters
                    "overflow": "integrated",
                    "faucet": "floor_mounted",
                    "insulation": "foam_backing"
                }
            },
            
            # Kitchen Fixtures
            "kitchen_island": {
                "name": "Kitchen Island",
                "type": "fixture",
                "category": "kitchen",
                "dimensions": ComponentDimensions(2400, 1000, 900).to_dict(),
                "materials": ComponentMaterials("timber_oak", secondary="stone_quartz").to_dict(),
                "features": {
                    "seating": "breakfast_bar",
                    "storage": "deep_drawers",
                    "electrical": "pop_up_outlets",
                    "plumbing": "prep_sink",
                    "ventilation": "downdraft_ready"
                }
            },
            
            "range_professional": {
                "name": "Professional Range",
                "type": "fixture",
                "category": "kitchen",
                "dimensions": ComponentDimensions(900, 600, 900).to_dict(),
                "materials": ComponentMaterials("steel_stainless").to_dict(),
                "features": {
                    "fuel_type": "gas",
                    "burners": 6,
                    "oven_count": 2,
                    "convection": True,
                    "self_cleaning": True,
                    "commercial_grade": True
                }
            },
            
            "refrigerator_integrated": {
                "name": "Integrated Refrigerator",
                "type": "fixture", 
                "category": "kitchen",
                "dimensions": ComponentDimensions(600, 600, 2100).to_dict(),
                "materials": ComponentMaterials("steel_stainless").to_dict(),
                "features": {
                    "style": "panel_ready",
                    "capacity": 500,  # liters
                    "configuration": "french_door",
                    "energy_rating": "5_star",
                    "smart_features": True
                }
            }
        }
    
    def _initialize_furniture_types(self) -> Dict[str, Dict]:
        """Initialize furniture components"""
        
        return {
            "built_in_wardrobe": {
                "name": "Built-in Wardrobe",
                "type": "furniture",
                "category": "bedroom",
                "dimensions": ComponentDimensions(2400, 600, 2400).to_dict(),
                "materials": ComponentMaterials("timber_oak").to_dict(),
                "features": {
                    "doors": "sliding_mirror",
                    "interior": "adjustable_shelving",
                    "lighting": "led_strip",
                    "accessories": "pull_out_shoe_rack"
                }
            },
            
            "kitchen_pantry": {
                "name": "Kitchen Pantry",
                "type": "furniture",
                "category": "kitchen",
                "dimensions": ComponentDimensions(900, 600, 2400).to_dict(),
                "materials": ComponentMaterials("timber_oak").to_dict(),
                "features": {
                    "shelving": "adjustable",
                    "drawers": "soft_close",
                    "lighting": "motion_sensor",
                    "ventilation": "passive"
                }
            },
            
            "study_desk_built_in": {
                "name": "Built-in Study Desk",
                "type": "furniture",
                "category": "office",
                "dimensions": ComponentDimensions(1800, 700, 750).to_dict(),
                "materials": ComponentMaterials("timber_oak").to_dict(),
                "features": {
                    "surface": "leather_inlay",
                    "storage": "integrated_drawers",
                    "cable_management": "built_in",
                    "lighting": "task_lighting"
                }
            }
        }
    
    def _initialize_equipment_types(self) -> Dict[str, Dict]:
        """Initialize mechanical equipment"""
        
        return {
            "hvac_split_system": {
                "name": "Split System Air Conditioner",
                "type": "equipment",
                "category": "hvac",
                "dimensions": ComponentDimensions(800, 300, 550).to_dict(),
                "performance": {
                    "cooling_capacity": 5.5,  # kW
                    "heating_capacity": 6.0,  # kW
                    "energy_rating": "5_star",
                    "refrigerant": "R32",
                    "noise_level": 19  # dBA
                },
                "features": {
                    "inverter": True,
                    "wifi_control": True,
                    "air_purification": True,
                    "self_cleaning": True
                }
            },
            
            "water_heater_heat_pump": {
                "name": "Heat Pump Water Heater",
                "type": "equipment",
                "category": "plumbing",
                "dimensions": ComponentDimensions(600, 600, 1800).to_dict(),
                "performance": {
                    "capacity": 270,  # liters
                    "efficiency": "COP_4.2",
                    "recovery_time": 2.5,  # hours
                    "energy_rating": "5_star"
                },
                "features": {
                    "smart_controller": True,
                    "frost_protection": True,
                    "quiet_operation": True,
                    "backup_element": True
                }
            },
            
            "solar_inverter": {
                "name": "Solar Inverter",
                "type": "equipment",
                "category": "electrical",
                "dimensions": ComponentDimensions(400, 200, 600).to_dict(),
                "performance": {
                    "capacity": 5.0,  # kW
                    "efficiency": 97.5,  # percent
                    "max_dc_voltage": 1000,  # volts
                    "mppt_trackers": 2
                },
                "features": {
                    "wifi_monitoring": True,
                    "arc_fault_detection": True,
                    "rapid_shutdown": True,
                    "weatherproof": "IP65"
                }
            }
        }
    
    def _initialize_structural_components(self) -> Dict[str, Dict]:
        """Initialize structural components"""
        
        return {
            "steel_beam_universal": {
                "name": "Universal Steel Beam",
                "type": "structural",
                "category": "beam",
                "standard_sizes": [
                    {"designation": "150UB18", "depth": 150, "width": 75, "weight": 18},
                    {"designation": "200UB25", "depth": 200, "width": 100, "weight": 25},
                    {"designation": "250UB31", "depth": 250, "width": 125, "weight": 31}
                ],
                "materials": ComponentMaterials("steel_structural").to_dict(),
                "properties": {
                    "yield_strength": 350,  # MPa
                    "tensile_strength": 430,  # MPa
                    "elastic_modulus": 200000,  # MPa
                    "fire_rating": "with_protection"
                }
            },
            
            "concrete_column": {
                "name": "Reinforced Concrete Column",
                "type": "structural",
                "category": "column",
                "standard_sizes": [
                    {"size": "300x300", "reinforcement": "8N20"},
                    {"size": "400x400", "reinforcement": "12N20"},
                    {"size": "500x500", "reinforcement": "16N25"}
                ],
                "materials": ComponentMaterials("concrete_structural").to_dict(),
                "properties": {
                    "concrete_strength": 32,  # MPa
                    "steel_grade": "N_class",
                    "cover": 40,  # mm
                    "fire_rating": "240_minutes"
                }
            },
            
            "timber_truss": {
                "name": "Engineered Timber Truss",
                "type": "structural",
                "category": "roof_structure",
                "dimensions": ComponentDimensions(12000, 600, 300).to_dict(),
                "materials": ComponentMaterials("timber_engineered").to_dict(),
                "properties": {
                    "span_capability": 12000,  # mm
                    "load_capacity": 15,  # kN/m
                    "deflection_limit": "L/300",
                    "moisture_content": 12  # percent
                }
            }
        }
    
    def _initialize_mep_components(self) -> Dict[str, Dict]:
        """Initialize MEP system components"""
        
        return {
            "electrical_outlet_gpo": {
                "name": "General Purpose Outlet",
                "type": "electrical",
                "category": "power",
                "dimensions": ComponentDimensions(80, 80, 40).to_dict(),
                "specifications": {
                    "voltage": 240,  # volts
                    "current": 10,  # amps
                    "protection": "RCD_required",
                    "standard": "AS3112",
                    "earthing": "required"
                },
                "installation": {
                    "height": 300,  # mm above floor
                    "box_type": "plastic_flush",
                    "cable_size": "2.5mm2"
                }
            },
            
            "lighting_downlight_led": {
                "name": "LED Downlight",
                "type": "electrical",
                "category": "lighting",
                "dimensions": ComponentDimensions(90, 90, 100).to_dict(),
                "specifications": {
                    "power": 10,  # watts
                    "lumens": 900,
                    "color_temperature": 3000,  # kelvin
                    "beam_angle": 36,  # degrees
                    "dimming": "phase_cut"
                },
                "installation": {
                    "cutout": 75,  # mm
                    "clearance": 100,  # mm above
                    "fire_rating": "IC_F_rated"
                }
            },
            
            "plumbing_pipe_copper": {
                "name": "Copper Water Pipe",
                "type": "plumbing",
                "category": "water_supply",
                "standard_sizes": [
                    {"size": "15mm", "wall_thickness": 0.7, "flow_rate": 0.3},
                    {"size": "20mm", "wall_thickness": 0.8, "flow_rate": 0.6},
                    {"size": "25mm", "wall_thickness": 0.9, "flow_rate": 1.0}
                ],
                "materials": ComponentMaterials("copper_type_b").to_dict(),
                "specifications": {
                    "pressure_rating": 2100,  # kPa
                    "temperature_rating": 150,  # celsius
                    "standard": "AS1432",
                    "joining": "capillary_fittings"
                }
            },
            
            "hvac_duct_rectangular": {
                "name": "Rectangular Duct",
                "type": "hvac",
                "category": "ductwork",
                "standard_sizes": [
                    {"width": 300, "height": 150, "area": 0.045},
                    {"width": 400, "height": 200, "area": 0.08},
                    {"width": 500, "height": 250, "area": 0.125}
                ],
                "materials": ComponentMaterials("steel_galvanized").to_dict(),
                "specifications": {
                    "gauge": 0.6,  # mm
                    "velocity": 8.0,  # m/s maximum
                    "pressure_class": "low",
                    "insulation": "external_required"
                }
            }
        }
    
    def _initialize_accessibility_components(self) -> Dict[str, Dict]:
        """Initialize accessibility components"""
        
        return {
            "ramp_concrete": {
                "name": "Concrete Access Ramp",
                "type": "accessibility",
                "category": "circulation",
                "specifications": {
                    "gradient": 8.33,  # percent (1:12)
                    "width": 1000,  # mm minimum
                    "landing_length": 1200,  # mm
                    "edge_protection": 75,  # mm height
                    "handrail_height": 865  # mm
                },
                "materials": ComponentMaterials("concrete_slip_resistant").to_dict(),
                "compliance": ["AS1428.1", "DDA", "BCA"]
            },
            
            "lift_passenger": {
                "name": "Passenger Lift",
                "type": "accessibility",
                "category": "vertical_transport",
                "dimensions": ComponentDimensions(1600, 1400, 2200).to_dict(),
                "specifications": {
                    "capacity": 630,  # kg (8 persons)
                    "travel_speed": 1.0,  # m/s
                    "door_width": 900,  # mm
                    "car_depth": 1400,  # mm
                    "car_width": 1100  # mm
                },
                "features": {
                    "accessible_controls": True,
                    "audio_announcements": True,
                    "braille_buttons": True,
                    "emergency_communication": True
                }
            },
            
            "grab_rail_fixed": {
                "name": "Fixed Grab Rail",
                "type": "accessibility",
                "category": "bathroom_aid",
                "dimensions": ComponentDimensions(600, 40, 40).to_dict(),
                "materials": ComponentMaterials("steel_stainless").to_dict(),
                "specifications": {
                    "load_capacity": 1300,  # N
                    "mounting_height": 810,  # mm
                    "projection": 60,  # mm
                    "grip_diameter": 32  # mm
                },
                "installation": {
                    "wall_type": "masonry_required",
                    "fixing": "through_bolt",
                    "backing": "timber_nogging"
                }
            }
        }
    
    def _customize_component(self, base_component: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Apply customizations to base component"""
        
        customized = base_component.copy()
        
        # Apply dimensional customizations
        if "width" in kwargs:
            customized["dimensions"]["width"] = kwargs["width"]
        if "height" in kwargs:
            customized["dimensions"]["height"] = kwargs["height"]
        if "depth" in kwargs:
            customized["dimensions"]["depth"] = kwargs["depth"]
        
        # Apply material customizations
        if "primary_material" in kwargs:
            customized["materials"]["primary"] = kwargs["primary_material"]
        if "secondary_material" in kwargs:
            customized["materials"]["secondary"] = kwargs["secondary_material"]
        
        # Apply performance customizations
        if "thermal_rating" in kwargs and "performance" in customized:
            customized["performance"]["thermal_rating"] = kwargs["thermal_rating"]
        
        # Apply accessibility requirements
        if "ada_compliant" in kwargs:
            customized["accessibility"] = {"ada_compliant": kwargs["ada_compliant"]}
        
        return customized
    
    def _get_default_component(self, component_type: str) -> Dict[str, Any]:
        """Get default component for unknown types"""
        
        return {
            "name": f"Default {component_type.title()}",
            "type": component_type,
            "dimensions": ComponentDimensions(1000, 1000, 100).to_dict(),
            "materials": ComponentMaterials("default_material").to_dict()
        }
    
    def _generate_door_geometry(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D geometry for door component"""
        
        dims = component["dimensions"]
        operation = component.get("operation", "hinged")
        
        geometry = {
            "type": "door",
            "operation": operation,
            "frame": {
                "width": dims["width"] + 100,  # Frame wider than opening
                "height": dims["height"] + 100,
                "depth": dims["depth"] + 50
            },
            "panel": {
                "width": dims["width"],
                "height": dims["height"], 
                "thickness": dims["depth"]
            },
            "hardware_positions": self._calculate_door_hardware_positions(component)
        }
        
        return geometry
    
    def _generate_window_geometry(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D geometry for window component"""
        
        dims = component["dimensions"]
        operation = component.get("operation", "fixed")
        
        geometry = {
            "type": "window",
            "operation": operation,
            "frame": {
                "width": dims["width"],
                "height": dims["height"],
                "depth": dims["depth"]
            },
            "glazing": {
                "width": dims["width"] - 100,  # Glazing smaller than frame
                "height": dims["height"] - 100,
                "thickness": component.get("glazing", {}).get("thickness", 24)
            },
            "hardware_positions": self._calculate_window_hardware_positions(component)
        }
        
        return geometry
    
    def _generate_fixture_geometry(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D geometry for fixture component"""
        
        dims = component["dimensions"]
        category = component.get("category", "general")
        
        geometry = {
            "type": "fixture",
            "category": category,
            "main_body": {
                "width": dims["width"],
                "height": dims["height"],
                "depth": dims["depth"]
            },
            "connections": self._calculate_fixture_connections(component)
        }
        
        return geometry
    
    def _generate_generic_geometry(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic 3D geometry"""
        
        dims = component["dimensions"]
        
        return {
            "type": "generic",
            "bounding_box": {
                "width": dims["width"],
                "height": dims["height"],
                "depth": dims["depth"]
            }
        }
    
    def _calculate_door_hardware_positions(self, component: Dict[str, Any]) -> Dict[str, List]:
        """Calculate door hardware positions"""
        
        hardware = component.get("hardware", {})
        dims = component["dimensions"]
        
        positions = {}
        
        # Handle position
        if "handle" in hardware:
            positions["handle"] = [{
                "x": dims["width"] - 100,
                "y": hardware["handle"].get("height", 1050),
                "z": dims["depth"] / 2
            }]
        
        # Hinge positions
        if "hinges" in hardware:
            hinge_count = hardware["hinges"].get("count", 3)
            positions["hinges"] = []
            for i in range(hinge_count):
                y_pos = 150 + (i * (dims["height"] - 300) / (hinge_count - 1))
                positions["hinges"].append({
                    "x": 0,
                    "y": y_pos,
                    "z": dims["depth"] / 2
                })
        
        return positions
    
    def _calculate_window_hardware_positions(self, component: Dict[str, Any]) -> Dict[str, List]:
        """Calculate window hardware positions"""
        
        hardware = component.get("hardware", {})
        dims = component["dimensions"]
        
        positions = {}
        
        # Handle position
        if "handle" in hardware:
            positions["handle"] = [{
                "x": dims["width"] - 100,
                "y": dims["height"] / 2,
                "z": dims["depth"]
            }]
        
        return positions
    
    def _calculate_fixture_connections(self, component: Dict[str, Any]) -> Dict[str, Dict]:
        """Calculate fixture connection points"""
        
        category = component.get("category", "general")
        dims = component["dimensions"]
        
        connections = {}
        
        if category == "bathroom":
            # Water supply connections
            connections["water_supply"] = {
                "hot": {"x": dims["width"] * 0.3, "y": 0, "z": dims["depth"] / 2},
                "cold": {"x": dims["width"] * 0.7, "y": 0, "z": dims["depth"] / 2}
            }
            
            # Drain connection
            connections["drain"] = {
                "x": dims["width"] / 2,
                "y": 0,
                "z": dims["depth"] / 2
            }
        
        return connections


def create_component_library() -> ParametricComponentLibrary:
    """Create parametric component library instance"""
    
    return ParametricComponentLibrary()


if __name__ == "__main__":
    # Test the parametric component library
    component_lib = create_component_library()
    
    print("🔧 Parametric Component Library Test")
    print("=" * 50)
    
    # Test door component
    pivot_door = component_lib.get_component("door", "pivot", width=1500, height=3000)
    print(f"Pivot door: {pivot_door['name']}")
    print(f"Dimensions: {pivot_door['dimensions']}")
    
    # Test window component
    bay_window = component_lib.get_component("window", "bay")
    print(f"Bay window: {bay_window['name']}")
    print(f"Features: {bay_window.get('hardware', {})}")
    
    # Test 3D geometry generation
    door_geometry = component_lib.generate_component_3d_geometry(pivot_door)
    print(f"Door geometry: {door_geometry['type']}")
    
    print("✅ Parametric Component Library initialized successfully!")