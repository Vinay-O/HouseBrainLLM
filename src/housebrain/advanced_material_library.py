"""
Advanced Material Library for HouseBrain Professional

This module provides an extensive library of PBR materials with:
- High-quality texture definitions
- Regional material variations
- Procedural material generation
- Advanced material properties
- Sustainability material ratings
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional


class AdvancedMaterialLibrary:
    """Comprehensive material library with advanced PBR properties"""
    
    def __init__(self):
        self.material_categories = self._initialize_material_categories()
        self.regional_materials = self._initialize_regional_materials()
        self.sustainability_ratings = self._initialize_sustainability_ratings()
        self.texture_library = self._initialize_texture_library()
        
    def get_material(self, material_name: str, region: str = "global") -> Dict[str, Any]:
        """Get material with regional variations"""
        
        # Check for regional specific material first
        regional_key = f"{region}_{material_name}"
        if regional_key in self.regional_materials:
            base_material = self.regional_materials[regional_key]
        else:
            # Fall back to global material
            base_material = self._find_material_in_categories(material_name)
        
        if not base_material:
            return self._get_default_material()
        
        # Enhance with sustainability rating
        enhanced_material = base_material.copy()
        enhanced_material["sustainability"] = self.sustainability_ratings.get(
            material_name, {"rating": "C", "embodied_carbon": "medium"}
        )
        
        return enhanced_material
    
    def get_materials_by_category(self, category: str) -> Dict[str, Dict]:
        """Get all materials in a specific category"""
        
        return self.material_categories.get(category, {})
    
    def get_regional_materials(self, region: str) -> Dict[str, Dict]:
        """Get materials specific to a region"""
        
        regional_materials = {}
        for key, material in self.regional_materials.items():
            if key.startswith(f"{region}_"):
                material_name = key.replace(f"{region}_", "")
                regional_materials[material_name] = material
        
        return regional_materials
    
    def get_all_materials(self) -> List[Dict[str, Any]]:
        """Get all materials in the library"""
        all_materials = []
        
        # Add materials from all categories
        for category, materials in self.material_categories.items():
            for material_id, material in materials.items():
                all_materials.append({
                    "id": material_id,
                    "category": category,
                    **material
                })
        
        return all_materials
    
    def generate_material_palette(self, building_type: str, region: str, style: str) -> List[Dict]:
        """Generate curated material palette for specific project"""
        
        palette_rules = {
            "residential": {
                "contemporary": ["concrete_polished", "timber_oak", "steel_brushed", "glass_clear"],
                "traditional": ["brick_red", "timber_pine", "stone_limestone", "slate_grey"],
                "mediterranean": ["stucco_white", "terracotta_natural", "stone_travertine", "timber_cedar"]
            },
            "commercial": {
                "modern": ["concrete_exposed", "steel_weathering", "glass_structural", "aluminum_anodized"],
                "traditional": ["brick_buff", "stone_granite", "timber_hardwood", "bronze_patina"]
            },
            "industrial": {
                "functional": ["concrete_industrial", "steel_galvanized", "metal_corrugated", "epoxy_floor"]
            }
        }
        
        base_materials = palette_rules.get(building_type, {}).get(style, ["concrete_smooth"])
        
        # Get regional variations of base materials
        palette = []
        for material_name in base_materials:
            material = self.get_material(material_name, region)
            palette.append(material)
        
        return palette
    
    def _initialize_material_categories(self) -> Dict[str, Dict]:
        """Initialize comprehensive material categories"""
        
        return {
            "concrete": {
                "concrete_polished": {
                    "name": "Polished Concrete",
                    "baseColorFactor": [0.75, 0.75, 0.75, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.2,
                    "normalTexture": "concrete_polished_normal",
                    "roughnessTexture": "concrete_polished_roughness",
                    "occlusionTexture": "concrete_polished_ao",
                    "properties": {
                        "density": 2400,  # kg/m³
                        "compressive_strength": 40,  # MPa
                        "thermal_conductivity": 1.7,  # W/mK
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low"
                    }
                },
                "concrete_exposed": {
                    "name": "Exposed Concrete",
                    "baseColorFactor": [0.6, 0.6, 0.6, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                    "normalTexture": "concrete_exposed_normal",
                    "properties": {
                        "density": 2400,
                        "compressive_strength": 35,
                        "thermal_conductivity": 1.7,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "medium"
                    }
                },
                "concrete_precast": {
                    "name": "Precast Concrete",
                    "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                    "normalTexture": "concrete_precast_normal",
                    "properties": {
                        "density": 2400,
                        "compressive_strength": 50,
                        "thermal_conductivity": 1.7,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low"
                    }
                }
            },
            
            "masonry": {
                "brick_red": {
                    "name": "Red Clay Brick",
                    "baseColorFactor": [0.8, 0.35, 0.25, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "normalTexture": "brick_red_normal",
                    "properties": {
                        "density": 1900,
                        "compressive_strength": 25,
                        "thermal_conductivity": 0.7,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low"
                    }
                },
                "brick_buff": {
                    "name": "Buff Brick",
                    "baseColorFactor": [0.9, 0.8, 0.6, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "normalTexture": "brick_buff_normal",
                    "properties": {
                        "density": 1900,
                        "compressive_strength": 25,
                        "thermal_conductivity": 0.7,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low"
                    }
                },
                "concrete_block": {
                    "name": "Concrete Masonry Unit",
                    "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                    "normalTexture": "cmu_normal",
                    "properties": {
                        "density": 2000,
                        "compressive_strength": 15,
                        "thermal_conductivity": 0.9,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "low"
                    }
                }
            },
            
            "timber": {
                "timber_oak": {
                    "name": "European Oak",
                    "baseColorFactor": [0.7, 0.5, 0.3, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                    "normalTexture": "timber_oak_normal",
                    "properties": {
                        "density": 700,
                        "strength": "high",
                        "thermal_conductivity": 0.15,
                        "fire_rating": "combustible",
                        "durability": "excellent",
                        "maintenance": "medium",
                        "sustainability": "renewable"
                    }
                },
                "timber_pine": {
                    "name": "Scandinavian Pine",
                    "baseColorFactor": [0.9, 0.8, 0.6, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.5,
                    "normalTexture": "timber_pine_normal",
                    "properties": {
                        "density": 520,
                        "strength": "medium",
                        "thermal_conductivity": 0.12,
                        "fire_rating": "combustible",
                        "durability": "good",
                        "maintenance": "medium",
                        "sustainability": "renewable"
                    }
                },
                "timber_cedar": {
                    "name": "Western Red Cedar",
                    "baseColorFactor": [0.8, 0.4, 0.3, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.7,
                    "normalTexture": "timber_cedar_normal",
                    "properties": {
                        "density": 350,
                        "strength": "medium",
                        "thermal_conductivity": 0.1,
                        "fire_rating": "combustible",
                        "durability": "excellent",
                        "maintenance": "low",
                        "sustainability": "renewable"
                    }
                },
                "timber_bamboo": {
                    "name": "Engineered Bamboo",
                    "baseColorFactor": [0.9, 0.85, 0.7, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.4,
                    "normalTexture": "bamboo_normal",
                    "properties": {
                        "density": 600,
                        "strength": "high",
                        "thermal_conductivity": 0.12,
                        "fire_rating": "combustible",
                        "durability": "good",
                        "maintenance": "low",
                        "sustainability": "highly_renewable"
                    }
                }
            },
            
            "metals": {
                "steel_brushed": {
                    "name": "Brushed Stainless Steel",
                    "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                    "metallicFactor": 0.9,
                    "roughnessFactor": 0.3,
                    "normalTexture": "steel_brushed_normal",
                    "properties": {
                        "density": 7850,
                        "strength": "very_high",
                        "thermal_conductivity": 50,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low",
                        "corrosion_resistance": "excellent"
                    }
                },
                "steel_weathering": {
                    "name": "Weathering Steel (Corten)",
                    "baseColorFactor": [0.6, 0.3, 0.2, 1.0],
                    "metallicFactor": 0.7,
                    "roughnessFactor": 0.8,
                    "normalTexture": "steel_weathering_normal",
                    "properties": {
                        "density": 7850,
                        "strength": "very_high",
                        "thermal_conductivity": 50,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "very_low",
                        "corrosion_resistance": "self_protecting"
                    }
                },
                "aluminum_anodized": {
                    "name": "Anodized Aluminum",
                    "baseColorFactor": [0.85, 0.85, 0.85, 1.0],
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.2,
                    "normalTexture": "aluminum_anodized_normal",
                    "properties": {
                        "density": 2700,
                        "strength": "medium",
                        "thermal_conductivity": 200,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "very_low",
                        "corrosion_resistance": "excellent"
                    }
                },
                "copper_patina": {
                    "name": "Copper with Patina",
                    "baseColorFactor": [0.3, 0.6, 0.5, 1.0],
                    "metallicFactor": 0.6,
                    "roughnessFactor": 0.7,
                    "normalTexture": "copper_patina_normal",
                    "properties": {
                        "density": 8960,
                        "strength": "medium",
                        "thermal_conductivity": 400,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "very_low",
                        "corrosion_resistance": "self_protecting"
                    }
                }
            },
            
            "glass": {
                "glass_clear": {
                    "name": "Clear Glass",
                    "baseColorFactor": [0.9, 0.9, 1.0, 0.1],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.0,
                    "transmissionFactor": 0.9,
                    "ior": 1.52,
                    "properties": {
                        "density": 2500,
                        "thermal_conductivity": 1.0,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "medium",
                        "light_transmission": 0.9
                    }
                },
                "glass_low_e": {
                    "name": "Low-E Glass",
                    "baseColorFactor": [0.85, 0.9, 0.95, 0.15],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.0,
                    "transmissionFactor": 0.8,
                    "ior": 1.52,
                    "properties": {
                        "density": 2500,
                        "thermal_conductivity": 0.8,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "medium",
                        "light_transmission": 0.8,
                        "thermal_performance": "high"
                    }
                },
                "glass_frosted": {
                    "name": "Frosted Glass",
                    "baseColorFactor": [0.9, 0.9, 1.0, 0.3],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "transmissionFactor": 0.6,
                    "properties": {
                        "density": 2500,
                        "thermal_conductivity": 1.0,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "medium",
                        "light_transmission": 0.6,
                        "privacy": "high"
                    }
                }
            },
            
            "stone": {
                "stone_granite": {
                    "name": "Granite",
                    "baseColorFactor": [0.4, 0.4, 0.4, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                    "normalTexture": "granite_normal",
                    "properties": {
                        "density": 2700,
                        "compressive_strength": 130,
                        "thermal_conductivity": 2.8,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low"
                    }
                },
                "stone_limestone": {
                    "name": "Limestone",
                    "baseColorFactor": [0.8, 0.8, 0.75, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.7,
                    "normalTexture": "limestone_normal",
                    "properties": {
                        "density": 2600,
                        "compressive_strength": 60,
                        "thermal_conductivity": 2.0,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "medium"
                    }
                },
                "stone_travertine": {
                    "name": "Travertine",
                    "baseColorFactor": [0.9, 0.85, 0.75, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "normalTexture": "travertine_normal",
                    "properties": {
                        "density": 2500,
                        "compressive_strength": 50,
                        "thermal_conductivity": 1.8,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "medium"
                    }
                }
            },
            
            "finishes": {
                "plaster_smooth": {
                    "name": "Smooth Plaster",
                    "baseColorFactor": [0.95, 0.95, 0.95, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "properties": {
                        "density": 1400,
                        "thermal_conductivity": 0.5,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "medium"
                    }
                },
                "plaster_venetian": {
                    "name": "Venetian Plaster",
                    "baseColorFactor": [0.9, 0.9, 0.85, 1.0],
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.3,
                    "normalTexture": "venetian_plaster_normal",
                    "properties": {
                        "density": 1600,
                        "thermal_conductivity": 0.6,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low"
                    }
                },
                "stucco_textured": {
                    "name": "Textured Stucco",
                    "baseColorFactor": [0.85, 0.8, 0.75, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                    "normalTexture": "stucco_textured_normal",
                    "properties": {
                        "density": 1800,
                        "thermal_conductivity": 0.7,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "medium"
                    }
                }
            },
            
            "ceramics": {
                "tile_porcelain": {
                    "name": "Porcelain Tile",
                    "baseColorFactor": [0.9, 0.9, 0.9, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.2,
                    "properties": {
                        "density": 2400,
                        "water_absorption": 0.1,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "very_low"
                    }
                },
                "tile_ceramic": {
                    "name": "Ceramic Tile",
                    "baseColorFactor": [0.85, 0.85, 0.85, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.3,
                    "properties": {
                        "density": 2300,
                        "water_absorption": 3.0,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "low"
                    }
                },
                "terracotta_natural": {
                    "name": "Natural Terracotta",
                    "baseColorFactor": [0.8, 0.4, 0.3, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "normalTexture": "terracotta_normal",
                    "properties": {
                        "density": 1900,
                        "water_absorption": 8.0,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "medium"
                    }
                }
            },
            
            "composite": {
                "fiber_cement": {
                    "name": "Fiber Cement Board",
                    "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.7,
                    "properties": {
                        "density": 1400,
                        "thermal_conductivity": 0.2,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low"
                    }
                },
                "composite_wood": {
                    "name": "Wood Composite Decking",
                    "baseColorFactor": [0.6, 0.4, 0.3, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                    "normalTexture": "composite_wood_normal",
                    "properties": {
                        "density": 1000,
                        "thermal_conductivity": 0.2,
                        "fire_rating": "combustible",
                        "durability": "excellent",
                        "maintenance": "very_low"
                    }
                }
            },
            
            "sustainable": {
                "rammed_earth": {
                    "name": "Rammed Earth",
                    "baseColorFactor": [0.7, 0.5, 0.4, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                    "normalTexture": "rammed_earth_normal",
                    "properties": {
                        "density": 2200,
                        "thermal_conductivity": 1.3,
                        "fire_rating": "non_combustible",
                        "durability": "good",
                        "maintenance": "low",
                        "embodied_carbon": "very_low",
                        "sustainability": "excellent"
                    }
                },
                "hempcrete": {
                    "name": "Hempcrete",
                    "baseColorFactor": [0.8, 0.8, 0.7, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "properties": {
                        "density": 300,
                        "thermal_conductivity": 0.06,
                        "fire_rating": "fire_resistant",
                        "durability": "good",
                        "maintenance": "low",
                        "embodied_carbon": "negative",
                        "sustainability": "excellent"
                    }
                },
                "recycled_steel": {
                    "name": "Recycled Steel",
                    "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.4,
                    "properties": {
                        "density": 7850,
                        "strength": "very_high",
                        "thermal_conductivity": 50,
                        "fire_rating": "non_combustible",
                        "durability": "excellent",
                        "maintenance": "low",
                        "recycled_content": 90,
                        "sustainability": "good"
                    }
                }
            }
        }
    
    def _initialize_regional_materials(self) -> Dict[str, Dict]:
        """Initialize region-specific material variations"""
        
        return {
            # Tropical region materials
            "tropical_timber_teak": {
                "name": "Tropical Teak",
                "baseColorFactor": [0.8, 0.6, 0.4, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.5,
                "properties": {
                    "density": 650,
                    "strength": "high",
                    "durability": "excellent",
                    "moisture_resistance": "excellent",
                    "termite_resistance": "excellent"
                }
            },
            "tropical_bamboo_structural": {
                "name": "Structural Bamboo",
                "baseColorFactor": [0.9, 0.85, 0.6, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.6,
                "properties": {
                    "density": 700,
                    "strength": "very_high",
                    "durability": "good",
                    "growth_rate": "very_fast",
                    "carbon_sequestration": "high"
                }
            },
            
            # Arid region materials
            "arid_adobe": {
                "name": "Adobe Block",
                "baseColorFactor": [0.8, 0.6, 0.4, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
                "properties": {
                    "density": 1600,
                    "thermal_mass": "high",
                    "thermal_conductivity": 0.9,
                    "fire_rating": "non_combustible",
                    "cost": "very_low",
                    "local_availability": "excellent"
                }
            },
            "arid_stone_sandstone": {
                "name": "Desert Sandstone",
                "baseColorFactor": [0.9, 0.7, 0.5, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
                "properties": {
                    "density": 2200,
                    "thermal_mass": "high",
                    "durability": "excellent",
                    "local_availability": "excellent",
                    "cost": "low"
                }
            },
            
            # Temperate region materials
            "temperate_brick_clay": {
                "name": "Local Clay Brick",
                "baseColorFactor": [0.7, 0.4, 0.3, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
                "properties": {
                    "density": 1800,
                    "thermal_mass": "medium",
                    "durability": "excellent",
                    "frost_resistance": "good",
                    "local_availability": "good"
                }
            },
            "temperate_timber_hardwood": {
                "name": "Local Hardwood",
                "baseColorFactor": [0.6, 0.4, 0.3, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.6,
                "properties": {
                    "density": 750,
                    "strength": "high",
                    "durability": "good",
                    "local_availability": "good",
                    "sustainability": "renewable"
                }
            },
            
            # Continental region materials
            "continental_timber_softwood": {
                "name": "Softwood Lumber",
                "baseColorFactor": [0.9, 0.8, 0.6, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.7,
                "properties": {
                    "density": 450,
                    "strength": "medium",
                    "thermal_conductivity": 0.1,
                    "frost_resistance": "excellent",
                    "local_availability": "excellent"
                }
            },
            "continental_stone_granite": {
                "name": "Regional Granite",
                "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.6,
                "properties": {
                    "density": 2700,
                    "strength": "very_high",
                    "frost_resistance": "excellent",
                    "durability": "excellent",
                    "local_availability": "good"
                }
            },
            
            # Maritime region materials
            "maritime_timber_cedar": {
                "name": "Marine Grade Cedar",
                "baseColorFactor": [0.8, 0.5, 0.3, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.7,
                "properties": {
                    "density": 350,
                    "moisture_resistance": "excellent",
                    "salt_resistance": "excellent",
                    "durability": "excellent",
                    "maintenance": "low"
                }
            },
            "maritime_concrete_marine": {
                "name": "Marine Concrete",
                "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.6,
                "properties": {
                    "density": 2400,
                    "salt_resistance": "excellent",
                    "durability": "excellent",
                    "chloride_resistance": "high",
                    "maintenance": "low"
                }
            }
        }
    
    def _initialize_sustainability_ratings(self) -> Dict[str, Dict]:
        """Initialize sustainability ratings for materials"""
        
        return {
            "concrete_polished": {
                "rating": "C",
                "embodied_carbon": "high",
                "recycled_content": 20,
                "end_of_life": "recyclable",
                "certifications": ["LEED_applicable"]
            },
            "timber_oak": {
                "rating": "A",
                "embodied_carbon": "low",
                "renewable": True,
                "carbon_sequestration": "high",
                "certifications": ["FSC", "PEFC"]
            },
            "steel_brushed": {
                "rating": "B",
                "embodied_carbon": "high",
                "recycled_content": 90,
                "recyclability": "excellent",
                "certifications": ["LEED_applicable"]
            },
            "bamboo_structural": {
                "rating": "A+",
                "embodied_carbon": "very_low",
                "renewable": True,
                "growth_rate": "very_fast",
                "certifications": ["FSC", "Cradle_to_Cradle"]
            },
            "rammed_earth": {
                "rating": "A+",
                "embodied_carbon": "very_low",
                "local_sourcing": "excellent",
                "end_of_life": "biodegradable",
                "certifications": ["Living_Building_Challenge"]
            }
        }
    
    def _initialize_texture_library(self) -> Dict[str, Dict]:
        """Initialize procedural texture library"""
        
        return {
            "concrete_polished_normal": {
                "type": "procedural",
                "generator": "concrete_polish_pattern",
                "parameters": {"roughness": 0.1, "scale": 2.0}
            },
            "brick_red_normal": {
                "type": "procedural", 
                "generator": "brick_pattern",
                "parameters": {"mortar_width": 10, "brick_variation": 0.1}
            },
            "timber_oak_normal": {
                "type": "procedural",
                "generator": "wood_grain_pattern",
                "parameters": {"grain_intensity": 0.3, "knot_frequency": 0.1}
            }
        }
    
    def _find_material_in_categories(self, material_name: str) -> Optional[Dict]:
        """Find material in category hierarchy"""
        
        for category, materials in self.material_categories.items():
            if material_name in materials:
                return materials[material_name]
        
        return None
    
    def _get_default_material(self) -> Dict[str, Any]:
        """Get default fallback material"""
        
        return {
            "name": "Default Material",
            "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
            "properties": {
                "density": 1000,
                "durability": "unknown",
                "maintenance": "unknown"
            }
        }


def create_material_database() -> AdvancedMaterialLibrary:
    """Create advanced material database instance"""
    
    return AdvancedMaterialLibrary()


if __name__ == "__main__":
    # Test the advanced material library
    material_lib = create_material_database()
    
    print("🏗️ Advanced Material Library Test")
    print("=" * 50)
    
    # Test regional materials
    tropical_materials = material_lib.get_regional_materials("tropical")
    print(f"Tropical materials: {len(tropical_materials)}")
    
    # Test material palette generation
    residential_palette = material_lib.generate_material_palette("residential", "temperate", "contemporary")
    print(f"Residential palette: {len(residential_palette)} materials")
    
    # Test specific material
    oak_material = material_lib.get_material("timber_oak", "temperate")
    print(f"Oak material properties: {oak_material.get('properties', {})}")
    
    print("✅ Advanced Material Library initialized successfully!")