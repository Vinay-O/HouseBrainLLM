#!/usr/bin/env python3
"""
HouseBrain 1M Super-Quality Reasoning Dataset Generator (R1)
- Advanced problem types (incl. structural, sustainability, smart home)
- India-specific features, bye-laws, heritage constraints
- Comprehensive building code compliance
- Strict quality gates, deduplication, sharded saving
- Compatible with current training scripts ({"input":..., "output":...})
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

@dataclass
class SuperQualityConfig:
    target_samples: int = 1_000_000
    quality_threshold: float = 0.90  # Increased to 0.90 for maximum quality
    train_ratio: float = 0.90
    shard_size: int = 100_000
    min_reasoning_steps: int = 8  # Increased to 8 for maximum reasoning depth
    min_output_chars: int = 800  # Increased to 800 for maximum content richness
    max_output_chars: int = 20_000
    india_ratio: float = 0.40
    seed: int = 42

    complexity_levels: List[str] = field(default_factory=lambda: [
        "Basic_Design", "Code_Compliance", "Multi_Constraint", "Optimization",
        "Conflict_Resolution", "Advanced_Reasoning", "Mathematical_Analysis",
        "Structural_Engineering", "Sustainability_Design", "Smart_Home_Integration",
        "Performance_Optimization"
    ])

    building_codes: List[str] = field(default_factory=lambda: [
        "NBC_2016", "IS_456", "IS_875", "IS_1893", "ECBC_2017",
        "Local_Bye_Laws", "Fire_Safety_Norms", "Seismic_Design_Codes",
        "Accessibility_Standards"
    ])

    regions_india: List[str] = field(default_factory=lambda: [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore",
        "Nagpur", "Vadodara", "Surat", "Bhopal", "Patna", "Ranchi"
    ])
    climate_zones: List[str] = field(default_factory=lambda: [
        "Tropical_Hot_Humid", "Tropical_Warm_Humid", "Subtropical_Hot_Dry",
        "Subtropical_Warm_Humid", "Composite", "Arid_Hot_Dry", "Arid_Warm_Dry",
        "Cold_Climate"
    ])
    building_types: List[str] = field(default_factory=lambda: [
        "Residential", "Commercial", "Mixed_Use", "Institutional"
    ])


class SuperQualityGenerator:
    def __init__(self, config: SuperQualityConfig):
        self.config = config
        random.seed(self.config.seed)
        self.generated = 0
        self.accepted = 0
        self.uniques: set[str] = set()

    # ---------- helpers ----------
    def _safe_sample(self, items: List[Any], k_min: int, k_max: int) -> List[Any]:
        if not items:
            return []
        k_max_eff = min(max(k_min, k_max), len(items))
        k_min_eff = min(k_min, k_max_eff)
        k = random.randint(k_min_eff, k_max_eff)
        return random.sample(items, k)

    def _rand_range(self, lo: int, hi: int) -> int:
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    def _hash(self, sample: Dict[str, Any]) -> str:
        payload = json.dumps(sample["input"], sort_keys=True) + "||" + json.dumps(sample["output"], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ---------- content ----------
    def _plot(self) -> Dict[str, Any]:
        shape = random.choice(["Rectangular", "Square", "L_Shape", "T_Shape", "Irregular", "Corner_Plot", "Sloped_Site", "Narrow_Plot", "Wide_Plot"])
        if shape == "Square":
            width = self._rand_range(30, 120)
            length = width
        elif shape == "Rectangular":
            width = self._rand_range(25, 100)
            length = self._rand_range(width + 10, int(width * 2.5))
        elif shape == "Narrow_Plot":
            width = self._rand_range(15, 35)
            length = self._rand_range(60, 120)
        elif shape == "Wide_Plot":
            width = self._rand_range(60, 100)
            length = self._rand_range(25, 60)
        else:
            width = self._rand_range(30, 90)
            length = self._rand_range(35, 95)
        area = width * length
        area = max(500, min(15000, area))
        return {
            "plot_shape": shape,
            "width_ft": width,
            "length_ft": length,
            "area_sqft": area,
            "orientation": random.choice(["North", "South", "East", "West", "North_East", "North_West", "South_East", "South_West"]),
            "slope_percentage": random.uniform(0, 15) if shape == "Sloped_Site" else 0,
            "soil_type": random.choice(["Hard_Rock", "Soft_Rock", "Sandy_Soil", "Clay_Soil", "Mixed_Soil"]),
            "access_road_width": self._rand_range(12, 30)
        }

    def _indian_features(self) -> Dict[str, Any]:
        region = random.choice(self.config.regions_india)
        climate_map = {
            "Mumbai": ["Tropical_Hot_Humid", "Tropical_Warm_Humid"],
            "Delhi": ["Composite", "Subtropical_Hot_Dry"],
            "Chennai": ["Tropical_Hot_Humid", "Tropical_Warm_Humid"],
            "Kolkata": ["Tropical_Hot_Humid", "Tropical_Warm_Humid"],
            "Bangalore": ["Composite", "Subtropical_Warm_Humid"],
            "Hyderabad": ["Composite", "Subtropical_Hot_Dry"],
            "Ahmedabad": ["Arid_Hot_Dry", "Arid_Warm_Dry"],
        }
        climate = random.choice(climate_map.get(region, ["Composite"]))
        style_map = {
            "Mumbai": ["Modern_Indian", "Contemporary_Indian", "Luxury_Indian"],
            "Delhi": ["Traditional_Indian", "Indo_Saracenic", "Modern_Indian"],
            "Chennai": ["Traditional_Indian", "Contemporary_Indian", "Heritage_Indian"],
            "Bangalore": ["Modern_Indian", "Contemporary_Indian", "Eco_Friendly_Indian"],
        }
        style = random.choice(style_map.get(region, ["Modern_Indian", "Contemporary_Indian"]))
        return {
            "region": region,
            "climate_zone": climate,
            "architectural_style": style,
            "vastu_compliance": random.choice([True, False])
        }

    # ---------- problems ----------
    def _problem_basic(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Basic_Design",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "plot_details": self._plot(),
            "requirements": {
                "family_size": self._rand_range(3, 8),
                "floors": self._rand_range(1, 4),
                "budget_inr": self._rand_range(2_000_000, 10_000_000),
                "lifestyle": random.choice(["Modern", "Traditional", "Minimalist", "Luxury", "Eco_Friendly"])
            },
            "reasoning_steps": [
                "Analyze site conditions including soil type, orientation, and access",
                "Calculate space requirements based on family size and lifestyle preferences",
                "Determine optimal room distribution considering privacy and functionality",
                "Plan circulation patterns for efficient movement between spaces",
                "Optimize natural light and ventilation based on site orientation",
                "Select appropriate materials and finishes within budget constraints",
                "Ensure compliance with local building codes and regulations",
                "Validate design efficiency and cost-effectiveness through analysis"
            ]
        }

    def _problem_code(self, indian: bool) -> Dict[str, Any]:
        floors = self._rand_range(2, 12)
        return {
            "problem_type": "Code_Compliance",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "location": {
                "zone_type": random.choice(["Residential", "Commercial", "Industrial", "Mixed_Use"]),
                "plot_area": self._rand_range(2_000, 15_000)
            },
            "building_specs": {
                "type": random.choice(["Residential", "Commercial", "Mixed_Use", "Institutional"]),
                "floors": floors,
                "height_m": round(floors * 3.2, 2),
                "total_area": self._rand_range(5_000, 30_000)
            },
            "compliance_challenges": self._safe_sample([
                "Setback_violations", "Floor_area_ratio_exceeded", "Height_restrictions",
                "Parking_shortage", "Fire_safety_requirements", "Structural_safety",
                "Accessibility_violations", "Ventilation_insufficient", "Natural_light_insufficient"
            ], 2, 5),
            "applicable_codes": self._safe_sample(self.config.building_codes, 4, 7),
            "reasoning_steps": [
                "Identify all applicable building codes and regulations for the region and building type",
                "Calculate setback requirements based on plot size, building height, and zone classification",
                "Verify floor area ratio (FAR) compliance with local development control regulations",
                "Check parking requirements and accessibility standards for the building type and size",
                "Ensure fire safety compliance including escape routes, suppression systems, and refuge areas",
                "Validate structural safety requirements and seismic design considerations",
                "Assess environmental compliance including ventilation, natural light, and waste management",
                "Propose specific modifications and estimate cost implications for achieving full compliance"
            ]
        }

    def _problem_multi(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Multi_Constraint",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "constraints": {
                "budget_inr": self._rand_range(3_000_000, 20_000_000),
                "timeline_months": self._rand_range(6, 24),
                "area_limit_sqft": self._rand_range(1_500, 10_000),
                "energy_goal": random.choice(["Net_Zero", "Green_Building", "High_Efficiency", "Standard"]),
                "aesthetic": random.choice(["Heritage_Compatible", "Modern", "Traditional", "Contemporary"])
            },
            "reasoning_steps": [
                "Prioritize constraints and success metrics",
                "Quantify trade-offs between constraints",
                "Develop alternatives and evaluate costs",
                "Select balanced plan with contingencies",
                "Define phased implementation and monitoring"
            ]
        }

    def _problem_opt(self, indian: bool) -> Dict[str, Any]:
        opt_type = random.choice(["Cost_Optimization", "Energy_Optimization", "Space_Optimization", "Performance_Optimization"])
        if opt_type == "Cost_Optimization":
            return {
                "problem_type": "Cost_Optimization",
                "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
                "plot_area": self._rand_range(2_000, 10_000),
                "total_budget": self._rand_range(5_000_000, 30_000_000),
                "cost_breakdown": {
                    "foundation_per_sqft": self._rand_range(800, 1500),
                    "structure_per_sqft": self._rand_range(1200, 2000),
                    "finishing_per_sqft": self._rand_range(800, 1800),
                    "mep_per_sqft": self._rand_range(400, 1000),
                    "landscaping_per_sqft": self._rand_range(200, 800)
                },
                "reasoning_steps": [
                    "Compute baseline construction cost",
                    "Identify optimization opportunities per component",
                    "Quantify savings and risks",
                    "Propose phased plan and QC measures",
                    "Validate against budget and quality"
                ]
            }
        if opt_type == "Energy_Optimization":
            return {
                "problem_type": "Energy_Optimization",
                "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
                "climate_zone": random.choice(self.config.climate_zones),
                "energy_target": random.choice(["Net_Zero", "Green_Building_5_Star", "ECBC_Compliant", "High_Efficiency"]),
                "site_conditions": {
                    "solar_exposure": random.choice(["High", "Medium", "Low"]),
                    "wind_patterns": random.choice(["Prevailing_Wind", "Variable", "Calm", "Seasonal"]),
                    "vegetation": random.choice(["Dense", "Moderate", "Sparse", "None"])
                },
                "reasoning_steps": [
                    "Analyze climate and targets",
                    "Optimize orientation/shading",
                    "Select insulation/glazing",
                    "Design efficient HVAC and renewables",
                    "Model performance and compliance"
                ]
            }
        if opt_type == "Space_Optimization":
            return {
                "problem_type": "Space_Optimization",
                "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
                "total_area": self._rand_range(1_500, 8_000),
                "family_composition": {
                    "adults": self._rand_range(2, 6),
                    "children": self._rand_range(0, 4),
                    "elderly": self._rand_range(0, 3)
                },
                "reasoning_steps": [
                    "Compute functional area requirements",
                    "Analyze adjacencies and circulation",
                    "Plan flexible spaces and storage",
                    "Maximize daylight and ventilation",
                    "Future-proof for adaptability"
                ]
            }
        return {
            "problem_type": "Performance_Optimization",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "performance_metrics": ["Thermal", "Acoustic", "Visual", "IAQ"],
            "targets": {"thermal": "ASHRAE_55", "acoustic": "STC_50", "visual": "DF_2%", "iaq": "ASHRAE_62.1"},
            "reasoning_steps": [
                "Set quantitative targets",
                "Optimize envelope and systems",
                "Integrate daylighting and controls",
                "Validate via simulation and monitoring"
            ]
        }

    def _problem_conflict(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Conflict_Resolution",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "conflicts": self._safe_sample([
                "Client_vs_Code", "Budget_vs_Quality", "Aesthetic_vs_Function",
                "Timeline_vs_Customization", "Site_vs_Design", "Sustainability_vs_Cost"
            ], 3, 5),
            "reasoning_steps": [
                "Identify root causes",
                "Analyze stakeholder priorities",
                "Develop resolution options",
                "Evaluate impact and select integrated plan",
                "Define communication and monitoring"
            ]
        }

    def _problem_advanced(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Advanced_Reasoning",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "complexity_factors": ["Multi_Story", "Mixed_Use", "Heritage_Integration", "Seismic_Design"],
            "reasoning_steps": [
                "Identify critical success factors",
                "Integrate structural/functional/aesthetic requirements",
                "Assess environmental impact and mitigation",
                "Evaluate economic feasibility (CBA)",
                "Plan stakeholder engagement and iterations"
            ]
        }

    def _problem_math(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Mathematical_Analysis",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "calculations": ["Load_Calculations", "Beam_Design", "Cost_Estimates", "Energy_Consumption", "Payback_Period"],
            "reasoning_steps": [
                "List required calculations and inputs",
                "Apply safety factors and standards",
                "Compute and cross-verify results",
                "Summarize ROI and payback",
                "Validate against codes and constraints"
            ]
        }

    def _problem_struct(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Structural_Engineering",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "elements": ["Foundation", "Column", "Beam", "Slab", "Roof"],
            "reasoning_steps": [
                "Analyze site conditions and soil properties to determine foundation requirements",
                "Calculate structural loads including dead loads, live loads, and environmental loads",
                "Design foundation system based on soil bearing capacity and building loads",
                "Determine structural system and member sizes considering safety factors and codes",
                "Perform seismic analysis and design for the specific seismic zone requirements",
                "Analyze wind loads and lateral stability considering building height and location",
                "Detail structural connections and joints for optimal load transfer and constructability",
                "Validate design through structural analysis and ensure compliance with all safety standards"
            ]
        }

    def _problem_sustain(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Sustainability_Design",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "goals": ["Energy", "Water", "Materials", "IEQ", "Site", "Waste"],
            "cert_targets": ["LEED_Platinum", "GRIHA_5_Star", "IGBC_Platinum"],
            "reasoning_steps": [
                "Define sustainability goals and certification targets based on project requirements",
                "Design energy-efficient building envelope with optimal insulation and glazing systems",
                "Integrate renewable energy systems including solar PV and solar water heating",
                "Plan water conservation strategies including rainwater harvesting and greywater recycling",
                "Select sustainable materials with low environmental impact and high durability",
                "Optimize indoor environmental quality through natural ventilation and daylighting",
                "Implement site sustainability measures including landscaping and stormwater management",
                "Establish monitoring and verification systems for ongoing performance assessment"
            ]
        }

    def _problem_smarthome(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Smart_Home_Integration",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "smart_systems": ["Automation", "Security", "Energy", "HVAC", "Lighting", "Entertainment"],
            "reasoning_steps": [
                "Analyze client requirements for smart home functionality and lifestyle integration",
                "Design integrated smart systems including automation, security, and energy management",
                "Plan IoT infrastructure with robust wireless networks and connectivity solutions",
                "Integrate energy management systems for optimal efficiency and cost savings",
                "Ensure data security and privacy protection for all smart home systems",
                "Provide user-friendly interfaces and mobile control applications for easy operation",
                "Design scalable architecture for future technology upgrades and system expansion",
                "Establish maintenance protocols and support systems for long-term reliability"
            ]
        }

    def _generate_problem(self) -> Dict[str, Any]:
        indian = random.random() < self.config.india_ratio
        p = random.choice(self.config.complexity_levels)
        mapping = {
            "Basic_Design": self._problem_basic,
            "Code_Compliance": self._problem_code,
            "Multi_Constraint": self._problem_multi,
            "Optimization": self._problem_opt,
            "Conflict_Resolution": self._problem_conflict,
            "Advanced_Reasoning": self._problem_advanced,
            "Mathematical_Analysis": self._problem_math,
            "Structural_Engineering": self._problem_struct,
            "Sustainability_Design": self._problem_sustain,
            "Smart_Home_Integration": self._problem_smarthome,
            "Performance_Optimization": self._problem_opt,
        }
        return mapping[p](indian)

    # ---------- solutions ----------
    def _solution_basic(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        area = problem["plot_details"]["area_sqft"]
        family = problem["requirements"]["family_size"]
        budget = problem["requirements"]["budget_inr"]
        bedrooms = max(2, min(6, family))
        living = int(area * 0.35)
        service = int(area * 0.22)
        circulation = int(area * 0.10)
        
        # Calculate realistic room areas
        bedroom_area = bedrooms * random.randint(120, 180)
        kitchen_area = int(service * 0.55)
        bathroom_area = int(service * 0.45)
        utility_area = int(area * 0.05)
        balcony_area = int(area * 0.08)
        
        # Determine material quality based on budget
        cost_per_sqft = budget / area
        if cost_per_sqft < 1500:
            quality = "Economy"
            materials = {
                "foundation": "RCC_Footing",
                "structure": "Load_Bearing",
                "walls": "Brick_Masonry",
                "roofing": "RCC_Slab",
                "flooring": "Cement_Flooring"
            }
        elif cost_per_sqft < 2500:
            quality = "Standard"
            materials = {
                "foundation": "RCC_Footing",
                "structure": "RCC_Frame",
                "walls": "Brick_Masonry",
                "roofing": "RCC_Slab",
                "flooring": "Vitrified_Tiles"
            }
        else:
            quality = "Premium"
            materials = {
                "foundation": "RCC_Footing",
                "structure": "RCC_Frame",
                "walls": "AAC_Blocks",
                "roofing": "RCC_Slab",
                "flooring": "Marble"
            }
        
        return {
            "design_solution": {
                "room_distribution": {
                    "bedrooms": bedrooms,
                    "bedroom_area_sqft": bedroom_area,
                    "living_area_sqft": living,
                    "kitchen_area_sqft": kitchen_area,
                    "bathroom_area_sqft": bathroom_area,
                    "circulation_area_sqft": circulation,
                    "utility_area_sqft": utility_area,
                    "balcony_area_sqft": balcony_area
                },
                "layout": random.choice([
                    "Open_plan_with_private_zones",
                    "Traditional_compartmentalized",
                    "Modern_split_level",
                    "Contemporary_loft_style"
                ]),
                "daylight": random.choice([
                    "North_South_optimized",
                    "East_West_orientation",
                    "Skylight_integrated",
                    "Courtyard_centered"
                ]),
                "ventilation": random.choice([
                    "Cross_ventilation",
                    "Stack_ventilation",
                    "Mechanical_ventilation",
                    "Natural_ventilation_enhanced"
                ])
            },
            "materials_and_finishes": materials,
            "analysis": {
                "space_efficiency": f"{random.randint(80, 95)}%",
                "natural_light_score": f"{random.uniform(0.75, 0.95):.2f}",
                "budget_alignment": random.choice(["Within_5_percent", "Within_10_percent", "Within_15_percent"]),
                "energy_efficiency": random.choice(["ECBC_Compliant", "Green_Building_Ready", "Net_Zero_Capable"])
            },
            "implementation": [
                "Site_preparation_and_excavation",
                "Foundation_construction",
                "Structural_framework",
                "Wall_and_roof_construction",
                "MEP_installation",
                "Interior_finishes",
                "Exterior_finishes",
                "Landscaping"
            ],
            "quality_level": quality,
            "estimated_cost_per_sqft": cost_per_sqft
        }

    def _solution_code(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        region = problem.get("context", {}).get("region", "Mumbai")
        family = 4  # Default family size for code compliance
        
        # Region-specific bye-laws
        bye_laws = {
            "Mumbai": {"setback_front": 3.0, "setback_rear": 3.0, "setback_sides": 1.5, "max_height": 24.0, "far_limit": 2.5},
            "Delhi": {"setback_front": 4.0, "setback_rear": 3.0, "setback_sides": 2.0, "max_height": 21.0, "far_limit": 2.0},
            "Chennai": {"setback_front": 3.0, "setback_rear": 3.0, "setback_sides": 1.5, "max_height": 18.0, "far_limit": 2.0}
        }
        
        local_bye_laws = bye_laws.get(region, bye_laws["Mumbai"])
        
        return {
            "compliance_solution": {
                "setbacks": {
                    "front": f"{local_bye_laws['setback_front']}m",
                    "rear": f"{local_bye_laws['setback_rear']}m", 
                    "side_each": f"{local_bye_laws['setback_sides']}m"
                },
                "far_limit": f"{local_bye_laws['far_limit']}",
                "height": f"Within_{local_bye_laws['max_height']}m_limit",
                "parking": f"{family + 1}_spaces_as_per_code",
                "fire_safety": "Staircase_refuge_suppression_system"
            },
            "modifications": self._safe_sample([
                f"Increase north setback by 0.5m to {local_bye_laws['setback_front'] + 0.5}m",
                f"Reduce height by 0.5m to {local_bye_laws['max_height'] - 0.5}m",
                f"Add {family + 1} parking bays as per requirement",
                "Upgrade fire suppression system to automatic",
                "Install fire escape staircase",
                "Add accessibility ramp for disabled access"
            ], 3, 4),
            "verification": {
                "setback": "Passed",
                "far": "Passed", 
                "height": "Passed",
                "parking": "Passed",
                "fire_safety": "Passed",
                "accessibility": "Passed"
            },
            "applicable_codes": problem.get("context", {}).get("applicable_codes", ["NBC_2016", "Local_Bye_Laws"]),
            "compliance_score": f"{random.randint(85, 98)}%"
        }

    def _solution_multi(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        budget = problem["constraints"]["budget_inr"]
        return {
            "integrated_solution": {
                "budget_allocation": {"structure": 0.60, "finishes": 0.25, "mep": 0.10, "landscaping": 0.05},
                "timeline": {"phases": 3, "critical_path": ["Foundation", "Structure", "Finishes"]},
                "space_efficiency": 0.85
            },
            "tradeoffs": {"budget_vs_quality": "Premium_in_key_zones", "space_vs_function": "Multi_use_spaces"},
            "risks": {"budget_overrun": 0.10, "schedule_slip": 0.12, "mitigations": ["Early_procurement", "Parallel_works", "QC"]},
            "budget_summary": {"total_budget_inr": budget, "contingency_inr": int(budget * 0.1)}
        }

    def _solution_cost(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        area = problem["plot_area"]
        cb = problem["cost_breakdown"]
        baseline = area * (cb["foundation_per_sqft"] + cb["structure_per_sqft"] + cb["finishing_per_sqft"] + cb["mep_per_sqft"] + cb["landscaping_per_sqft"]) 
        optimized = int(baseline * 0.85)
        return {
            "costs": {"baseline_inr": int(baseline), "optimized_inr": optimized, "savings_inr": int(baseline - optimized), "savings_pct": 15},
            "strategies": ["Optimized_beam_sizing", "Modular_components", "Focus_finishes", "Value_engineering_MEP"],
            "quality": {"structural": "Maintained", "functionality": "Improved", "durability": "Maintained"}
        }

    def _solution_energy(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "energy_measures": {"orientation": "NS_optimal", "insulation": "High_perf", "glazing": "Low_E_DG", "hvac": "High_SEER", "renewables": "PV_30%"},
            "performance": {"annual_kwh_m2": 45, "savings_pct": 55, "carbon_reduction_pct": 60},
            "compliance": {"ecbc": "Exceeds", "green_rating": "5_star"}
        }

    def _solution_space(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "space_plan": {"efficiency_ratio": 0.90, "flexible_spaces_pct": 0.35, "built_in_storage_pct": 0.15, "circulation_efficiency": 0.85},
            "zones": {"private": "Bedrooms_ensuite", "social": "Open_LDK", "service": "Kitchen_utility", "flex": "Study_guest"}
        }

    def _solution_performance(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "targets": problem.get("targets", {}),
            "measures": {"envelope": "Optimized", "controls": "Smart_controls", "commissioning": "Cx_plan"},
            "validation": {"simulation": "OK", "monitoring": "Plan_ready"}
        }

    def _solution_advanced(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "strategy": {"success_factors": ["Structure", "Function", "Aesthetics"], "iterations": 3},
            "feasibility": {"economic": "CBA_positive", "environmental": "Mitigated"}
        }

    def _solution_math(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "calculations": problem["calculations"],
            "results_summary": {"roi": ">18%", "payback_years": 5},
            "validation": "Cross-checked"
        }

    def _solution_struct(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Use default values since structural problems don't have plot_details
        area = 2500  # Default area
        soil_type = random.choice(["Hard_Rock", "Soft_Rock", "Sandy_Soil", "Clay_Soil", "Mixed_Soil"])
        family = 4  # Default family size
        
        # Determine foundation type based on soil
        if soil_type in ["Hard_Rock", "Soft_Rock"]:
            foundation_type = "Isolated_Footing"
            foundation_depth = random.randint(3, 5)
        elif soil_type == "Sandy_Soil":
            foundation_type = "Strip_Footing"
            foundation_depth = random.randint(4, 6)
        elif soil_type == "Clay_Soil":
            foundation_type = "Raft_Foundation"
            foundation_depth = random.randint(5, 8)
        else:
            foundation_type = "Isolated_Footing"
            foundation_depth = random.randint(3, 6)
        
        # Calculate structural loads
        dead_load = area * 150  # kg/m2
        live_load = area * 200  # kg/m2
        total_load = dead_load + live_load
        
        # Determine seismic zone
        seismic_zones = ["Zone_II", "Zone_III", "Zone_IV", "Zone_V"]
        seismic_zone = random.choice(seismic_zones)
        
        return {
            "structural_design": {
                "foundation": {
                    "type": foundation_type,
                    "depth_m": foundation_depth,
                    "bearing_capacity": f"{random.randint(150, 300)} kN/m2",
                    "soil_type": soil_type
                },
                "superstructure": {
                    "system": "RCC_Frame_Structure",
                    "columns": f"{random.randint(8, 16)}_columns",
                    "beams": "Primary_and_secondary_beams",
                    "slab": "RCC_slab_150mm_thick"
                },
                "load_analysis": {
                    "dead_load_kg_m2": dead_load,
                    "live_load_kg_m2": live_load,
                    "total_load_kg_m2": total_load,
                    "wind_load_kg_m2": random.randint(50, 150)
                }
            },
            "seismic_design": {
                "zone": seismic_zone,
                "response_reduction_factor": random.uniform(3.0, 5.0),
                "importance_factor": 1.0,
                "ductility_detailing": "As_per_IS_13920",
                "base_shear": f"{random.randint(800, 1500)} kN"
            },
            "safety_factors": {
                "concrete_safety_factor": random.uniform(1.5, 2.0),
                "steel_safety_factor": random.uniform(1.15, 1.25),
                "load_combination_factor": random.uniform(1.2, 1.5)
            },
            "construction_details": {
                "concrete_grade": random.choice(["M20", "M25", "M30"]),
                "steel_grade": "Fe_415",
                "cover_thickness": "25mm_for_columns_20mm_for_beams",
                "joint_details": "As_per_IS_456_2000"
            },
            "quality_control": {
                "testing_frequency": "As_per_IS_456",
                "curing_period": "28_days",
                "strength_verification": "Cube_testing_required"
            }
        }

    def _solution_sustain(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Get sustainability goals and targets
        goals = problem.get("goals", ["Energy", "Water", "Materials", "IEQ", "Site", "Waste"])
        cert_targets = problem.get("cert_targets", ["LEED_Platinum", "GRIHA_5_Star", "IGBC_Platinum"])
        target_cert = random.choice(cert_targets)
        
        # Calculate energy performance
        energy_consumption = random.randint(30, 80)  # kWh/m2/year
        renewable_contribution = random.randint(40, 80)  # %
        carbon_reduction = random.randint(50, 85)  # %
        
        # Water conservation calculations
        water_consumption = random.randint(80, 150)  # liters/person/day
        rainwater_harvesting = random.randint(60, 90)  # % of roof area
        greywater_recycling = random.randint(30, 70)  # % of wastewater
        
        return {
            "sustainability_strategy": {
                "certification_target": target_cert,
                "overall_rating": f"{random.randint(85, 98)}%",
                "implementation_phases": ["Design", "Construction", "Operation", "Monitoring"]
            },
            "energy_efficiency": {
                "building_envelope": {
                    "insulation": "High_performance_thermal_insulation_R30",
                    "glazing": "Low_E_double_glazing_U_value_1.8",
                    "air_tightness": "Airtight_construction_ACH_0.6",
                    "thermal_bridge": "Minimized_thermal_bridges"
                },
                "hvac_systems": {
                    "system_type": "High_efficiency_VRV_with_heat_recovery",
                    "efficiency_rating": "SEER_18_plus",
                    "controls": "Smart_thermostats_with_occupancy_sensing",
                    "maintenance": "Regular_maintenance_schedule"
                },
                "renewable_energy": {
                    "solar_pv": f"{random.randint(5, 15)}_kW_system",
                    "solar_water_heating": "Evacuated_tube_collectors",
                    "contribution": f"{renewable_contribution}%_of_total_energy",
                    "battery_storage": "Optional_lithium_ion_battery_system"
                },
                "performance_metrics": {
                    "annual_energy_consumption": f"{energy_consumption}_kWh_m2_year",
                    "energy_savings": f"{random.randint(40, 70)}%_vs_baseline",
                    "carbon_reduction": f"{carbon_reduction}%_vs_conventional",
                    "payback_period": f"{random.randint(5, 12)}_years"
                }
            },
            "water_conservation": {
                "fixtures": {
                    "toilets": "Dual_flush_ultra_low_flow_4.8L",
                    "faucets": "Aerated_faucets_2.5L_min",
                    "showerheads": "Low_flow_showerheads_7.5L_min",
                    "urinals": "Waterless_urinals"
                },
                "rainwater_harvesting": {
                    "collection_area": f"{rainwater_harvesting}%_of_roof_area",
                    "storage_capacity": f"{random.randint(20, 50)}_kl_storage_tank",
                    "filtration": "Multi_stage_filtration_system",
                    "usage": "Landscaping_and_toilet_flushing"
                },
                "greywater_recycling": {
                    "collection": "Separate_greywater_plumbing",
                    "treatment": "Biological_treatment_system",
                    "recycling_rate": f"{greywater_recycling}%_of_wastewater",
                    "usage": "Landscaping_and_cooling_towers"
                },
                "performance_metrics": {
                    "water_consumption": f"{water_consumption}_liters_person_day",
                    "water_savings": f"{random.randint(30, 60)}%_vs_conventional",
                    "rainwater_utilization": f"{random.randint(40, 80)}%_of_landscape_water"
                }
            },
            "sustainable_materials": {
                "structural_materials": {
                    "concrete": "Low_carbon_concrete_with_30%_fly_ash",
                    "steel": "Recycled_steel_with_90%_recycled_content",
                    "timber": "FSC_certified_sustainable_timber",
                    "masonry": "Local_clay_bricks_with_low_embodied_energy"
                },
                "finishes": {
                    "flooring": "Bamboo_flooring_or_recycled_tiles",
                    "paints": "Low_VOC_water_based_paints",
                    "carpets": "Recycled_content_carpets",
                    "furniture": "FSC_certified_or_recycled_furniture"
                },
                "insulation": {
                    "wall_insulation": "Recycled_denim_or_cellulose_insulation",
                    "roof_insulation": "High_performance_rigid_insulation",
                    "acoustic_insulation": "Recycled_rubber_or_cork"
                }
            },
            "indoor_environmental_quality": {
                "ventilation": {
                    "natural_ventilation": "Cross_ventilation_design",
                    "mechanical_ventilation": "Energy_recovery_ventilation",
                    "air_filtration": "MERV_13_air_filters",
                    "fresh_air_rates": "ASHRAE_62.1_compliant"
                },
                "daylighting": {
                    "daylight_factor": f"{random.uniform(2.0, 4.0):.1f}%",
                    "glare_control": "Automated_shading_systems",
                    "light_shelves": "Reflective_light_shelves",
                    "skylights": "Tubular_skylights_for_deep_spaces"
                },
                "acoustic_comfort": {
                    "sound_absorption": "Acoustic_ceiling_tiles_and_wall_panels",
                    "sound_isolation": "STC_50_plus_wall_assemblies",
                    "background_noise": "NC_35_or_better"
                }
            },
            "site_sustainability": {
                "landscaping": {
                    "native_plants": "90%_native_drought_resistant_plants",
                    "irrigation": "Drip_irrigation_with_smart_controls",
                    "green_roof": "Extensive_green_roof_system",
                    "permeable_surfaces": "Permeable_paving_for_stormwater"
                },
                "stormwater_management": {
                    "retention": "On_site_stormwater_retention",
                    "infiltration": "Rain_gardens_and_bioswales",
                    "treatment": "Natural_filtration_systems",
                    "reuse": "Stormwater_harvesting_for_irrigation"
                },
                "heat_island_reduction": {
                    "cool_roofs": "High_albedo_roof_materials",
                    "shading": "Deciduous_trees_for_summer_shading",
                    "pavement": "Light_colored_permeable_pavement"
                }
            },
            "waste_management": {
                "construction_waste": {
                    "recycling_target": "90%_construction_waste_recycling",
                    "separation": "On_site_waste_separation_system",
                    "documentation": "Waste_management_plan_and_tracking"
                },
                "operational_waste": {
                    "recycling_program": "Comprehensive_recycling_program",
                    "composting": "On_site_composting_system",
                    "waste_reduction": "Waste_audit_and_reduction_strategies"
                }
            },
            "monitoring_and_verification": {
                "energy_monitoring": "Real_time_energy_monitoring_system",
                "water_monitoring": "Smart_water_metering_and_leak_detection",
                "indoor_air_quality": "Continuous_IAQ_monitoring",
                "performance_tracking": "Annual_sustainability_performance_reports"
            }
        }

    def _solution_smarthome(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Get smart systems requirements
        smart_systems = problem.get("smart_systems", ["Automation", "Security", "Energy", "HVAC", "Lighting", "Entertainment"])
        
        # Calculate system specifications
        total_devices = random.randint(25, 60)
        network_bandwidth = random.randint(100, 500)  # Mbps
        energy_savings = random.randint(15, 35)  # %
        security_level = random.choice(["Basic", "Advanced", "Premium"])
        
        return {
            "smart_home_strategy": {
                "overall_approach": "Integrated_ecosystem_with_centralized_control",
                "implementation_phases": ["Infrastructure", "Core_Systems", "Advanced_Features", "Integration"],
                "scalability": "Modular_design_for_future_expansion",
                "user_experience": "Intuitive_interface_with_voice_control"
            },
            "home_automation": {
                "lighting_control": {
                    "system_type": "Zigbee_based_smart_lighting",
                    "features": ["Dimmer_control", "Color_temperature_adjustment", "Motion_sensing", "Schedule_automation"],
                    "devices": f"{random.randint(15, 30)}_smart_bulbs_and_switches",
                    "integration": "Works_with_Google_Home_and_Amazon_Alexa"
                },
                "climate_control": {
                    "hvac_integration": "Smart_thermostat_with_zone_control",
                    "features": ["Temperature_scheduling", "Occupancy_detection", "Weather_integration", "Energy_optimization"],
                    "sensors": ["Temperature", "Humidity", "Air_quality", "Occupancy"],
                    "automation": "Automated_adjustment_based_on_occupancy_and_weather"
                },
                "entertainment_systems": {
                    "audio_distribution": "Multi_zone_audio_system",
                    "video_management": "Centralized_media_server",
                    "streaming_integration": "Smart_TV_with_streaming_apps",
                    "control": "Unified_remote_control_via_mobile_app"
                }
            },
            "security_systems": {
                "access_control": {
                    "entry_systems": ["Smart_locks", "Video_doorbell", "Keypad_entry"],
                    "authentication": ["PIN_codes", "Biometric_scanners", "Mobile_app_access"],
                    "monitoring": "24_7_remote_monitoring_service",
                    "notifications": "Instant_alerts_for_unauthorized_access"
                },
                "surveillance": {
                    "cameras": f"{random.randint(4, 12)}_HD_security_cameras",
                    "coverage": "Complete_property_coverage_with_night_vision",
                    "storage": "Cloud_and_local_storage_options",
                    "analytics": "AI_powered_motion_detection_and_face_recognition"
                },
                "intrusion_detection": {
                    "sensors": ["Door_window_sensors", "Motion_detectors", "Glass_break_sensors"],
                    "alarm_system": "Loud_siren_with_silent_alerts",
                    "integration": "Connected_to_local_police_station",
                    "backup": "Battery_backup_for_power_outages"
                }
            },
            "energy_management": {
                "smart_metering": {
                    "electricity_monitoring": "Real_time_electricity_consumption_tracking",
                    "water_monitoring": "Smart_water_meter_with_leak_detection",
                    "gas_monitoring": "Gas_consumption_tracking_and_safety_alerts",
                    "analytics": "Usage_patterns_and_optimization_recommendations"
                },
                "load_balancing": {
                    "peak_shaving": "Automatic_load_reduction_during_peak_hours",
                    "demand_response": "Integration_with_utility_demand_response_programs",
                    "battery_management": "Smart_battery_charging_and_discharging",
                    "efficiency": f"{energy_savings}%_energy_savings_through_optimization"
                },
                "renewable_integration": {
                    "solar_monitoring": "Real_time_solar_panel_performance_tracking",
                    "battery_storage": "Smart_battery_management_system",
                    "grid_interaction": "Bidirectional_power_flow_with_grid",
                    "optimization": "AI_optimized_energy_usage_patterns"
                }
            },
            "iot_infrastructure": {
                "network_setup": {
                    "wifi_system": f"WiFi_6_mesh_network_with_{network_bandwidth}_Mbps_bandwidth",
                    "zigbee_network": "Zigbee_3.0_mesh_network_for_smart_devices",
                    "bluetooth_mesh": "Bluetooth_LE_mesh_for_proximity_devices",
                    "cellular_backup": "4G_LTE_backup_connection"
                },
                "connectivity": {
                    "device_count": f"Supports_up_to_{total_devices}_smart_devices",
                    "protocols": ["WiFi", "Zigbee", "Z_Wave", "Bluetooth_LE", "Thread"],
                    "interoperability": "Cross_platform_device_compatibility",
                    "reliability": "99.9%_uptime_with_redundant_connections"
                },
                "data_management": {
                    "local_storage": "On_premise_data_storage_for_privacy",
                    "cloud_backup": "Encrypted_cloud_backup_service",
                    "edge_computing": "Local_processing_for_fast_response",
                    "analytics": "AI_powered_data_analytics_and_insights"
                }
            },
            "data_security": {
                "encryption": {
                    "data_transmission": "End_to_end_encryption_for_all_communications",
                    "data_storage": "AES_256_encryption_for_stored_data",
                    "authentication": "Multi_factor_authentication_for_access",
                    "privacy": "GDPR_compliant_data_handling"
                },
                "access_control": {
                    "user_management": "Role_based_access_control_system",
                    "device_authentication": "Secure_device_registration_and_authentication",
                    "network_security": "Firewall_and_intrusion_detection_system",
                    "updates": "Automatic_security_updates_and_patches"
                },
                "privacy_protection": {
                    "data_minimization": "Collect_only_necessary_data",
                    "user_consent": "Explicit_consent_for_data_collection",
                    "data_retention": "Configurable_data_retention_policies",
                    "rights_management": "User_rights_to_access_and_delete_data"
                }
            },
            "user_interfaces": {
                "mobile_app": {
                    "platforms": ["iOS", "Android"],
                    "features": ["Remote_control", "Real_time_monitoring", "Automation_setup", "Notifications"],
                    "usability": "Intuitive_interface_with_voice_commands",
                    "accessibility": "Accessibility_features_for_disabled_users"
                },
                "voice_control": {
                    "assistants": ["Amazon_Alexa", "Google_Assistant", "Apple_Siri"],
                    "commands": "Natural_language_voice_commands",
                    "customization": "Custom_voice_commands_and_routines",
                    "multi_language": "Support_for_multiple_languages"
                },
                "web_interface": {
                    "dashboard": "Comprehensive_web_dashboard_for_management",
                    "analytics": "Detailed_analytics_and_reporting",
                    "configuration": "Advanced_configuration_and_customization",
                    "remote_access": "Secure_remote_access_from_any_location"
                }
            },
            "maintenance_and_support": {
                "monitoring": {
                    "system_health": "Continuous_system_health_monitoring",
                    "predictive_maintenance": "AI_powered_predictive_maintenance_alerts",
                    "performance_tracking": "Real_time_performance_metrics",
                    "troubleshooting": "Automated_troubleshooting_and_diagnostics"
                },
                "support_system": {
                    "technical_support": "24_7_technical_support_service",
                    "remote_assistance": "Remote_troubleshooting_and_repair",
                    "warranty": "Extended_warranty_coverage_for_all_components",
                    "upgrades": "Regular_software_updates_and_feature_upgrades"
                },
                "documentation": {
                    "user_manuals": "Comprehensive_user_manuals_and_guides",
                    "installation_guides": "Detailed_installation_and_setup_guides",
                    "troubleshooting_guides": "Step_by_step_troubleshooting_procedures",
                    "video_tutorials": "Video_tutorials_for_all_features"
                }
            }
        }

    def _solution_conflict_resolution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        conflicts = problem.get("conflicts", [])
        return {
            "resolution": {
                "approach": "Compromise_design_with_value_engineering",
                "stakeholder_alignment": 0.9,
                "implementation": "Phased_with_regular_reviews"
            },
            "strategies": {
                "client_vs_code": "Design_mods_within_regulations",
                "budget_vs_quality": "Value_engineering_focused",
                "aesthetic_vs_function": "Integrated_design"
            },
            "risk_mitigation": {
                "communication_plan": "Weekly_stakeholder_updates",
                "change_management": "Documented_approval_process",
                "quality_control": "Regular_inspections_and_reviews"
            }
        }

    def _solution_advanced(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "strategy": {"success_factors": ["Structure", "Function", "Aesthetics"], "iterations": 3},
            "feasibility": {"economic": "CBA_positive", "environmental": "Mitigated"}
        }

    def _solution_math(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "calculations": problem["calculations"],
            "results_summary": {"roi": ">18%", "payback_years": 5},
            "validation": "Cross-checked"
        }

    def _solution_struct(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Use default values since structural problems don't have plot_details
        area = 2500  # Default area
        soil_type = random.choice(["Hard_Rock", "Soft_Rock", "Sandy_Soil", "Clay_Soil", "Mixed_Soil"])
        family = 4  # Default family size
        
        # Determine foundation type based on soil
        if soil_type in ["Hard_Rock", "Soft_Rock"]:
            foundation_type = "Isolated_Footing"
            foundation_depth = random.randint(3, 5)
        elif soil_type == "Sandy_Soil":
            foundation_type = "Strip_Footing"
            foundation_depth = random.randint(4, 6)
        elif soil_type == "Clay_Soil":
            foundation_type = "Raft_Foundation"
            foundation_depth = random.randint(5, 8)
        else:
            foundation_type = "Isolated_Footing"
            foundation_depth = random.randint(3, 6)
        
        # Calculate structural loads
        dead_load = area * 150  # kg/m2
        live_load = area * 200  # kg/m2
        total_load = dead_load + live_load
        
        # Determine seismic zone
        seismic_zones = ["Zone_II", "Zone_III", "Zone_IV", "Zone_V"]
        seismic_zone = random.choice(seismic_zones)
        
        return {
            "structural_design": {
                "foundation": {
                    "type": foundation_type,
                    "depth_m": foundation_depth,
                    "bearing_capacity": f"{random.randint(150, 300)} kN/m2",
                    "soil_type": soil_type
                },
                "superstructure": {
                    "system": "RCC_Frame_Structure",
                    "columns": f"{random.randint(8, 16)}_columns",
                    "beams": "Primary_and_secondary_beams",
                    "slab": "RCC_slab_150mm_thick"
                },
                "load_analysis": {
                    "dead_load_kg_m2": dead_load,
                    "live_load_kg_m2": live_load,
                    "total_load_kg_m2": total_load,
                    "wind_load_kg_m2": random.randint(50, 150)
                }
            },
            "seismic_design": {
                "zone": seismic_zone,
                "response_reduction_factor": random.uniform(3.0, 5.0),
                "importance_factor": 1.0,
                "ductility_detailing": "As_per_IS_13920",
                "base_shear": f"{random.randint(800, 1500)} kN"
            },
            "safety_factors": {
                "concrete_safety_factor": random.uniform(1.5, 2.0),
                "steel_safety_factor": random.uniform(1.15, 1.25),
                "load_combination_factor": random.uniform(1.2, 1.5)
            },
            "construction_details": {
                "concrete_grade": random.choice(["M20", "M25", "M30"]),
                "steel_grade": "Fe_415",
                "cover_thickness": "25mm_for_columns_20mm_for_beams",
                "joint_details": "As_per_IS_456_2000"
            },
            "quality_control": {
                "testing_frequency": "As_per_IS_456",
                "curing_period": "28_days",
                "strength_verification": "Cube_testing_required"
            }
        }

    def _solution_sustain(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "measures": {"energy": "Efficient_envelope", "water": "RWH+low_flow", "materials": "Low_impact"},
            "certification": {"target": random.choice(problem["cert_targets"])},
            "monitoring": "Plan_ready"
        }

    def _solution_performance(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "targets": problem.get("targets", {}),
            "measures": {"envelope": "Optimized", "controls": "Smart_controls", "commissioning": "Cx_plan"},
            "validation": {"simulation": "OK", "monitoring": "Plan_ready"}
        }

    def _generate_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        pt = problem["problem_type"]
        if pt == "Basic_Design": return self._solution_basic(problem)
        if pt == "Code_Compliance": return self._solution_code(problem)
        if pt == "Multi_Constraint": return self._solution_multi(problem)
        if pt == "Optimization":
            # branch by embedded type in fields
            if "cost_breakdown" in problem: return self._solution_cost(problem)
            if "climate_zone" in problem: return self._solution_energy(problem)
            if "total_area" in problem: return self._solution_space(problem)
            return self._solution_performance(problem)
        if pt == "Conflict_Resolution": return self._solution_conflict_resolution(problem)
        if pt == "Advanced_Reasoning": return self._solution_advanced(problem)
        if pt == "Mathematical_Analysis": return self._solution_math(problem)
        if pt == "Structural_Engineering": return self._solution_struct(problem)
        if pt == "Sustainability_Design": return self._solution_sustain(problem)
        if pt == "Smart_Home_Integration": return self._solution_smarthome(problem)
        return {"note": "No solution mapping"}

    # ---------- quality ----------
    def _quality_score(self, sample: Dict[str, Any]) -> float:
        score = 0.0
        checks = 0
        inp = sample["input"]
        out = sample["output"]
        
        # Basic checks that should always pass for synthetic data
        checks += 1
        if inp.get("problem_type") and out:
            score += 1.0
        
        # Output length check (using config value)
        checks += 1
        out_len = len(json.dumps(out))
        if self.config.min_output_chars <= out_len <= self.config.max_output_chars:
            score += 1.0
        
        # Reasoning steps check (using config value)
        checks += 1
        steps = inp.get("reasoning_steps", [])
        if isinstance(steps, list) and len(steps) >= self.config.min_reasoning_steps:
            score += 1.0
        
        # India-specific validation (only when applicable)
        if inp.get("context", {}).get("indian_market"):
            checks += 1
            if inp["context"].get("region") and inp["context"].get("climate_zone") in self.config.climate_zones:
                score += 1.0
        
        return score / max(1, checks)

    # ---------- io ----------
    def _ensure_dirs(self, base: Path, shard_id: int) -> Dict[str, Path]:
        shard = f"shard_{shard_id:02d}"
        train = base / "train" / shard
        val = base / "validation" / shard
        train.mkdir(parents=True, exist_ok=True)
        val.mkdir(parents=True, exist_ok=True)
        return {"train": train, "val": val}

    def generate(self, output_dir: str):
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        shard_id = 1
        dirs = self._ensure_dirs(base, shard_id)

        print(f"🎯 Target: {self.config.target_samples:,} samples | India: {int(self.config.india_ratio*100)}% | Q>= {self.config.quality_threshold}")
        print(f"📊 Problem types: {len(self.config.complexity_levels)} | Building codes: {len(self.config.building_codes)}")
        print(f"🌍 Indian regions: {len(self.config.regions_india)} | Climate zones: {len(self.config.climate_zones)}")
        
        pbar = tqdm(total=self.config.target_samples, desc="Generating super-quality samples")
        start_time = datetime.now()
        
        try:
            while self.accepted < self.config.target_samples:
                problem = self._generate_problem()
                solution = self._generate_solution(problem)
                sample = {
                    "input": problem, 
                    "output": solution, 
                    "metadata": {
                        "generated_at": datetime.now().isoformat(), 
                        "version": "R1_Super_v1.0", 
                        "sample_id": f"R1-SUP-{self.generated:07d}",
                        "problem_type": problem.get("problem_type", "Unknown")
                    }
                }
                self.generated += 1
                
                # Deduplication
                h = self._hash(sample)
                if h in self.uniques:
                    continue
                
                # Quality gate
                q = self._quality_score(sample)
                if q < self.config.quality_threshold:
                    continue
                
                # Accept and save
                self.uniques.add(h)
                subset = dirs["train"] if random.random() < self.config.train_ratio else dirs["val"]
                with open(subset / f"sample_{self.accepted:07d}.json", "w") as f:
                    json.dump(sample, f, indent=2)
                self.accepted += 1
                pbar.update(1)
                
                # Rotate shard
                if self.accepted % self.config.shard_size == 0:
                    shard_id += 1
                    dirs = self._ensure_dirs(base, shard_id)
                
                # Progress updates
                if self.accepted % 10000 == 0:
                    rate = self.accepted / max(1, self.generated)
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    eta = (elapsed / self.accepted) * (self.config.target_samples - self.accepted) if self.accepted > 0 else 0
                    print(f"✅ Accepted {self.accepted:,}/{self.config.target_samples:,} | Rate: {rate:.2%} | Elapsed: {elapsed:.1f}h | ETA: {eta:.1f}h")
                    
        except KeyboardInterrupt:
            print(f"\n⚠️ Generation interrupted. Saved {self.accepted:,} samples.")
        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            raise
        finally:
            pbar.close()
        info = {"name": "housebrain_dataset_r1_super_1M", "version": "R1_Super_v1.0", "total_samples": self.accepted, "train_ratio": self.config.train_ratio, "india_ratio": self.config.india_ratio, "quality_threshold": self.config.quality_threshold, "shard_size": self.config.shard_size, "min_reasoning_steps": self.config.min_reasoning_steps}
        with open(base / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print("\n🎉 Super-quality dataset generation complete!")
        print(f"📊 Total generated: {self.generated:,}")
        print(f"✅ Accepted: {self.accepted:,}")
        print(f"📁 Saved to: {base}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate 1M super-quality reasoning dataset for HouseBrain")
    p.add_argument("--output", type=str, default="housebrain_dataset_r1_super_1M")
    p.add_argument("--target", type=int, default=1_000_000)
    p.add_argument("--quality", type=float, default=0.90)
    p.add_argument("--india", type=float, default=0.40)
    p.add_argument("--shard", type=int, default=100_000)
    a = p.parse_args()
    cfg = SuperQualityConfig(target_samples=a.target, quality_threshold=a.quality, india_ratio=a.india, shard_size=a.shard)
    gen = SuperQualityGenerator(cfg)
    gen.generate(a.output)

if __name__ == "__main__":
    main()
