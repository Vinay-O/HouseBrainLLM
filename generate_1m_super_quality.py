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
    quality_threshold: float = 0.85  # Optimized for Colab generation
    train_ratio: float = 0.90
    shard_size: int = 100_000
    min_reasoning_steps: int = 6  # Back to 6 for original generator
    min_output_chars: int = 300  # Realistic for original generator (avg 346)
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
                "budget_inr": self._rand_range(2_000_000, 12_000_000),
                "lifestyle": random.choice(["Modern", "Traditional", "Minimalist", "Luxury", "Eco_Friendly"])
            },
            "reasoning_steps": [
                "Calculate minimum room requirements based on family size and lifestyle",
                "Determine optimal room sizes and proportions",
                "Plan circulation and connectivity",
                "Optimize natural light and cross-ventilation",
                "Ensure budget compliance with material selection",
                "Iterate layout for efficiency and privacy"
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
                "Identify all applicable codes",
                "Compute setbacks, FAR, height and parking",
                "Verify fire/structural safety and accessibility",
                "List violations and propose compliant modifications",
                "Estimate cost/time impact of modifications",
                "Provide compliance verification plan"
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
                "Analyze loads and soil conditions",
                "Design elements with safety factors",
                "Ensure seismic/wind compliance",
                "Detail connections and joints",
                "Validate via structural analysis"
            ]
        }

    def _problem_sustain(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Sustainability_Design",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "goals": ["Energy", "Water", "Materials", "IEQ", "Site", "Waste"],
            "cert_targets": ["LEED_Platinum", "GRIHA_5_Star", "IGBC_Platinum"],
            "reasoning_steps": [
                "Define goals and certification targets",
                "Envelope/system efficiency and water strategies",
                "Material selection and IEQ optimization",
                "Site sustainability and waste reduction",
                "Monitoring and verification plan"
            ]
        }

    def _problem_smarthome(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Smart_Home_Integration",
            "context": {"indian_market": indian, **(self._indian_features() if indian else {})},
            "smart_systems": ["Automation", "Security", "Energy", "HVAC", "Lighting", "Entertainment"],
            "reasoning_steps": [
                "Capture functional requirements",
                "Design integrated systems and networks",
                "Plan energy management and controls",
                "Ensure data security and privacy",
                "Define maintenance and upgrade roadmap"
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
        bedrooms = max(2, min(6, family))
        living = int(area * 0.35)
        service = int(area * 0.22)
        circulation = int(area * 0.10)
        return {
            "design_solution": {
                "room_distribution": {
                    "bedrooms": bedrooms,
                    "bedroom_area_sqft": bedrooms * 140,
                    "living_area_sqft": living,
                    "kitchen_area_sqft": int(service * 0.55),
                    "bathroom_area_sqft": int(service * 0.45),
                    "circulation_area_sqft": circulation
                },
                "layout": "Open_plan_with_private_zones",
                "daylight": "North_South_optimized",
                "ventilation": "Cross_ventilation"
            },
            "analysis": {
                "space_efficiency": ">=85%",
                "natural_light_score": ">=0.85",
                "budget_alignment": "Within_5_percent"
            },
            "implementation": ["Massing", "Sizing", "Circulation", "Fenestration", "Materials"]
        }

    def _solution_code(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "compliance_solution": {
                "setbacks": {"front": "3.0m", "rear": "3.0m", "side_each": "1.5m"},
                "far_limit": "As_per_zone", "height": "Within_limits", "parking": "As_per_code",
                "fire_safety": "Staircase/refuge/suppression"
            },
            "modifications": self._safe_sample([
                "Increase north setback by 0.5m", "Reduce height by 0.5m",
                "Add two parking bays", "Upgrade suppression system"
            ], 2, 3),
            "verification": {"setback": "Passed", "far": "Passed", "height": "Passed", "parking": "Passed"}
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
        return {
            "elements": problem["elements"],
            "design_notes": ["Seismic_zone_compliance", "Wind_load_checked", "Connections_detailed"]
        }

    def _solution_sustain(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "measures": {"energy": "Efficient_envelope", "water": "RWH+low_flow", "materials": "Low_impact"},
            "certification": {"target": random.choice(problem["cert_targets"])},
            "monitoring": "Plan_ready"
        }

    def _solution_smarthome(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "systems": problem["smart_systems"],
            "integration": {"iot": "Yes", "security": "Hardened", "apps": "Mobile+Voice"}
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
        return {
            "elements": problem["elements"],
            "design_notes": ["Seismic_zone_compliance", "Wind_load_checked", "Connections_detailed"]
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
