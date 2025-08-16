#!/usr/bin/env python3
"""
Generate 1M Reasoning-Optimized Dataset for HouseBrain (R1)
- Fresh dataset (keeps old datasets aside)
- Strict quality gates and validations
- Balanced India-focused scenarios
- Sharded saving for scalability
- Safe random ranges and consistent cost math
- Compatible with current training scripts (expects {"input": ..., "output": ...})
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Reasoning1MConfig:
    target_samples: int = 1_000_000
    quality_threshold: float = 0.90
    train_ratio: float = 0.90
    shard_size: int = 100_000              # Save in shards for scalability
    min_reasoning_steps: int = 5
    min_output_chars: int = 800
    max_output_chars: int = 12_000
    india_ratio: float = 0.40
    seed: int = 42

    regions_india: List[str] = field(default_factory=lambda: [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore"
    ])
    climate_zones: List[str] = field(default_factory=lambda: [
        "Tropical_Hot_Humid", "Tropical_Warm_Humid", "Subtropical_Hot_Dry",
        "Subtropical_Warm_Humid", "Composite", "Arid_Hot_Dry", "Arid_Warm_Dry"
    ])
    building_codes: List[str] = field(default_factory=lambda: [
        "NBC_2016", "IS_456", "IS_875", "IS_1893", "ECBC_2017", "Local_Byelaws"
    ])
    building_types: List[str] = field(default_factory=lambda: [
        "Residential", "Commercial", "Mixed_Use", "Institutional"
    ])
    plot_shapes: List[str] = field(default_factory=lambda: [
        "Rectangular", "Square", "L_Shape", "T_Shape", "Irregular", "Corner_Plot"
    ])
    reasoning_problem_types: List[str] = field(default_factory=lambda: [
        "Code_Compliance", "Multi_Constraint", "Cost_Optimization",
        "Energy_Optimization", "Space_Optimization", "Conflict_Resolution",
        "Basic_Design"
    ])


# ----------------------------
# Generator
# ----------------------------

class Reasoning1MDatasetGenerator:
    def __init__(self, config: Reasoning1MConfig):
        self.config = config
        random.seed(self.config.seed)
        self.generated: int = 0
        self.accepted: int = 0
        self.unique_hashes: set[str] = set()

    # --------- Helpers ---------
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

    def _hash_sample(self, sample: Dict[str, Any]) -> str:
        payload = json.dumps(sample["input"], sort_keys=True) + "||" + json.dumps(sample["output"], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # --------- Content Generators ---------
    def _generate_plot(self) -> Dict[str, Any]:
        shape = random.choice(self.config.plot_shapes)
        if shape == "Square":
            width = self._rand_range(28, 100)
            length = width
        elif shape == "Rectangular":
            width = self._rand_range(25, 80)
            length = self._rand_range(width + 5, min(width * 3, 200))
        else:
            width = self._rand_range(25, 95)
            length = self._rand_range(28, 110)
        area = width * length
        return {
            "shape": shape,
            "width_ft": width,
            "length_ft": length,
            "area_sqft": area,
            "orientation": random.choice([
                "North", "South", "East", "West", "North_East", "North_West", "South_East", "South_West"
            ])
        }

    def _generate_basic_design_problem(self, indian: bool) -> Dict[str, Any]:
        plot = self._generate_plot()
        family_size = self._rand_range(3, 8)
        budget = self._rand_range(2_000_000, 12_000_000)
        return {
            "problem_type": "Basic_Design",
            "context": {
                "indian_market": indian,
                "region": random.choice(self.config.regions_india) if indian else None,
                "climate_zone": random.choice(self.config.climate_zones) if indian else None,
                "building_type": random.choice(self.config.building_types)
            },
            "plot_details": plot,
            "requirements": {
                "family_size": family_size,
                "floors": self._rand_range(1, 4),
                "budget_inr": budget,
                "lifestyle": random.choice(["Modern", "Traditional", "Minimalist", "Luxury", "Eco_Friendly"])
            },
            "reasoning_steps": [
                "Calculate minimum room requirements based on family size",
                "Determine optimal room sizes and proportions",
                "Plan circulation and connectivity between spaces",
                "Consider natural light and cross ventilation",
                "Ensure budget compliance with material selection"
            ]
        }

    def _generate_code_compliance_problem(self, indian: bool) -> Dict[str, Any]:
        floors = self._rand_range(2, 8)
        return {
            "problem_type": "Code_Compliance",
            "context": {
                "indian_market": indian,
                "region": random.choice(self.config.regions_india) if indian else None,
                "climate_zone": random.choice(self.config.climate_zones) if indian else None,
                "building_type": random.choice(self.config.building_types)
            },
            "location": {
                "zone_type": random.choice(["Residential", "Commercial", "Mixed_Use"]),
                "plot_area_sqft": self._rand_range(1800, 12_000)
            },
            "building_specs": {
                "floors": floors,
                "height_m": round(floors * 3.0 + random.uniform(-0.4, 0.6), 2),
                "total_builtup_sqft": self._rand_range(3_000, 25_000)
            },
            "applicable_codes": self._safe_sample(self.config.building_codes, 3, 5),
            "compliance_challenges": self._safe_sample([
                "Setback_violations", "Floor_area_ratio_exceeded", "Height_restrictions",
                "Parking_shortage", "Fire_safety_requirements", "Structural_safety"
            ], 2, 4),
            "reasoning_steps": [
                "Identify applicable building codes",
                "Compute setbacks and FAR limits",
                "Check height and parking requirements",
                "Verify fire and structural safety",
                "Propose modifications for violations"
            ]
        }

    def _generate_multi_constraint_problem(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Multi_Constraint",
            "context": {
                "indian_market": indian,
                "region": random.choice(self.config.regions_india) if indian else None,
                "climate_zone": random.choice(self.config.climate_zones) if indian else None,
                "building_type": random.choice(self.config.building_types)
            },
            "constraints": {
                "budget_inr": self._rand_range(3_000_000, 20_000_000),
                "timeline_months": self._rand_range(6, 18),
                "area_limit_sqft": self._rand_range(1_500, 8_000),
                "energy_goal": random.choice(["Net_Zero", "Green_Building", "Standard"]),
                "aesthetic": random.choice(["Heritage_Compatible", "Modern", "Traditional"])
            },
            "reasoning_steps": [
                "Prioritize constraints based on client requirements",
                "Analyze trade-offs between conflicting constraints",
                "Develop alternatives and estimate impacts",
                "Quantify cost-benefit for each alternative",
                "Select optimal balanced solution"
            ]
        }

    def _generate_cost_optimization_problem(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Cost_Optimization",
            "context": {
                "indian_market": indian,
                "region": random.choice(self.config.regions_india) if indian else None,
                "building_type": random.choice(self.config.building_types)
            },
            "plot_area_sqft": self._rand_range(2_000, 10_000),
            "budget_inr": self._rand_range(5_000_000, 25_000_000),
            "cost_rates_per_sqft": {
                "foundation": self._rand_range(800, 1200),
                "structure": self._rand_range(1100, 1800),
                "finishes": self._rand_range(800, 1500),
                "mep": self._rand_range(400, 800),
                "landscaping": self._rand_range(200, 500)
            },
            "reasoning_steps": [
                "Compute baseline total construction cost",
                "Identify components for cost saving",
                "Evaluate trade-offs between cost and quality",
                "Quantify potential savings and risks",
                "Propose phased plan to stay within budget"
            ]
        }

    def _generate_energy_optimization_problem(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Energy_Optimization",
            "context": {
                "indian_market": indian,
                "region": random.choice(self.config.regions_india) if indian else None,
                "climate_zone": random.choice(self.config.climate_zones) if indian else None,
            },
            "energy_target": random.choice(["Net_Zero", "5_Star_Green", "ECBC_Compliant"]),
            "site_conditions": {
                "solar_exposure": random.choice(["High", "Medium", "Low"]),
                "wind_patterns": random.choice(["Prevailing", "Variable", "Calm"]),
                "vegetation": random.choice(["Dense", "Moderate", "Sparse"]) 
            },
            "reasoning_steps": [
                "Analyze climate and set energy targets",
                "Optimize orientation and shading",
                "Select insulation and glazing systems",
                "Design efficient HVAC and renewables",
                "Estimate performance and compliance"
            ]
        }

    def _generate_space_optimization_problem(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Space_Optimization",
            "context": {
                "indian_market": indian,
                "region": random.choice(self.config.regions_india) if indian else None,
                "building_type": random.choice(self.config.building_types)
            },
            "total_area_sqft": self._rand_range(1_500, 6_000),
            "family": {
                "adults": self._rand_range(2, 4),
                "children": self._rand_range(0, 3),
                "elderly": self._rand_range(0, 2)
            },
            "goals": self._safe_sample([
                "Maximize_Functionality", "Improve_Circulation", "Enhance_Privacy",
                "Optimize_Storage", "Create_Flexible_Spaces", "Improve_Natural_Light"
            ], 3, 6),
            "reasoning_steps": [
                "Calculate space requirements per function",
                "Analyze relationships and circulation",
                "Propose flexible multi-use areas",
                "Optimize storage and built-ins",
                "Plan for future adaptability"
            ]
        }

    def _generate_conflict_resolution_problem(self, indian: bool) -> Dict[str, Any]:
        return {
            "problem_type": "Conflict_Resolution",
            "context": {
                "indian_market": indian,
                "region": random.choice(self.config.regions_india) if indian else None,
            },
            "conflicts": self._safe_sample([
                "Client_Requirements_vs_Code_Compliance",
                "Budget_vs_Quality_Expectations",
                "Aesthetics_vs_Functional_Needs",
                "Timeline_vs_Customization",
                "Site_Constraints_vs_Design_Vision"
            ], 2, 4),
            "reasoning_steps": [
                "Identify root causes for each conflict",
                "Analyze stakeholder interests and priorities",
                "Develop and compare resolution options",
                "Select integrated solution and plan rollout",
                "Define monitoring and adjustments"
            ]
        }

    def _generate_problem(self) -> Dict[str, Any]:
        indian = random.random() < self.config.india_ratio
        ptype = random.choice(self.config.reasoning_problem_types)
        if ptype == "Basic_Design":
            return self._generate_basic_design_problem(indian)
        if ptype == "Code_Compliance":
            return self._generate_code_compliance_problem(indian)
        if ptype == "Multi_Constraint":
            return self._generate_multi_constraint_problem(indian)
        if ptype == "Cost_Optimization":
            return self._generate_cost_optimization_problem(indian)
        if ptype == "Energy_Optimization":
            return self._generate_energy_optimization_problem(indian)
        if ptype == "Space_Optimization":
            return self._generate_space_optimization_problem(indian)
        return self._generate_conflict_resolution_problem(indian)

    # --------- Solutions ---------
    def _solution_basic_design(self, problem: Dict[str, Any]) -> Dict[str, Any]:
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
                "layout_principle": "Open_Plan_with_Private_Zones",
                "orientation_strategy": "North_South_optimized_for_daylight",
                "ventilation_strategy": "Cross_ventilation_with_window_alignment"
            },
            "analysis": {
                "space_efficiency": ">=85%",
                "natural_light_score": ">=0.85",
                "budget_alignment": "Within_5_percent"
            },
            "implementation_steps": [
                "Site analysis and massing",
                "Room sizing and adjacency planning",
                "Circulation optimization",
                "Window/shading placement",
                "Material selection within budget"
            ]
        }

    def _solution_code_compliance(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "compliance_solution": {
                "setbacks": {
                    "front": "3.0m",
                    "rear": "3.0m",
                    "side_each": "1.5m"
                },
                "far_limit": "As_per_zone",
                "height_compliance": "Within_limits",
                "parking_provision": "As_per_code",
                "fire_safety": "Staircase, refuge, suppression per code"
            },
            "modifications": self._safe_sample([
                "Increase north setback by 0.5m",
                "Reduce building height by 0.5m",
                "Add two parking bays",
                "Upgrade fire suppression system"
            ], 2, 3),
            "verification": {
                "setback_check": "Passed",
                "far_check": "Passed",
                "height_check": "Passed",
                "parking_check": "Passed"
            }
        }

    def _solution_multi_constraint(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        budget = problem["constraints"]["budget_inr"]
        return {
            "integrated_solution": {
                "budget_allocation": {
                    "structure": 0.60,
                    "finishes": 0.25,
                    "mep": 0.10,
                    "landscaping": 0.05
                },
                "timeline": {
                    "phases": 3,
                    "critical_path": ["Foundation", "Structure", "Finishes"]
                },
                "space_efficiency": 0.85
            },
            "tradeoffs": {
                "budget_vs_quality": "Premium_in_high_impact_zones",
                "space_vs_function": "Multi_functional_rooms",
                "aesthetics_vs_energy": "Passive_design_with_modern_aesthetics"
            },
            "risks": {
                "budget_overrun": 0.10,
                "schedule_slip": 0.12,
                "mitigations": ["Early_procurement", "Parallel_works", "Quality_control"]
            },
            "budget_summary": {
                "total_budget_inr": budget,
                "contingency_inr": int(budget * 0.1)
            }
        }

    def _solution_cost_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        area = problem["plot_area_sqft"]
        rates = problem["cost_rates_per_sqft"]
        baseline = area * (rates["foundation"] + rates["structure"] + rates["finishes"] + rates["mep"] + rates["landscaping"]) 
        optimized = int(baseline * 0.85)
        return {
            "costs": {
                "baseline_inr": int(baseline),
                "optimized_inr": optimized,
                "savings_inr": int(baseline - optimized),
                "savings_pct": 15
            },
            "strategies": [
                "Optimized_beam_sizing",
                "Standardize_modular_components",
                "Focus_finishes_in_high_impact_areas",
                "Value_engineering_MEP"
            ],
            "quality": {
                "structural_integrity": "Maintained",
                "functionality": "Improved",
                "durability": "Maintained"
            }
        }

    def _solution_energy_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "energy_measures": {
                "orientation": "North_South_for_solar_control",
                "insulation": "High_performance_envelope",
                "glazing": "Low_E_double_glazed",
                "hvac": "High_SEER_split_systems",
                "renewables": "Solar_PV_30_percent"
            },
            "performance": {
                "annual_consumption_kwh_m2": 45,
                "savings_pct": 55,
                "carbon_reduction_pct": 60
            },
            "compliance": {
                "ecbc": "Exceeds",
                "green_rating": "5_star"
            }
        }

    def _solution_space_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "space_plan": {
                "efficiency_ratio": 0.90,
                "flexible_spaces_pct": 0.35,
                "built_in_storage_pct": 0.15,
                "circulation_efficiency": 0.85
            },
            "zones": {
                "private": "Bedrooms_with_ensuite",
                "social": "Open_living_dining_kitchen",
                "service": "Efficient_kitchen_utility",
                "flex": "Multi_purpose_study_guest"
            }
        }

    def _solution_conflict_resolution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
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
            }
        }

    def _generate_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        ptype = problem["problem_type"]
        if ptype == "Basic_Design":
            return self._solution_basic_design(problem)
        if ptype == "Code_Compliance":
            return self._solution_code_compliance(problem)
        if ptype == "Multi_Constraint":
            return self._solution_multi_constraint(problem)
        if ptype == "Cost_Optimization":
            return self._solution_cost_optimization(problem)
        if ptype == "Energy_Optimization":
            return self._solution_energy_optimization(problem)
        if ptype == "Space_Optimization":
            return self._solution_space_optimization(problem)
        return self._solution_conflict_resolution(problem)

    # --------- Quality Gates ---------
    def _quality_score(self, sample: Dict[str, Any]) -> float:
        score = 0.0
        checks = 0
        input_obj = sample["input"]
        output_obj = sample["output"]

        # Steps check
        steps = input_obj.get("reasoning_steps", [])
        if isinstance(steps, list) and len(steps) >= self.config.min_reasoning_steps:
            score += 1.0
        checks += 1

        # Output length check (proxy for richness)
        out_len = len(json.dumps(output_obj))
        if self.config.min_output_chars <= out_len <= self.config.max_output_chars:
            score += 1.0
        checks += 1

        # Problem-solution alignment presence
        if input_obj.get("problem_type") and len(output_obj.keys()) >= 1:
            score += 1.0
        checks += 1

        # India balance (no direct score, but ensure fields when indian)
        if input_obj.get("context", {}).get("indian_market"):
            if input_obj["context"].get("region") and input_obj["context"].get("climate_zone") in self.config.climate_zones:
                score += 1.0
        checks += 1

        # Numeric sanity for cost tasks
        if input_obj.get("problem_type") == "Cost_Optimization":
            costs = output_obj.get("costs", {})
            if costs.get("baseline_inr", 0) > costs.get("optimized_inr", 0) > 0:
                score += 1.0
        checks += 1

        return score / max(checks, 1)

    # --------- Save / Shard ---------
    def _ensure_dirs(self, base: Path, shard_id: int) -> Dict[str, Path]:
        shard_name = f"shard_{shard_id:02d}"
        train_dir = base / "train" / shard_name
        val_dir = base / "validation" / shard_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        return {"train": train_dir, "val": val_dir}

    # --------- Main generation ---------
    def generate(self, output_dir: str):
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        shard_id = 1
        dirs = self._ensure_dirs(base, shard_id)

        print(f"🎯 Target: {self.config.target_samples:,} samples")
        print(f"🧠 India ratio: {int(self.config.india_ratio*100)}% | Quality threshold: {self.config.quality_threshold}")

        pbar = tqdm(total=self.config.target_samples, desc="Generating reasoning samples")
        while self.accepted < self.config.target_samples:
            problem = self._generate_problem()
            solution = self._generate_solution(problem)
            sample = {
                "input": problem,
                "output": solution,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "version": "R1_1M_v1.0",
                    "sample_id": f"R1-1M-{self.generated:07d}"
                }
            }

            self.generated += 1

            # Deduplicate by hash
            h = self._hash_sample(sample)
            if h in self.unique_hashes:
                continue

            # Quality gate
            q = self._quality_score(sample)
            if q < self.config.quality_threshold:
                continue

            # Accept & save
            self.unique_hashes.add(h)
            is_train = random.random() < self.config.train_ratio
            subset_dir = dirs["train"] if is_train else dirs["val"]
            filename = f"sample_{self.accepted:07d}.json"
            with open(subset_dir / filename, "w") as f:
                json.dump(sample, f, indent=2)

            self.accepted += 1
            pbar.update(1)

            # Rotate shard
            if self.accepted % self.config.shard_size == 0:
                shard_id += 1
                dirs = self._ensure_dirs(base, shard_id)

            # Occasional stats
            if self.accepted % 10000 == 0:
                acc_rate = self.accepted / max(self.generated, 1)
                print(f"✅ Accepted {self.accepted:,}/{self.config.target_samples:,} | Acceptance rate: {acc_rate:.2%}")

        pbar.close()

        # Dataset info
        info = {
            "name": "housebrain_dataset_r1_1M",
            "version": "R1_1M_v1.0",
            "description": "1M reasoning-optimized synthetic dataset for HouseBrain (DeepSeek-R1-Distill-Qwen-7B)",
            "total_samples": self.accepted,
            "train_ratio": self.config.train_ratio,
            "india_ratio": self.config.india_ratio,
            "quality_threshold": self.config.quality_threshold,
            "shard_size": self.config.shard_size,
            "min_reasoning_steps": self.config.min_reasoning_steps,
            "min_output_chars": self.config.min_output_chars,
            "max_output_chars": self.config.max_output_chars,
            "problem_types": self.config.reasoning_problem_types,
        }
        with open(base / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print("\n🎉 Dataset generation complete!")
        print(f"📊 Total generated: {self.generated:,}")
        print(f"✅ Accepted: {self.accepted:,}")
        print(f"📁 Saved to: {base}")


# ----------------------------
# CLI
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate 1M reasoning dataset for HouseBrain")
    parser.add_argument("--output", type=str, default="housebrain_dataset_r1_1M", help="Output directory")
    parser.add_argument("--target", type=int, default=1_000_000, help="Target number of accepted samples")
    parser.add_argument("--quality", type=float, default=0.90, help="Quality threshold [0-1]")
    parser.add_argument("--india", type=float, default=0.40, help="India ratio [0-1]")
    parser.add_argument("--shard", type=int, default=100_000, help="Shard size")
    args = parser.parse_args()

    cfg = Reasoning1MConfig(
        target_samples=args.target,
        quality_threshold=args.quality,
        india_ratio=args.india,
        shard_size=args.shard,
    )
    gen = Reasoning1MDatasetGenerator(cfg)
    gen.generate(args.output)


if __name__ == "__main__":
    main()
