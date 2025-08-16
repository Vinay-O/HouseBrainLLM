#!/usr/bin/env python3
"""
R1-Enhanced Dataset Generation for HouseBrain
Optimized for DeepSeek-R1-Distill-Qwen-7B reasoning capabilities
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

@dataclass
class R1EnhancedDatasetConfig:
    target_samples: int = 100000
    quality_threshold: float = 0.85
    
    complexity_levels: List[str] = field(default_factory=lambda: [
        "Basic_Design", "Code_Compliance", "Multi_Constraint", "Optimization", "Conflict_Resolution"
    ])
    
    reasoning_tasks: List[str] = field(default_factory=lambda: [
        "Step_By_Step_Analysis", "Code_Interpretation", "Constraint_Balancing", 
        "Cost_Optimization", "Energy_Efficiency", "Structural_Validation"
    ])

class R1EnhancedHouseBrainDatasetGenerator:
    def __init__(self, config: R1EnhancedDatasetConfig):
        self.config = config
        self.generated_samples = 0
        
    def generate_complex_problem_scenario(self) -> Dict[str, Any]:
        complexity = random.choice(self.config.complexity_levels)
        
        if complexity == "Basic_Design":
            return self._generate_basic_design_problem()
        elif complexity == "Code_Compliance":
            return self._generate_code_compliance_problem()
        elif complexity == "Multi_Constraint":
            return self._generate_multi_constraint_problem()
        elif complexity == "Optimization":
            return self._generate_optimization_problem()
        else:
            return self._generate_conflict_resolution_problem()
    
    def _generate_basic_design_problem(self) -> Dict[str, Any]:
        plot_area = random.randint(1000, 5000)
        family_size = random.randint(3, 8)
        budget = random.randint(2000000, 10000000)
        
        return {
            "problem_type": "Basic_Design",
            "plot_details": {
                "area_sqft": plot_area,
                "shape": random.choice(["Rectangular", "Square", "L_Shape"]),
                "orientation": random.choice(["North", "South", "East", "West"])
            },
            "requirements": {
                "family_size": family_size,
                "budget_inr": budget,
                "lifestyle": random.choice(["Modern", "Traditional", "Minimalist"])
            },
            "reasoning_steps": [
                "Calculate minimum room requirements based on family size",
                "Determine optimal room sizes and proportions",
                "Plan circulation and connectivity between spaces",
                "Consider natural light and ventilation requirements",
                "Ensure budget compliance with material selection"
            ]
        }
    
    def _generate_code_compliance_problem(self) -> Dict[str, Any]:
        return {
            "problem_type": "Code_Compliance",
            "location": {
                "region": random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai"]),
                "zone_type": random.choice(["Residential", "Commercial", "Mixed_Use"]),
                "plot_area": random.randint(2000, 10000)
            },
            "building_specs": {
                "type": random.choice(["Residential", "Commercial"]),
                "floors": random.randint(2, 8),
                "height_m": random.randint(6, 24),
                "total_area": random.randint(5000, 20000)
            },
            "compliance_challenges": [
                "Setback_violations", "Floor_area_ratio_exceeded", "Height_restrictions"
            ],
            "reasoning_steps": [
                "Identify applicable building codes for the region and building type",
                "Calculate setback requirements based on plot size and height",
                "Verify floor area ratio compliance with local regulations",
                "Check parking requirements and accessibility standards",
                "Propose solutions for any violations found"
            ]
        }
    
    def _generate_multi_constraint_problem(self) -> Dict[str, Any]:
        return {
            "problem_type": "Multi_Constraint",
            "constraints": {
                "budget_constraint": random.randint(3000000, 15000000),
                "time_constraint": random.randint(6, 18),
                "space_constraint": random.randint(1500, 8000),
                "energy_constraint": random.choice(["Net_Zero", "Green_Building", "Standard"])
            },
            "reasoning_steps": [
                "Prioritize constraints based on client requirements",
                "Analyze trade-offs between conflicting constraints",
                "Develop multiple design alternatives",
                "Evaluate cost-benefit analysis for each alternative",
                "Propose optimal solution balancing all constraints"
            ]
        }
    
    def _generate_optimization_problem(self) -> Dict[str, Any]:
        return {
            "problem_type": "Optimization",
            "optimization_type": random.choice(["Cost_Optimization", "Energy_Optimization", "Space_Optimization"]),
            "plot_area": random.randint(2000, 8000),
            "budget": random.randint(5000000, 20000000),
            "reasoning_steps": [
                "Calculate current costs and identify optimization opportunities",
                "Analyze trade-offs between cost and quality",
                "Develop optimization strategies",
                "Calculate potential savings and impact",
                "Provide implementation plan"
            ]
        }
    
    def _generate_conflict_resolution_problem(self) -> Dict[str, Any]:
        return {
            "problem_type": "Conflict_Resolution",
            "conflicts": [
                "Client_Requirements_vs_Code_Compliance",
                "Budget_vs_Quality_Expectations",
                "Aesthetic_Preferences_vs_Functional_Needs"
            ],
            "reasoning_steps": [
                "Identify root causes of each conflict",
                "Analyze stakeholder interests and priorities",
                "Develop multiple resolution strategies",
                "Evaluate impact of each resolution",
                "Propose integrated solution"
            ]
        }
    
    def generate_reasoning_output(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        problem_type = problem["problem_type"]
        
        if problem_type == "Basic_Design":
            return self._generate_basic_design_solution(problem)
        elif problem_type == "Code_Compliance":
            return self._generate_code_compliance_solution(problem)
        elif problem_type == "Multi_Constraint":
            return self._generate_multi_constraint_solution(problem)
        elif problem_type == "Optimization":
            return self._generate_optimization_solution(problem)
        else:
            return self._generate_conflict_resolution_solution(problem)
    
    def _generate_basic_design_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        plot_area = problem["plot_details"]["area_sqft"]
        family_size = problem["requirements"]["family_size"]
        
        bedroom_area = family_size * 150
        common_area = plot_area * 0.4
        service_area = plot_area * 0.2
        
        return {
            "design_solution": {
                "room_distribution": {
                    "bedrooms": family_size,
                    "bedroom_area_sqft": bedroom_area,
                    "living_area_sqft": common_area,
                    "kitchen_area_sqft": service_area * 0.6
                },
                "layout_principle": "Open_Plan_With_Private_Zones",
                "orientation_strategy": "North_South_Orientation_For_Natural_Light"
            },
            "reasoning_analysis": {
                "space_efficiency": "85%",
                "natural_light_optimization": "90%",
                "budget_compliance": "95%"
            }
        }
    
    def _generate_code_compliance_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "compliance_solution": {
                "setback_adjustments": {
                    "front_setback": "3.0m",
                    "rear_setback": "3.0m"
                },
                "floor_area_ratio": "2.5",
                "height_compliance": "Within_limits"
            },
            "modifications_required": [
                "Reduce building height by 0.5m",
                "Increase setback on north side by 0.5m"
            ]
        }
    
    def _generate_multi_constraint_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "integrated_solution": {
                "budget_optimization": {
                    "total_cost": problem["constraints"]["budget_constraint"],
                    "cost_distribution": {
                        "structure": "60%",
                        "finishes": "25%",
                        "mep": "10%"
                    }
                },
                "time_optimization": {
                    "construction_phases": "3_phases",
                    "critical_path": "Foundation_Structure_Finishes"
                }
            }
        }
    
    def _generate_optimization_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        plot_area = problem["plot_area"]
        budget = problem["budget"]
        
        optimized_cost = plot_area * 2800  # Optimized rate
        original_cost = plot_area * 3500
        
        return {
            "cost_optimization": {
                "original_cost": original_cost,
                "optimized_cost": optimized_cost,
                "savings": original_cost - optimized_cost,
                "savings_percentage": "20%"
            },
            "optimization_strategies": {
                "foundation": "RCC_footing_with_optimized_design",
                "structure": "Standard_RCC_frame_with_efficient_beam_sizing",
                "finishes": "Quality_finishes_in_high_impact_areas_only"
            }
        }
    
    def _generate_conflict_resolution_solution(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "conflict_resolution": {
                "primary_conflicts": "Resolved_through_compromise_design",
                "stakeholder_satisfaction": "90%_consensus_achieved"
            },
            "resolution_strategies": {
                "client_vs_code": "Modified_design_within_regulatory_framework",
                "budget_vs_quality": "Value_engineering_with_quality_focus"
            }
        }
    
    def calculate_quality_score(self, sample: Dict[str, Any]) -> float:
        score = 0.0
        total_checks = 0
        
        if len(sample["input"]["reasoning_steps"]) >= 5:
            score += 1.0
        total_checks += 1
        
        if "design_solution" in sample["output"] or "compliance_solution" in sample["output"]:
            score += 1.0
        total_checks += 1
        
        if any(keyword in str(sample["output"]) for keyword in ["calculation", "percentage", "optimization"]):
            score += 1.0
        total_checks += 1
        
        if len(str(sample["output"])) > 1000:
            score += 1.0
        total_checks += 1
        
        return score / total_checks
    
    def generate_sample(self) -> Dict[str, Any]:
        problem = self.generate_complex_problem_scenario()
        solution = self.generate_reasoning_output(problem)
        
        sample = {
            "input": problem,
            "output": solution,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "R1_v1.0",
                "complexity_level": problem["problem_type"],
                "reasoning_required": True,
                "sample_id": f"R1-ENH-{self.generated_samples:06d}"
            }
        }
        
        self.generated_samples += 1
        quality_score = self.calculate_quality_score(sample)
        sample["metadata"]["quality_score"] = quality_score
        
        return sample
    
    def generate_dataset(self, output_dir: str, num_samples: int):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_dir = output_path / "train"
        val_dir = output_path / "validation"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        print(f"🧠 Generating {num_samples} R1-enhanced reasoning samples...")
        
        generated = 0
        accepted = 0
        
        with tqdm(total=num_samples, desc="Generating R1 samples") as pbar:
            while accepted < num_samples:
                sample = self.generate_sample()
                generated += 1
                
                if sample["metadata"]["quality_score"] >= self.config.quality_threshold:
                    is_train = random.random() < 0.9
                    target_dir = train_dir if is_train else val_dir
                    
                    filename = f"r1_sample_{accepted:06d}.json"
                    with open(target_dir / filename, 'w') as f:
                        json.dump(sample, f, indent=2)
                    
                    accepted += 1
                    pbar.update(1)
                
                if generated % 50 == 0:
                    pbar.set_postfix({
                        "Generated": generated,
                        "Accepted": accepted,
                        "Acceptance_Rate": f"{accepted/generated*100:.1f}%"
                    })
        
        dataset_info = {
            "total_samples": accepted,
            "train_samples": len(list(train_dir.glob("*.json"))),
            "validation_samples": len(list(val_dir.glob("*.json"))),
            "quality_threshold": self.config.quality_threshold,
            "version": "R1_v1.0",
            "description": "R1-enhanced HouseBrain dataset with complex reasoning tasks",
            "model_optimized_for": "DeepSeek-R1-Distill-Qwen-7B"
        }
        
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n🧠 R1 Dataset generation complete!")
        print(f"📊 Total generated: {generated}")
        print(f"✅ Accepted: {accepted}")
        print(f"📈 Acceptance rate: {accepted/generated*100:.1f}%")

def main():
    config = R1EnhancedDatasetConfig()
    generator = R1EnhancedHouseBrainDatasetGenerator(config)
    
    output_dir = "housebrain_dataset_r1_enhanced"
    num_samples = config.target_samples
    
    print(f"🧠 Generating R1-enhanced dataset for DeepSeek-R1-Distill-Qwen-7B")
    print(f"📊 Target: {num_samples:,} high-quality reasoning samples")
    
    generator.generate_dataset(output_dir, num_samples)

if __name__ == "__main__":
    main()
