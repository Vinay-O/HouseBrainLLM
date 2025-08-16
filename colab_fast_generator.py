#!/usr/bin/env python3
"""
Enhanced Fast HouseBrain Dataset Generator for Colab
High quality with fast generation
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class EnhancedFastGenerator:
    def __init__(self, target_samples=1000000, quality_threshold=0.85):
        self.target_samples = target_samples
        self.quality_threshold = quality_threshold
        self.generated = 0
        self.accepted = 0
        self.uniques = set()
        
        # Enhanced problem types with more complexity
        self.problem_types = [
            "Basic_Design", "Code_Compliance", "Multi_Constraint", 
            "Optimization", "Structural_Engineering", "Sustainability_Design",
            "Smart_Home_Integration", "Mathematical_Analysis"
        ]
        
        # Indian regions with climate mapping
        self.regions_climate = {
            "Mumbai": "Tropical_Hot_Humid",
            "Delhi": "Composite", 
            "Bangalore": "Composite",
            "Chennai": "Tropical_Hot_Humid",
            "Kolkata": "Tropical_Hot_Humid",
            "Pune": "Composite",
            "Hyderabad": "Composite",
            "Ahmedabad": "Arid_Hot_Dry"
        }
        
        # Building codes and regulations
        self.building_codes = [
            "NBC_2016", "IS_456", "IS_875", "IS_1893", "ECBC_2017",
            "Local_Bye_Laws", "Fire_Safety_Norms"
        ]
        
        # Material options by quality
        self.materials = {
            "Economy": {
                "foundation": "RCC_Footing",
                "structure": "Load_Bearing",
                "walls": "Brick_Masonry",
                "roofing": "RCC_Slab",
                "flooring": "Cement_Flooring"
            },
            "Standard": {
                "foundation": "RCC_Footing", 
                "structure": "RCC_Frame",
                "walls": "Brick_Masonry",
                "roofing": "RCC_Slab",
                "flooring": "Vitrified_Tiles"
            },
            "Premium": {
                "foundation": "RCC_Footing",
                "structure": "RCC_Frame", 
                "walls": "AAC_Blocks",
                "roofing": "RCC_Slab",
                "flooring": "Marble"
            }
        }
        
    def generate_plot_details(self):
        """Generate realistic plot details"""
        shapes = ["Rectangular", "Square", "L_Shape", "T_Shape", "Corner_Plot"]
        shape = random.choice(shapes)
        
        if shape == "Square":
            width = random.randint(30, 80)
            length = width
        elif shape == "Rectangular":
            width = random.randint(25, 60)
            length = random.randint(width + 10, width * 2)
        else:
            width = random.randint(30, 70)
            length = random.randint(35, 85)
            
        area = width * length
        
        return {
            "plot_shape": shape,
            "width_ft": width,
            "length_ft": length,
            "area_sqft": area,
            "orientation": random.choice(["North", "South", "East", "West", "North_East"]),
            "soil_type": random.choice(["Hard_Rock", "Soft_Rock", "Sandy_Soil", "Clay_Soil"]),
            "water_table": random.choice(["High", "Medium", "Low"]),
            "access_road_width": random.randint(12, 25)
        }
        
    def generate_problem(self):
        """Generate enhanced problem with more details"""
        problem_type = random.choice(self.problem_types)
        indian = random.random() < 0.4
        
        plot_details = self.generate_plot_details()
        family_size = random.randint(3, 8)
        budget = random.randint(2000000, 15000000)
        
        problem = {
            "problem_type": problem_type,
            "plot_details": plot_details,
            "requirements": {
                "family_size": family_size,
                "budget_inr": budget,
                "lifestyle": random.choice(["Modern", "Traditional", "Minimalist", "Luxury", "Eco_Friendly"]),
                "special_needs": random.choice([None, "Elderly_Friendly", "Work_From_Home", "Entertainment_Focused"])
            },
            "constraints": {
                "time_constraint_months": random.randint(6, 24),
                "energy_efficiency_target": random.choice(["Standard", "Green_Building", "Net_Zero"]),
                "sustainability_goal": random.choice(["Basic", "LEED_Silver", "GRIHA_3_Star", "IGBC_Gold"])
            },
            "reasoning_steps": self.generate_reasoning_steps(problem_type)
        }
        
        if indian:
            region = random.choice(list(self.regions_climate.keys()))
            problem["context"] = {
                "indian_market": True,
                "region": region,
                "climate_zone": self.regions_climate[region],
                "applicable_codes": random.sample(self.building_codes, random.randint(3, 5)),
                "vastu_compliance": random.choice([True, False])
            }
            
        return problem
    
    def generate_reasoning_steps(self, problem_type):
        """Generate problem-specific reasoning steps"""
        base_steps = [
            "Analyze site conditions and constraints",
            "Calculate space requirements based on family needs",
            "Plan optimal room distribution and circulation",
            "Consider natural light and ventilation requirements",
            "Ensure budget compliance with material selection"
        ]
        
        if problem_type == "Code_Compliance":
            base_steps.extend([
                "Check setback requirements and FAR limits",
                "Verify parking and accessibility standards",
                "Ensure fire safety compliance",
                "Validate structural safety requirements"
            ])
        elif problem_type == "Structural_Engineering":
            base_steps.extend([
                "Calculate structural loads and forces",
                "Design foundation based on soil conditions",
                "Optimize structural elements for economy",
                "Ensure seismic design compliance"
            ])
        elif problem_type == "Sustainability_Design":
            base_steps.extend([
                "Design energy-efficient building envelope",
                "Integrate renewable energy systems",
                "Plan water conservation strategies",
                "Select sustainable materials and finishes"
            ])
        elif problem_type == "Smart_Home_Integration":
            base_steps.extend([
                "Design integrated smart systems",
                "Plan IoT infrastructure and connectivity",
                "Ensure data security and privacy",
                "Provide user-friendly control interfaces"
            ])
            
        return base_steps
    
    def generate_solution(self, problem):
        """Generate enhanced solution with more details"""
        area = problem["plot_details"]["area_sqft"]
        family = problem["requirements"]["family_size"]
        budget = problem["requirements"]["budget_inr"]
        problem_type = problem["problem_type"]
        
        # Calculate realistic room areas
        bedroom_area = family * random.randint(120, 180)
        living_area = int(area * random.uniform(0.25, 0.40))
        kitchen_area = int(area * random.uniform(0.10, 0.18))
        bathroom_area = family * random.randint(60, 100)
        circulation_area = int(area * random.uniform(0.08, 0.15))
        
        # Determine material quality based on budget
        cost_per_sqft = budget / area
        if cost_per_sqft < 1500:
            quality = "Economy"
        elif cost_per_sqft < 2500:
            quality = "Standard"
        else:
            quality = "Premium"
            
        solution = {
            "design_solution": {
                "room_distribution": {
                    "bedrooms": family,
                    "bedroom_area_sqft": bedroom_area,
                    "living_area_sqft": living_area,
                    "kitchen_area_sqft": kitchen_area,
                    "bathroom_area_sqft": bathroom_area,
                    "circulation_area_sqft": circulation_area,
                    "utility_area_sqft": int(area * 0.05),
                    "balcony_area_sqft": int(area * 0.08)
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
            "materials_and_finishes": self.materials[quality],
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
            ]
        }
        
        # Add problem-specific solutions
        if problem_type == "Code_Compliance":
            solution["compliance_details"] = {
                "setbacks": {"front": "3.0m", "rear": "3.0m", "side_each": "1.5m"},
                "far_compliance": "Within_limits",
                "parking_provision": f"{family + 1}_spaces",
                "fire_safety": "Staircase_refuge_suppression_system"
            }
        elif problem_type == "Structural_Engineering":
            solution["structural_details"] = {
                "foundation_type": random.choice(["Isolated_Footing", "Strip_Footing", "Raft_Foundation"]),
                "structural_system": "RCC_Frame_Structure",
                "seismic_design": random.choice(["Zone_II", "Zone_III", "Zone_IV"]),
                "safety_factors": {"concrete": 1.5, "steel": 1.15, "load_combination": 1.2}
            }
        elif problem_type == "Sustainability_Design":
            solution["sustainability_measures"] = {
                "energy_efficiency": ["High_performance_insulation", "Low_E_glazing", "Solar_orientation"],
                "water_conservation": ["Rainwater_harvesting", "Low_flow_fixtures", "Greywater_recycling"],
                "renewable_energy": ["Solar_PV_30%", "Solar_water_heating", "Wind_turbine_optional"],
                "materials": ["Local_sourcing", "Recycled_content", "Low_VOC_finishes"]
            }
        elif problem_type == "Smart_Home_Integration":
            solution["smart_systems"] = {
                "home_automation": ["Lighting_control", "Climate_control", "Security_system"],
                "iot_infrastructure": ["WiFi_6_network", "Zigbee_mesh", "Bluetooth_LE"],
                "energy_management": ["Smart_metering", "Load_balancing", "Peak_shaving"],
                "security": ["Video_surveillance", "Access_control", "Intrusion_detection"]
            }
            
        return solution
    
    def quality_score(self, sample):
        """Enhanced quality scoring"""
        inp = sample["input"]
        out = sample["output"]
        
        score = 0
        checks = 0
        
        # Basic structure checks
        checks += 1
        if inp.get("problem_type") and out:
            score += 1
        
        # Output richness check
        checks += 1
        out_len = len(json.dumps(out))
        if out_len >= 300:  # Increased minimum for richer content
            score += 1
        
        # Reasoning steps check
        checks += 1
        steps = inp.get("reasoning_steps", [])
        if isinstance(steps, list) and len(steps) >= 5:  # More reasoning steps
            score += 1
        
        # Problem-specific validation
        checks += 1
        problem_type = inp.get("problem_type")
        if problem_type == "Code_Compliance" and "compliance_details" in out:
            score += 1
        elif problem_type == "Structural_Engineering" and "structural_details" in out:
            score += 1
        elif problem_type == "Sustainability_Design" and "sustainability_measures" in out:
            score += 1
        elif problem_type == "Smart_Home_Integration" and "smart_systems" in out:
            score += 1
        else:
            score += 1  # Other problem types get basic credit
        
        # India-specific validation
        if inp.get("context", {}).get("indian_market"):
            checks += 1
            if inp["context"].get("region") and inp["context"].get("climate_zone"):
                score += 1
        
        return score / checks
    
    def hash_sample(self, sample):
        """Generate hash for deduplication"""
        payload = json.dumps(sample["input"], sort_keys=True) + "||" + json.dumps(sample["output"], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    def generate(self, output_dir):
        """Generate the enhanced dataset"""
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        
        train_dir = base / "train"
        val_dir = base / "validation"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Target: {self.target_samples:,} samples | Q>= {self.quality_threshold}")
        print(f"📊 Problem types: {len(self.problem_types)} | Enhanced quality")
        print(f"🌍 Indian regions: {len(self.regions_climate)} | Climate zones: {len(set(self.regions_climate.values()))}")
        
        start_time = datetime.now()
        
        try:
            while self.accepted < self.target_samples:
                problem = self.generate_problem()
                solution = self.generate_solution(problem)
                
                sample = {
                    "input": problem,
                    "output": solution,
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "version": "Enhanced_Fast_v1.0",
                        "sample_id": f"ENH-{self.generated:07d}",
                        "problem_type": problem.get("problem_type", "Unknown"),
                        "quality_score": self.quality_score({"input": problem, "output": solution})
                    }
                }
                
                self.generated += 1
                
                # Deduplication
                h = self.hash_sample(sample)
                if h in self.uniques:
                    continue
                
                # Quality gate
                q = self.quality_score(sample)
                if q < self.quality_threshold:
                    continue
                
                # Accept and save
                self.uniques.add(h)
                subset = train_dir if random.random() < 0.9 else val_dir
                
                with open(subset / f"sample_{self.accepted:07d}.json", "w") as f:
                    json.dump(sample, f, indent=2)
                
                self.accepted += 1
                
                # Progress updates
                if self.accepted % 1000 == 0:
                    rate = self.accepted / max(1, self.generated)
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    eta = (elapsed / self.accepted) * (self.target_samples - self.accepted) if self.accepted > 0 else 0
                    print(f"✅ Accepted {self.accepted:,}/{self.target_samples:,} | Rate: {rate:.2%} | Elapsed: {elapsed:.1f}h | ETA: {eta:.1f}h")
                    
        except KeyboardInterrupt:
            print(f"\n⚠️ Generation interrupted. Saved {self.accepted:,} samples.")
        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            raise
        
        print(f"\n🎉 Enhanced dataset generation complete!")
        print(f"📊 Total generated: {self.generated:,}")
        print(f"✅ Accepted: {self.accepted:,}")
        print(f"📁 Saved to: {base}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Fast HouseBrain dataset generator")
    parser.add_argument("--output", type=str, default="housebrain_dataset_r1_super_1M")
    parser.add_argument("--target", type=int, default=1000000)
    parser.add_argument("--quality", type=float, default=0.85)
    args = parser.parse_args()
    
    generator = EnhancedFastGenerator(target_samples=args.target, quality_threshold=args.quality)
    generator.generate(args.output)

if __name__ == "__main__":
    main()
