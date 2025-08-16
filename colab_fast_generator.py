#!/usr/bin/env python3
"""
Fast HouseBrain Dataset Generator for Colab
Optimized for speed and minimal dependencies
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class FastGenerator:
    def __init__(self, target_samples=1000000, quality_threshold=0.85):
        self.target_samples = target_samples
        self.quality_threshold = quality_threshold
        self.generated = 0
        self.accepted = 0
        self.uniques = set()
        
        # Problem types
        self.problem_types = [
            "Basic_Design", "Code_Compliance", "Multi_Constraint", 
            "Optimization", "Structural_Engineering", "Sustainability_Design"
        ]
        
        # Indian regions
        self.regions = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Pune"]
        
    def generate_problem(self):
        """Generate a simple problem"""
        problem_type = random.choice(self.problem_types)
        indian = random.random() < 0.4
        
        problem = {
            "problem_type": problem_type,
            "plot_details": {
                "area_sqft": random.randint(1000, 5000),
                "shape": random.choice(["Rectangular", "Square", "L_Shape"]),
                "orientation": random.choice(["North", "South", "East", "West"])
            },
            "requirements": {
                "family_size": random.randint(3, 8),
                "budget_inr": random.randint(2000000, 10000000),
                "lifestyle": random.choice(["Modern", "Traditional", "Minimalist"])
            },
            "reasoning_steps": [
                "Calculate space requirements",
                "Plan room distribution", 
                "Optimize circulation",
                "Consider natural light",
                "Ensure budget compliance"
            ]
        }
        
        if indian:
            problem["context"] = {
                "indian_market": True,
                "region": random.choice(self.regions),
                "climate_zone": random.choice(["Tropical_Hot_Humid", "Composite", "Subtropical_Hot_Dry"])
            }
            
        return problem
    
    def generate_solution(self, problem):
        """Generate a simple solution"""
        area = problem["plot_details"]["area_sqft"]
        family = problem["requirements"]["family_size"]
        
        return {
            "design_solution": {
                "room_distribution": {
                    "bedrooms": family,
                    "bedroom_area_sqft": family * 140,
                    "living_area_sqft": int(area * 0.35),
                    "kitchen_area_sqft": int(area * 0.12),
                    "bathroom_area_sqft": family * 80
                },
                "layout": "Open_plan_with_private_zones",
                "daylight": "North_South_optimized",
                "ventilation": "Cross_ventilation"
            },
            "analysis": {
                "space_efficiency": "85%",
                "natural_light_score": "0.85",
                "budget_alignment": "Within_5_percent"
            },
            "implementation": ["Massing", "Sizing", "Circulation", "Fenestration"]
        }
    
    def quality_score(self, sample):
        """Simple quality scoring"""
        inp = sample["input"]
        out = sample["output"]
        
        score = 0
        checks = 0
        
        # Basic checks
        checks += 1
        if inp.get("problem_type") and out:
            score += 1
        
        checks += 1
        if len(json.dumps(out)) >= 100:
            score += 1
        
        checks += 1
        if len(inp.get("reasoning_steps", [])) >= 3:
            score += 1
        
        return score / checks
    
    def hash_sample(self, sample):
        """Generate hash for deduplication"""
        payload = json.dumps(sample["input"], sort_keys=True) + "||" + json.dumps(sample["output"], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    def generate(self, output_dir):
        """Generate the dataset"""
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        
        train_dir = base / "train"
        val_dir = base / "validation"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Target: {self.target_samples:,} samples | Q>= {self.quality_threshold}")
        print(f"📊 Problem types: {len(self.problem_types)}")
        
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
                        "version": "Fast_v1.0",
                        "sample_id": f"FAST-{self.generated:07d}"
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
        
        print(f"\n🎉 Dataset generation complete!")
        print(f"📊 Total generated: {self.generated:,}")
        print(f"✅ Accepted: {self.accepted:,}")
        print(f"📁 Saved to: {base}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast HouseBrain dataset generator")
    parser.add_argument("--output", type=str, default="housebrain_dataset_r1_super_1M")
    parser.add_argument("--target", type=int, default=1000000)
    parser.add_argument("--quality", type=float, default=0.85)
    args = parser.parse_args()
    
    generator = FastGenerator(target_samples=args.target, quality_threshold=args.quality)
    generator.generate(args.output)

if __name__ == "__main__":
    main()
