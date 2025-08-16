#!/usr/bin/env python3
"""
Quick test to debug generator performance
"""

import time
from generate_1m_super_quality import SuperQualityConfig, SuperQualityGenerator

def test_generator():
    print("🧪 Testing generator performance...")
    
    # Create config with small target
    config = SuperQualityConfig(
        target_samples=10,
        quality_threshold=0.85,
        min_output_chars=200
    )
    
    generator = SuperQualityGenerator(config)
    
    # Test problem generation
    print("📝 Testing problem generation...")
    start = time.time()
    for i in range(5):
        problem = generator._generate_problem()
        print(f"  Problem {i+1}: {problem.get('problem_type', 'Unknown')}")
    problem_time = time.time() - start
    print(f"  Problem generation: {problem_time:.3f}s per problem")
    
    # Test solution generation
    print("🔧 Testing solution generation...")
    start = time.time()
    for i in range(5):
        problem = generator._generate_problem()
        solution = generator._generate_solution(problem)
        print(f"  Solution {i+1}: {len(str(solution))} chars")
    solution_time = time.time() - start
    print(f"  Solution generation: {solution_time:.3f}s per solution")
    
    # Test quality scoring
    print("✅ Testing quality scoring...")
    start = time.time()
    for i in range(5):
        problem = generator._generate_problem()
        solution = generator._generate_solution(problem)
        sample = {"input": problem, "output": solution}
        score = generator._quality_score(sample)
        print(f"  Sample {i+1}: score = {score:.3f}")
    scoring_time = time.time() - start
    print(f"  Quality scoring: {scoring_time:.3f}s per sample")
    
    # Test full generation
    print("🚀 Testing full generation...")
    start = time.time()
    generator.generate("test_output")
    full_time = time.time() - start
    print(f"  Full generation: {full_time:.3f}s for 10 samples")
    
    print(f"\n📊 Summary:")
    print(f"  Problem gen: {problem_time:.3f}s")
    print(f"  Solution gen: {solution_time:.3f}s") 
    print(f"  Quality scoring: {scoring_time:.3f}s")
    print(f"  Full generation: {full_time:.3f}s")

if __name__ == "__main__":
    test_generator()
