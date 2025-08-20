#!/usr/bin/env python3
"""
LLM Integration Module for HouseBrain
Handles all LLM inference for different frontend sections
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
from typing import Dict, Any, List

# Add src to path
sys.path.append('src')

class HouseBrainLLM:
    def __init__(self, model_path: str):
        """Initialize the fine-tuned HouseBrain model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate_house_design(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete house design from user requirements
        Args:
            requirements: {
                'rooms': ['living', 'kitchen', 'bedroom', 'bathroom'],
                'style': 'modern',
                'budget': 150000,
                'location': 'suburban',
                'area_sqft': 1800,
                'stories': 1,
                'special_requirements': ['garage', 'home_office']
            }
        Returns:
            Complete v2 plan JSON
        """
        prompt = self._create_house_design_prompt(requirements)
        response = self._generate_response(prompt)
        plan_json = self._extract_json_from_response(response)
        return plan_json
    
    def generate_cost_estimate(self, plan_or_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed cost estimate
        Can work from either a complete plan or basic requirements
        """
        if 'walls' in plan_or_requirements:  # Full plan provided
            prompt = self._create_cost_prompt_from_plan(plan_or_requirements)
        else:  # Basic requirements provided
            prompt = self._create_cost_prompt_from_requirements(plan_or_requirements)
            
        response = self._generate_response(prompt)
        cost_data = self._extract_cost_data(response)
        return cost_data
    
    def generate_interior_design(self, room_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate interior design recommendations
        Args:
            room_requirements: {
                'room_type': 'living_room',
                'style': 'modern',
                'color_scheme': 'neutral',
                'budget': 15000,
                'existing_furniture': [],
                'preferences': ['minimalist', 'natural_light']
            }
        """
        prompt = self._create_interior_design_prompt(room_requirements)
        response = self._generate_response(prompt)
        interior_data = self._extract_interior_data(response)
        return interior_data
    
    def generate_construction_sequence(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed construction timeline and sequence
        """
        prompt = self._create_construction_prompt(plan)
        response = self._generate_response(prompt)
        construction_data = self._extract_construction_data(response)
        return construction_data
    
    def _create_house_design_prompt(self, requirements: Dict[str, Any]) -> str:
        """Create prompt for house design generation"""
        rooms_list = ", ".join(requirements.get('rooms', []))
        style = requirements.get('style', 'modern')
        budget = requirements.get('budget', 150000)
        area = requirements.get('area_sqft', 1800)
        location = requirements.get('location', 'suburban')
        
        prompt = f"""Generate a complete architectural floor plan in HouseBrain v2 JSON format.

Requirements:
- Rooms needed: {rooms_list}
- Architectural style: {style}
- Budget: ${budget:,}
- Total area: {area} sq ft
- Location: {location}
- Stories: {requirements.get('stories', 1)}

Special requirements: {requirements.get('special_requirements', [])}

Generate a complete plan with:
1. All required rooms with proper dimensions
2. Exterior and interior walls with appropriate thickness
3. Doors and windows with proper handing and operation types
4. Electrical and plumbing considerations
5. Code-compliant egress and accessibility where needed

Output only valid JSON in HouseBrain v2 schema format:"""

        return prompt
    
    def _create_cost_prompt_from_plan(self, plan: Dict[str, Any]) -> str:
        """Create cost estimation prompt from complete plan"""
        total_area = self._calculate_plan_area(plan)
        wall_count = len(plan.get('walls', []))
        opening_count = len(plan.get('openings', []))
        
        prompt = f"""Generate detailed cost estimate for this architectural plan:

Plan Summary:
- Total floor area: {total_area:.0f} sq ft
- Number of walls: {wall_count}
- Number of doors/windows: {opening_count}
- Spaces: {[s.get('name') for s in plan.get('spaces', [])]}

Provide detailed cost breakdown including:
1. Foundation costs
2. Framing and structural
3. Roofing and exterior
4. Electrical systems
5. Plumbing systems
6. Interior finishes
7. Labor costs by trade
8. Permits and fees
9. Contingency
10. Timeline estimate

Format as JSON with detailed breakdown."""

        return prompt
    
    def _create_cost_prompt_from_requirements(self, requirements: Dict[str, Any]) -> str:
        """Create cost estimation prompt from basic requirements"""
        prompt = f"""Generate cost estimate for house with these requirements:

Requirements:
- Area: {requirements.get('area_sqft', 1800)} sq ft
- Style: {requirements.get('style', 'modern')}
- Budget target: ${requirements.get('budget', 150000):,}
- Location: {requirements.get('location', 'suburban')}
- Bedrooms: {requirements.get('bedrooms', 3)}
- Bathrooms: {requirements.get('bathrooms', 2)}

Provide detailed cost breakdown in JSON format."""

        return prompt
    
    def _create_interior_design_prompt(self, room_requirements: Dict[str, Any]) -> str:
        """Create interior design prompt"""
        room_type = room_requirements.get('room_type', 'living_room')
        style = room_requirements.get('style', 'modern')
        budget = room_requirements.get('budget', 15000)
        
        prompt = f"""Generate interior design recommendations for a {room_type}:

Requirements:
- Style: {style}
- Budget: ${budget:,}
- Color scheme: {room_requirements.get('color_scheme', 'neutral')}
- Preferences: {room_requirements.get('preferences', [])}
- Existing furniture: {room_requirements.get('existing_furniture', [])}

Provide detailed recommendations for:
1. Color palette (wall colors, accent colors)
2. Flooring materials and finishes
3. Furniture layout and specifications
4. Lighting plan (ambient, task, accent)
5. Window treatments
6. Decorative elements
7. Budget allocation by category
8. Shopping list with specific products

Format as JSON with detailed specifications."""

        return prompt
    
    def _create_construction_prompt(self, plan: Dict[str, Any]) -> str:
        """Create construction sequence prompt"""
        spaces = [s.get('name') for s in plan.get('spaces', [])]
        
        prompt = f"""Generate detailed construction sequence for this house plan:

Plan includes: {spaces}
Number of walls: {len(plan.get('walls', []))}
Number of openings: {len(plan.get('openings', []))}

Generate detailed construction timeline including:
1. Pre-construction (permits, site prep)
2. Foundation phase with specific tasks
3. Framing sequence (walls, roof)
4. MEP rough-in coordination
5. Insulation and drywall
6. Flooring installation order
7. Finish carpentry and trim
8. Final MEP and fixtures
9. Paint and cleanup
10. Final inspections

For each phase provide:
- Duration in days
- Required trades/crews
- Critical dependencies
- Quality checkpoints
- Potential delays/risks

Format as JSON with detailed timeline."""

        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: return basic structure
                return self._create_fallback_plan()
                
        except json.JSONDecodeError:
            return self._create_fallback_plan()
    
    def _extract_cost_data(self, response: str) -> Dict[str, Any]:
        """Extract cost data from response"""
        try:
            return self._extract_json_from_response(response)
        except:
            return {"error": "Could not parse cost estimate", "total": 0}
    
    def _extract_interior_data(self, response: str) -> Dict[str, Any]:
        """Extract interior design data from response"""
        try:
            return self._extract_json_from_response(response)
        except:
            return {"error": "Could not parse interior design recommendations"}
    
    def _extract_construction_data(self, response: str) -> Dict[str, Any]:
        """Extract construction timeline data from response"""
        try:
            return self._extract_json_from_response(response)
        except:
            return {"error": "Could not parse construction timeline"}
    
    def _calculate_plan_area(self, plan: Dict[str, Any]) -> float:
        """Calculate total area of plan in sq ft"""
        total_area = 0.0
        for space in plan.get('spaces', []):
            boundary = space.get('boundary', [])
            if len(boundary) >= 3:
                # Calculate polygon area (mm²) and convert to sq ft
                area_mm2 = self._poly_area(boundary)
                total_area += area_mm2 / 92903.04  # Convert mm² to sq ft
        return total_area
    
    def _poly_area(self, points: List[List[float]]) -> float:
        """Calculate polygon area using shoelace formula"""
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0
    
    def _create_fallback_plan(self) -> Dict[str, Any]:
        """Create fallback plan if LLM fails"""
        from generate_synthetic_v2 import make_simple_house
        return make_simple_house(12000, 9000)

# Global model instance (load once)
_model_instance = None

def get_llm_instance(model_path: str = "path/to/your/fine-tuned-model"):
    """Get singleton LLM instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = HouseBrainLLM(model_path)
    return _model_instance

def generate_house_from_requirements(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for house generation"""
    llm = get_llm_instance()
    return llm.generate_house_design(requirements)

def generate_cost_from_plan_or_requirements(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for cost estimation"""
    llm = get_llm_instance()
    return llm.generate_cost_estimate(data)

def generate_interior_design_recommendations(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for interior design"""
    llm = get_llm_instance()
    return llm.generate_interior_design(requirements)
