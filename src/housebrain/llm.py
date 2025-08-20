import json
import os
from typing import Dict, Any, Optional, List
from .schema import HouseInput, HouseOutput
from .layout import solve_house_layout
from .validate import validate_house_design


class HouseBrainLLM:
    """Main LLM interface for HouseBrain architectural design"""
    
    def __init__(self, demo_mode: bool = False, model_name: str = "deepseek-r1:8b", 
                 finetuned_model_path: Optional[str] = None):
        self.demo_mode = demo_mode
        self.model_name = model_name
        self.finetuned_model_path = finetuned_model_path
        self.system_prompt = self._get_system_prompt()
        self.model = None
        self.tokenizer = None
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are HouseBrain, an expert architectural AI that designs residential houses.

Your expertise includes:
- Architectural design principles
- Building codes and regulations
- Construction methods and materials
- Interior design and space planning
- Cost estimation and project management

You always respond with valid JSON that matches the HouseOutput schema. Your designs must be:
- Functional and practical
- Code compliant
- Cost-effective
- Aesthetically pleasing
- Optimized for the given plot and requirements

Think like an experienced architect who has designed hundreds of successful homes."""

    def generate_house_design(self, house_input: HouseInput) -> HouseOutput:
        """Generate a complete house design using AI reasoning"""
        
        if self.demo_mode:
            return self._generate_demo_design(house_input)
        elif self.finetuned_model_path and os.path.exists(self.finetuned_model_path):
            return self._generate_finetuned_design(house_input)
        else:
            return self._generate_ollama_design(house_input)
    
    def _generate_demo_design(self, house_input: HouseInput) -> HouseOutput:
        """Generate a demo design using the layout solver"""
        print("🎨 Generating demo house design...")
        
        # Use the layout solver to generate a basic design
        house_output = solve_house_layout(house_input)
        
        # Enhance the design with AI reasoning
        enhanced_output = self._enhance_design_with_ai_reasoning(house_output)
        
        # Validate the design
        validation = validate_house_design(enhanced_output)
        
        if not validation.is_valid:
            print(f"⚠️  Design validation issues: {validation.errors}")
            print(f"📊 Compliance score: {validation.compliance_score}")
        
        return enhanced_output
    
    def _generate_finetuned_design(self, house_input: HouseInput) -> HouseOutput:
        """Generate design using fine-tuned model"""
        print("🤖 Generating design with fine-tuned model...")
        
        try:
            # Load the fine-tuned model
            if self.model is None:
                self._load_finetuned_model()
            
            # Create the input prompt
            prompt = self._create_inference_prompt(house_input)
            
            # Generate response
            response = self._generate_with_model(prompt)
            
            # Parse the response
            house_output = self._parse_model_response(response, house_input)
            
            # Validate the design
            validation = validate_house_design(house_output)
            
            if not validation.is_valid:
                print(f"⚠️  Design validation issues: {validation.errors}")
                print(f"📊 Compliance score: {validation.compliance_score}")
            
            return house_output
            
        except Exception as e:
            print(f"❌ Fine-tuned model generation failed: {e}")
            print("🔄 Falling back to demo mode...")
            return self._generate_demo_design(house_input)
    
    def _generate_ollama_design(self, house_input: HouseInput) -> HouseOutput:
        """Generate design using Ollama"""
        print("🤖 Generating design with Ollama...")
        
        try:
            import requests
            
            # Create the input prompt
            prompt = self._create_inference_prompt(house_input)
            
            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                model_response = result["response"]
                
                # Parse the response
                house_output = self._parse_model_response(model_response, house_input)
                
                # Validate the design
                validation = validate_house_design(house_output)
                
                if not validation.is_valid:
                    print(f"⚠️  Design validation issues: {validation.errors}")
                    print(f"📊 Compliance score: {validation.compliance_score}")
                
                return house_output
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ollama generation failed: {e}")
            print("🔄 Falling back to demo mode...")
            return self._generate_demo_design(house_input)
    
    def _load_finetuned_model(self):
        """Load the fine-tuned model"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print(f"📥 Loading fine-tuned model from {self.finetuned_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.finetuned_model_path)
        self.model.eval()
        
        print("✅ Fine-tuned model loaded successfully")
    
    def _create_inference_prompt(self, house_input: HouseInput) -> str:
        """Create the inference prompt that mirrors the training format."""
        system = (
            "You are HouseBrain, an expert architectural AI. "
            "Generate detailed house designs in JSON format."
        )
        user = (
            "Design a house with these specifications:\n"
            f"{json.dumps(house_input.model_dump() if hasattr(house_input, 'model_dump') else house_input.__dict__, indent=2)}"
        )
        prompt = (
            f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"<|im_start|>user\n{user}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt
    
    def _generate_with_model(self, prompt: str) -> str:
        """Generate response using the loaded model"""
        import torch
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the generated part if echo present
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
    
    def _parse_model_response(self, response: str, house_input: HouseInput) -> HouseOutput:
        """Parse the model response into HouseOutput"""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                output_data = json.loads(json_str)
                
                # Validate and create HouseOutput
                house_output = HouseOutput(**output_data)
                return house_output
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"❌ Failed to parse model response: {e}")
            print("🔄 Falling back to layout solver...")
            
            # Fall back to layout solver
            return solve_house_layout(house_input)
    
    def _enhance_design_with_ai_reasoning(self, house_output: HouseOutput) -> HouseOutput:
        """Enhance the basic design with AI reasoning"""
        
        # Analyze the design and make improvements
        improvements = self._analyze_and_improve_design(house_output)
        
        # Apply improvements
        for improvement in improvements:
            house_output = self._apply_improvement(house_output, improvement)
        
        # Update render paths
        house_output.render_paths = self._generate_render_paths(house_output)
        
        return house_output
    
    def _analyze_and_improve_design(self, house_output: HouseOutput) -> List[Dict[str, Any]]:
        """Analyze design and suggest improvements"""
        improvements = []
        
        # Analyze each level
        for level in house_output.levels:
            # Check room adjacencies
            adjacency_improvements = self._check_room_adjacencies(level)
            improvements.extend(adjacency_improvements)
            
            # Check circulation patterns
            circulation_improvements = self._check_circulation(level)
            improvements.extend(circulation_improvements)
            
            # Check daylight and ventilation
            daylight_improvements = self._check_daylight_ventilation(level)
            improvements.extend(daylight_improvements)
        
        # Check multi-floor connectivity
        if len(house_output.levels) > 1:
            connectivity_improvements = self._check_floor_connectivity(house_output.levels)
            improvements.extend(connectivity_improvements)
        
        return improvements
    
    def _check_room_adjacencies(self, level) -> List[Dict[str, Any]]:
        """Check and improve room adjacencies"""
        improvements = []
        
        # Find kitchen and dining room
        kitchen = next((r for r in level.rooms if r.type.value == "kitchen"), None)
        dining = next((r for r in level.rooms if r.type.value == "dining_room"), None)
        
        if kitchen and dining:
            # Check if they're adjacent
            if not self._rooms_adjacent(kitchen, dining):
                improvements.append({
                    "type": "adjacency",
                    "rooms": [kitchen.id, dining.id],
                    "priority": "high",
                    "description": "Kitchen and dining room should be adjacent"
                })
        
        # Check bedroom-bathroom adjacencies
        bedrooms = [r for r in level.rooms if r.type.value in ["bedroom", "master_bedroom"]]
        bathrooms = [r for r in level.rooms if r.type.value in ["bathroom", "half_bath"]]
        
        for bedroom in bedrooms:
            nearby_bathroom = False
            for bathroom in bathrooms:
                if self._rooms_adjacent(bedroom, bathroom):
                    nearby_bathroom = True
                    break
            
            if not nearby_bathroom:
                improvements.append({
                    "type": "adjacency",
                    "rooms": [bedroom.id, "bathroom"],
                    "priority": "medium",
                    "description": f"Bedroom {bedroom.id} should be near a bathroom"
                })
        
        return improvements
    
    def _check_circulation(self, level) -> List[Dict[str, Any]]:
        """Check and improve circulation patterns"""
        improvements = []
        
        # Check if all rooms are accessible
        accessible_rooms = self._find_accessible_rooms(level.rooms)
        inaccessible_rooms = [r.id for r in level.rooms if r.id not in accessible_rooms]
        
        if inaccessible_rooms:
            improvements.append({
                "type": "circulation",
                "rooms": inaccessible_rooms,
                "priority": "high",
                "description": "Some rooms are not accessible"
            })
        
        # Check corridor widths
        corridors = [r for r in level.rooms if r.type.value == "corridor"]
        for corridor in corridors:
            if corridor.bounds.width < 3.0:
                improvements.append({
                    "type": "circulation",
                    "rooms": [corridor.id],
                    "priority": "medium",
                    "description": f"Corridor {corridor.id} is too narrow"
                })
        
        return improvements
    
    def _check_daylight_ventilation(self, level) -> List[Dict[str, Any]]:
        """Check and improve daylight and ventilation"""
        improvements = []
        
        habitable_rooms = ["living_room", "bedroom", "master_bedroom", "study", "dining_room"]
        
        for room in level.rooms:
            if room.type.value in habitable_rooms:
                if not room.windows:
                    improvements.append({
                        "type": "daylight",
                        "rooms": [room.id],
                        "priority": "high",
                        "description": f"Habitable room {room.id} needs windows"
                    })
        
        return improvements
    
    def _check_floor_connectivity(self, levels) -> List[Dict[str, Any]]:
        """Check and improve floor connectivity"""
        improvements = []
        
        # Check stair connections
        stair_connections = set()
        for level in levels:
            for stair in level.stairs:
                stair_connections.add((stair.floor_from, stair.floor_to))
        
        # Check for missing connections
        for i in range(len(levels) - 1):
            if (i, i + 1) not in stair_connections and (i + 1, i) not in stair_connections:
                improvements.append({
                    "type": "connectivity",
                    "floors": [i, i + 1],
                    "priority": "high",
                    "description": f"Missing stair connection between floors {i} and {i + 1}"
                })
        
        return improvements
    
    def _apply_improvement(self, house_output: HouseOutput, improvement: Dict[str, Any]) -> HouseOutput:
        """Apply a design improvement"""
        # This is a simplified improvement system
        # In a real implementation, this would make actual geometry changes
        
        if improvement["type"] == "adjacency":
            # For demo purposes, just log the improvement
            print(f"💡 Improvement: {improvement['description']}")
        
        elif improvement["type"] == "circulation":
            print(f"💡 Improvement: {improvement['description']}")
        
        elif improvement["type"] == "daylight":
            print(f"💡 Improvement: {improvement['description']}")
        
        elif improvement["type"] == "connectivity":
            print(f"💡 Improvement: {improvement['description']}")
        
        return house_output
    
    def _generate_render_paths(self, house_output: HouseOutput) -> Dict[str, str]:
        """Generate paths for rendered outputs"""
        base_path = "outputs"
        os.makedirs(base_path, exist_ok=True)
        
        return {
            "plan_json": f"{base_path}/plan.json",
            "level_0_svg": f"{base_path}/level_0.svg",
            "level_1_svg": f"{base_path}/level_1.svg",
            "scene_obj": f"{base_path}/scene.obj",
            "front_elevation": f"{base_path}/front_elevation.png",
            "isometric": f"{base_path}/isometric.png"
        }
    
    def _rooms_adjacent(self, room1, room2) -> bool:
        """Check if two rooms are adjacent"""
        # Simple adjacency check
        r1 = room1.bounds
        r2 = room2.bounds
        
        # Check if rooms share a wall
        if (abs(r1.x + r1.width - r2.x) < 0.1 or abs(r2.x + r2.width - r1.x) < 0.1):
            return not (r1.y + r1.height <= r2.y or r2.y + r2.height <= r1.y)
        
        if (abs(r1.y + r1.height - r2.y) < 0.1 or abs(r2.y + r2.height - r1.y) < 0.1):
            return not (r1.x + r1.width <= r2.x or r2.x + r2.width <= r1.x)
        
        return False
    
    def _find_accessible_rooms(self, rooms) -> List[str]:
        """Find all accessible rooms from entrance"""
        if not rooms:
            return []
        
        # Find entrance room
        entrance_room = next((r for r in rooms if r.type.value == "entrance"), None)
        
        if not entrance_room:
            # If no entrance, assume first room is accessible
            accessible = {rooms[0].id}
        else:
            accessible = {entrance_room.id}
        
        # Simple flood fill
        changed = True
        while changed:
            changed = False
            for room in rooms:
                if room.id in accessible:
                    continue
                
                # Check if room is connected to any accessible room
                for door in room.doors:
                    if door.room1 in accessible or door.room2 in accessible:
                        accessible.add(room.id)
                        changed = True
                        break
        
        return list(accessible)


# Convenience function
def generate_house_design(house_input: HouseInput, demo_mode: bool = True, 
                         finetuned_model_path: Optional[str] = None) -> HouseOutput:
    """Main function to generate house design"""
    llm = HouseBrainLLM(demo_mode=demo_mode, finetuned_model_path=finetuned_model_path)
    return llm.generate_house_design(house_input)