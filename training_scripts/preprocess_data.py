#!/usr/bin/env python3
"""
Data Preprocessing for DeepSeek R1 HouseBrain Enhancement
"""

import json
import os
from typing import List, Dict, Any
from transformers import AutoTokenizer

class HouseBrainDataPreprocessor:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = 8192
    
    def preprocess_architectural_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """Preprocess architectural training data"""
        
        processed_examples = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        example = json.load(f)
                    
                    # Create training pair
                    input_text = self._create_input_prompt(example.get('input', {}))
                    output_text = self._create_output_target(example.get('expected_output', {}))
                    
                    # Tokenize
                    tokenized = self._tokenize_pair(input_text, output_text)
                    if tokenized:
                        processed_examples.append(tokenized)
        
        return processed_examples
    
    def _create_input_prompt(self, input_data: Dict[str, Any]) -> str:
        """Create structured input prompt for architectural generation"""
        
        prompt = "Generate a comprehensive architectural design with professional 2D floor plans and 3D models.\n\n"
        prompt += "Requirements:\n"
        
        # Basic details
        basic = input_data.get('basicDetails', {})
        prompt += f"- Total Area: {basic.get('totalArea', 0)} {basic.get('unit', 'sqft')}\n"
        prompt += f"- Floors: {basic.get('floors', 1)}\n"
        prompt += f"- Style: {basic.get('style', 'Modern')}\n"
        
        # Enhanced requirements
        enhanced = input_data.get('enhanced_requirements', {})
        if enhanced:
            prompt += "\nProfessional Standards:\n"
            
            # 2D requirements
            req_2d = enhanced.get('2d_requirements', {})
            if req_2d:
                prompt += f"- Drawing Set: {req_2d.get('drawing_set', '17 professional sheets')}\n"
                prompt += f"- Precision: {req_2d.get('precision', '0.001mm')}\n"
                prompt += f"- Standards: {', '.join(req_2d.get('cad_standards', ['AIA']))}\n"
            
            # 3D requirements  
            req_3d = enhanced.get('3d_requirements', {})
            if req_3d:
                prompt += f"- 3D Quality: {req_3d.get('quality_level', 'ultra')}\n"
                prompt += f"- Materials: {req_3d.get('material_type', 'PBR')}\n"
                prompt += f"- Export Formats: {', '.join(req_3d.get('export_formats', ['obj', 'gltf']))}\n"
        
        prompt += "\nGenerate:"
        return prompt
    
    def _create_output_target(self, output_data: Dict[str, Any]) -> str:
        """Create structured output target for training"""
        
        output = ""
        
        # 2D deliverables
        deliverables_2d = output_data.get('2d_deliverables', {})
        if deliverables_2d:
            output += "2D DELIVERABLES:\n"
            for category, items in deliverables_2d.items():
                output += f"{category.upper()}: {', '.join(items) if isinstance(items, list) else items}\n"
        
        # 3D deliverables
        deliverables_3d = output_data.get('3d_deliverables', {})
        if deliverables_3d:
            output += "\n3D DELIVERABLES:\n"
            for key, value in deliverables_3d.items():
                output += f"{key.replace('_', ' ').title()}: {value}\n"
        
        # Quality metrics
        metrics = output_data.get('quality_metrics', {})
        if metrics:
            output += "\nQUALITY METRICS:\n"
            for metric, value in metrics.items():
                output += f"{metric.replace('_', ' ').title()}: {value}\n"
        
        return output
    
    def _tokenize_pair(self, input_text: str, output_text: str) -> Dict[str, Any]:
        """Tokenize input-output pair for training"""
        
        # Combine input and output for training
        full_text = f"{input_text}\n\n{output_text}"
        
        # Tokenize
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        if len(tokens["input_ids"][0]) < self.max_length * 0.1:  # Skip very short examples
            return None
        
        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels": tokens["input_ids"][0].clone()
        }

if __name__ == "__main__":
    preprocessor = HouseBrainDataPreprocessor("models/deepseek_r1_600k")
    examples = preprocessor.preprocess_architectural_data("training_dataset")
    print(f"Processed {len(examples)} training examples")\n