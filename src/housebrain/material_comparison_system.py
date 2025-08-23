"""
Material Comparison System for HouseBrain Professional

This module provides comprehensive material comparison capabilities:
- Side-by-side material property comparison
- Performance analysis and scoring
- Cost-benefit analysis
- Sustainability impact comparison
- Regional suitability assessment
- Professional recommendation engine
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class MaterialScore:
    """Material performance score in a specific category"""
    category: str
    score: float  # 0-100
    weight: float  # Importance weight
    explanation: str
    
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class ComparisonResult:
    """Result of material comparison"""
    materials: List[str]
    comparison_type: str
    scores: Dict[str, List[MaterialScore]]
    overall_scores: Dict[str, float]
    recommendation: str
    detailed_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "materials": self.materials,
            "comparison_type": self.comparison_type,
            "scores": {
                material: [
                    {"category": s.category, "score": s.score, "weight": s.weight, "explanation": s.explanation}
                    for s in scores
                ]
                for material, scores in self.scores.items()
            },
            "overall_scores": self.overall_scores,
            "recommendation": self.recommendation,
            "detailed_analysis": self.detailed_analysis
        }


class MaterialComparisonSystem:
    """Advanced material comparison and analysis system"""
    
    def __init__(self, material_library):
        self.material_library = material_library
        self.comparison_criteria = self._initialize_comparison_criteria()
        self.scoring_algorithms = self._initialize_scoring_algorithms()
        self.regional_factors = self._initialize_regional_factors()
        
        print("🔬 Material Comparison System Initialized")
    
    def compare_materials(
        self,
        material_names: List[str],
        comparison_type: str = "general",
        region: str = "global",
        project_context: Dict[str, Any] = None
    ) -> ComparisonResult:
        """Compare multiple materials with detailed analysis"""
        
        if len(material_names) < 2:
            raise ValueError("At least 2 materials required for comparison")
        
        print(f"🔍 Comparing {len(material_names)} materials for {comparison_type} use...")
        
        # Get material data
        materials_data = {}
        for name in material_names:
            material = self.material_library.get_material(name, region)
            if material:
                materials_data[name] = material
            else:
                print(f"⚠️ Material '{name}' not found, skipping...")
        
        if len(materials_data) < 2:
            raise ValueError("Not enough valid materials found for comparison")
        
        # Get comparison criteria for this type
        criteria = self.comparison_criteria.get(comparison_type, self.comparison_criteria["general"])
        
        # Score each material
        material_scores = {}
        for material_name, material_data in materials_data.items():
            scores = self._score_material(material_data, criteria, region, project_context)
            material_scores[material_name] = scores
        
        # Calculate overall scores
        overall_scores = {}
        for material_name, scores in material_scores.items():
            overall_score = sum(score.weighted_score() for score in scores) / sum(score.weight for score in scores)
            overall_scores[material_name] = overall_score
        
        # Generate recommendation
        best_material = max(overall_scores.keys(), key=lambda x: overall_scores[x])
        recommendation = self._generate_recommendation(
            materials_data, material_scores, overall_scores, best_material, comparison_type
        )
        
        # Create detailed analysis
        detailed_analysis = self._create_detailed_analysis(
            materials_data, material_scores, comparison_type, region, project_context
        )
        
        return ComparisonResult(
            materials=list(materials_data.keys()),
            comparison_type=comparison_type,
            scores=material_scores,
            overall_scores=overall_scores,
            recommendation=recommendation,
            detailed_analysis=detailed_analysis
        )
    
    def compare_for_application(
        self,
        material_names: List[str],
        application: str,
        climate_zone: str = "temperate",
        building_type: str = "residential"
    ) -> ComparisonResult:
        """Compare materials for specific application"""
        
        application_context = {
            "application": application,
            "climate_zone": climate_zone,
            "building_type": building_type
        }
        
        comparison_type = self._get_comparison_type_for_application(application)
        
        return self.compare_materials(
            material_names,
            comparison_type,
            climate_zone,
            application_context
        )
    
    def cost_benefit_analysis(
        self,
        material_names: List[str],
        project_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform cost-benefit analysis for materials"""
        
        analysis = {
            "materials": material_names,
            "project_parameters": project_parameters,
            "cost_analysis": {},
            "benefit_analysis": {},
            "roi_analysis": {},
            "lifecycle_analysis": {}
        }
        
        # Calculate costs
        for material_name in material_names:
            material = self.material_library.get_material(material_name)
            if material:
                cost_data = self._calculate_material_costs(material, project_parameters)
                analysis["cost_analysis"][material_name] = cost_data
        
        # Calculate benefits
        for material_name in material_names:
            material = self.material_library.get_material(material_name)
            if material:
                benefit_data = self._calculate_material_benefits(material, project_parameters)
                analysis["benefit_analysis"][material_name] = benefit_data
        
        # Calculate ROI
        for material_name in material_names:
            if material_name in analysis["cost_analysis"] and material_name in analysis["benefit_analysis"]:
                roi_data = self._calculate_roi(
                    analysis["cost_analysis"][material_name],
                    analysis["benefit_analysis"][material_name],
                    project_parameters
                )
                analysis["roi_analysis"][material_name] = roi_data
        
        return analysis
    
    def sustainability_comparison(
        self,
        material_names: List[str],
        focus_areas: List[str] = None
    ) -> Dict[str, Any]:
        """Compare materials on sustainability criteria"""
        
        if focus_areas is None:
            focus_areas = ["embodied_carbon", "recyclability", "durability", "toxicity", "local_sourcing"]
        
        sustainability_analysis = {
            "materials": material_names,
            "focus_areas": focus_areas,
            "scores": {},
            "rankings": {},
            "detailed_metrics": {}
        }
        
        # Score each material on sustainability criteria
        for material_name in material_names:
            material = self.material_library.get_material(material_name)
            if material:
                sustainability_scores = self._score_sustainability(material, focus_areas)
                sustainability_analysis["scores"][material_name] = sustainability_scores
        
        # Generate rankings
        for focus_area in focus_areas:
            area_scores = {
                material: scores.get(focus_area, 0)
                for material, scores in sustainability_analysis["scores"].items()
            }
            ranked_materials = sorted(area_scores.keys(), key=lambda x: area_scores[x], reverse=True)
            sustainability_analysis["rankings"][focus_area] = ranked_materials
        
        return sustainability_analysis
    
    def generate_comparison_report(
        self,
        comparison_result: ComparisonResult,
        format_type: str = "html"
    ) -> str:
        """Generate formatted comparison report"""
        
        if format_type == "html":
            return self._generate_html_comparison_report(comparison_result)
        elif format_type == "json":
            return json.dumps(comparison_result.to_dict(), indent=2)
        elif format_type == "text":
            return self._generate_text_comparison_report(comparison_result)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _initialize_comparison_criteria(self) -> Dict[str, Dict]:
        """Initialize comparison criteria for different use cases"""
        
        return {
            "general": {
                "performance": {"weight": 0.25, "subcriteria": ["strength", "durability", "thermal_performance"]},
                "cost": {"weight": 0.20, "subcriteria": ["initial_cost", "maintenance_cost", "lifecycle_cost"]},
                "sustainability": {"weight": 0.20, "subcriteria": ["embodied_carbon", "recyclability", "local_sourcing"]},
                "aesthetics": {"weight": 0.15, "subcriteria": ["appearance", "texture", "color_options"]},
                "installation": {"weight": 0.10, "subcriteria": ["ease_of_installation", "skill_required", "tools_needed"]},
                "maintenance": {"weight": 0.10, "subcriteria": ["maintenance_frequency", "cleaning_ease", "repair_ease"]}
            },
            
            "structural": {
                "strength": {"weight": 0.35, "subcriteria": ["compressive_strength", "tensile_strength", "flexural_strength"]},
                "durability": {"weight": 0.25, "subcriteria": ["weather_resistance", "chemical_resistance", "fatigue_resistance"]},
                "cost": {"weight": 0.20, "subcriteria": ["material_cost", "installation_cost", "lifecycle_cost"]},
                "sustainability": {"weight": 0.15, "subcriteria": ["embodied_carbon", "recyclability", "resource_depletion"]},
                "workability": {"weight": 0.05, "subcriteria": ["fabrication_ease", "connection_options", "modification_ease"]}
            },
            
            "insulation": {
                "thermal_performance": {"weight": 0.40, "subcriteria": ["r_value", "thermal_bridging", "air_sealing"]},
                "moisture_management": {"weight": 0.20, "subcriteria": ["vapor_permeability", "water_resistance", "mold_resistance"]},
                "fire_safety": {"weight": 0.15, "subcriteria": ["fire_rating", "smoke_production", "toxic_emissions"]},
                "cost": {"weight": 0.15, "subcriteria": ["material_cost", "installation_cost", "energy_savings"]},
                "sustainability": {"weight": 0.10, "subcriteria": ["embodied_carbon", "recyclability", "health_impact"]}
            },
            
            "finishes": {
                "aesthetics": {"weight": 0.30, "subcriteria": ["appearance", "texture", "color_stability"]},
                "durability": {"weight": 0.25, "subcriteria": ["wear_resistance", "stain_resistance", "uv_resistance"]},
                "maintenance": {"weight": 0.20, "subcriteria": ["cleaning_ease", "repair_ease", "replacement_frequency"]},
                "cost": {"weight": 0.15, "subcriteria": ["initial_cost", "maintenance_cost", "replacement_cost"]},
                "health": {"weight": 0.10, "subcriteria": ["voc_emissions", "allergen_potential", "antimicrobial_properties"]}
            },
            
            "roofing": {
                "weather_protection": {"weight": 0.35, "subcriteria": ["water_resistance", "wind_resistance", "hail_resistance"]},
                "durability": {"weight": 0.25, "subcriteria": ["uv_resistance", "thermal_cycling", "expected_lifespan"]},
                "thermal_performance": {"weight": 0.20, "subcriteria": ["solar_reflectance", "thermal_emittance", "insulation_value"]},
                "cost": {"weight": 0.15, "subcriteria": ["material_cost", "installation_cost", "maintenance_cost"]},
                "sustainability": {"weight": 0.05, "subcriteria": ["embodied_carbon", "end_of_life", "local_availability"]}
            }
        }
    
    def _initialize_scoring_algorithms(self) -> Dict[str, Any]:
        """Initialize scoring algorithms for different material properties"""
        
        return {
            "linear_scale": lambda value, min_val, max_val: max(0, min(100, (value - min_val) / (max_val - min_val) * 100)),
            "inverse_scale": lambda value, min_val, max_val: max(0, min(100, 100 - (value - min_val) / (max_val - min_val) * 100)),
            "logarithmic": lambda value, base: max(0, min(100, math.log(value) / math.log(base) * 100)),
            "categorical": lambda value, mapping: mapping.get(value, 0)
        }
    
    def _initialize_regional_factors(self) -> Dict[str, Dict]:
        """Initialize regional factors for material performance"""
        
        return {
            "tropical": {
                "humidity_factor": 1.5,
                "temperature_factor": 1.3,
                "uv_factor": 1.4,
                "precipitation_factor": 1.6,
                "key_concerns": ["mold_resistance", "decay_resistance", "thermal_expansion"]
            },
            "arid": {
                "humidity_factor": 0.3,
                "temperature_factor": 1.5,
                "uv_factor": 1.8,
                "precipitation_factor": 0.2,
                "key_concerns": ["thermal_shock", "dust_resistance", "thermal_mass"]
            },
            "temperate": {
                "humidity_factor": 1.0,
                "temperature_factor": 1.0,
                "uv_factor": 1.0,
                "precipitation_factor": 1.0,
                "key_concerns": ["freeze_thaw", "moisture_management", "seasonal_cycling"]
            },
            "polar": {
                "humidity_factor": 0.8,
                "temperature_factor": 2.0,
                "uv_factor": 0.5,
                "precipitation_factor": 0.7,
                "key_concerns": ["freeze_resistance", "thermal_bridging", "extreme_cold"]
            }
        }
    
    def _score_material(
        self,
        material_data: Dict[str, Any],
        criteria: Dict[str, Dict],
        region: str,
        project_context: Dict[str, Any]
    ) -> List[MaterialScore]:
        """Score a material against comparison criteria"""
        
        scores = []
        
        for criterion_name, criterion_config in criteria.items():
            weight = criterion_config["weight"]
            subcriteria = criterion_config.get("subcriteria", [criterion_name])
            
            # Calculate score for this criterion
            criterion_score = self._calculate_criterion_score(
                material_data, criterion_name, subcriteria, region, project_context
            )
            
            # Generate explanation
            explanation = self._generate_score_explanation(
                material_data, criterion_name, criterion_score, region
            )
            
            scores.append(MaterialScore(
                category=criterion_name,
                score=criterion_score,
                weight=weight,
                explanation=explanation
            ))
        
        return scores
    
    def _calculate_criterion_score(
        self,
        material_data: Dict[str, Any],
        criterion_name: str,
        subcriteria: List[str],
        region: str,
        project_context: Dict[str, Any]
    ) -> float:
        """Calculate score for a specific criterion"""
        
        properties = material_data.get("properties", {})
        
        if criterion_name == "performance":
            return self._score_performance(properties, subcriteria, region)
        elif criterion_name == "cost":
            return self._score_cost(properties, subcriteria, project_context)
        elif criterion_name == "sustainability":
            return self._score_sustainability_basic(properties, subcriteria)
        elif criterion_name == "aesthetics":
            return self._score_aesthetics(properties, subcriteria)
        elif criterion_name == "installation":
            return self._score_installation(properties, subcriteria)
        elif criterion_name == "maintenance":
            return self._score_maintenance(properties, subcriteria)
        elif criterion_name == "strength":
            return self._score_strength(properties, subcriteria)
        elif criterion_name == "durability":
            return self._score_durability(properties, subcriteria, region)
        elif criterion_name == "thermal_performance":
            return self._score_thermal_performance(properties, subcriteria, region)
        elif criterion_name == "moisture_management":
            return self._score_moisture_management(properties, subcriteria, region)
        elif criterion_name == "fire_safety":
            return self._score_fire_safety(properties, subcriteria)
        elif criterion_name == "weather_protection":
            return self._score_weather_protection(properties, subcriteria, region)
        else:
            return 50.0  # Default neutral score
    
    def _score_performance(self, properties: Dict, subcriteria: List[str], region: str) -> float:
        """Score material performance"""
        
        scores = []
        
        if "strength" in subcriteria:
            strength = properties.get("compressive_strength", properties.get("strength", "medium"))
            if isinstance(strength, (int, float)):
                scores.append(min(100, strength / 50 * 100))  # Normalize to 50 MPa = 100 score
            else:
                strength_mapping = {"very_high": 95, "high": 80, "medium": 60, "low": 40, "very_low": 20}
                scores.append(strength_mapping.get(strength, 50))
        
        if "durability" in subcriteria:
            durability = properties.get("durability", "medium")
            durability_mapping = {"excellent": 95, "very_good": 85, "good": 70, "fair": 55, "poor": 30}
            scores.append(durability_mapping.get(durability, 50))
        
        if "thermal_performance" in subcriteria:
            thermal_conductivity = properties.get("thermal_conductivity", 1.0)
            # Lower thermal conductivity = better insulation = higher score
            scores.append(max(0, min(100, 100 - thermal_conductivity * 20)))
        
        return sum(scores) / len(scores) if scores else 50.0
    
    def _score_cost(self, properties: Dict, subcriteria: List[str], project_context: Dict) -> float:
        """Score material cost factors"""
        
        # This would integrate with real cost databases
        # For now, use simplified scoring
        
        scores = []
        
        if "initial_cost" in subcriteria:
            # Estimate based on material type and properties
            base_score = 70  # Default
            if properties.get("density", 1000) > 2000:  # Heavy materials often more expensive
                base_score -= 10
            if properties.get("strength") == "high":
                base_score -= 5
            scores.append(base_score)
        
        if "maintenance_cost" in subcriteria:
            maintenance = properties.get("maintenance", "medium")
            maintenance_mapping = {"very_low": 95, "low": 80, "medium": 60, "high": 40, "very_high": 20}
            scores.append(maintenance_mapping.get(maintenance, 60))
        
        if "lifecycle_cost" in subcriteria:
            durability = properties.get("durability", "medium")
            durability_mapping = {"excellent": 90, "good": 75, "fair": 60, "poor": 40}
            scores.append(durability_mapping.get(durability, 60))
        
        return sum(scores) / len(scores) if scores else 60.0
    
    def _score_sustainability_basic(self, properties: Dict, subcriteria: List[str]) -> float:
        """Score basic sustainability factors"""
        
        scores = []
        
        if "embodied_carbon" in subcriteria:
            # Lower embodied carbon = higher score
            embodied_carbon = properties.get("embodied_carbon", "medium")
            carbon_mapping = {"very_low": 95, "low": 80, "medium": 60, "high": 40, "very_high": 20}
            scores.append(carbon_mapping.get(embodied_carbon, 60))
        
        if "recyclability" in subcriteria:
            recyclability = properties.get("recyclability", properties.get("end_of_life", "moderate"))
            recycle_mapping = {"excellent": 95, "good": 80, "moderate": 60, "poor": 40, "none": 20}
            scores.append(recycle_mapping.get(recyclability, 60))
        
        if "local_sourcing" in subcriteria:
            local_availability = properties.get("local_availability", "moderate")
            local_mapping = {"excellent": 90, "good": 75, "moderate": 60, "poor": 40, "very_poor": 20}
            scores.append(local_mapping.get(local_availability, 60))
        
        return sum(scores) / len(scores) if scores else 60.0
    
    # Additional scoring methods (simplified implementations)
    
    def _score_aesthetics(self, properties: Dict, subcriteria: List[str]) -> float:
        return 75.0  # Simplified - would need more detailed aesthetic database
    
    def _score_installation(self, properties: Dict, subcriteria: List[str]) -> float:
        return 70.0  # Simplified - would need installation complexity database
    
    def _score_maintenance(self, properties: Dict, subcriteria: List[str]) -> float:
        maintenance = properties.get("maintenance", "medium")
        maintenance_mapping = {"very_low": 95, "low": 85, "medium": 70, "high": 50, "very_high": 30}
        return maintenance_mapping.get(maintenance, 70)
    
    def _score_strength(self, properties: Dict, subcriteria: List[str]) -> float:
        strength = properties.get("compressive_strength", properties.get("strength", "medium"))
        if isinstance(strength, (int, float)):
            return min(100, strength / 50 * 100)  # Normalize to 50 MPa = 100 score
        else:
            strength_mapping = {"very_high": 95, "high": 85, "medium": 70, "low": 50, "very_low": 30}
            return strength_mapping.get(strength, 70)
    
    def _score_durability(self, properties: Dict, subcriteria: List[str], region: str) -> float:
        base_durability = properties.get("durability", "medium")
        durability_mapping = {"excellent": 90, "good": 75, "fair": 60, "poor": 40}
        base_score = durability_mapping.get(base_durability, 60)
        
        # Apply regional factors
        regional_factor = self.regional_factors.get(region, self.regional_factors["temperate"])
        if any(concern in properties for concern in regional_factor["key_concerns"]):
            base_score += 10  # Bonus for addressing regional concerns
        
        return min(100, base_score)
    
    def _score_thermal_performance(self, properties: Dict, subcriteria: List[str], region: str) -> float:
        thermal_conductivity = properties.get("thermal_conductivity", 1.0)
        # Lower thermal conductivity = better insulation = higher score
        base_score = max(0, min(100, 100 - thermal_conductivity * 15))
        
        # Apply regional thermal factors
        regional_factor = self.regional_factors.get(region, self.regional_factors["temperate"])
        temp_factor = regional_factor.get("temperature_factor", 1.0)
        
        if temp_factor > 1.2:  # Extreme temperature regions
            base_score *= 1.1  # Thermal performance more important
        
        return min(100, base_score)
    
    def _score_moisture_management(self, properties: Dict, subcriteria: List[str], region: str) -> float:
        base_score = 70.0
        
        # Check for moisture-related properties
        if properties.get("moisture_resistance") == "excellent":
            base_score += 15
        elif properties.get("water_resistance") == "excellent":
            base_score += 10
        
        # Apply regional humidity factors
        regional_factor = self.regional_factors.get(region, self.regional_factors["temperate"])
        humidity_factor = regional_factor.get("humidity_factor", 1.0)
        
        if humidity_factor > 1.2:  # High humidity regions
            base_score *= 1.2  # Moisture management more critical
        
        return min(100, base_score)
    
    def _score_fire_safety(self, properties: Dict, subcriteria: List[str]) -> float:
        fire_rating = properties.get("fire_rating", "standard")
        
        fire_mapping = {
            "non_combustible": 95,
            "fire_resistant": 85,
            "fire_retardant": 75,
            "standard": 60,
            "combustible": 40
        }
        
        return fire_mapping.get(fire_rating, 60)
    
    def _score_weather_protection(self, properties: Dict, subcriteria: List[str], region: str) -> float:
        base_score = 70.0
        
        # Check weather resistance properties
        if properties.get("weather_resistance") == "excellent":
            base_score += 20
        elif properties.get("uv_resistance") == "excellent":
            base_score += 10
        
        return min(100, base_score)
    
    def _generate_score_explanation(
        self,
        material_data: Dict[str, Any],
        criterion_name: str,
        score: float,
        region: str
    ) -> str:
        """Generate explanation for material score"""
        
        material_name = material_data.get("name", "Material")
        
        if score >= 85:
            performance = "excellent"
        elif score >= 75:
            performance = "very good"
        elif score >= 65:
            performance = "good"
        elif score >= 50:
            performance = "fair"
        else:
            performance = "poor"
        
        explanations = {
            "performance": f"{material_name} shows {performance} overall performance with score {score:.1f}",
            "cost": f"{material_name} has {performance} cost-effectiveness with score {score:.1f}",
            "sustainability": f"{material_name} demonstrates {performance} sustainability with score {score:.1f}",
            "durability": f"{material_name} exhibits {performance} durability characteristics with score {score:.1f}",
            "strength": f"{material_name} provides {performance} structural strength with score {score:.1f}"
        }
        
        return explanations.get(criterion_name, f"{material_name} scored {score:.1f} for {criterion_name}")
    
    def _generate_recommendation(
        self,
        materials_data: Dict[str, Dict],
        material_scores: Dict[str, List[MaterialScore]],
        overall_scores: Dict[str, float],
        best_material: str,
        comparison_type: str
    ) -> str:
        """Generate recommendation text"""
        
        best_score = overall_scores[best_material]
        
        recommendation = f"Based on {comparison_type} comparison criteria, "
        recommendation += f"{best_material} is the recommended choice with an overall score of {best_score:.1f}%. "
        
        # Find the strongest category for the best material
        best_material_scores = material_scores[best_material]
        strongest_category = max(best_material_scores, key=lambda x: x.score)
        
        recommendation += f"It excels particularly in {strongest_category.category} "
        recommendation += f"with a score of {strongest_category.score:.1f}%. "
        
        # Add consideration notes
        if best_score < 80:
            recommendation += "However, consider project-specific requirements and consult with specialists for final selection."
        else:
            recommendation += "This material meets high performance standards for the intended application."
        
        return recommendation
    
    def _create_detailed_analysis(
        self,
        materials_data: Dict[str, Dict],
        material_scores: Dict[str, List[MaterialScore]],
        comparison_type: str,
        region: str,
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed analysis data"""
        
        return {
            "comparison_summary": {
                "materials_evaluated": len(materials_data),
                "comparison_type": comparison_type,
                "region": region,
                "criteria_count": len(material_scores[list(material_scores.keys())[0]])
            },
            "regional_considerations": self.regional_factors.get(region, {}),
            "project_context": project_context or {},
            "methodology": "Multi-criteria decision analysis with weighted scoring",
            "confidence_level": "High" if len(materials_data) <= 5 else "Medium"
        }
    
    def _get_comparison_type_for_application(self, application: str) -> str:
        """Get appropriate comparison type for application"""
        
        application_mapping = {
            "structural_walls": "structural",
            "insulation": "insulation",
            "exterior_cladding": "finishes",
            "roofing": "roofing",
            "flooring": "finishes",
            "interior_walls": "finishes"
        }
        
        return application_mapping.get(application, "general")
    
    # Cost-benefit analysis methods (simplified)
    
    def _calculate_material_costs(self, material: Dict, project_params: Dict) -> Dict[str, float]:
        return {
            "material_cost_per_unit": 100.0,  # Placeholder
            "installation_cost_per_unit": 50.0,
            "total_initial_cost": 150.0
        }
    
    def _calculate_material_benefits(self, material: Dict, project_params: Dict) -> Dict[str, float]:
        return {
            "energy_savings_annual": 200.0,  # Placeholder
            "maintenance_savings_annual": 100.0,
            "durability_benefit": 500.0
        }
    
    def _calculate_roi(self, costs: Dict, benefits: Dict, project_params: Dict) -> Dict[str, float]:
        annual_savings = benefits.get("energy_savings_annual", 0) + benefits.get("maintenance_savings_annual", 0)
        initial_cost = costs.get("total_initial_cost", 0)
        
        payback_period = initial_cost / annual_savings if annual_savings > 0 else float('inf')
        
        return {
            "annual_savings": annual_savings,
            "payback_period_years": payback_period,
            "roi_10_year": (annual_savings * 10 - initial_cost) / initial_cost * 100 if initial_cost > 0 else 0
        }
    
    def _score_sustainability(self, material: Dict, focus_areas: List[str]) -> Dict[str, float]:
        scores = {}
        
        for area in focus_areas:
            if area == "embodied_carbon":
                scores[area] = 75.0  # Placeholder
            elif area == "recyclability":
                scores[area] = 80.0
            elif area == "durability":
                scores[area] = 85.0
            elif area == "toxicity":
                scores[area] = 90.0
            elif area == "local_sourcing":
                scores[area] = 70.0
            else:
                scores[area] = 75.0
        
        return scores
    
    def _generate_html_comparison_report(self, comparison_result: ComparisonResult) -> str:
        """Generate HTML comparison report"""
        
        scores = comparison_result.overall_scores
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Material Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .material {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .score {{ font-weight: bold; color: #2e7d32; }}
        .recommendation {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Material Comparison Report</h1>
    <h2>Comparison Type: {comparison_result.comparison_type.title()}</h2>
    
    <div class="recommendation">
        <h3>Recommendation</h3>
        <p>{comparison_result.recommendation}</p>
    </div>
    
    <h3>Overall Scores</h3>
    <table>
        <tr><th>Material</th><th>Overall Score</th></tr>
        {"".join(f"<tr><td>{material}</td><td class='score'>{score:.1f}%</td></tr>" 
                for material, score in sorted(scores.items(), key=lambda x: x[1], reverse=True))}
    </table>
    
    <h3>Detailed Analysis</h3>
    {self._format_detailed_scores_html(comparison_result.scores)}
    
</body>
</html>'''
        
        return html
    
    def _format_detailed_scores_html(self, scores: Dict[str, List[MaterialScore]]) -> str:
        html = ""
        for material, material_scores in scores.items():
            html += f"<div class='material'><h4>{material}</h4><ul>"
            for score in material_scores:
                html += f"<li>{score.category.title()}: {score.score:.1f}% (Weight: {score.weight:.2f}) - {score.explanation}</li>"
            html += "</ul></div>"
        return html
    
    def _generate_text_comparison_report(self, comparison_result: ComparisonResult) -> str:
        """Generate text comparison report"""
        
        lines = [
            "Material Comparison Report",
            "========================",
            f"Comparison Type: {comparison_result.comparison_type.title()}",
            "",
            "Overall Scores:",
            "--------------"
        ]
        
        for material, score in sorted(comparison_result.overall_scores.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{material}: {score:.1f}%")
        
        lines.extend([
            "",
            "Recommendation:",
            "--------------",
            comparison_result.recommendation
        ])
        
        return "\n".join(lines)


def create_material_comparison_system(material_library) -> MaterialComparisonSystem:
    """Create material comparison system instance"""
    
    return MaterialComparisonSystem(material_library)


if __name__ == "__main__":
    # Test material comparison system
    from advanced_material_library import create_material_database
    
    print("🔬 Material Comparison System Test")
    print("=" * 50)
    
    # Create material library and comparison system
    material_lib = create_material_database()
    comparison_system = create_material_comparison_system(material_lib)
    
    # Test material comparison
    materials_to_compare = ["concrete_polished", "timber_oak", "steel_brushed"]
    
    result = comparison_system.compare_materials(
        materials_to_compare,
        comparison_type="structural",
        region="temperate"
    )
    
    print(f"Compared {len(result.materials)} materials:")
    for material, score in result.overall_scores.items():
        print(f"  {material}: {score:.1f}%")
    
    print(f"\nRecommendation: {result.recommendation}")
    
    # Test sustainability comparison
    sustainability_result = comparison_system.sustainability_comparison(materials_to_compare)
    print(f"\nSustainability analysis completed for {len(sustainability_result['materials'])} materials")
    
    print("✅ Material Comparison System initialized successfully!")