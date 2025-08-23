"""
Professional Building Code Validation and Architectural Quality Assessment

This module provides comprehensive validation for architect-level designs including:
- Building code compliance (IBC, ADA, IECC, etc.)
- Structural adequacy checks
- MEP system validation
- Accessibility compliance
- Fire safety requirements
- Energy code compliance
- Zoning and setback verification
- Construction feasibility assessment
"""

from __future__ import annotations

import json
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    CODE_COMPLIANCE = "code_compliance"
    PERMIT_READY = "permit_ready"


class ViolationSeverity(Enum):
    """Violation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    code: str
    severity: ViolationSeverity
    category: str
    description: str
    location: Optional[str] = None
    element_id: Optional[str] = None
    code_reference: Optional[str] = None
    suggested_fix: Optional[str] = None
    impact_on_permit: bool = False


@dataclass
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    compliance_score: float  # 0-100
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    code_compliance: Dict[str, bool]
    recommendations: List[str]


class BuildingCodeValidator:
    """Validates against building codes (IBC, ADA, etc.)"""
    
    def __init__(self, jurisdiction: str = "IBC_2021"):
        self.jurisdiction = jurisdiction
        self.load_code_requirements()
    
    def load_code_requirements(self):
        """Load code requirements for jurisdiction"""
        self.requirements = {
            "occupancy_loads": {
                "residential": {
                    "sleeping_rooms": 200,  # sq ft per person
                    "assembly": 15,
                    "concentrated_business": 100,
                    "unconcentrated_business": 300
                }
            },
            "egress": {
                "door_width": {
                    "min_clear": 815,  # mm (32")
                    "main_exit": 915   # mm (36")
                },
                "corridor_width": {
                    "min": 1118,      # mm (44")
                    "occupant_load_50": 1219  # mm (48")
                },
                "travel_distance": {
                    "residential": 76200,  # mm (250 ft)
                    "sprinklered": 91400   # mm (300 ft)
                }
            },
            "accessibility": {
                "door_clear_width": 813,  # mm (32")
                "door_maneuvering": {
                    "front_approach_pull": 457,  # mm (18")
                    "front_approach_push": 305,  # mm (12")
                    "latch_approach": 610        # mm (24")
                },
                "ramp_slope": 8.33,  # % (1:12)
                "bathroom_clear_space": 762,  # mm (30")
                "toilet_centerline": 457     # mm (18" from wall)
            },
            "structural": {
                "live_loads": {
                    "residential_floors": 1916,    # Pa (40 psf)
                    "bedrooms": 1436,             # Pa (30 psf)
                    "stairs": 4788,               # Pa (100 psf)
                    "balconies": 4788             # Pa (100 psf)
                },
                "wind_loads": {
                    "basic_speed": 45,  # m/s (100 mph)
                    "exposure_b": 1.0,
                    "importance_factor": 1.0
                }
            },
            "fire_safety": {
                "sprinkler_required": {
                    "area_threshold": 464516,  # mm² (5000 sq ft)
                    "height_threshold": 9144   # mm (30 ft)
                },
                "smoke_detectors": {
                    "bedrooms": True,
                    "hallways": True,
                    "living_areas": True,
                    "max_spacing": 9144  # mm (30 ft)
                }
            },
            "energy": {
                "iecc_2021": {
                    "u_factors": {
                        "walls": 0.6,      # W/m²·K
                        "windows": 2.8,
                        "roof": 0.3
                    },
                    "air_leakage": 2.0,   # ACH50
                    "duct_leakage": 4.0   # CFM25/100 sq ft
                }
            }
        }
    
    def validate_egress(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate egress requirements"""
        issues = []
        spaces = plan_data.get("spaces", [])
        openings = plan_data.get("openings", [])
        
        # Check exit door widths
        for opening in openings:
            if opening.get("type") == "door":
                wall_id = opening.get("wall_id")
                wall = next((w for w in plan_data.get("walls", []) if w.get("id") == wall_id), None)
                
                if wall and wall.get("type") == "exterior":
                    # Main exit door
                    clear_width = opening.get("dimensions", {}).get("width", 0) - 100  # Account for frame
                    if clear_width < self.requirements["egress"]["door_width"]["main_exit"]:
                        issues.append(ValidationIssue(
                            code="EGRESS_001",
                            severity=ViolationSeverity.ERROR,
                            category="egress",
                            description=f"Main exit door too narrow: {clear_width}mm < {self.requirements['egress']['door_width']['main_exit']}mm required",
                            element_id=opening.get("id"),
                            code_reference="IBC 1010.1.1",
                            suggested_fix="Increase door width to minimum 36\" clear",
                            impact_on_permit=True
                        ))
                else:
                    # Interior door
                    clear_width = opening.get("dimensions", {}).get("width", 0) - 100
                    if clear_width < self.requirements["egress"]["door_width"]["min_clear"]:
                        issues.append(ValidationIssue(
                            code="EGRESS_002",
                            severity=ViolationSeverity.WARNING,
                            category="egress",
                            description=f"Interior door narrow: {clear_width}mm < {self.requirements['egress']['door_width']['min_clear']}mm recommended",
                            element_id=opening.get("id"),
                            code_reference="IBC 1010.1.1",
                            suggested_fix="Consider increasing door width to 32\" clear minimum"
                        ))
        
        # Check travel distances
        for space in spaces:
            space_type = space.get("type", "").lower()
            if "bed" in space_type:
                # Calculate travel distance to nearest exit
                travel_distance = self._calculate_travel_distance(space, plan_data)
                max_allowed = self.requirements["egress"]["travel_distance"]["residential"]
                
                if travel_distance > max_allowed:
                    issues.append(ValidationIssue(
                        code="EGRESS_003",
                        severity=ViolationSeverity.ERROR,
                        category="egress",
                        description=f"Travel distance too long: {travel_distance:.0f}mm > {max_allowed}mm max",
                        element_id=space.get("id"),
                        code_reference="IBC 1017.1",
                        suggested_fix="Add additional exit or reduce travel path",
                        impact_on_permit=True
                    ))
        
        return issues
    
    def validate_accessibility(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate ADA accessibility requirements"""
        issues = []
        
        # Check door clear widths
        for opening in plan_data.get("openings", []):
            if opening.get("type") == "door":
                clear_width = opening.get("dimensions", {}).get("width", 0) - 100
                if clear_width < self.requirements["accessibility"]["door_clear_width"]:
                    issues.append(ValidationIssue(
                        code="ADA_001",
                        severity=ViolationSeverity.ERROR,
                        category="accessibility",
                        description=f"Door not ADA compliant: {clear_width}mm < {self.requirements['accessibility']['door_clear_width']}mm required",
                        element_id=opening.get("id"),
                        code_reference="ADA 404.2.3",
                        suggested_fix="Increase door width to 32\" clear minimum",
                        impact_on_permit=True
                    ))
        
        # Check bathroom accessibility
        for space in plan_data.get("spaces", []):
            space_type = space.get("type", "").lower()
            if "bath" in space_type:
                area = space.get("area", 0)
                boundary = space.get("boundary", [])
                
                if boundary:
                    # Calculate room dimensions
                    xs = [p[0] for p in boundary]
                    ys = [p[1] for p in boundary]
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                    
                    # Check for 30" x 48" clear floor space
                    min_clear_area = 762 * 1219  # mm²
                    if min(width, height) < 762 or area < min_clear_area * 1.5:  # Allow some tolerance
                        issues.append(ValidationIssue(
                            code="ADA_002",
                            severity=ViolationSeverity.WARNING,
                            category="accessibility",
                            description="Bathroom may not have adequate clear floor space for accessibility",
                            element_id=space.get("id"),
                            code_reference="ADA 606.2",
                            suggested_fix="Ensure 30\" x 48\" clear floor space at fixtures"
                        ))
        
        return issues
    
    def validate_structural(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate structural requirements"""
        issues = []
        
        # Check span limitations
        structural = plan_data.get("structural", {})
        beams = structural.get("beams", [])
        
        for beam in beams:
            start = beam.get("start", [0, 0, 0])
            end = beam.get("end", [0, 0, 0])
            span = math.sqrt(sum((e - s)**2 for s, e in zip(start, end)))
            size = beam.get("size", "")
            
            # Simplified span check for wood beams
            if size.startswith("2x"):
                depth = int(size.split("x")[1]) * 25.4  # Convert inches to mm
                max_span = depth * 20  # Rule of thumb: depth * 20 for residential loads
                
                if span > max_span:
                    issues.append(ValidationIssue(
                        code="STRUCT_001",
                        severity=ViolationSeverity.ERROR,
                        category="structural",
                        description=f"Beam span may be excessive: {span:.0f}mm > {max_span:.0f}mm recommended",
                        element_id=beam.get("id"),
                        code_reference="IRC R502.3",
                        suggested_fix="Increase beam size or add intermediate support",
                        impact_on_permit=True
                    ))
        
        # Check column spacing
        columns = structural.get("columns", [])
        if len(columns) < 2:
            # No column spacing to check
            pass
        else:
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    loc1 = col1.get("location", [0, 0])
                    loc2 = col2.get("location", [0, 0])
                    distance = math.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)
                    
                    # Maximum spacing check (simplified)
                    max_spacing = 7315  # mm (24 feet)
                    if distance > max_spacing:
                        issues.append(ValidationIssue(
                            code="STRUCT_002",
                            severity=ViolationSeverity.WARNING,
                            category="structural",
                            description=f"Large column spacing: {distance:.0f}mm > {max_spacing}mm typical max",
                            code_reference="IRC R407.3",
                            suggested_fix="Consider additional columns or larger beam sizes"
                        ))
        
        return issues
    
    def validate_fire_safety(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate fire safety requirements"""
        issues = []
        
        # Check smoke detector coverage
        spaces = plan_data.get("spaces", [])
        bedroom_count = sum(1 for s in spaces if "bed" in s.get("type", "").lower())
        
        mep_systems = plan_data.get("mep_systems", {})
        fire_protection = mep_systems.get("fire_protection", {})
        smoke_detectors = fire_protection.get("smoke_detection", [])
        
        required_detectors = bedroom_count + 1  # One per bedroom + hallway
        if len(smoke_detectors) < required_detectors:
            issues.append(ValidationIssue(
                code="FIRE_001",
                severity=ViolationSeverity.ERROR,
                category="fire_safety",
                description=f"Insufficient smoke detectors: {len(smoke_detectors)} < {required_detectors} required",
                code_reference="IRC R314.3",
                suggested_fix="Add smoke detectors in each bedroom and hallway",
                impact_on_permit=True
            ))
        
        # Check sprinkler requirements
        total_area = sum(s.get("area", 0) for s in spaces)
        sprinkler_req = self.requirements["fire_safety"]["sprinkler_required"]
        
        if total_area > sprinkler_req["area_threshold"]:
            sprinkler_system = fire_protection.get("sprinkler_system", {})
            if not sprinkler_system.get("required", False):
                issues.append(ValidationIssue(
                    code="FIRE_002",
                    severity=ViolationSeverity.WARNING,
                    category="fire_safety",
                    description=f"Large building area may require sprinkler system: {total_area:.0f}mm² > {sprinkler_req['area_threshold']}mm²",
                    code_reference="IBC 903.2.8",
                    suggested_fix="Consider automatic sprinkler system"
                ))
        
        return issues
    
    def validate_energy_code(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate energy code compliance (IECC)"""
        issues = []
        
        energy_model = plan_data.get("energy_model", {})
        building_envelope = energy_model.get("building_envelope", {})
        
        # Check U-factors
        requirements = self.requirements["energy"]["iecc_2021"]["u_factors"]
        
        wall_u = building_envelope.get("wall_u_factor", 999)  # Default high value
        if wall_u > requirements["walls"]:
            issues.append(ValidationIssue(
                code="ENERGY_001",
                severity=ViolationSeverity.WARNING,
                category="energy",
                description=f"Wall U-factor too high: {wall_u:.2f} > {requirements['walls']:.2f} W/m²·K max",
                code_reference="IECC C402.1.3",
                suggested_fix="Increase wall insulation or improve thermal bridging"
            ))
        
        window_u = building_envelope.get("window_u_factor", 999)
        if window_u > requirements["windows"]:
            issues.append(ValidationIssue(
                code="ENERGY_002",
                severity=ViolationSeverity.WARNING,
                category="energy",
                description=f"Window U-factor too high: {window_u:.2f} > {requirements['windows']:.2f} W/m²·K max",
                code_reference="IECC C402.1.3",
                suggested_fix="Specify higher performance windows"
            ))
        
        # Check air leakage
        air_leakage = building_envelope.get("air_leakage_rate", 999)
        max_leakage = self.requirements["energy"]["iecc_2021"]["air_leakage"]
        if air_leakage > max_leakage:
            issues.append(ValidationIssue(
                code="ENERGY_003",
                severity=ViolationSeverity.ERROR,
                category="energy",
                description=f"Building air leakage too high: {air_leakage:.1f} > {max_leakage:.1f} ACH50 max",
                code_reference="IECC C402.4.1.2",
                suggested_fix="Improve air sealing details and construction",
                impact_on_permit=True
            ))
        
        return issues
    
    def _calculate_travel_distance(self, space: Dict[str, Any], plan_data: Dict[str, Any]) -> float:
        """Calculate travel distance from space to nearest exit"""
        
        # Get space center
        boundary = space.get("boundary", [])
        if not boundary:
            return 0
        
        space_center = [
            sum(p[0] for p in boundary) / len(boundary),
            sum(p[1] for p in boundary) / len(boundary)
        ]
        
        # Find nearest exterior door
        min_distance = float('inf')
        
        for opening in plan_data.get("openings", []):
            if opening.get("type") == "door":
                wall_id = opening.get("wall_id")
                wall = next((w for w in plan_data.get("walls", []) if w.get("id") == wall_id), None)
                
                if wall and wall.get("type") == "exterior":
                    # Calculate door location
                    geometry = wall.get("geometry", {})
                    if geometry.get("type") == "straight":
                        start = geometry["start"]
                        end = geometry["end"]
                        position = opening.get("position", 0.5)
                        
                        door_location = [
                            start[0] + position * (end[0] - start[0]),
                            start[1] + position * (end[1] - start[1])
                        ]
                        
                        # Calculate distance (simplified straight-line)
                        distance = math.sqrt(
                            (door_location[0] - space_center[0])**2 +
                            (door_location[1] - space_center[1])**2
                        )
                        
                        min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0


class MEPSystemValidator:
    """Validates MEP (Mechanical, Electrical, Plumbing) systems"""
    
    def validate_hvac_system(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate HVAC system design"""
        issues = []
        
        mep_systems = plan_data.get("mep_systems", {})
        hvac = mep_systems.get("hvac", {})
        
        # Check if HVAC system is defined
        if not hvac:
            issues.append(ValidationIssue(
                code="HVAC_001",
                severity=ViolationSeverity.ERROR,
                category="hvac",
                description="No HVAC system defined",
                code_reference="IMC 403.1",
                suggested_fix="Add HVAC system design",
                impact_on_permit=True
            ))
            return issues
        
        # Check equipment sizing
        equipment = hvac.get("equipment", [])
        spaces = plan_data.get("spaces", [])
        total_area = sum(s.get("area", 0) for s in spaces) / 1_000_000  # Convert mm² to m²
        
        total_capacity = sum(eq.get("capacity", 0) for eq in equipment)
        
        # Rule of thumb: 100-150 W/m² for residential
        required_capacity = total_area * 125  # W
        
        if total_capacity < required_capacity * 0.8:  # 80% tolerance
            issues.append(ValidationIssue(
                code="HVAC_002",
                severity=ViolationSeverity.WARNING,
                category="hvac",
                description=f"HVAC capacity may be insufficient: {total_capacity:.0f}W < {required_capacity:.0f}W estimated",
                code_reference="IMC 403.2",
                suggested_fix="Increase equipment capacity or add additional units"
            ))
        
        # Check ductwork
        ductwork = hvac.get("ductwork", [])
        zones = hvac.get("zones", [])
        
        if zones and not ductwork:
            issues.append(ValidationIssue(
                code="HVAC_003",
                severity=ViolationSeverity.ERROR,
                category="hvac",
                description="HVAC zones defined but no ductwork specified",
                code_reference="IMC 601.1",
                suggested_fix="Add ductwork distribution system",
                impact_on_permit=True
            ))
        
        return issues
    
    def validate_plumbing_system(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate plumbing system design"""
        issues = []
        
        mep_systems = plan_data.get("mep_systems", {})
        plumbing = mep_systems.get("plumbing", {})
        
        fixtures = plumbing.get("fixtures", [])
        plumbing.get("supply_lines", [])
        plumbing.get("waste_lines", [])
        
        # Check fixture count vs. spaces
        bathroom_spaces = [s for s in plan_data.get("spaces", []) if "bath" in s.get("type", "").lower()]
        toilet_fixtures = [f for f in fixtures if f.get("type") == "toilet"]
        
        if len(bathroom_spaces) != len(toilet_fixtures):
            issues.append(ValidationIssue(
                code="PLUMB_001",
                severity=ViolationSeverity.ERROR,
                category="plumbing",
                description=f"Toilet fixture count mismatch: {len(toilet_fixtures)} fixtures for {len(bathroom_spaces)} bathrooms",
                code_reference="IPC 403.1",
                suggested_fix="Add toilet fixture for each bathroom",
                impact_on_permit=True
            ))
        
        # Check water heater capacity
        water_heater = plumbing.get("water_heater", {})
        if water_heater:
            capacity = water_heater.get("capacity", 0)  # Liters
            len(fixtures)
            
            # Rule of thumb: 150L for first bathroom, 75L for each additional
            required_capacity = 150 + (len(bathroom_spaces) - 1) * 75
            
            if capacity < required_capacity * 0.8:
                issues.append(ValidationIssue(
                    code="PLUMB_002",
                    severity=ViolationSeverity.WARNING,
                    category="plumbing",
                    description=f"Water heater capacity may be insufficient: {capacity}L < {required_capacity}L recommended",
                    code_reference="IPC 504.1",
                    suggested_fix="Increase water heater capacity"
                ))
        
        return issues
    
    def validate_electrical_system(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate electrical system design"""
        issues = []
        
        mep_systems = plan_data.get("mep_systems", {})
        electrical = mep_systems.get("electrical", {})
        
        service = electrical.get("service", {})
        electrical.get("panels", [])
        outlets = electrical.get("outlets", [])
        
        # Check service size
        amperage = service.get("amperage", 0)
        spaces = plan_data.get("spaces", [])
        total_area = sum(s.get("area", 0) for s in spaces) / 92903  # Convert mm² to sq ft
        
        # NEC requirement: minimum 100A for dwelling
        min_service = 100
        if total_area > 1200:  # sq ft
            min_service = 200
        
        if amperage < min_service:
            issues.append(ValidationIssue(
                code="ELEC_001",
                severity=ViolationSeverity.ERROR,
                category="electrical",
                description=f"Electrical service undersized: {amperage}A < {min_service}A required",
                code_reference="NEC 220.82",
                suggested_fix=f"Upgrade to minimum {min_service}A service",
                impact_on_permit=True
            ))
        
        # Check outlet spacing
        living_spaces = [s for s in spaces if s.get("type", "").lower() in ["living", "bedroom", "dining", "family"]]
        
        for space in living_spaces:
            space_outlets = [o for o in outlets if self._point_in_polygon(o.get("location", [0, 0]), space.get("boundary", []))]
            
            # NEC requires outlets every 12 feet (3.6m) along walls
            boundary = space.get("boundary", [])
            if boundary:
                perimeter = self._calculate_perimeter(boundary)
                required_outlets = max(2, perimeter / 3600)  # mm to 12-foot spacing
                
                if len(space_outlets) < required_outlets * 0.8:  # 80% tolerance
                    issues.append(ValidationIssue(
                        code="ELEC_002",
                        severity=ViolationSeverity.WARNING,
                        category="electrical",
                        description=f"Insufficient outlets in {space.get('name', 'space')}: {len(space_outlets)} < {required_outlets:.0f} estimated",
                        element_id=space.get("id"),
                        code_reference="NEC 210.52(A)",
                        suggested_fix="Add outlets to meet 12-foot spacing requirement"
                    ))
        
        return issues
    
    def _point_in_polygon(self, point: List[float], polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _calculate_perimeter(self, boundary: List[List[float]]) -> float:
        """Calculate polygon perimeter"""
        perimeter = 0
        for i in range(len(boundary)):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % len(boundary)]
            perimeter += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return perimeter


class ArchitecturalQualityValidator:
    """Validates architectural design quality and best practices"""
    
    def validate_space_planning(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate space planning and layout quality"""
        issues = []
        
        spaces = plan_data.get("spaces", [])
        
        # Check room sizes
        for space in spaces:
            space_type = space.get("type", "").lower()
            area = space.get("area", 0) / 1_000_000  # Convert mm² to m²
            
            min_areas = {
                "bedroom": 9.3,      # 100 sq ft
                "master_bedroom": 14.9,  # 160 sq ft
                "bathroom": 3.7,     # 40 sq ft
                "kitchen": 9.3,      # 100 sq ft
                "living": 18.6,      # 200 sq ft
                "dining": 11.1       # 120 sq ft
            }
            
            if space_type in min_areas and area < min_areas[space_type]:
                issues.append(ValidationIssue(
                    code="ARCH_001",
                    severity=ViolationSeverity.WARNING,
                    category="architectural",
                    description=f"{space.get('name', space_type)} undersized: {area:.1f}m² < {min_areas[space_type]:.1f}m² recommended",
                    element_id=space.get("id"),
                    suggested_fix="Increase room size for better functionality"
                ))
        
        # Check room proportions
        for space in spaces:
            boundary = space.get("boundary", [])
            if len(boundary) >= 4:
                xs = [p[0] for p in boundary]
                ys = [p[1] for p in boundary]
                width = (max(xs) - min(xs)) / 1000  # Convert to meters
                height = (max(ys) - min(ys)) / 1000
                
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    if aspect_ratio > 3.0:  # Very elongated room
                        issues.append(ValidationIssue(
                            code="ARCH_002",
                            severity=ViolationSeverity.INFO,
                            category="architectural",
                            description=f"{space.get('name', 'Room')} has poor proportions: {aspect_ratio:.1f}:1 ratio",
                            element_id=space.get("id"),
                            suggested_fix="Consider subdividing or reconfiguring room"
                        ))
        
        # Check circulation paths
        circulation_issues = self._analyze_circulation(plan_data)
        issues.extend(circulation_issues)
        
        return issues
    
    def validate_daylighting(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate natural lighting and window placement"""
        issues = []
        
        spaces = plan_data.get("spaces", [])
        openings = plan_data.get("openings", [])
        walls = plan_data.get("walls", [])
        
        # Check window to floor area ratios
        for space in spaces:
            space_type = space.get("type", "").lower()
            
            # Skip non-habitable spaces
            if space_type in ["bathroom", "utility", "storage", "garage"]:
                continue
            
            space_area = space.get("area", 0)
            space_windows = self._find_space_windows(space, openings, walls)
            
            total_window_area = sum(
                opening.get("dimensions", {}).get("width", 0) * 
                opening.get("dimensions", {}).get("height", 1400)
                for opening in space_windows
            )
            
            if space_area > 0:
                window_ratio = total_window_area / space_area
                min_ratio = 0.10  # 10% minimum
                
                if window_ratio < min_ratio:
                    issues.append(ValidationIssue(
                        code="LIGHT_001",
                        severity=ViolationSeverity.WARNING,
                        category="daylighting",
                        description=f"{space.get('name', 'Space')} insufficient daylighting: {window_ratio:.1%} < {min_ratio:.1%} recommended",
                        element_id=space.get("id"),
                        suggested_fix="Add windows or increase window size"
                    ))
        
        return issues
    
    def validate_accessibility_design(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate accessibility design beyond code compliance"""
        issues = []
        
        # Check for accessible route through building
        spaces = plan_data.get("spaces", [])
        plan_data.get("openings", [])
        
        # Find entrance
        entrance_spaces = [s for s in spaces if "entrance" in s.get("type", "").lower()]
        
        if not entrance_spaces:
            issues.append(ValidationIssue(
                code="ACCESS_001",
                severity=ViolationSeverity.WARNING,
                category="accessibility",
                description="No designated entrance space identified",
                suggested_fix="Designate and design accessible entrance"
            ))
        
        # Check bathroom accessibility features
        bathroom_spaces = [s for s in spaces if "bath" in s.get("type", "").lower()]
        
        for bathroom in bathroom_spaces:
            # Check if at least one bathroom is fully accessible
            area = bathroom.get("area", 0) / 1_000_000  # m²
            
            # Full accessible bathroom needs ~6m² minimum
            if area < 5.5:
                issues.append(ValidationIssue(
                    code="ACCESS_002",
                    severity=ViolationSeverity.INFO,
                    category="accessibility",
                    description=f"Bathroom {bathroom.get('name', '')} may not accommodate full accessibility: {area:.1f}m²",
                    element_id=bathroom.get("id"),
                    suggested_fix="Consider enlarging at least one bathroom for full accessibility"
                ))
        
        return issues
    
    def _analyze_circulation(self, plan_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Analyze circulation paths and flow"""
        issues = []
        
        spaces = plan_data.get("spaces", [])
        openings = plan_data.get("openings", [])
        walls = plan_data.get("walls", [])
        
        # Check for dead-end corridors
        corridor_spaces = [s for s in spaces if "corridor" in s.get("type", "").lower() or "hall" in s.get("name", "").lower()]
        
        for corridor in corridor_spaces:
            # Count doors opening to corridor
            corridor_doors = self._find_space_openings(corridor, openings, walls)
            
            if len(corridor_doors) < 2:
                issues.append(ValidationIssue(
                    code="CIRC_001",
                    severity=ViolationSeverity.WARNING,
                    category="circulation",
                    description=f"Potential dead-end corridor: {corridor.get('name', 'corridor')}",
                    element_id=corridor.get("id"),
                    suggested_fix="Provide circulation loop or second exit"
                ))
        
        # Check bedroom access (shouldn't go through other bedrooms)
        bedroom_spaces = [s for s in spaces if "bed" in s.get("type", "").lower()]
        
        for bedroom in bedroom_spaces:
            # This would require more complex graph analysis
            # For now, just flag private bedrooms with multiple doors
            bedroom_doors = self._find_space_openings(bedroom, openings, walls)
            
            if len(bedroom_doors) > 1:
                issues.append(ValidationIssue(
                    code="CIRC_002",
                    severity=ViolationSeverity.INFO,
                    category="circulation",
                    description=f"Bedroom {bedroom.get('name', '')} has multiple doors - check privacy",
                    element_id=bedroom.get("id"),
                    suggested_fix="Ensure bedroom privacy in circulation design"
                ))
        
        return issues
    
    def _find_space_windows(self, space: Dict[str, Any], openings: List[Dict[str, Any]], walls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find windows that serve a specific space"""
        space_windows = []
        space_boundary = space.get("boundary", [])
        
        for opening in openings:
            if opening.get("type") == "window":
                wall_id = opening.get("wall_id")
                wall = next((w for w in walls if w.get("id") == wall_id), None)
                
                if wall and self._wall_bounds_space(wall, space_boundary):
                    space_windows.append(opening)
        
        return space_windows
    
    def _find_space_openings(self, space: Dict[str, Any], openings: List[Dict[str, Any]], walls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find all openings (doors/windows) that serve a specific space"""
        space_openings = []
        space_boundary = space.get("boundary", [])
        
        for opening in openings:
            wall_id = opening.get("wall_id")
            wall = next((w for w in walls if w.get("id") == wall_id), None)
            
            if wall and self._wall_bounds_space(wall, space_boundary):
                space_openings.append(opening)
        
        return space_openings
    
    def _wall_bounds_space(self, wall: Dict[str, Any], space_boundary: List[List[float]]) -> bool:
        """Check if wall forms part of space boundary"""
        geometry = wall.get("geometry", {})
        
        if geometry.get("type") == "straight":
            start = geometry["start"]
            end = geometry["end"]
            
            # Check if wall endpoints are close to space boundary
            tolerance = 100  # mm
            
            for i in range(len(space_boundary)):
                p1 = space_boundary[i]
                p2 = space_boundary[(i + 1) % len(space_boundary)]
                
                # Check if wall aligns with boundary segment
                if (self._point_distance(start, p1) < tolerance and self._point_distance(end, p2) < tolerance) or \
                   (self._point_distance(start, p2) < tolerance and self._point_distance(end, p1) < tolerance):
                    return True
        
        return False
    
    def _point_distance(self, p1: List[float], p2: List[float]) -> float:
        """Calculate distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


class ProfessionalValidator:
    """Main professional validation class that orchestrates all validation checks"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.PROFESSIONAL):
        self.validation_level = validation_level
        self.building_code_validator = BuildingCodeValidator()
        self.mep_validator = MEPSystemValidator()
        self.architectural_validator = ArchitecturalQualityValidator()
    
    def validate_plan(self, plan_data: Dict[str, Any]) -> ValidationResult:
        """Perform comprehensive validation of architectural plan"""
        
        all_issues = []
        
        # Building code compliance
        all_issues.extend(self.building_code_validator.validate_egress(plan_data))
        all_issues.extend(self.building_code_validator.validate_accessibility(plan_data))
        all_issues.extend(self.building_code_validator.validate_structural(plan_data))
        all_issues.extend(self.building_code_validator.validate_fire_safety(plan_data))
        all_issues.extend(self.building_code_validator.validate_energy_code(plan_data))
        
        # MEP systems
        all_issues.extend(self.mep_validator.validate_hvac_system(plan_data))
        all_issues.extend(self.mep_validator.validate_plumbing_system(plan_data))
        all_issues.extend(self.mep_validator.validate_electrical_system(plan_data))
        
        # Architectural quality
        if self.validation_level in [ValidationLevel.PROFESSIONAL, ValidationLevel.PERMIT_READY]:
            all_issues.extend(self.architectural_validator.validate_space_planning(plan_data))
            all_issues.extend(self.architectural_validator.validate_daylighting(plan_data))
            all_issues.extend(self.architectural_validator.validate_accessibility_design(plan_data))
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(all_issues)
        
        # Determine overall validity
        critical_errors = [i for i in all_issues if i.severity == ViolationSeverity.CRITICAL]
        errors = [i for i in all_issues if i.severity == ViolationSeverity.ERROR]
        permit_blockers = [i for i in all_issues if i.impact_on_permit]
        
        is_valid = len(critical_errors) == 0 and (
            self.validation_level == ValidationLevel.BASIC or 
            (self.validation_level == ValidationLevel.PROFESSIONAL and len(errors) == 0) or
            (self.validation_level == ValidationLevel.PERMIT_READY and len(permit_blockers) == 0)
        )
        
        # Generate summary
        summary = self._generate_summary(all_issues, plan_data)
        
        # Check code compliance by category
        code_compliance = self._assess_code_compliance(all_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, plan_data)
        
        return ValidationResult(
            is_valid=is_valid,
            compliance_score=compliance_score,
            issues=all_issues,
            summary=summary,
            code_compliance=code_compliance,
            recommendations=recommendations
        )
    
    def _calculate_compliance_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall compliance score (0-100)"""
        
        if not issues:
            return 100.0
        
        # Weight by severity
        severity_weights = {
            ViolationSeverity.CRITICAL: 25,
            ViolationSeverity.ERROR: 10,
            ViolationSeverity.WARNING: 3,
            ViolationSeverity.INFO: 1
        }
        
        total_deductions = sum(severity_weights.get(issue.severity, 1) for issue in issues)
        
        # Base score starts at 100, deduct based on issues
        score = max(0, 100 - total_deductions)
        
        return score
    
    def _generate_summary(self, issues: List[ValidationIssue], plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        
        issue_counts = {
            "critical": len([i for i in issues if i.severity == ViolationSeverity.CRITICAL]),
            "errors": len([i for i in issues if i.severity == ViolationSeverity.ERROR]),
            "warnings": len([i for i in issues if i.severity == ViolationSeverity.WARNING]),
            "info": len([i for i in issues if i.severity == ViolationSeverity.INFO])
        }
        
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        permit_blockers = len([i for i in issues if i.impact_on_permit])
        
        # Basic plan statistics
        spaces = plan_data.get("spaces", [])
        total_area = sum(s.get("area", 0) for s in spaces) / 1_000_000  # m²
        
        return {
            "total_issues": len(issues),
            "issue_counts": issue_counts,
            "category_breakdown": category_counts,
            "permit_blocking_issues": permit_blockers,
            "plan_statistics": {
                "total_area_m2": total_area,
                "space_count": len(spaces),
                "level_count": len(plan_data.get("levels", [])),
                "wall_count": len(plan_data.get("walls", [])),
                "opening_count": len(plan_data.get("openings", []))
            }
        }
    
    def _assess_code_compliance(self, issues: List[ValidationIssue]) -> Dict[str, bool]:
        """Assess compliance by code category"""
        
        categories = ["egress", "accessibility", "structural", "fire_safety", "energy", "hvac", "plumbing", "electrical"]
        compliance = {}
        
        for category in categories:
            category_errors = [i for i in issues if i.category == category and i.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.ERROR]]
            compliance[category] = len(category_errors) == 0
        
        return compliance
    
    def _generate_recommendations(self, issues: List[ValidationIssue], plan_data: Dict[str, Any]) -> List[str]:
        """Generate prioritized recommendations"""
        
        recommendations = []
        
        # Prioritize by permit impact and severity
        critical_issues = [i for i in issues if i.severity == ViolationSeverity.CRITICAL]
        permit_issues = [i for i in issues if i.impact_on_permit]
        
        if critical_issues:
            recommendations.append("Address critical safety issues immediately - these prevent occupancy")
        
        if permit_issues:
            recommendations.append("Resolve permit-blocking issues before submission")
        
        # Category-specific recommendations
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in top_categories[:3]:  # Top 3 problem areas
            if count >= 3:
                recommendations.append(f"Focus on {category} system - {count} issues identified")
        
        # Generic quality recommendations
        if len(issues) > 10:
            recommendations.append("Consider design review with licensed architect")
        
        return recommendations


def validate_professional_plan(plan_path: str, validation_level: ValidationLevel = ValidationLevel.PROFESSIONAL) -> ValidationResult:
    """Main function to validate a professional architectural plan"""
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan_data = json.load(f)
    
    validator = ProfessionalValidator(validation_level)
    result = validator.validate_plan(plan_data)
    
    return result


def generate_validation_report(validation_result: ValidationResult, output_path: str):
    """Generate detailed validation report"""
    
    report = {
        "validation_summary": {
            "is_valid": validation_result.is_valid,
            "compliance_score": validation_result.compliance_score,
            "total_issues": len(validation_result.issues)
        },
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity.value,
                "category": issue.category,
                "description": issue.description,
                "location": issue.location,
                "element_id": issue.element_id,
                "code_reference": issue.code_reference,
                "suggested_fix": issue.suggested_fix,
                "impact_on_permit": issue.impact_on_permit
            }
            for issue in validation_result.issues
        ],
        "summary": validation_result.summary,
        "code_compliance": validation_result.code_compliance,
        "recommendations": validation_result.recommendations,
        "validation_metadata": {
            "generated_by": "HouseBrain Professional Validator v3.0",
            "validation_level": "professional",
            "standards_checked": ["IBC 2021", "ADA 2010", "IECC 2021", "NEC 2020", "IMC 2021", "IPC 2021"]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Validation report generated: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Building Code Validation")
    parser.add_argument("--input", required=True, help="HouseBrain plan JSON file")
    parser.add_argument("--output", help="Validation report output file")
    parser.add_argument("--level", choices=["basic", "professional", "code_compliance", "permit_ready"], 
                       default="professional", help="Validation level")
    
    args = parser.parse_args()
    
    validation_level = ValidationLevel(args.level)
    result = validate_professional_plan(args.input, validation_level)
    
    print("🏗️ Validation Complete")
    print(f"   Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")
    print(f"   Compliance Score: {result.compliance_score:.1f}/100")
    print(f"   Total Issues: {len(result.issues)}")
    
    if result.issues:
        print("\n📋 Issues Summary:")
        for category, compliant in result.code_compliance.items():
            status = "✅" if compliant else "❌"
            print(f"   {status} {category.replace('_', ' ').title()}")
    
    if args.output:
        generate_validation_report(result, args.output)