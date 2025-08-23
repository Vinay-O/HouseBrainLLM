"""
Advanced Measurement Tools for HouseBrain 3D Viewer

This module provides comprehensive measurement capabilities:
- Distance measurements (point-to-point, multi-point)
- Area measurements (polygon areas, surface areas)
- Angle measurements (2D and 3D angles)
- Volume calculations
- Material quantity takeoffs
- Real-time measurement display
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class Point3D:
    """3D point with x, y, z coordinates"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate distance to another point"""
        return math.sqrt(
            (self.x - other.x) ** 2 + 
            (self.y - other.y) ** 2 + 
            (self.z - other.z) ** 2
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class MeasurementResult:
    """Result of a measurement operation"""
    measurement_type: str
    value: float
    unit: str
    points: List[Point3D]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.measurement_type,
            "value": self.value,
            "unit": self.unit,
            "points": [p.to_dict() for p in self.points],
            "metadata": self.metadata
        }


class AdvancedMeasurementTools:
    """Advanced measurement tools for 3D architectural models"""
    
    def __init__(self, unit_system: str = "metric"):
        self.unit_system = unit_system
        self.measurements = []
        self.active_measurement = None
        
        # Unit conversion factors (to mm)
        self.unit_factors = {
            "mm": 1.0,
            "cm": 10.0,
            "m": 1000.0,
            "inches": 25.4,
            "feet": 304.8
        }
        
        # Default units by system
        self.default_units = {
            "metric": {"linear": "mm", "area": "m²", "volume": "m³", "angle": "degrees"},
            "imperial": {"linear": "feet", "area": "sq ft", "volume": "cu ft", "angle": "degrees"}
        }
        
        print("📐 Advanced Measurement Tools Initialized")
    
    def measure_distance(
        self, 
        start_point: Point3D, 
        end_point: Point3D,
        display_unit: str = None
    ) -> MeasurementResult:
        """Measure distance between two points"""
        
        distance_mm = start_point.distance_to(end_point)
        
        if display_unit is None:
            display_unit = self.default_units[self.unit_system]["linear"]
        
        # Convert to display unit
        display_value = distance_mm / self.unit_factors[display_unit]
        
        measurement = MeasurementResult(
            measurement_type="distance",
            value=display_value,
            unit=display_unit,
            points=[start_point, end_point],
            metadata={
                "raw_value_mm": distance_mm,
                "measurement_method": "point_to_point"
            }
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def measure_multi_point_distance(
        self, 
        points: List[Point3D],
        display_unit: str = None
    ) -> MeasurementResult:
        """Measure total distance along a path of multiple points"""
        
        if len(points) < 2:
            raise ValueError("At least 2 points required for distance measurement")
        
        total_distance_mm = 0.0
        
        for i in range(len(points) - 1):
            segment_distance = points[i].distance_to(points[i + 1])
            total_distance_mm += segment_distance
        
        if display_unit is None:
            display_unit = self.default_units[self.unit_system]["linear"]
        
        display_value = total_distance_mm / self.unit_factors[display_unit]
        
        measurement = MeasurementResult(
            measurement_type="multi_point_distance",
            value=display_value,
            unit=display_unit,
            points=points,
            metadata={
                "raw_value_mm": total_distance_mm,
                "segment_count": len(points) - 1,
                "measurement_method": "multi_point_path"
            }
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def measure_area_polygon(
        self, 
        points: List[Point3D],
        display_unit: str = None
    ) -> MeasurementResult:
        """Measure area of a polygon defined by points"""
        
        if len(points) < 3:
            raise ValueError("At least 3 points required for area measurement")
        
        # Project to 2D plane for area calculation (assume Z=0 plane for simplicity)
        area_mm2 = self._calculate_polygon_area_2d([(p.x, p.y) for p in points])
        
        if display_unit is None:
            display_unit = self.default_units[self.unit_system]["area"]
        
        # Convert to display unit
        if display_unit == "m²":
            display_value = area_mm2 / (1000.0 * 1000.0)
        elif display_unit == "sq ft":
            display_value = area_mm2 / (304.8 * 304.8)
        elif display_unit == "sq in":
            display_value = area_mm2 / (25.4 * 25.4)
        else:
            display_value = area_mm2  # mm²
            display_unit = "mm²"
        
        measurement = MeasurementResult(
            measurement_type="area_polygon",
            value=display_value,
            unit=display_unit,
            points=points,
            metadata={
                "raw_value_mm2": area_mm2,
                "vertex_count": len(points),
                "measurement_method": "polygon_area"
            }
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def measure_angle_3_points(
        self, 
        point1: Point3D, 
        vertex: Point3D, 
        point2: Point3D
    ) -> MeasurementResult:
        """Measure angle defined by three points (vertex is the angle point)"""
        
        # Create vectors from vertex to the other two points
        v1 = Point3D(point1.x - vertex.x, point1.y - vertex.y, point1.z - vertex.z)
        v2 = Point3D(point2.x - vertex.x, point2.y - vertex.y, point2.z - vertex.z)
        
        # Calculate angle using dot product
        dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        magnitude1 = math.sqrt(v1.x**2 + v1.y**2 + v1.z**2)
        magnitude2 = math.sqrt(v2.x**2 + v2.y**2 + v2.z**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            raise ValueError("Invalid points for angle measurement")
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        # Clamp to valid range for acos
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        angle_radians = math.acos(cos_angle)
        angle_degrees = math.degrees(angle_radians)
        
        measurement = MeasurementResult(
            measurement_type="angle_3_points",
            value=angle_degrees,
            unit="degrees",
            points=[point1, vertex, point2],
            metadata={
                "raw_value_radians": angle_radians,
                "measurement_method": "three_point_angle"
            }
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def measure_volume_box(
        self, 
        corner1: Point3D, 
        corner2: Point3D,
        display_unit: str = None
    ) -> MeasurementResult:
        """Measure volume of a rectangular box defined by two opposite corners"""
        
        width = abs(corner2.x - corner1.x)
        height = abs(corner2.y - corner1.y)
        depth = abs(corner2.z - corner1.z)
        
        volume_mm3 = width * height * depth
        
        if display_unit is None:
            display_unit = self.default_units[self.unit_system]["volume"]
        
        # Convert to display unit
        if display_unit == "m³":
            display_value = volume_mm3 / (1000.0 ** 3)
        elif display_unit == "cu ft":
            display_value = volume_mm3 / (304.8 ** 3)
        elif display_unit == "liters":
            display_value = volume_mm3 / (1000.0 ** 3) * 1000.0  # m³ to liters
        else:
            display_value = volume_mm3  # mm³
            display_unit = "mm³"
        
        measurement = MeasurementResult(
            measurement_type="volume_box",
            value=display_value,
            unit=display_unit,
            points=[corner1, corner2],
            metadata={
                "raw_value_mm3": volume_mm3,
                "dimensions_mm": {"width": width, "height": height, "depth": depth},
                "measurement_method": "rectangular_volume"
            }
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def calculate_material_quantities(
        self,
        geometry_data: Dict[str, Any],
        material_name: str
    ) -> Dict[str, MeasurementResult]:
        """Calculate material quantities from geometry data"""
        
        quantities = {}
        
        # Calculate wall areas for wall materials
        if "walls" in geometry_data:
            wall_areas = []
            total_wall_area = 0.0
            
            for wall in geometry_data["walls"]:
                if wall.get("material") == material_name:
                    area = self._calculate_wall_area(wall)
                    wall_areas.append(area)
                    total_wall_area += area
            
            if total_wall_area > 0:
                quantities["wall_area"] = MeasurementResult(
                    measurement_type="material_quantity_area",
                    value=total_wall_area / (1000.0 * 1000.0),  # Convert to m²
                    unit="m²",
                    points=[],
                    metadata={
                        "material": material_name,
                        "wall_count": len(wall_areas),
                        "individual_areas": wall_areas
                    }
                )
        
        # Calculate floor areas
        if "spaces" in geometry_data:
            floor_areas = []
            total_floor_area = 0.0
            
            for space in geometry_data["spaces"]:
                if space.get("floor_material") == material_name:
                    area = space.get("area", 0) / (1000.0 * 1000.0)  # Convert to m²
                    floor_areas.append(area)
                    total_floor_area += area
            
            if total_floor_area > 0:
                quantities["floor_area"] = MeasurementResult(
                    measurement_type="material_quantity_area",
                    value=total_floor_area,
                    unit="m²",
                    points=[],
                    metadata={
                        "material": material_name,
                        "space_count": len(floor_areas),
                        "individual_areas": floor_areas
                    }
                )
        
        return quantities
    
    def get_measurement_summary(self) -> Dict[str, Any]:
        """Get summary of all measurements"""
        
        summary = {
            "total_measurements": len(self.measurements),
            "measurement_types": {},
            "unit_system": self.unit_system,
            "measurements": [m.to_dict() for m in self.measurements]
        }
        
        # Group by measurement type
        for measurement in self.measurements:
            mtype = measurement.measurement_type
            if mtype not in summary["measurement_types"]:
                summary["measurement_types"][mtype] = 0
            summary["measurement_types"][mtype] += 1
        
        return summary
    
    def clear_measurements(self):
        """Clear all measurements"""
        self.measurements.clear()
        self.active_measurement = None
    
    def export_measurements(self, format_type: str = "json") -> str:
        """Export measurements to specified format"""
        
        summary = self.get_measurement_summary()
        
        if format_type == "json":
            import json
            return json.dumps(summary, indent=2)
        
        elif format_type == "csv":
            csv_lines = ["Type,Value,Unit,Points"]
            for measurement in self.measurements:
                points_str = f"{len(measurement.points)} points"
                csv_lines.append(f"{measurement.measurement_type},{measurement.value},{measurement.unit},{points_str}")
            return "\n".join(csv_lines)
        
        elif format_type == "text":
            lines = [f"Measurement Summary ({self.unit_system})"]
            lines.append("=" * 40)
            
            for i, measurement in enumerate(self.measurements, 1):
                lines.append(f"{i}. {measurement.measurement_type.replace('_', ' ').title()}")
                lines.append(f"   Value: {measurement.value:.3f} {measurement.unit}")
                lines.append(f"   Points: {len(measurement.points)}")
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def generate_measurement_report(self, house_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive measurement report for a house design"""
        
        report = {
            "report_metadata": {
                "unit_system": self.unit_system,
                "measurement_count": len(self.measurements),
                "report_type": "comprehensive_measurement_analysis"
            },
            "dimensional_analysis": {},
            "area_analysis": {},
            "volume_analysis": {},
            "material_quantities": {},
            "compliance_checks": {}
        }
        
        # Analyze building dimensions
        if "geometry" in house_data:
            geometry = house_data["geometry"]
            
            # Calculate overall building dimensions
            if "spaces" in geometry:
                all_points = []
                for space in geometry["spaces"]:
                    boundary = space.get("boundary", [])
                    for point in boundary:
                        if len(point) >= 2:
                            all_points.append(Point3D(point[0], point[1], point[2] if len(point) > 2 else 0))
                
                if all_points:
                    min_x = min(p.x for p in all_points)
                    max_x = max(p.x for p in all_points)
                    min_y = min(p.y for p in all_points)
                    max_y = max(p.y for p in all_points)
                    
                    building_width = max_x - min_x
                    building_depth = max_y - min_y
                    
                    report["dimensional_analysis"] = {
                        "building_width_mm": building_width,
                        "building_depth_mm": building_depth,
                        "building_width_m": building_width / 1000.0,
                        "building_depth_m": building_depth / 1000.0,
                        "footprint_area_m2": (building_width * building_depth) / (1000.0 * 1000.0)
                    }
            
            # Calculate total areas by space type
            if "spaces" in geometry:
                space_areas = {}
                total_area = 0.0
                
                for space in geometry["spaces"]:
                    space_type = space.get("type", "unknown")
                    area_mm2 = space.get("area", 0)
                    area_m2 = area_mm2 / (1000.0 * 1000.0)
                    
                    if space_type not in space_areas:
                        space_areas[space_type] = 0.0
                    space_areas[space_type] += area_m2
                    total_area += area_m2
                
                report["area_analysis"] = {
                    "total_floor_area_m2": total_area,
                    "areas_by_type": space_areas,
                    "space_count": len(geometry["spaces"])
                }
        
        return report
    
    # Helper methods
    
    def _calculate_polygon_area_2d(self, points: List[Tuple[float, float]]) -> float:
        """Calculate area of 2D polygon using shoelace formula"""
        
        if len(points) < 3:
            return 0.0
        
        # Ensure polygon is closed
        if points[0] != points[-1]:
            points = points + [points[0]]
        
        area = 0.0
        for i in range(len(points) - 1):
            area += points[i][0] * points[i + 1][1]
            area -= points[i + 1][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def _calculate_wall_area(self, wall: Dict[str, Any]) -> float:
        """Calculate area of a wall from geometry data"""
        
        geometry = wall.get("geometry", {})
        height = wall.get("height", 2700)  # Default ceiling height
        
        if geometry.get("type") == "straight":
            start = geometry.get("start", [0, 0])
            end = geometry.get("end", [1000, 0])
            
            length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            area = length * height
            
            # Subtract openings
            openings = wall.get("openings", [])
            for opening in openings:
                opening_area = opening.get("width", 900) * opening.get("height", 2100)
                area -= opening_area
            
            return max(0, area)  # Ensure non-negative
        
        return 0.0


def create_measurement_tools(unit_system: str = "metric") -> AdvancedMeasurementTools:
    """Create advanced measurement tools instance"""
    
    return AdvancedMeasurementTools(unit_system)


# JavaScript integration for 3D viewer
def generate_measurement_tools_js() -> str:
    """Generate JavaScript code for 3D viewer measurement tools"""
    
    return '''
// Advanced Measurement Tools for Three.js 3D Viewer
class MeasurementTools {
    constructor(scene, camera, renderer, domElement) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        this.domElement = domElement;
        
        this.measurements = [];
        this.activeMeasurement = null;
        this.measurementMode = null;
        
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.setupEventListeners();
        this.createMeasurementUI();
    }
    
    setupEventListeners() {
        this.domElement.addEventListener('click', (event) => {
            if (this.measurementMode) {
                this.handleMeasurementClick(event);
            }
        });
        
        this.domElement.addEventListener('mousemove', (event) => {
            if (this.measurementMode) {
                this.updateMousePosition(event);
            }
        });
    }
    
    createMeasurementUI() {
        const toolbar = document.createElement('div');
        toolbar.id = 'measurement-toolbar';
        toolbar.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-family: Arial, sans-serif;
            z-index: 1000;
        `;
        
        const buttons = [
            {name: 'Distance', mode: 'distance'},
            {name: 'Area', mode: 'area'},
            {name: 'Angle', mode: 'angle'},
            {name: 'Clear', mode: 'clear'}
        ];
        
        buttons.forEach(btn => {
            const button = document.createElement('button');
            button.textContent = btn.name;
            button.style.cssText = `
                margin: 2px;
                padding: 5px 10px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            `;
            
            button.addEventListener('click', () => {
                if (btn.mode === 'clear') {
                    this.clearMeasurements();
                } else {
                    this.setMeasurementMode(btn.mode);
                }
            });
            
            toolbar.appendChild(button);
        });
        
        document.body.appendChild(toolbar);
        
        // Create measurement display
        const display = document.createElement('div');
        display.id = 'measurement-display';
        display.style.cssText = `
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-family: Arial, sans-serif;
            max-width: 300px;
            z-index: 1000;
        `;
        
        document.body.appendChild(display);
    }
    
    setMeasurementMode(mode) {
        this.measurementMode = mode;
        this.activeMeasurement = {
            type: mode,
            points: [],
            objects: []
        };
        
        this.updateCursor();
    }
    
    handleMeasurementClick(event) {
        this.updateMousePosition(event);
        
        // Raycast to find intersection
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);
        
        if (intersects.length > 0) {
            const point = intersects[0].point;
            this.addMeasurementPoint(point);
        }
    }
    
    addMeasurementPoint(point) {
        if (!this.activeMeasurement) return;
        
        this.activeMeasurement.points.push(point);
        
        // Create point marker
        const pointGeometry = new THREE.SphereGeometry(10, 8, 8);
        const pointMaterial = new THREE.MeshBasicMaterial({color: 0xff0000});
        const pointMarker = new THREE.Mesh(pointGeometry, pointMaterial);
        pointMarker.position.copy(point);
        
        this.scene.add(pointMarker);
        this.activeMeasurement.objects.push(pointMarker);
        
        // Check if measurement is complete
        this.checkMeasurementComplete();
    }
    
    checkMeasurementComplete() {
        const measurement = this.activeMeasurement;
        
        if (measurement.type === 'distance' && measurement.points.length === 2) {
            this.completeMeasurement();
        } else if (measurement.type === 'angle' && measurement.points.length === 3) {
            this.completeMeasurement();
        } else if (measurement.type === 'area' && measurement.points.length >= 3) {
            // Area measurement can be completed with right-click or Enter key
        }
    }
    
    completeMeasurement() {
        const measurement = this.activeMeasurement;
        
        if (measurement.type === 'distance') {
            const distance = this.calculateDistance(measurement.points[0], measurement.points[1]);
            this.createDistanceLine(measurement.points[0], measurement.points[1], distance);
            measurement.value = distance;
            measurement.unit = 'mm';
        } else if (measurement.type === 'angle') {
            const angle = this.calculateAngle(measurement.points[0], measurement.points[1], measurement.points[2]);
            this.createAngleArc(measurement.points[0], measurement.points[1], measurement.points[2], angle);
            measurement.value = angle;
            measurement.unit = 'degrees';
        }
        
        this.measurements.push(measurement);
        this.updateMeasurementDisplay();
        
        this.activeMeasurement = null;
        this.measurementMode = null;
        this.updateCursor();
    }
    
    calculateDistance(point1, point2) {
        return point1.distanceTo(point2);
    }
    
    calculateAngle(point1, vertex, point2) {
        const v1 = new THREE.Vector3().subVectors(point1, vertex).normalize();
        const v2 = new THREE.Vector3().subVectors(point2, vertex).normalize();
        const angle = Math.acos(v1.dot(v2));
        return THREE.MathUtils.radToDeg(angle);
    }
    
    createDistanceLine(start, end, distance) {
        const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
        const material = new THREE.LineBasicMaterial({color: 0x00ff00, linewidth: 2});
        const line = new THREE.Line(geometry, material);
        
        this.scene.add(line);
        this.activeMeasurement.objects.push(line);
        
        // Add distance label
        this.createLabel(start.clone().lerp(end, 0.5), `${(distance/1000).toFixed(2)}m`);
    }
    
    createLabel(position, text) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;
        
        context.fillStyle = 'rgba(0,0,0,0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        context.fillStyle = 'white';
        context.font = '20px Arial';
        context.textAlign = 'center';
        context.fillText(text, canvas.width/2, canvas.height/2 + 7);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({map: texture});
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(100, 25, 1);
        
        this.scene.add(sprite);
        this.activeMeasurement.objects.push(sprite);
    }
    
    updateMeasurementDisplay() {
        const display = document.getElementById('measurement-display');
        
        let html = '<h3>Measurements</h3>';
        this.measurements.forEach((measurement, index) => {
            const value = measurement.value;
            const unit = measurement.unit;
            const displayValue = unit === 'mm' && value > 1000 ? 
                `${(value/1000).toFixed(2)}m` : 
                `${value.toFixed(2)}${unit}`;
            
            html += `<div>${index + 1}. ${measurement.type}: ${displayValue}</div>`;
        });
        
        display.innerHTML = html;
    }
    
    clearMeasurements() {
        this.measurements.forEach(measurement => {
            measurement.objects.forEach(obj => {
                this.scene.remove(obj);
            });
        });
        
        this.measurements = [];
        this.activeMeasurement = null;
        this.measurementMode = null;
        this.updateMeasurementDisplay();
        this.updateCursor();
    }
    
    updateMousePosition(event) {
        const rect = this.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    }
    
    updateCursor() {
        if (this.measurementMode) {
            this.domElement.style.cursor = 'crosshair';
        } else {
            this.domElement.style.cursor = 'default';
        }
    }
}

// Integration function for existing 3D viewers
function addMeasurementTools(scene, camera, renderer, domElement) {
    return new MeasurementTools(scene, camera, renderer, domElement);
}
'''


if __name__ == "__main__":
    # Test measurement tools
    tools = create_measurement_tools("metric")
    
    print("📐 Advanced Measurement Tools Test")
    print("=" * 50)
    
    # Test distance measurement
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(3000, 4000, 0)
    
    distance_result = tools.measure_distance(p1, p2)
    print(f"Distance measurement: {distance_result.value:.2f} {distance_result.unit}")
    
    # Test area measurement
    polygon_points = [
        Point3D(0, 0, 0),
        Point3D(5000, 0, 0),
        Point3D(5000, 3000, 0),
        Point3D(0, 3000, 0)
    ]
    
    area_result = tools.measure_area_polygon(polygon_points)
    print(f"Area measurement: {area_result.value:.2f} {area_result.unit}")
    
    # Test angle measurement
    angle_result = tools.measure_angle_3_points(
        Point3D(1000, 0, 0),
        Point3D(0, 0, 0),
        Point3D(0, 1000, 0)
    )
    print(f"Angle measurement: {angle_result.value:.2f} {angle_result.unit}")
    
    # Test measurement summary
    summary = tools.get_measurement_summary()
    print(f"Total measurements: {summary['total_measurements']}")
    
    print("✅ Advanced Measurement Tools initialized successfully!")