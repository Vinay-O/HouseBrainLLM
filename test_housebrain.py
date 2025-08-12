#!/usr/bin/env python3
"""
HouseBrain v1.1.0 - Test Script

This script demonstrates the HouseBrain architectural design system.
"""

import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from housebrain import generate_house_design, validate_house_design, HouseInput


def test_housebrain():
    """Test the HouseBrain system with sample input"""
    
    print("🏠 HouseBrain v1.1.0 - Test Script")
    print("=" * 50)
    
    # Load sample input
    with open("data/sample_input.json", "r") as f:
        sample_input = json.load(f)
    
    print("📋 Input Requirements:")
    print(f"   - Plot: {sample_input['plot']['width']}' x {sample_input['plot']['length']}'")
    print(f"   - Floors: {sample_input['basicDetails']['floors']}")
    print(f"   - Bedrooms: {sample_input['basicDetails']['bedrooms']}")
    print(f"   - Bathrooms: {sample_input['basicDetails']['bathrooms']}")
    print(f"   - Style: {sample_input['basicDetails']['style']}")
    print(f"   - Budget: ${sample_input['basicDetails']['budget']:,}")
    
    # Generate design
    print("\n🎨 Generating house design...")
    house_input = HouseInput(**sample_input)
    house_output = generate_house_design(house_input, demo_mode=True)
    
    # Validate design
    print("\n🔍 Validating design...")
    validation = validate_house_design(house_output)
    
    # Display results
    print("\n📊 Results:")
    print(f"   - Total Area: {house_output.total_area:.1f} sqft")
    print(f"   - Construction Cost: ${house_output.construction_cost:,.0f}")
    print(f"   - Compliance Score: {validation.compliance_score:.1f}/100")
    print(f"   - Valid: {'✅ Yes' if validation.is_valid else '❌ No'}")
    
    # Show room breakdown
    print("\n🏠 Room Breakdown:")
    for i, level in enumerate(house_output.levels):
        print(f"   Level {i}:")
        for room in level.rooms:
            area = room.bounds.area
            print(f"     - {room.type.value.replace('_', ' ').title()}: {area:.1f} sqft")
    
    # Show validation issues
    if validation.errors:
        print(f"\n⚠️  Validation Errors ({len(validation.errors)}):")
        for error in validation.errors[:5]:  # Show first 5
            print(f"   - {error}")
        if len(validation.errors) > 5:
            print(f"   ... and {len(validation.errors) - 5} more")
    
    if validation.warnings:
        print(f"\n💡 Warnings ({len(validation.warnings)}):")
        for warning in validation.warnings[:3]:  # Show first 3
            print(f"   - {warning}")
        if len(validation.warnings) > 3:
            print(f"   ... and {len(validation.warnings) - 3} more")
    
    # Show material estimates
    print(f"\n🏗️  Material Estimates:")
    for material, amount in house_output.materials.items():
        unit = "sqft" if "sqft" in material else "tons" if "tons" in material else "cubic yards" if "cubic" in material else "board feet" if "board" in material else "sheets"
        print(f"   - {material.replace('_', ' ').title()}: {amount:.1f} {unit}")
    
    print(f"\n✅ Test completed! Check outputs/ directory for generated files.")
    print(f"🌐 API available at: http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    test_housebrain()
