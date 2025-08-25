# scripts/generate_prompts.py
import random
import argparse
from pathlib import Path

# --- Components for Prompt Generation ---
STYLES = [
    "Modern", "Contemporary", "Minimalist", "Traditional Kerala-style 'Nalukettu'",
    "Colonial", "Industrial", "Scandinavian", "Bohemian", "Farmhouse", "Chettinad-style",
    "Eco-friendly", "Brutalist", "Art Deco", "Mediterranean"
]
SIZES_BHK = ["1BHK", "2BHK", "3BHK", "4BHK", "5BHK", "6BHK", "studio apartment"]
SIZES_SQFT = [
    "800 sqft", "1000 sqft", "1200 sqft", "1500 sqft", "1800 sqft",
    "2000 sqft", "2500 sqft", "3000 sqft", "4000 sqft", "5000 sqft"
]
FLOORS = [
    "single-story", "two-story", "G+1", "G+2", "duplex", "triplex",
    "split-level", "penthouse"
]
STRUCTURE_TYPES = ["house", "villa", "apartment", "bungalow", "farmhouse", "townhouse", "cottage"]
PLOT_SIZES = [
    "30x40 feet", "30x50 feet", "40x60 feet", "50x80 feet", "60x90 feet",
    "80x100 feet", "100x100 feet", "corner", "irregular"
]
FEATURES = [
    "with an open-plan kitchen and living area", "with a swimming pool", "with a home theater",
    "with a large garden", "with a central courtyard", "with a dedicated home office",
    "with a private gym", "featuring floor-to-ceiling windows", "with a rooftop terrace",
    "with a two-car garage", "with servant's quarters", "with a library", "with a spacious balcony for each bedroom"
]
CONSTRAINTS = [
    "and be Vastu-compliant", "with a North-facing entrance", "with a West-facing plot",
    "on a tight budget", "for a luxury segment", "designed for a family of four",
    "with a focus on natural light and ventilation", "for a joint family", "as a bachelor pad",
    "to be wheelchair accessible"
]

def generate_prompt():
    """Generates a single, diverse architectural prompt."""
    prompt_parts = []
    style = random.choice(STYLES)
    num_floors = random.choice(FLOORS)
    bhk = random.choice(SIZES_BHK)
    structure = random.choice(STRUCTURE_TYPES)
    prompt_parts.append(f"Design a {style}, {num_floors} {bhk} {structure}")

    # 70% chance to specify plot size, 30% chance to specify total area
    if random.random() < 0.7:
        plot = random.choice(PLOT_SIZES)
        prompt_parts.append(f"for a {plot} plot")
    else:
        area = random.choice(SIZES_SQFT)
        prompt_parts.append(f"with a total area of {area}")

    num_features = random.randint(1, 3)
    selected_features = random.sample(FEATURES, num_features)
    prompt_parts.extend(selected_features)

    # 60% chance to add constraints
    if random.random() < 0.6:
        num_constraints = random.randint(1, 2)
        selected_constraints = random.sample(CONSTRAINTS, num_constraints)
        prompt_parts.extend(selected_constraints)

    return ". ".join(prompt_parts) + "."

def main():
    parser = argparse.ArgumentParser(description="Generate a large number of architectural prompts.")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10000,
        help="The number of prompts to generate."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default="platinum_prompts.txt",
        help="The file to save the generated prompts to."
    )
    args = parser.parse_args()

    print(f"Generating {args.num_prompts} prompts and saving to {args.output_file}...")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w') as f:
        for _ in range(args.num_prompts):
            prompt = generate_prompt()
            f.write(f"{prompt}\n")

    print(f"✅ Successfully generated and saved {args.num_prompts} prompts.")
    print("\nFirst 5 prompts generated:")
    with open(args.output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(line.strip())

if __name__ == "__main__":
    main()
