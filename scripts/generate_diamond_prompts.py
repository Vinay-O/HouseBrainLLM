import random
import argparse

# --- Components for Diamond-Tier Prompt Generation ---

# Standard components, but we'll use them more sparingly
STYLES = [
    "Modern", "Contemporary", "Minimalist", "Traditional Kerala-style 'Nalukettu'", "Colonial",
    "Industrial", "Scandinavian", "Eco-friendly", "Brutalist", "Art Deco", "Mediterranean"
]
STRUCTURE_TYPES = ["house", "villa", "bungalow", "farmhouse", "cottage"]
SIZES_BHK = ["2BHK", "3BHK", "4BHK", "5BHK"]

# --- NEW: Components focused on complexity and constraints ---

UNUSUAL_PLOTS = [
    "a narrow, triangular plot", "a pie-shaped corner plot with street access on two sides",
    "a steep hillside plot with a significant slope", "a plot with a protected stream running through the middle",
    "an irregularly shaped plot with five sides", "a long, thin plot (30x100 feet)",
    "a plot landlocked between two existing buildings"
]

COMPLEX_FEATURES = [
    "must incorporate a double-height living room", "with a cantilevered master bedroom",
    "featuring a central atrium that spans all floors", "with a sunken seating area in the living room",
    "with an integrated greenhouse for year-round gardening", "designed around a large, ancient tree that cannot be moved",
    "with a basement-level home theater and wine cellar", "featuring a rooftop infinity pool"
]

CONFLICTING_CONSTRAINTS = [
    "that feels open and spacious despite a small plot size",
    "that is both luxurious and built with sustainable, locally-sourced materials",
    "that adheres to strict Vastu principles while having a modern, minimalist aesthetic",
    "that maximizes privacy from neighbors while using extensive floor-to-ceiling glass",
    "a design for an elderly couple that is fully wheelchair accessible but also includes a second-story guest suite",
    "a budget-friendly home that must also meet high-end energy efficiency standards (LEED Platinum)"
]

# --- Diamond Prompt Generation Logic ---

def generate_diamond_prompt():
    """Generates a single, complex architectural prompt for the Diamond dataset."""
    
    prompt_parts = []

    # Core structure is always complex
    style = random.choice(STYLES)
    structure = random.choice(STRUCTURE_TYPES)
    bhk = random.choice(SIZES_BHK)
    plot = random.choice(UNUSUAL_PLOTS)
    
    prompt_parts.append(f"Design a {style} {bhk} {structure} on {plot}")

    # Add 1 to 2 complex features
    num_features = random.randint(1, 2)
    selected_features = random.sample(COMPLEX_FEATURES, num_features)
    prompt_parts.extend(selected_features)

    # Add exactly one conflicting constraint to force reasoning
    selected_constraint = random.choice(CONFLICTING_CONSTRAINTS)
    prompt_parts.append(selected_constraint)
        
    return ". ".join(prompt_parts) + "."


def main():
    parser = argparse.ArgumentParser(description="Generate a list of complex, Diamond-tier architectural prompts.")
    parser.add_argument(
        "-n", "--num-prompts",
        type=int,
        default=2500,
        help="The number of prompts to generate."
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default="diamond_prompts.txt",
        help="The file to save the generated prompts to."
    )
    args = parser.parse_args()

    print(f"Generating {args.num_prompts} Diamond-tier prompts and saving to {args.output_file}...")

    with open(args.output_file, 'w') as f:
        for _ in range(args.num_prompts):
            prompt = generate_diamond_prompt()
            f.write(f"{prompt}\n")
    
    print(f"✅ Successfully generated and saved {args.num_prompts} prompts.")


if __name__ == "__main__":
    main()
