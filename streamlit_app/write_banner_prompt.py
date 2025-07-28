#!/usr/bin/env python3
"""
Optimized script that generates updated banner prompts based on extract_layout.py output and user text.
This script accepts layout JSON from extract_layout.py and user text, then updates descriptions
in both text elements and background parameters based on the user's input.
"""

import json
import os
import sys
import argparse
from typing import Dict, Any, List, Optional
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

class BannerPromptOptimizer:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Gemini API."""
        self.client = genai.Client(
            api_key=api_key or os.environ.get("YOUR_GEMINI_API_KEY")
        )
        self.model = "gemini-2.5-pro"
    
    def update_text_descriptions(self, text_elements: List[Dict[str, Any]], user_text: str) -> List[Dict[str, Any]]:
        """Update description field for each text element based on user text."""
        updated_elements = []
        
        for element in text_elements:
            updated_element = element.copy()
            
            # Generate new description based on user text and original description
            prompt = f"""
            As a professional marketer, create a trendy and intuitive sentence for a banner based on the user's request.
            User's request: "{user_text}"
            Original text description: "{element['description']}"

            Your task is to rewrite the description to be catchy, engaging, and aligned with the latest marketing trends.
            The new description should be concise and directly related to the user's theme.
            Return only the new, optimized description text, without any JSON formatting.
            """
            
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt]
                )
                new_description = response.text.strip().strip('"').strip("'")
                updated_element['description'] = new_description
            except Exception as e:
                print(f"Warning: Could not update text description: {e}")
                updated_element['description'] = f"Text related to: {user_text}"
            
            updated_elements.append(updated_element)
        
        return updated_elements
    
    def update_background_parameters(self, background: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        """Update background parameters based on user text."""
        updated_background = background.copy()
        
        # Generate updated background content
        prompt = f"""
        As a creative director, generate a detailed and trendy background concept for a banner based on the user's request.
        User's request: "{user_text}"

        Update the following background parameters to create a visually stunning and modern design.
        The new design should be highly relevant to the user's theme and optimized for visual impact.

        Original scene: {background.get('scene', '')}
        Original subjects: {background.get('subjects', [])}
        Original style: {background.get('style', '')}
        Original mood: {background.get('mood', '')}

        Generate a JSON object with the following keys, ensuring the content is fresh, creative, and detailed:
        {{
            "scene": "A highly descriptive and imaginative scene that captures the essence of the user's request.",
            "subjects": ["Primary subject(s) that are the main focus", "Secondary subject(s) that add context"],
            "style": "A modern and visually appealing art style (e.g., '3D render', 'vector illustration', 'photo-realistic')",
            "color_palette": ["A vibrant and harmonious color palette that evokes the desired mood (e.g., 'neon pastels', 'earthy tones')"],
            "lighting": "Creative lighting that enhances the mood and highlights key elements (e.g., 'dramatic backlighting', 'soft, diffused light')",
            "mood": "The overall feeling or atmosphere the banner should convey (e.g., 'energetic and youthful', 'calm and sophisticated')",
            "background_elements": "Subtle elements that enrich the scene without cluttering it (e.g., 'geometric patterns', 'abstract shapes')",
            "composition": "A dynamic and balanced composition (e.g., 'rule of thirds', 'asymmetrical balance')",
            "camera": {{
                "angle": "A compelling camera angle (e.g., 'low-angle shot', 'dutch angle')",
                "distance": "The shot distance that best captures the scene (e.g., 'close-up', 'wide shot')",
                "focus": "The focus of the shot (e.g., 'sharp focus on the main subject with a blurred background')"
            }}
        }}

        Ensure the entire output is a single, valid JSON object.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            updated_data = json.loads(response.text)
            
            # Merge updated data with original structure
            for key in updated_data:
                if key in updated_background:
                    updated_background[key] = updated_data[key]
                    
        except Exception as e:
            print(f"Warning: Could not update background parameters: {e}")
            # Fallback: append user text to existing descriptions
            updated_background['scene'] = f"{background.get('scene', '')} - themed around: {user_text}"
            updated_background['mood'] = f"{background.get('mood', '')} with {user_text} theme"
        
        return updated_background
    
    def optimize_prompt(self, layout_data: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        """Main method to optimize the prompt based on user text."""
        # Create a deep copy to avoid modifying original
        optimized_data = json.loads(json.dumps(layout_data))
        
        # Update text descriptions
        if 'text' in optimized_data:
            optimized_data['text'] = self.update_text_descriptions(
                optimized_data['text'], 
                user_text
            )
        
        # Update background parameters
        if 'background' in optimized_data:
            optimized_data['background'] = self.update_background_parameters(
                optimized_data['background'], 
                user_text
            )
        
        return optimized_data

def main():
    """Main function to handle command line arguments and process the prompt."""
    parser = argparse.ArgumentParser(description='Optimize banner prompt based on user text')
    parser.add_argument('user_text', help='User text to base the prompt on')
    parser.add_argument('--input', '-i', default='layout_output.json', 
                       help='Input layout JSON file (default: layout_output.json)')
    parser.add_argument('--output', '-o', default='optimized_prompt.json',
                       help='Output JSON file (default: optimized_prompt.json)')
    parser.add_argument('--api-key', help='Gemini API key (optional if in .env.local)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Read layout data
    try:
        with open(args.input, 'r') as f:
            layout_data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Initialize optimizer
    try:
        optimizer = BannerPromptOptimizer(args.api_key)
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        sys.exit(1)
    
    # Optimize the prompt
    try:
        optimized_data = optimizer.optimize_prompt(layout_data, args.user_text)
        
        # Save optimized data
        with open(args.output, 'w') as f:
            json.dump(optimized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Optimized prompt saved to {args.output}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Updated {len(optimized_data.get('text', []))} text descriptions")
        print(f"- Updated background parameters for theme: {args.user_text}")
        
    except Exception as e:
        print(f"Error optimizing prompt: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
