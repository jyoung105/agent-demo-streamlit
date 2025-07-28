# This script uses Google's Gemini API to analyze a poster image and extract layout information as JSON.
# Optimized to return precise JSON structure with 'text' and 'background' parameters as specified.
import base64
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env.local file
load_dotenv('.env.local')

def extract_layout_from_image(image_path):
    """
    Extract layout information from an image file.
    
    Args:
        image_path (str): Path to the image file to analyze
        
    Returns:
        dict: JSON structure with 'text' and 'background' parameters
    """
    # Create the Gemini client using the API key from environment variables
    client = genai.Client(
        api_key=os.environ.get("YOUR_GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"
    
    # Read the image file in binary mode
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    
    # Encode the image data as base64 for transmission
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    
    # Prepare the prompt to extract the exact JSON structure required
    prompt = """Analyze this banner/poster image and extract the layout information in the following exact JSON format:

{
  "text": [
    {
      "bbox": [x, y, width, height],
      "font_size": 24,
      "font_style": "bold",
      "font_color": "#000000",
      "description": "Main title text describing what the text says"
    }
  ],
  "background": {
    "scene": "description of the overall scene",
    "subjects": [
      {
        "type": "type of subject (person, object, landscape element)",
        "description": "detailed description of the subject",
        "pose": "pose or position of the subject",
        "position": "position in frame (foreground, middleground, background)"
      }
    ],
    "style": "art style (digital painting, photograph, illustration, etc.)",
    "color_palette": ["color1", "color2", "color3"],
    "lighting": "description of lighting conditions",
    "mood": "overall mood or atmosphere",
    "background": "description of background elements",
    "composition": "composition technique used",
    "camera": {
      "angle": "camera angle (eye level, low angle, high angle, etc.)",
      "distance": "shot distance (close-up, medium shot, long shot, etc.)",
      "focus": "focus description (sharp, soft, depth of field, etc.)"
    }
  }
}

Instructions:
1. Identify ALL text elements in the image and provide each as a separate object in the "text" array
2. For each text element, provide exact bounding box coordinates [x, y, width, height]
3. Estimate font size in pixels based on visual appearance
4. Describe font style (bold, italic, regular, serif, sans-serif, etc.)
5. Provide font color as hex code or color name
6. Write a brief description of what each text element says
7. For the background, provide detailed scene analysis
8. Include any human subjects, objects, or landscape elements in "subjects"
9. List exactly 3 main colors in the color_palette
10. Be specific and descriptive in all string fields

Return ONLY the JSON object, no additional text or markdown formatting."""
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(image_b64),
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
    )

    # Generate content and get response
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    # Parse the JSON response
    try:
        layout_data = json.loads(response.text)
        return layout_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print("Raw response:", response.text)
        return None

def save_layout_to_file(layout_data, output_path="layout_output.json"):
    """Save layout data to a JSON file."""
    if layout_data:
        with open(output_path, "w") as f:
            json.dump(layout_data, f, indent=2)
        print(f"Layout JSON has been saved to {output_path}")
        return True
    return False

def main():
    """Main function to demonstrate usage."""
    import sys
    
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to example image if no argument provided
        image_path = "banner-agent/example/poster.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    print(f"Analyzing image: {image_path}")
    layout_data = extract_layout_from_image(image_path)
    
    if layout_data:
        # Save to file
        save_layout_to_file(layout_data)
        
        # Print formatted JSON
        print("\nExtracted Layout:")
        print(json.dumps(layout_data, indent=2))
    else:
        print("Failed to extract layout information.")

if __name__ == "__main__":
    main()
