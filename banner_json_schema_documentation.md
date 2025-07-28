# Banner Generation Tool JSON Schema Documentation

## Overview
This document defines the standardized JSON schemas for data flow between the three banner generation tools.

## Tool Chain Flow
```
extract_layout → write_banner_prompt → generate_banner
   (Layout JSON) → (Optimized JSON) → (Image Output)
```

---

## 1. extract_layout Tool

### Input Parameters
- `image_data` (string): Base64 encoded image data of reference banner

### Output JSON Schema
```json
{
  "text": [
    {
      "bbox": [x, y, width, height],        // Array of numbers [x, y, w, h]
      "font_size": 24,                      // Number (pixels)
      "font_style": "bold",                 // String (bold/italic/regular/serif/sans-serif)
      "font_color": "#000000",              // String (hex color or color name)
      "description": "Main title text..."  // String (what the text says)
    }
  ],
  "background": {
    "scene": "description of overall scene",
    "subjects": [
      {
        "type": "person/object/landscape",
        "description": "detailed description",
        "pose": "pose or position",
        "position": "foreground/middleground/background"
      }
    ],
    "style": "digital painting/photograph/illustration",
    "color_palette": ["color1", "color2", "color3"],  // Array of exactly 3 colors
    "lighting": "lighting conditions description",
    "mood": "mood or atmosphere",
    "background": "background elements description",
    "composition": "composition technique",
    "camera": {
      "angle": "eye level/low angle/high angle",
      "distance": "close-up/medium shot/long shot",
      "focus": "sharp/soft/depth of field"
    }
  }
}
```

### Error Output Schema
```json
{
  "error": "Error description string"
}
```

---

## 2. write_banner_prompt Tool

### Input Parameters  
- `layout_json` (string): JSON string from extract_layout tool
- `user_requirements` (string): User's specific requirements for the banner

### Output JSON Schema
Same structure as extract_layout output, but with optimized content:

```json
{
  "text": [
    {
      "bbox": [x, y, width, height],
      "font_size": 24,
      "font_style": "bold",
      "font_color": "#000000",
      "description": "OPTIMIZED text based on user requirements"
    }
  ],
  "background": {
    "scene": "OPTIMIZED scene description",
    "subjects": ["OPTIMIZED subject descriptions"],
    "style": "OPTIMIZED art style",
    "color_palette": ["OPTIMIZED color palette"],
    "lighting": "OPTIMIZED lighting description", 
    "mood": "OPTIMIZED mood based on user requirements",
    "background_elements": "OPTIMIZED background elements",
    "composition": "OPTIMIZED composition technique",
    "camera": {
      "angle": "OPTIMIZED camera angle",
      "distance": "OPTIMIZED shot distance", 
      "focus": "OPTIMIZED focus description"
    }
  }
}
```

### Error Output Schema
```json
{
  "error": "Error description string"
}
```

---

## 3. generate_banner Tool

### Input Parameters
- `optimized_prompt_json` (string): JSON string from write_banner_prompt tool
- `size` (string, optional): Image size (default: "1792x1024")
- `include_image_data` (boolean, optional): Whether to include base64 image data

### Output JSON Schema (Success)
```json
{
  "success": true,
  "image_data": "base64_encoded_image_string",
  "original_prompt": "text_prompt_sent_to_gpt_image_1",
  "size": "1792x1024"
}
```

### Output JSON Schema (Error)
```json
{
  "success": false,
  "error": "Error description string",
  "original_prompt": "prompt_that_failed"
}
```

---

## Data Flow Validation Rules

### 1. extract_layout → write_banner_prompt
- `extract_layout` output must be valid JSON
- `write_banner_prompt` must parse `layout_json` parameter successfully
- Error handling: If layout_json is invalid, write_banner_prompt returns error JSON

### 2. write_banner_prompt → generate_banner  
- `write_banner_prompt` output must be valid JSON
- `generate_banner` must parse `optimized_prompt_json` parameter successfully
- Error handling: If optimized_prompt_json is invalid, generate_banner returns error JSON

### 3. generate_banner → Streamlit Display
- `generate_banner` must return JSON with either success=true + image_data OR success=false + error
- Streamlit main.py checks for "success" field and "image_data" presence
- Image data must be valid base64 that can be decoded for display

---

## Error Handling Standards

All tools follow consistent error handling:

1. **JSON Parse Errors**: Return `{"error": "Invalid JSON: [details]"}`
2. **API Errors**: Return `{"error": "API call failed: [details]"}`  
3. **Processing Errors**: Return `{"error": "Processing failed: [details]"}`
4. **Success Check**: All outputs can be validated with `json.loads()` and checking for "error" key

---

## Integration with main.py

### Tool Call Sequence
```python
# Step 1: Extract Layout
extract_layout_output = extract_layout(image_data=base64_image)
layout_json = extract_layout_output  # JSON string

# Step 2: Optimize Prompt
write_prompt_output = write_banner_prompt(
    layout_json=layout_json,
    user_requirements=user_input_text
)
optimized_json = write_prompt_output  # JSON string

# Step 3: Generate Banner  
generate_output = generate_banner(
    optimized_prompt_json=optimized_json,
    size="1792x1024",
    include_image_data=True
)
result = json.loads(generate_output)  # Parse for display
```

### Validation Steps
1. Each tool output is validated as valid JSON
2. Error outputs are caught and displayed to user
3. Success outputs are passed to next tool
4. Final image data is decoded and displayed in Streamlit

This schema ensures robust, predictable data flow between all banner generation tools.