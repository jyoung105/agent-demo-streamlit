"""
Enhanced banner prompt optimization service with Korean marketing focus.
Provides step-by-step processing with persistent data storage.
"""
import time
import json
import logging
import os
import re
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from sqlalchemy.orm import Session
from shared.banner import BannerLayoutData, BannerPromptResponse, TextElement, Subject
from database import (
    PromptGeneration, LayoutExtraction, update_step_status, 
    StepStatus, JobStatus, update_job_status
)
from ..config import get_config, OPENAI_TEXT_MODEL, OPENAI_TEXT_MAX_TOKENS, OPENAI_TEXT_TEMPERATURE

logger = logging.getLogger(__name__)

# Korean marketing templates for authentic banner text
KOREAN_MARKETING_TEMPLATES = {
    'discount': [
        "최대 {discount}% 할인",
        "~{discount}% SALE", 
        "{discount}% 파격세일",
        "UP TO {discount}%"
    ],
    'time_limited': [
        "단 {days}일간",
        "오늘만 특가",
        "품절임박",
        "한정수량",
        "이번주만!"
    ],
    'bonus': [
        "{item} 증정",
        "1+1 이벤트",
        "사은품 대박",
        "무료배송"
    ],
    'cta': [
        "지금 바로 구매하기",
        "놓치면 후회!",
        "서두르세요!",
        "클릭!"
    ],
    'trust': [
        "정품보장",
        "당일발송",
        "만족보장",
        "베스트셀러"
    ]
}

# Product keywords for recognition
PRODUCT_KEYWORDS = {
    'fashion': ['의류', '옷', '패션', '셔츠', '원피스', '바지', '자켓', '코트', '가방', '신발'],
    'cosmetics': ['화장품', '스킨케어', '크림', '로션', '세럼', '마스크팩', '선크림', '립스틱'],
    'food': ['음료', '과자', '스낵', '커피', '차', '음식', '도시락', '샐러드'],
    'electronics': ['전자제품', '폰', '노트북', '이어폰', '스마트', '가전'],
    'lifestyle': ['생활용품', '인테리어', '가구', '침구', '주방'],
    'education': ['강의', '클래스', '수업', '교육', '학습', '강좌']
}

class BannerPromptService:
    def __init__(self, openai_client: OpenAI, db_session: Session):
        self.client = openai_client
        self.db = db_session
        self.model = OPENAI_TEXT_MODEL
        self.max_tokens = OPENAI_TEXT_MAX_TOKENS
        self.temperature = OPENAI_TEXT_TEMPERATURE
    
    def extract_product_info(self, user_text: str) -> Dict[str, Any]:
        """
        Extract product information from user input for better targeting.
        
        Args:
            user_text: User's input text
            
        Returns:
            Dict containing product type, features, and target audience
        """
        product_info = {
            'product_type': 'general',
            'brand': '',
            'features': [],
            'target': 'general',
            'category': 'general'
        }
        
        # Convert to lowercase for matching
        text_lower = user_text.lower()
        
        # Identify product category
        for category, keywords in PRODUCT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    product_info['category'] = category
                    product_info['product_type'] = keyword
                    break
        
        # Extract brand names (common Korean brands)
        brand_patterns = [
            r'토리든|torriden', r'클래스101|class101', r'올리브영', 
            r'gs25|gs 25', r'cu|씨유', r'이마트', r'쿠팡', r'네이버'
        ]
        for pattern in brand_patterns:
            match = re.search(pattern, text_lower)
            if match:
                product_info['brand'] = match.group()
                break
        
        # Extract features
        if '세일' in text_lower or '할인' in text_lower:
            product_info['features'].append('discount')
        if '신제품' in text_lower or '신상' in text_lower:
            product_info['features'].append('new')
        if '한정' in text_lower or '품절' in text_lower:
            product_info['features'].append('limited')
        if '무료' in text_lower or '증정' in text_lower:
            product_info['features'].append('gift')
        
        # Identify target audience
        if '여성' in text_lower or '여자' in text_lower:
            product_info['target'] = 'women'
        elif '남성' in text_lower or '남자' in text_lower:
            product_info['target'] = 'men'
        elif '아이' in text_lower or '키즈' in text_lower:
            product_info['target'] = 'kids'
        elif 'mz' in text_lower or '젊은' in text_lower:
            product_info['target'] = 'mz'
        
        return product_info
    
    def calculate_bbox_scaling(self, original_size: Tuple[int, int], target_size: str) -> Tuple[float, float]:
        """
        Calculate scaling factors for bounding boxes when image size changes.
        
        Args:
            original_size: (width, height) of original image
            target_size: Target image size string like "1792x1024"
            
        Returns:
            Tuple of (x_scale, y_scale) scaling factors
        """
        try:
            original_width, original_height = original_size
            target_width, target_height = map(int, target_size.split('x'))
            
            x_scale = target_width / original_width
            y_scale = target_height / original_height
            
            logger.info(f"Bbox scaling: {original_width}x{original_height} -> {target_width}x{target_height} "
                       f"(scale: {x_scale:.3f}, {y_scale:.3f})")
            
            return (x_scale, y_scale)
            
        except Exception as e:
            logger.warning(f"Could not calculate bbox scaling: {e}")
            return (1.0, 1.0)  # No scaling
    
    def scale_text_elements(self, text_elements: List[TextElement], x_scale: float, y_scale: float) -> List[TextElement]:
        """
        Scale text element bounding boxes for new image dimensions.
        
        Args:
            text_elements: List of text elements with bbox coordinates
            x_scale: Horizontal scaling factor
            y_scale: Vertical scaling factor
            
        Returns:
            List of text elements with scaled bounding boxes
        """
        scaled_elements = []
        
        for elem in text_elements:
            if elem.bbox and len(elem.bbox) == 4:
                x, y, width, height = elem.bbox
                scaled_bbox = [
                    int(x * x_scale),
                    int(y * y_scale), 
                    int(width * x_scale),
                    int(height * y_scale)
                ]
                
                # Create new text element with scaled bbox
                # Apply more conservative font scaling to prevent overflow
                if elem.font_size:
                    # Scale font size conservatively (70% of the scaling factor)
                    scaled_font_size = int(elem.font_size * min(x_scale, y_scale) * 0.7)
                    # Also apply maximum font size limits
                    if scaled_font_size > 100:
                        scaled_font_size = 100
                    elif scaled_font_size > 60 and scaled_font_size <= 100:
                        scaled_font_size = int(scaled_font_size * 0.85)
                else:
                    scaled_font_size = elem.font_size
                
                scaled_elem = TextElement(
                    bbox=scaled_bbox,
                    font_size=scaled_font_size,
                    font_style=elem.font_style,
                    font_color=elem.font_color,
                    description=elem.description
                )
                scaled_elements.append(scaled_elem)
            else:
                # Keep original if bbox is invalid
                scaled_elements.append(elem)
        
        return scaled_elements
    
    def update_text_descriptions(self, text_elements: List[TextElement], user_text: str) -> List[TextElement]:
        """Update description field for each text element with Korean marketing focus."""
        updated_elements = []
        
        # Extract product info for better context
        product_info = self.extract_product_info(user_text)
        
        for element in text_elements:
            # Generate new description based on user text and original description
            prompt = f"""
당신은 한국 마케팅 전문 카피라이터입니다. 다음 요청에 맞는 트렌디한 배너 문구를 작성하세요.

사용자 요청: "{user_text}"
원본 문구 위치/역할: "{element.description}"
제품 정보: {json.dumps(product_info, ensure_ascii=False)}

작성 규칙:
1. 반드시 한국어로 작성
2. MZ세대가 좋아하는 짧고 임팩트 있는 표현 사용
3. 이모티콘이나 특수문자 적절히 활용 (!, ~, ♥ 등)
4. 다음 중 적절한 요소 포함:
   - 할인율 강조 (예: ~70% 할인, 최대 50% SALE)
   - 시간 한정 (예: 오늘만!, 단 3일간)
   - 혜택 강조 (예: 무료배송, 1+1)
   - 행동 유도 (예: 지금 클릭!, 서두르세요!)

원본 문구의 역할에 맞는 한국어 마케팅 문구만 반환하세요.
"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "당신은 한국 시장 전문 마케팅 카피라이터입니다. 트렌디하고 젊은 감성의 한국어 광고 문구를 작성하세요. 반드시 한국어로만 작성하세요."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.8  # More creative for marketing copy
                )
                
                new_description = response.choices[0].message.content.strip().strip('"').strip("'")
                
                # Ensure Korean text was generated
                if not any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in new_description):
                    # Fallback to Korean if no Korean characters detected
                    if 'discount' in product_info['features']:
                        new_description = "최대 70% 할인!"
                    elif 'new' in product_info['features']:
                        new_description = "신제품 출시!"
                    else:
                        new_description = "특별 이벤트!"
                
                # Create updated element
                updated_element = TextElement(
                    bbox=element.bbox,
                    font_size=element.font_size,
                    font_style=element.font_style,
                    font_color=element.font_color,
                    description=new_description
                )
                
            except Exception as e:
                logger.warning(f"Could not update text description: {e}")
                # Fallback with Korean text
                fallback_texts = {
                    'title': f"{user_text} 특가!",
                    'subtitle': "놓치면 후회하는 기회",
                    'cta': "지금 바로 구매하기",
                    'default': "특별 할인 이벤트"
                }
                
                fallback_key = 'default'
                for key in ['title', 'subtitle', 'cta']:
                    if key in element.description.lower():
                        fallback_key = key
                        break
                
                updated_element = TextElement(
                    bbox=element.bbox,
                    font_size=element.font_size,
                    font_style=element.font_style,
                    font_color=element.font_color,
                    description=fallback_texts[fallback_key]
                )
            
            updated_elements.append(updated_element)
        
        return updated_elements
    
    def update_image_parameters(self, image: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        """Update image parameters with product focus and no text overlays."""
        updated_image = image.copy()
        
        # Extract product information
        product_info = self.extract_product_info(user_text)
        
        # Preserve original style, composition, and camera from reference
        original_style = image.get('style', 'product photography')
        original_composition = image.get('composition', 'centered layout')
        original_camera = image.get('camera', {
            "angle": "eye level",
            "distance": "medium shot",
            "focus": "sharp focus"
        })
        
        # Generate updated background content
        prompt = f"""
제품 중심의 배너 배경을 생성하기 위한 파라미터를 만드세요. 텍스트는 절대 포함하지 마세요.

사용자 요청: "{user_text}"
제품 정보: {json.dumps(product_info, ensure_ascii=False)}

중요 규칙:
1. subjects는 반드시 실제 제품 사진이어야 함
2. 배경에 텍스트, 문자, 타이포그래피는 절대 금지
3. 제품이 선명하게 보이는 구도
4. 한국 광고 스타일의 깔끔한 배경

원본 설정 (반드시 유지):
- Style: {original_style}
- Composition: {original_composition}
- Camera: {json.dumps(original_camera)}

생성할 JSON:
{{
    "scene": "제품 촬영을 위한 스튜디오나 실제 사용 환경",
    "subjects": ["실제 {product_info['product_type']} 제품", "제품 패키지 클로즈업"],
    "style": "{original_style}",
    "color_palette": ["제품과 조화로운 색상 3-4개"],
    "lighting": "제품을 돋보이게 하는 조명",
    "mood": "구매욕구를 자극하는 분위기",
    "background": "심플하고 제품에 집중할 수 있는 배경, 텍스트 없음",
    "composition": "{original_composition}",
    "camera": {json.dumps(original_camera)}
}}

negative_prompt에 추가: "no text, no typography, no letters, no words, no writing"
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a visual designer specializing in product photography for Korean marketing. Create backgrounds that highlight products without any text elements."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            updated_data = json.loads(response_text)
            
            # Ensure no text-related elements in subjects
            if 'subjects' in updated_data:
                cleaned_subjects = []
                for subject in updated_data['subjects']:
                    if isinstance(subject, str):
                        # Remove any text-related descriptions
                        if not any(word in subject.lower() for word in ['text', 'typography', 'letter', 'word', 'writing']):
                            cleaned_subjects.append(subject)
                updated_data['subjects'] = cleaned_subjects
            
            # Add strong negative prompt to avoid text
            updated_data['negative_prompt'] = "text, typography, letters, words, writing, characters, numbers, alphabet, signs, labels, captions, titles, headlines, paragraphs, sentences, font, typeface, script"
            
            # Ensure scene doesn't contain text-related descriptions
            if 'scene' in updated_data:
                updated_data['scene'] = updated_data['scene'].replace('text', '').replace('typography', '').replace('letters', '').replace('words', '')
                
            # Double-check background field
            if 'background' in updated_data:
                updated_data['background'] = updated_data['background'] + " (absolutely no text or typography)"
            
            # Merge updated data with original structure
            for key in updated_data:
                if key in updated_image:
                    updated_image[key] = updated_data[key]
                    
        except Exception as e:
            logger.warning(f"Could not update background parameters: {e}")
            # Fallback to product-focused defaults
            
            product_subjects = {
                'fashion': ["clothing item on hanger", "folded clothes display"],
                'cosmetics': ["cosmetic bottle or jar", "skincare product close-up"],
                'food': ["beverage can or bottle", "food package display"],
                'electronics': ["electronic device", "tech product showcase"],
                'lifestyle': ["lifestyle product arrangement", "home goods display"],
                'education': ["online class interface", "educational materials"]
            }
            
            category = product_info.get('category', 'general')
            subjects = product_subjects.get(category, ["product display", "item showcase"])
            
            updated_image['scene'] = f"Clean product photography setup for {user_text}"
            updated_image['subjects'] = subjects
            updated_image['mood'] = "Professional and appealing"
            updated_image['background'] = "Minimal, clean background without any text"
            updated_image['negative_prompt'] = "text, typography, letters, words, writing"
        
        return updated_image
    
    def optimize_prompt_data(self, layout_data: BannerLayoutData, user_requirements: str) -> BannerLayoutData:
        """
        Main method to optimize the prompt based on user requirements.
        
        Args:
            layout_data: Original layout data
            user_requirements: User's requirements for the banner
            
        Returns:
            BannerLayoutData: Optimized layout data
        """
        # Update text descriptions with Korean marketing focus
        updated_text = self.update_text_descriptions(layout_data.text, user_requirements)
        
        # Update image parameters with product focus
        image_dict = layout_data.image.dict()  # Now using image instead of background
        updated_image_dict = self.update_image_parameters(image_dict, user_requirements)
        
        # Create optimized layout data
        try:
            # Convert subjects back to Subject objects if they were updated
            subjects = updated_image_dict.get('subjects', [])
            if subjects and isinstance(subjects[0], str):
                # Convert string subjects to Subject objects
                subject_objects = []
                for subj_str in subjects:
                    subject_objects.append(Subject(
                        type="product",  # Changed to product type
                        description=subj_str,
                        pose="display",  # Product display pose
                        position="center"  # Center position for product focus
                    ))
                updated_image_dict['subjects'] = subject_objects
            
            optimized_layout_data = BannerLayoutData(
                text=updated_text,
                image=updated_image_dict,  # Now using image instead of background
                image_info=layout_data.image_info  # Preserve the original image dimensions
            )
            
            return optimized_layout_data
            
        except Exception as e:
            logger.error(f"Error creating optimized layout data: {e}")
            # Return original data if optimization fails
            return layout_data
    
    def generate_final_prompt(self, optimized_data: BannerLayoutData) -> str:
        """
        Generate final text prompt for image generation from optimized data.
        
        Args:
            optimized_data: Optimized layout data
            
        Returns:
            str: Final prompt for image generation
        """
        # Extract key elements
        image = optimized_data.image  # Now using image instead of background
        text_descriptions = [elem.description for elem in optimized_data.text]
        
        # Build comprehensive prompt
        prompt_parts = []
        
        # Scene description
        prompt_parts.append(image.scene)
        
        # Subjects (products)
        if image.subjects:
            subjects_text = ", ".join([subj.description for subj in image.subjects])
            prompt_parts.append(f"featuring {subjects_text}")
        
        # Style and mood
        prompt_parts.append(f"in {image.style} style")
        prompt_parts.append(f"with {image.mood} mood")
        
        # Lighting and composition
        prompt_parts.append(f"using {image.lighting}")
        prompt_parts.append(f"composed with {image.composition}")
        
        # Camera settings
        if image.camera:
            camera_desc = f"{image.camera.distance} {image.camera.angle} with {image.camera.focus}"
            prompt_parts.append(f"shot as {camera_desc}")
        
        # Color palette
        if image.color_palette:
            colors = ", ".join(image.color_palette)
            prompt_parts.append(f"using color palette: {colors}")
        
        # Note about text areas (but no actual text in image)
        prompt_parts.append("clean areas reserved for text overlay")
        
        # Combine all parts
        final_prompt = ", ".join(prompt_parts)
        
        # Add quality enhancers
        quality_terms = [
            "high quality", "professional product photography", "commercial advertising",
            "Korean marketing style", "clean composition", "no text in image"
        ]
        
        final_prompt += f", {', '.join(quality_terms)}"
        
        # Add negative prompt reminder
        if hasattr(image, 'negative_prompt'):
            final_prompt += f". Negative: {image.negative_prompt}"
        
        return final_prompt
    
    def optimize_prompt(self, layout_data: BannerLayoutData, user_requirements: str) -> BannerPromptResponse:
        """
        Main service method to optimize banner prompt.
        
        Args:
            layout_data: Layout data from extraction
            user_requirements: User's requirements for the banner
            
        Returns:
            BannerPromptResponse: Service response with optimized data or error
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not user_requirements or not user_requirements.strip():
                return BannerPromptResponse(
                    success=False,
                    error="User requirements cannot be empty",
                    processing_time=time.time() - start_time
                )
            
            # Optimize the prompt data
            optimized_data = self.optimize_prompt_data(layout_data, user_requirements)
            
            return BannerPromptResponse(
                success=True,
                optimized_data=optimized_data,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return BannerPromptResponse(
                success=False,
                error=f"Prompt optimization failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def optimize_prompt_for_job(self, job_id: str, layout_data: BannerLayoutData, user_requirements: str, target_size: str = "1792x1024") -> BannerPromptResponse:
        """
        Optimize prompt for a specific job with database tracking.
        
        Args:
            job_id: Banner job ID
            layout_data: Layout data from extraction step
            user_requirements: User's requirements for the banner
            
        Returns:
            BannerPromptResponse: Service response with optimized data or error
        """
        start_time = time.time()
        
        # Update step status to in_progress
        update_step_status(
            self.db, job_id, 2, StepStatus.IN_PROGRESS,
            input_data={
                "user_requirements": user_requirements,
                "text_elements_count": len(layout_data.text) if layout_data.text else 0,
                "background_style": layout_data.background.style if layout_data.background else "unknown"
            },
            api_model=self.model
        )
        
        try:
            # Get layout extraction ID from database
            layout_extraction = self.db.query(LayoutExtraction).filter(
                LayoutExtraction.job_id == job_id
            ).first()
            
            # Perform prompt optimization
            response = self.optimize_prompt(layout_data, user_requirements)
            
            if response.success and response.optimized_data:
                # Generate final text prompt for image generation
                final_prompt = self.generate_final_prompt(response.optimized_data)
                
                # Store prompt generation results in database
                prompt_generation = PromptGeneration(
                    job_id=job_id,
                    layout_extraction_id=layout_extraction.id if layout_extraction else None,
                    user_requirements=user_requirements,
                    optimized_scene=response.optimized_data.dict(),
                    generation_prompt=final_prompt,
                    prompt_model=self.model,
                    processing_time=response.processing_time
                )
                
                self.db.add(prompt_generation)
                self.db.commit()
                
                # Update step status to completed
                output_data = {
                    "prompt_generation_id": prompt_generation.id,
                    "final_prompt": final_prompt,
                    "optimized_scene": response.optimized_data.dict(),
                    "prompt_length": len(final_prompt)
                }
                
                update_step_status(
                    self.db, job_id, 2, StepStatus.COMPLETED,
                    output_data=output_data,
                    processing_time=response.processing_time
                )
                
                logger.info(f"Prompt optimization completed for job {job_id}")
                
            else:
                # Update step status to failed
                update_step_status(
                    self.db, job_id, 2, StepStatus.FAILED,
                    error_message=response.error or "Unknown error",
                    processing_time=response.processing_time
                )
                
                # Update job status to failed
                update_job_status(
                    self.db, job_id, JobStatus.FAILED,
                    error_message=f"Prompt optimization failed: {response.error}"
                )
                
                logger.error(f"Prompt optimization failed for job {job_id}: {response.error}")
            
            return response
            
        except Exception as e:
            error_msg = f"Prompt optimization service error: {str(e)}"
            logger.error(error_msg)
            
            # Update step and job status to failed
            update_step_status(
                self.db, job_id, 2, StepStatus.FAILED,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )
            
            update_job_status(
                self.db, job_id, JobStatus.FAILED,
                error_message=error_msg
            )
            
            return BannerPromptResponse(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )