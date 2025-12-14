"""
Qwen Captioning Adapter.

Adapter to use Qwen models for region captioning.
This wraps the existing Qwen client to implement the captioning protocol.
"""

from __future__ import annotations

from typing import Tuple, Optional, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def _extend_bbox_for_crop(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Extend bounding box by 25% on each dimension (12.5% on each side).
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
    
    Returns:
        Extended bounding box (x1, y1, x2, y2) clamped to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    # Extend bbox by 25% on each dimension (12.5% on each side)
    width = x2 - x1
    height = y2 - y1
    extend_x = width * 0.125  # 12.5% of width on each side = 25% total
    extend_y = height * 0.125  # 12.5% of height on each side = 25% total
    
    # Apply extension
    x1_extended = max(0.0, x1 - extend_x)
    y1_extended = max(0.0, y1 - extend_y)
    x2_extended = min(1.0, x2 + extend_x)
    y2_extended = min(1.0, y2 + extend_y)
    
    return (x1_extended, y1_extended, x2_extended, y2_extended)


class QwenCaptioningAdapter:
    """
    Adapter to use Qwen for region captioning.
    
    This adapter wraps a Qwen client (Instruct or Thinking) and implements
    the captioning protocol by using Qwen's image understanding capabilities.
    """
    
    def __init__(self, qwen_client):
        """
        Initialize Qwen captioning adapter.
        
        Args:
            qwen_client: A Qwen client (Qwen3VLInstructClient or Qwen3VLThinkingClient)
        """
        self.client = qwen_client
    
    def caption_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
    ) -> str:
        """
        Generate caption for a specific region.
        
        Uses Qwen's image understanding to describe the cropped region.
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging
        
        Returns:
            Textual description of the region
        """
        try:
            # Extend bbox by 25% before cropping
            x1_ext, y1_ext, x2_ext, y2_ext = _extend_bbox_for_crop(bbox)
            
            # Crop the region
            w, h = image.size
            
            # Convert normalized to pixel coordinates
            left = int(x1_ext * w)
            top = int(y1_ext * h)
            right = int(x2_ext * w)
            bottom = int(y2_ext * h)
            
            # Ensure valid coordinates
            left = max(0, min(left, w-1))
            top = max(0, min(top, h-1))
            right = max(left+1, min(right, w))
            bottom = max(top+1, min(bottom, h))
            
            cropped = image.crop((left, top, right, bottom))
            
            # Use Qwen to caption the cropped region
            prompt = "Describe this image region in detail. Focus on objects, colors, and spatial relationships."
            
            if hasattr(self.client, '_chat'):
                response = self.client._chat(image=cropped, prompt=prompt, max_new_tokens=128)
            else:
                # Fallback: use a simple description
                response = f"Region at ({x1_ext:.2f}, {y1_ext:.2f}, {x2_ext:.2f}, {y2_ext:.2f})"
            
            if not response or not response.strip():
                logger.warning(f"Empty response from Qwen captioning for bbox: {bbox}")
                return "Unable to caption region"
            
            logger.debug(f"Qwen captioning generated: {response[:80]}")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Qwen captioning failed: {e}")
            return "Caption generation failed"
    
    def caption_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
    ) -> List[str]:
        """
        Generate captions for multiple regions in batch.
        
        Processes multiple cropped regions efficiently.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
        
        Returns:
            List of captions corresponding to each bbox
        """
        if not bboxes:
            return []
        
        logger.info(f"[Qwen Captioning] Batch captioning {len(bboxes)} regions")
        captions = []
        
        for bbox_idx, bbox in enumerate(bboxes):
            try:
                # Extend bbox by 25% before cropping
                x1_ext, y1_ext, x2_ext, y2_ext = _extend_bbox_for_crop(bbox)
                
                # Crop the region
                w, h = image.size
                
                # Convert normalized to pixel coordinates
                left = int(x1_ext * w)
                top = int(y1_ext * h)
                right = int(x2_ext * w)
                bottom = int(y2_ext * h)
                
                # Ensure valid coordinates
                left = max(0, min(left, w-1))
                top = max(0, min(top, h-1))
                right = max(left+1, min(right, w))
                bottom = max(top+1, min(bottom, h))
                
                cropped = image.crop((left, top, right, bottom))
                
                # Use Qwen to caption the cropped region
                prompt = "Describe this image region in detail. Focus on objects, colors, and spatial relationships."
                
                if hasattr(self.client, '_chat'):
                    response = self.client._chat(image=cropped, prompt=prompt, max_new_tokens=128)
                else:
                    # Fallback: use a simple description
                    response = f"Region at ({x1_ext:.2f}, {y1_ext:.2f}, {x2_ext:.2f}, {y2_ext:.2f})"
                
                if not response or not response.strip():
                    logger.warning(f"Empty response from Qwen captioning for bbox {bbox_idx}: {bbox}")
                    captions.append("Unable to caption region")
                else:
                    captions.append(response.strip())
                    logger.debug(f"Qwen captioning generated for bbox {bbox_idx}: {response[:80]}")
                    
            except Exception as e:
                logger.error(f"Qwen captioning failed for bbox {bbox_idx}: {e}")
                captions.append("Caption generation failed")
        
        logger.info(f"[Qwen Captioning] Batch captioning completed for {len(captions)} regions")
        return captions
    
    def ocr_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
    ) -> str:
        """
        Extract OCR text from a specific region using Qwen.
        
        Note: Qwen models may not have dedicated OCR capabilities.
        This method attempts to extract text by asking Qwen to read text.
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging
        
        Returns:
            Extracted text, or empty string if not supported
        """
        try:
            # Extend bbox by 25% before cropping
            x1_ext, y1_ext, x2_ext, y2_ext = _extend_bbox_for_crop(bbox)
            
            # Crop the region
            w, h = image.size
            
            # Convert normalized to pixel coordinates
            left = int(x1_ext * w)
            top = int(y1_ext * h)
            right = int(x2_ext * w)
            bottom = int(y2_ext * h)
            
            # Ensure valid coordinates
            left = max(0, min(left, w-1))
            top = max(0, min(top, h-1))
            right = max(left+1, min(right, w))
            bottom = max(top+1, min(bottom, h))
            
            cropped = image.crop((left, top, right, bottom))
            
            # Use Qwen to extract text from the cropped region
            prompt = "Extract and return all text visible in this image. Return only the text content, nothing else."
            
            if hasattr(self.client, '_chat'):
                response = self.client._chat(image=cropped, prompt=prompt, max_new_tokens=256)
            else:
                logger.warning("Qwen client does not support OCR extraction")
                return ""
            
            if not response or not response.strip():
                logger.warning(f"Empty OCR response from Qwen for bbox: {bbox}")
                return ""
            
            logger.debug(f"Qwen OCR extracted: {response[:80]}")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Qwen OCR failed: {e}")
            return ""
    
    def ocr_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
    ) -> List[str]:
        """
        Extract OCR text from multiple regions in batch.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
        
        Returns:
            List of OCR text corresponding to each bbox
        """
        if not bboxes:
            return []
        
        logger.info(f"[Qwen OCR] Batch OCR {len(bboxes)} regions")
        ocr_texts = []
        
        for bbox_idx, bbox in enumerate(bboxes):
            try:
                ocr_text = self.ocr_region(image, bbox, step_index, bbox_idx)
                ocr_texts.append(ocr_text)
            except Exception as e:
                logger.error(f"Qwen OCR failed for bbox {bbox_idx}: {e}")
                ocr_texts.append("")
        
        logger.info(f"[Qwen OCR] Batch OCR completed for {len(ocr_texts)} regions")
        return ocr_texts


__all__ = ["QwenCaptioningAdapter"]

