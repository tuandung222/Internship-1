"""
Tests for Florence-2 coordinate handling.

Tests that Florence-2 clients correctly handle coordinate conversion:
- Grounding returns normalized [0, 1] coordinates
- Captioning accepts normalized coordinates and crops correctly
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from corgi.models.florence.florence_grounding_client import Florence2GroundingClient
from corgi.models.florence.florence_captioning_client import Florence2CaptioningClient
from corgi.core.config import ModelConfig


@pytest.fixture
def mock_florence_model():
    """Mock Florence-2 model and processor."""
    with patch(
        "corgi.models.florence.florence_grounding_client._load_florence_backend"
    ) as mock_loader_ground:
        with patch(
            "corgi.models.florence.florence_captioning_client._load_florence_backend"
        ) as mock_loader_cap:
            mock_model = Mock()
            mock_processor = Mock()
            mock_loader_ground.return_value = (mock_model, mock_processor)
            mock_loader_cap.return_value = (mock_model, mock_processor)
            yield mock_model, mock_processor


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (1000, 800), color="white")


class TestFlorence2GroundingCoordinates:
    """Tests for Florence-2 grounding coordinate conversion."""

    def test_grounding_returns_normalized_coords(
        self, mock_florence_model, sample_image
    ):
        """Test that Florence-2 grounding returns normalized [0, 1] coordinates."""
        mock_model, mock_processor = mock_florence_model

        # Mock the generate method to return pixel coordinates
        mock_model.generate.return_value = MagicMock()

        # Mock processor to return pixel bboxes from Florence-2
        mock_processor.batch_decode.return_value = ["<s>mocked</s>"]
        mock_processor.post_process_generation.return_value = {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[100, 200, 500, 600]],  # Pixel coordinates
                "labels": ["object"],
            }
        }
        mock_processor.return_value = {"input_ids": Mock(), "pixel_values": Mock()}

        config = ModelConfig(
            model_id="microsoft/Florence-2-large", model_type="florence2", device="cpu"
        )
        client = Florence2GroundingClient(config)

        # Extract regions
        bboxes = client.extract_regions(sample_image, "test statement", max_regions=1)

        # Should return normalized coordinates [0, 1]
        assert len(bboxes) == 1
        bbox = bboxes[0]

        # Verify normalized to [0, 1]
        assert 0 <= bbox[0] <= 1
        assert 0 <= bbox[1] <= 1
        assert 0 <= bbox[2] <= 1
        assert 0 <= bbox[3] <= 1

        # Verify correct conversion from pixel to normalized
        # Original: [100, 200, 500, 600] with image size (1000, 800)
        expected = (100 / 1000, 200 / 800, 500 / 1000, 600 / 800)
        assert bbox == pytest.approx(expected)

    def test_grounding_handles_empty_results(self, mock_florence_model, sample_image):
        """Test that grounding handles empty results gracefully."""
        mock_model, mock_processor = mock_florence_model

        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["<s>mocked</s>"]
        mock_processor.post_process_generation.return_value = {
            "<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [], "labels": []}
        }
        mock_processor.return_value = {"input_ids": Mock(), "pixel_values": Mock()}

        config = ModelConfig(
            model_id="microsoft/Florence-2-large", model_type="florence2", device="cpu"
        )
        client = Florence2GroundingClient(config)

        bboxes = client.extract_regions(sample_image, "test statement", max_regions=3)
        assert bboxes == []


class TestFlorence2CaptioningCoordinates:
    """Tests for Florence-2 captioning coordinate handling."""

    def test_captioning_accepts_normalized_coords(
        self, mock_florence_model, sample_image
    ):
        """Test that captioning accepts normalized [0, 1] coordinates."""
        mock_model, mock_processor = mock_florence_model

        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["<s>mocked</s>"]
        mock_processor.post_process_generation.return_value = {
            "<DETAILED_CAPTION>": "A test caption"
        }
        mock_processor.return_value = {"input_ids": Mock(), "pixel_values": Mock()}

        config = ModelConfig(
            model_id="microsoft/Florence-2-large", model_type="florence2", device="cpu"
        )
        client = Florence2CaptioningClient(config)

        # Pass normalized coordinates
        normalized_bbox = (0.1, 0.2, 0.5, 0.6)
        caption = client.caption_region(sample_image, normalized_bbox)

        # Should return a caption
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_crop_region_converts_to_pixels(self, sample_image):
        """Test that _crop_region correctly converts normalized to pixel coords."""
        # This is a static method, can test directly
        normalized_bbox = (0.1, 0.2, 0.5, 0.6)

        cropped = Florence2CaptioningClient._crop_region(sample_image, normalized_bbox)

        # Verify cropped image size
        # Expected pixel coords: (100, 160, 500, 480) for 1000x800 image
        expected_width = int(0.5 * 1000) - int(0.1 * 1000)  # 500 - 100 = 400
        expected_height = int(0.6 * 800) - int(0.2 * 800)  # 480 - 160 = 320

        assert cropped.size == (expected_width, expected_height)

    def test_crop_region_handles_edge_cases(self, sample_image):
        """Test that crop_region handles edge cases (0, 1) correctly."""
        # Full image
        bbox = (0.0, 0.0, 1.0, 1.0)
        cropped = Florence2CaptioningClient._crop_region(sample_image, bbox)
        assert cropped.size == sample_image.size

        # Small region at corner
        bbox = (0.0, 0.0, 0.1, 0.1)
        cropped = Florence2CaptioningClient._crop_region(sample_image, bbox)
        expected_width = int(0.1 * 1000)
        expected_height = int(0.1 * 800)
        assert cropped.size == (expected_width, expected_height)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
