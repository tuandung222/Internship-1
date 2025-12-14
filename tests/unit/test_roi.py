import unittest

from corgi.utils.parsers import parse_roi_evidence


class ROIParserTests(unittest.TestCase):
    def test_normalizes_and_sorts_boxes(self):
        response = """
        ```json
        [
          {"step": 1, "bbox": [800, 200, 200, 900], "description": "Red car on the left", "confidence": 0.76},
          {"step": 2, "bbox": [0.1, 0.2, 0.3, 0.4], "description": "Scoreboard", "confidence": 0.55}
        ]
        ```
        """
        evidences = parse_roi_evidence(response, default_step_index=1)
        self.assertEqual(len(evidences), 2)
        self.assertEqual(evidences[0].bbox, (0.2, 0.2, 0.8, 0.9))
        self.assertTrue(0 <= evidences[1].bbox[0] <= 1)
        self.assertEqual(evidences[0].confidence, 0.76)
        self.assertEqual(evidences[0].description, "Red car on the left")
        self.assertEqual(evidences[0].step_index, 1)

    def test_handles_missing_fields(self):
        response = """
        [
          {"bbox": [10, 20, 30, 40]}
        ]
        """
        evidences = parse_roi_evidence(response, default_step_index=3)
        self.assertEqual(len(evidences), 1)
        ev = evidences[0]
        self.assertEqual(ev.step_index, 3)
        self.assertIsNone(ev.description)
        self.assertIsNone(ev.confidence)

    def test_fallback_parses_bbox_from_text(self):
        response = """
        Checking the platter confirms lobster regions.
        First lobster near the platter edge: [285, 275, 455, 670] – vivid red shell.
        Second lobster closer to the bottom plate: [382, 537, 555, 750].
        """
        evidences = parse_roi_evidence(response, default_step_index=2)
        self.assertEqual(len(evidences), 2)
        first = evidences[0]
        self.assertEqual(first.step_index, 2)
        self.assertIsNone(first.confidence)
        self.assertEqual(first.description, "First lobster near the platter edge")
        self.assertAlmostEqual(first.bbox[0], 0.285, places=3)
        self.assertAlmostEqual(first.bbox[1], 0.275, places=3)
        self.assertAlmostEqual(first.bbox[2], 0.455, places=3)
        self.assertAlmostEqual(first.bbox[3], 0.67, places=3)


if __name__ == "__main__":
    unittest.main()
