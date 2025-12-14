import unittest

from corgi.utils.parsers import parse_structured_reasoning


class StructuredReasoningParserTests(unittest.TestCase):
    def test_basic_parse(self):
        response = """
        ```json
        [
          {"index": 1, "statement": "The image shows a red car.", "needs_vision": true, "reason": "Requires visual color check."},
          {"index": 2, "statement": "Cars often park on streets.", "needs_vision": false, "reason": "General knowledge."}
        ]
        ```
        """
        steps = parse_structured_reasoning(response, max_steps=3)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].index, 1)
        self.assertEqual(steps[0].statement, "The image shows a red car.")
        self.assertTrue(steps[0].needs_vision)
        self.assertEqual(steps[0].reason, "Requires visual color check.")
        self.assertFalse(steps[1].needs_vision)

    def test_tolerates_strings_and_missing_reason(self):
        response = """
        [
          {"index": 3, "statement": "The player wears a blue jersey.", "needs_vision": "yes"},
          {"index": 4, "statement": "The match is likely soccer.", "needs_vision": "no", "reason": ""}
        ]
        """
        steps = parse_structured_reasoning(response, max_steps=5)
        self.assertEqual([s.index for s in steps], [3, 4])
        self.assertTrue(steps[0].needs_vision)
        self.assertIsNone(steps[0].reason)
        self.assertFalse(steps[1].needs_vision)

    def test_limits_steps(self):
        response = """
        [
          {"index": 1, "statement": "Step one", "needs_vision": true},
          {"index": 2, "statement": "Step two", "needs_vision": true},
          {"index": 3, "statement": "Step three", "needs_vision": true}
        ]
        """
        steps = parse_structured_reasoning(response, max_steps=2)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[-1].index, 2)

    def test_raises_on_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_structured_reasoning("not json at all", max_steps=3)

    def test_fallback_handles_thinking_style_text(self):
        response = """
        Got it, let's piece this together.
        Step 1: Lobster is a food item on the table. Needs vision: true. Reason: Visual confirmation is needed to see the lobster platter.
        Step 2: Shrimp dishes appear near the center of the spread. Needs vision: true. Reason: Only the image shows the shrimp bowls.
        Step 3: Chopsticks are arranged for diners on the right side. Needs vision: true. Reason: You must inspect the table setup to notice the chopsticks.
        """
        steps = parse_structured_reasoning(response, max_steps=4)
        self.assertEqual(len(steps), 3)
        self.assertEqual([step.index for step in steps], [1, 2, 3])
        self.assertTrue(all(step.needs_vision for step in steps))
        self.assertEqual(steps[0].statement, "Lobster is a food item on the table")
        self.assertEqual(
            steps[0].reason, "Visual confirmation is needed to see the lobster platter"
        )
        self.assertEqual(
            steps[1].statement, "Shrimp dishes appear near the center of the spread"
        )
        self.assertEqual(
            steps[2].statement, "Chopsticks are arranged for diners on the right side"
        )

    def test_handles_ordinal_step_markers(self):
        response = """
        First step: Check for any people in the scene. Statement: "No people are visible in the frame."
        Needs_vision: true. Reason: Visual inspection is required to confirm the absence of people.

        Second step: Determine if a white watch is present on anyone. Statement: "Since no people are present, no white watch can be seen."
        Needs_vision: true? Reason: Without people, visual evidence confirms that no accessories are worn.
        """
        steps = parse_structured_reasoning(response, max_steps=4)
        self.assertEqual(len(steps), 2)
        self.assertEqual([step.index for step in steps], [1, 2])
        self.assertTrue(all(step.needs_vision for step in steps))
        self.assertIn("no people are visible", steps[0].statement.lower())
        self.assertTrue(steps[0].reason.lower().startswith("visual inspection"))
        self.assertIn("no white watch", steps[1].statement.lower())

    def test_filters_meta_commentary_step(self):
        response = """
        Step 1: Count how many people are visible. Needs vision: true.
        Step 2: Check whether any person wears a white watch. Needs vision: true.
        Step 3: Maybe confirm the watch color? But the question is already answered in step 2, wait, the protocol says two steps.
        """
        steps = parse_structured_reasoning(response, max_steps=3)
        self.assertEqual(len(steps), 2)
        self.assertEqual([step.index for step in steps], [1, 2])


if __name__ == "__main__":
    unittest.main()
