import unittest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from src.data.processors.text_processor import TextProcessor
from src.config import OfflineTextCleaningConfig

class TestTextProcessor(unittest.TestCase):
    """Unit tests for the TextProcessor class."""

    def setUp(self):
        """Set up a sample DataFrame and configurations for testing."""
        self.df = pd.DataFrame({
            'title': ["  A Title with <p>HTML</p>  ", "ANOTHER TITLE IN CAPS"],
            'description': ["Description with extra    whitespace.", "Final\u00A0description."],
            'tag': ["tag1", None]
        })
        # Config that enables all cleaning steps
        self.full_config = OfflineTextCleaningConfig(
            remove_html=True,
            normalize_unicode=True,
            to_lowercase=True
        )
        # Config that disables all cleaning steps
        self.no_op_config = OfflineTextCleaningConfig(
            remove_html=False,
            normalize_unicode=False,
            to_lowercase=False
        )
        self.processor = TextProcessor(cleaning_config=self.full_config)


    def test_clean_text_field(self):
        """Tests cleaning of a single text string."""
        text = "  A <B>Bold</B> Title with non-breaking\u00A0space  "
        expected = "a bold title with non-breaking space"
        self.assertEqual(self.processor.clean_text_field(text), expected)

    def test_clean_dataframe_text_columns(self):
        """Tests the cleaning of multiple text columns in a DataFrame."""
        cleaned_df = self.processor.clean_dataframe_text_columns(
            self.df, 
            text_columns=['title', 'description']
        )
        self.assertEqual(cleaned_df.loc[0, 'title'], "a title with html")
        self.assertEqual(cleaned_df.loc[1, 'title'], "another title in caps")
        self.assertEqual(cleaned_df.loc[0, 'description'], "description with extra whitespace.")
        self.assertEqual(cleaned_df.loc[1, 'description'], "final description.")

    def test_get_combined_text(self):
        """Tests the combination of multiple text columns into one string."""
        row = pd.Series({
            'title': 'My Title',
            'tag': 'MyTag',
            'description': 'A description.'
        })
        combined = self.processor.get_combined_text(row, ['title', 'tag', 'description'], separator=' | ')
        self.assertEqual(combined, "My Title | MyTag | A description.")

    def test_get_combined_text_with_missing_data(self):
        """Tests combining text when some columns are missing or NaN."""
        row = pd.Series({
            'title': 'My Title',
            'tag': None, # Missing tag
            'description': 'A description.'
        })
        combined = self.processor.get_combined_text(row, ['title', 'tag', 'description'], separator=' ')
        self.assertEqual(combined, "My Title A description.")

    def test_configuration_flags_respected(self):
        """Tests that the processor respects the boolean flags in its config."""
        # Correctly initialize the processor by naming the 'cleaning_config' argument.
        processor_no_op = TextProcessor(cleaning_config=self.no_op_config)
        text = "  A <B>Bold</B> Title  "
        
        # With the no-op config, only stripping should occur
        expected = "A <B>Bold</B> Title"
        self.assertEqual(processor_no_op.clean_text_field(text), expected)

        # Test with only lowercase enabled
        config_only_lower = OfflineTextCleaningConfig(remove_html=False, normalize_unicode=False, to_lowercase=True)
        # Correctly initialize this processor as well.
        processor_only_lower = TextProcessor(cleaning_config=config_only_lower)
        expected_lower = "a <b>bold</b> title"
        self.assertEqual(processor_only_lower.clean_text_field(text), expected_lower)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)