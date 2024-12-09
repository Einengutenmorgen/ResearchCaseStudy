"""Test script for the persona creation pipeline components."""

import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import pytest
from unittest.mock import Mock, patch

from src.pipeline.conversation_pipeline.components.persona.persona_creator import PersonaCreator
from src.pipeline.conversation_pipeline.components.persona.post_analyzer import PostAnalyzer
from src.pipeline.conversation_pipeline.components.persona.persona_pipeline import PersonaPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to control whether to run tests that require OpenAI API
SKIP_API_TESTS = True

def create_test_data():
    """Create sample data for testing."""
    users = ['user1', 'user2']
    posts = []
    
    base_time = datetime.now()
    
    # Create test posts for each user
    for user_id in users:
        for i in range(15):  # 15 posts per user
            post_time = base_time - timedelta(days=i)
            posts.append({
                'user_id': user_id,
                'full_text': f"Test post {i} from {user_id} #test @mention{i} http://test.com",
                'created_at': post_time.isoformat(),
                'reply_to_id': None,
                'reply_to_user': None
            })
    
    return pd.DataFrame(posts)

def test_post_analyzer():
    """Test PostAnalyzer component."""
    logger.info("Testing PostAnalyzer...")
    
    analyzer = PostAnalyzer()
    test_data = create_test_data()
    user_posts = test_data[test_data['user_id'] == 'user1'].to_dict('records')
    
    analysis = analyzer.analyze_posts(user_posts)
    
    # Assert expected structure
    assert 'post_metrics' in analysis, "Post metrics missing from analysis"
    assert 'temporal_patterns' in analysis, "Temporal patterns missing from analysis"
    assert 'content_patterns' in analysis, "Content patterns missing from analysis"
    
    # Assert expected values
    assert analysis['post_metrics']['total_posts'] == 15, "Incorrect total posts count"
    assert analysis['post_metrics']['avg_word_count'] == 5.0, "Incorrect average word count"
    assert analysis['content_patterns']['url_frequency'] == 1.0, "Incorrect URL frequency"
    
    logger.info("PostAnalyzer test completed successfully")

@pytest.mark.skipif(SKIP_API_TESTS, reason="Skipping tests that require OpenAI API")
def test_persona_creator():
    """Test PersonaCreator component with real API."""
    logger.info("Testing PersonaCreator with real API...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip("OpenAI API key not found in environment variables")
        
    creator = PersonaCreator(api_key, cache_dir="test_cache")
    test_data = create_test_data()
    user_posts = test_data[test_data['user_id'] == 'user1'].to_dict('records')
    
    persona = creator.create_persona('user1', user_posts)
    
    assert 'user_id' in persona, "User ID missing from persona"
    assert 'analysis' in persona, "Analysis missing from persona"
    assert 'metadata' in persona, "Metadata missing from persona"
    
    logger.info("PersonaCreator test with real API completed successfully")

def test_persona_creator_mock():
    """Test PersonaCreator component with mocked API."""
    logger.info("Testing PersonaCreator with mocked API...")
    
    # Create patches for both openai module and OpenAI class
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Mocked persona analysis"))]
    
    class MockCompletions:
        def create(self, model, messages, temperature):
            return mock_response
    
    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()
    
    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = MockChat()
    
    with patch('src.pipeline.conversation_pipeline.components.persona.persona_creator.OpenAI', MockOpenAI):
        # Create PersonaCreator with mock
        creator = PersonaCreator("fake-api-key", cache_dir="test_cache")
        test_data = create_test_data()
        user_posts = test_data[test_data['user_id'] == 'user1'].to_dict('records')
        
        persona = creator.create_persona('user1', user_posts)
        
        # Verify the result
        assert persona['user_id'] == 'user1', "Incorrect user ID"
        assert persona['analysis'] == "Mocked persona analysis", "Incorrect analysis"
        assert 'metadata' in persona, "Missing metadata"
        assert persona['metadata']['model_used'] == "gpt-4o", "Incorrect model"
        
        logger.info("PersonaCreator test with mocked API completed successfully")

def test_persona_pipeline_mock():
    """Test complete PersonaPipeline with mocked API."""
    logger.info("Testing PersonaPipeline with mocked API...")
    
    # Create patches for both openai module and OpenAI class
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Mocked persona analysis"))]
    
    class MockCompletions:
        def create(self, model, messages, temperature):
            return mock_response
    
    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()
    
    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = MockChat()
    
    with patch('src.pipeline.conversation_pipeline.components.persona.persona_creator.OpenAI', MockOpenAI):
        # Create pipeline with mock
        pipeline = PersonaPipeline(
            api_key="fake-api-key",
            output_dir="test_output",
            cache_dir="test_cache",
            min_posts=5,
            max_posts=15
        )
        
        test_data = create_test_data()
        results = pipeline.process_user_data(test_data)
        
        # Verify results
        assert 'personas' in results, "Personas missing from results"
        assert 'post_analyses' in results, "Post analyses missing from results"
        assert 'stats' in results, "Statistics missing from results"
        assert results['stats']['total_users'] == 2, "Incorrect user count"
        assert results['stats']['processed_users'] == 2, "Not all users processed"
        assert results['stats']['errors'] == 0, "Unexpected errors"
        
        # Test loading results
        loaded_results = PersonaPipeline.load_results("test_output")
        assert loaded_results['stats'] == results['stats'], "Loaded results don't match original"
        
        logger.info("PersonaPipeline test with mocked API completed successfully")

def main():
    """Run all tests."""
    load_dotenv()
    
    try:
        # Test individual components
        test_post_analyzer()
        
        if not SKIP_API_TESTS:
            test_persona_creator()
        
        # Test with mocks
        test_persona_creator_mock()
        test_persona_pipeline_mock()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
