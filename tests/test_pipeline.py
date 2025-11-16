from sophia.pipeline import SophiaPipeline


def test_pipeline_basic() -> None:
    pipeline = SophiaPipeline()
    response = pipeline.answer("What is overfitting in machine learning?")
    assert response.topic
    assert response.technical_answer
    assert response.simple_explanation
    assert response.follow_up_question
