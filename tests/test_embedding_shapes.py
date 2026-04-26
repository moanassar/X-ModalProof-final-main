from src.models.text_model import SimpleTextWatermarkModel


def test_embedding_shape():
    model = SimpleTextWatermarkModel(vocab_size=128, embedding_dim=32, num_labels=4)
    batch = {"input_ids": [1, 2, 3, 4, 5]}
    emb = model.extract_embedding(batch)
    assert len(emb) == 32
