from src.losses.watermark_losses import watermark_loss


def test_watermark_loss_forward():
    embeddings = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    is_trigger = [1, 0, 1]
    signature = [1.0, 0.0]
    out = watermark_loss(0.2, embeddings, is_trigger, signature, 1.0, 0.1, 0.2)
    assert "total" in out
    assert out["total"] >= 0.0
