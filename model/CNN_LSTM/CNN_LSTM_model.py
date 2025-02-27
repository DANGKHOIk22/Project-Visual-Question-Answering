import torch
import torch.nn as nn
import timm

class VQAModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        img_model_name: str,
        embedding_dim: int,
        vocab,
        n_layers: int = 2,
        hidden_size: int = 256,
        drop_p: float = 0.2
    ):
        super(VQAModel, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)

        # Image Feature Extractor
        self.image_encoder = timm.create_model(
            img_model_name, pretrained=True, num_classes=hidden_size
        )

        # Unfreeze the image encoder for training
        for param in self.image_encoder.parameters():
            param.requires_grad = True

        # Text Embedding & LSTM Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=drop_p,
            bidirectional=True  
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # Concatenating (img + text)
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, img, text):
        """
        Forward pass for VQAModel.

        Args:
            img (Tensor): Image tensor (batch_size, C, H, W).
            text (Tensor): Tokenized question tensor (batch_size, seq_len).

        Returns:
            Tensor: Model output (batch_size, n_classes).
        """
        # Extract image features
        img_features = self.image_encoder(img)

        # Process text input
        text_features = self.embedding(text)  # Shape: (batch, seq_len, embedding_dim)
        text_features, (hn, cn) = self.lstm(text_features)  # LSTM output

        # Properly extract last hidden state from bidirectional LSTM
        text_features = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # Shape: (batch, hidden_size * 2)

        # Concatenate image and text features
        features = torch.cat((img_features, text_features), dim=1)  # Shape: (batch, hidden_size * 3)

        # Final classification layer
        output = self.fc(features)

        return output
