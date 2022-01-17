from torch import nn


class SimpleFeatureExtractor(nn.Module):
    def __init__(self, feature_dim, model_type):
        super(SimpleFeatureExtractor, self).__init__()

        if model_type == "mnist":
            self.feature_extractor = MNISTSimpleFeatureExtractor(feature_dim)
        elif model_type == "synthesis":
            self.feature_extractor = SynthesisSimpleFeatureExtractor(feature_dim)

    def forward(self, x):
        return self.feature_extractor(x)



class SynthesisSimpleFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )
        self.refactor = nn.Linear(16, feature_dim)

    def forward(self, x):
        feature = self.fc(x)
        feature = self.refactor(feature)
        return feature



class MNISTSimpleFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        self.fc = nn.Sequential(
            nn.Linear(179776, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.25),
            nn.Linear(200, 100)
        )

        self.refactor = nn.Linear(100, feature_dim)

    def forward(self, x):
        feature = self.feature(x)
        # print(feature.shape)
        # print("check")
        feature = self.fc(feature)
        feature = self.refactor(feature)
        return feature
