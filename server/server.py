import os
import pickle
from concurrent import futures

import grpc

from protos import model_pb2, model_pb2_grpc


class PredictionServicer(model_pb2_grpc.PredictionServiceServicer):
    def __init__(self, model_path: str, model_version: str):
        self.model_path = model_path
        self.model_version = model_version
        self.model = self._load_model()

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def Health(self, request, context):
        return model_pb2.HealthResponse(
            status="ok",
            model_version=self.model_version,
        )

    def Predict(self, request, context):
        features = list(request.features)
        prediction = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])[0]
        confidence = float(max(proba))

        return model_pb2.PredictResponse(
            prediction=str(prediction),
            confidence=confidence,
            model_version=self.model_version,
        )


def serve():
    port = os.getenv("PORT", "50051")
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    model_version = os.getenv("MODEL_VERSION", "v1.0.0")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServicer(model_path, model_version),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    print(f"сервер запущен на порту {port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
