import os
import grpc
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protos import model_pb2, model_pb2_grpc

def test_health():
    """тест health эндпоинта"""
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_pb2_grpc.PredictionServiceStub(channel)
        response = stub.Health(model_pb2.HealthRequest())
        print(f"Health: status={response.status}, model_version={response.model_version}")

def test_predict():
    """тест predict эндпоинта с данными iris"""
    # тестовые фичи для iris (setosa)
    features = [5.1, 3.5, 1.4, 0.2]
    
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_pb2_grpc.PredictionServiceStub(channel)
        request = model_pb2.PredictRequest(features=features)
        response = stub.Predict(request)
        print(f"Predict: prediction={response.prediction}, confidence={response.confidence:.2f}, model_version={response.model_version}")

def main():
    print("тестирование gRPC ML сервиса...")
    test_health()
    test_predict()

if __name__ == "__main__":
    main()
