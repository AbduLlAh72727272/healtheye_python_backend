# Import the prediction server app for Vercel serverless deployment
from prediction_server import app

def handler(request):
    return app(request, {})