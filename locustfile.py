from locust import HttpUser, task, between
import os

class ImageClassificationUser(HttpUser):
    """
    Simulates a user sending image prediction requests
    to the /predict endpoint of the FastAPI app.
    """
    wait_time = between(1, 3)  # time between tasks per user

    def on_start(self):
        """
        Called once when a simulated user starts.
        We load the sample image into memory to reuse it
        for all requests (faster than reading from disk each time).
        """
        sample_image_path = "sample_image.jpg"  

        if not os.path.exists(sample_image_path):
            raise FileNotFoundError(
                f"Sample image not found at {sample_image_path}. "
                "Please add a sample_image.jpg in the project root."
            )

        with open(sample_image_path, "rb") as f:
            self.image_bytes = f.read()

    @task
    def predict_image(self):
        """
        Task that sends a POST request to /predict
        with the sample image as form-data.
        """
        files = {
            "file": ("sample_image.jpg", self.image_bytes, "image/jpeg")
        }

        with self.client.post("/predict", files=files, name="/predict", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Non-200 status code: {response.status_code}")
            else:
                
                response.success()
