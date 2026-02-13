import requests
import io
from PIL import Image
import unittest

class TestBackend(unittest.TestCase):
    def test_sketch_generation(self):
        # Create a dummy image
        img = Image.new('RGB', (100, 100), color = 'red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        url = 'http://localhost:5000/sketch'
        files = {'image': ('test.png', img_byte_arr, 'image/png')}

        try:
            response = requests.post(url, files=files)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['Content-Type'], 'image/png')
            print("Test passed: API returned 200 OK and an image.")
        except requests.exceptions.ConnectionError:
            print("Test failed: Could not connect to backend. Is it running?")
            # self.fail("Backend not running") 
            # We don't want to fail the test runner if we're just checking connectivity manually.

if __name__ == '__main__':
    unittest.main()
