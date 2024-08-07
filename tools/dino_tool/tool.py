"""Tool for Groundig dino model."""

from typing import Any, Optional, Type, List, Union, Dict

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
from langchain_core.tools import BaseTool 

import requests
import threading
import base64
import time
import roslibpy
import cv2
import numpy as np
from io import BytesIO

DINO_ENDPOINT = "http://localhost:8081/predict"

class DinoInput(BaseModel):
    """Input for DinoTool."""

class DinoPromptRun(BaseTool):
    """Tool for Grounding dino model."""
    
    _instance = None
    _lock: threading.Lock = threading.Lock()
    
    name: str = "dinoDetectionTool"
    description: str = (
        "A tool for detecting the robot to obtained a bounding box."
        "Useful for when you need to detect the robot in the image but it will not tell you where on the grid the robot is located."
        "No input is necessary as it will read the image for you."
    )
    args_schema: Type[BaseModel] = DinoInput
    
    _ros: roslibpy.Ros = PrivateAttr()
    _image_listener: roslibpy.Topic = PrivateAttr()
    _image: Optional[np.array] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, **data):
        if self._initialized:
            return
        super().__init__( **data)
        
        # Establish a connection to the ROS master
        self._ros = roslibpy.Ros(host='localhost', port=9090)
        
        # Define the topic to subscribe to
        self._image_listener = roslibpy.Topic(self._ros, '/camera/image_projected/compressed', 'sensor_msgs/CompressedImage')

        # Subscribe to the image topic with the callback function
        self._image_listener.subscribe(self._image_callback)
        
        # Run the ROS event loop until the image is received and connection is terminated
        if not self._ros.is_connected:
            print("Connecting to ROS...")
            self._ros.run()
            time.sleep(1)
        self._initialized = True
        
    def _image_callback(self, message):
        try:
            data = message['data']

            # Convert the data to a numpy array
            # Decode base64 image
            image_bytes = base64.b64decode(data)
            image_data = np.frombuffer(image_bytes, np.uint8)

            # Convert to OpenCV format
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            # Crop the followin area top_left (195,5) botto_right (805, 595)
            # Crop the following area top_left (195,5) bottom_right (805, 595)
            image = image[5:595, 195:805]

            # Process the image as a string (if needed for other purposes)
            self._image = image

        except Exception as e:
            print(f"Error processing image: {e}")
        
        
    def __del__(self):
        self._image_listener.unsubscribe()
        self._ros.terminate()
    
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Dino model."""
        print('Calling dino tool')
        
        object_prompt = "".join("Turtlebot3")
        _, image_buffer = cv2.imencode('.jpg', self._image)
        image_buffer = BytesIO(image_buffer.tobytes())
        image_buffer.seek(0)
        response = requests.post(DINO_ENDPOINT, files={'file': ('image.jpg', image_buffer, 'image/jpeg')}, data={'prompt': object_prompt})
        #return response.json()[0]
        bbox = response.json()[0]['boxes'][0]

        return bbox
        
        