"""Tool for Locating the robot on the grid."""

from typing import Any, Optional, Type, List, Tuple

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr, Field
from langchain_core.tools import BaseTool 

import numpy as np
import json


# Grid dimensions (size of each square)
grid_width = 120
grid_height = 145

# Number of rows and columns in the grid
grid_rows = 4
grid_cols = 5

class LocationToolInput(BaseModel):
    """Input for Location Tool."""
    #image bytes
    bbox: List[str] = Field(description="Bounding box of the robot on the grid", example="[0.83, 0.69, 1.0, 0.99]")
    
class LocationToolRun(BaseTool):
    """Tool for Locating the robot on the grid."""
    
    name: str = "locationGridTool"
    description: str = (
        "A tool for locating the robot on the 4x5 grid from a bounding box."
        "Useful for when you need know location of the robot on the grid."
        "Input should be a bounding box."
    )
    args_schema: Type[BaseModel] = LocationToolInput
    
    _grid_centroids: List[Tuple[int, int]] = PrivateAttr()
    _image_width: int = PrivateAttr()
    _image_height: int = PrivateAttr()
    
    def __init__(self, **data):
        super().__init__(**data)
        # Precompute the centroids of all grid squares
        self._grid_centroids = [(col * grid_width + grid_width // 2, row * grid_height + grid_height // 2)
                  for row in range(grid_rows) for col in range(grid_cols)]
        # Projected Image Width and Height
        self._image_width = 610
        self._image_height = 590
    
    def _get_location(self, bounding_box):
        # Calculate the centroid of the bounding box
        centroid_x = (bounding_box[0] + bounding_box[2]) / 2
        centroid_y = (bounding_box[1] + bounding_box[3]) / 2
        #print(centroid_x, centroid_y)

        # Calculate distances to precomputed grid centroids
        distances = [np.sqrt((centroid_x - gc[0])**2 + (centroid_y - gc[1])**2) for gc in self._grid_centroids]
        #print(distances)

        # Find the index of the closest centroid
        closest_index = np.argmin(distances)
        #print(closest_index)

        # Convert the linear index to grid coordinates
        closest_row = closest_index // grid_cols
        closest_col = closest_index % grid_cols

        return f"({closest_row + 1},{closest_col + 1})"
    
    def _run(
        self,
        bbox: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the most likely location of the robot on the grid."""
        # Bounding box coordinates (x, y, width, height)
        print('Calling location tool')
        xmin, ymin, xmax, ymax = bbox
        xmin = int(max(0, min(float(xmin) * self._image_width, self._image_width)))
        ymin = int(max(0, min(float(ymin) * self._image_height, self._image_height)))
        xmax = int(max(0, min(float(xmax) * self._image_width, self._image_width)))
        ymax = int(max(0, min(float(ymax) * self._image_height, self._image_height)))

        bbox = [xmin, ymin, xmax, ymax]

        location = self._get_location(bbox)
        return location