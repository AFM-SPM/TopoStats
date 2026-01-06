"""Classes."""

from pydantic import BaseModel, ConfigDict


class TopoStatsBaseModel(BaseModel):
    """Base model for TopoStats classes."""

    # Allow numpy arrays
    model_config = ConfigDict(arbitrary_types_allowed=True)
