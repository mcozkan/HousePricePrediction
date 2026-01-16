
import numpy as np
import logging
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz as sv
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, Any

# We add the missing alert class into Numpy lirary in order to solve the version problem...
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = type('VisibleDeprecationWarning', (DeprecationWarning,), {})
import matplotlib.pyplot as plt
import seaborn as sns