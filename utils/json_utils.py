import json
from datetime import datetime, date
import pandas as pd

class ChainJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)