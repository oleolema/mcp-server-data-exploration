import sys
from io import StringIO
from typing import Optional, List, Dict
import os
import time

import numpy as np
## import common data analysis libraries
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
from mcp.shared.exceptions import McpError
## import mcp server
from mcp.types import (
    TextContent,
    INTERNAL_ERROR,
    ErrorData,
)
from pydantic import BaseModel


class RunScript(BaseModel):
    script: str
    save_to_disk: Optional[List[Dict[str, str]]] = None


### Python (Pandas, NumPy, SciPy) Script Runner
class ScriptRunner:
    def __init__(self, data_dir: Optional[str] = None, clean_days: Optional[float] = None):
        self.notes: list[str] = []
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(os.getcwd(), "data")  # Default data directory within project
        os.makedirs(self.data_dir, exist_ok=True)  # Create data directory if it doesn't exist
        self.clean_days = clean_days

        # Clean old files if requested, only once at initialization
        if self.clean_days is not None:
            deleted = self.clean_old_files(days=self.clean_days)
            if deleted:
                self.notes.append(f"Cleaned up files older than {self.clean_days} days: {deleted}")

    def clean_old_files(self, days: float = 1.0):
        """
        Delete files in the data directory whose creation time is older than the specified number of days.
        :param days: Number of days; files older than this will be deleted.
        """
        now = time.time()
        threshold = now - days * 86400  # 86400 seconds in a day
        deleted_files = []
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            for filename in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        ctime = os.path.getctime(file_path)
                        if ctime < threshold:
                            os.remove(file_path)
                            deleted_files.append(filename)
                            self.notes.append(f"Deleted old file '{filename}' (ctime: {ctime})")
                    except Exception as e:
                        self.notes.append(f"Error checking/deleting file '{filename}': {str(e)}")
        return deleted_files

    def safe_eval(self, script: str, save_to_disk: Optional[List[Dict[str, str]]] = None):
        """safely run a script, return the result if valid, otherwise return the error message
        """
        # first extract dataframes from the self.data
        local_dict = {}

        # Automatically load dataframes from the default data directory
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".csv"):
                    df_name = os.path.splitext(filename)[0]  # Use filename without extension as df name
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        local_dict[df_name] = pd.read_csv(file_path)
                        self.notes.append(f"Automatically loaded dataframe '{df_name}' from '{file_path}'")
                    except Exception as e:
                        self.notes.append(
                            f"Error automatically loading dataframe '{df_name}' from '{file_path}': {str(e)}")
        else:
            self.notes.append(
                f"Warning: Data directory '{self.data_dir}' does not exist or is not a directory. No dataframes automatically loaded.")

        # execute the script and return the result and if there is error, return the error message
        try:
            stdout_capture = StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            self.notes.append(f"Running script: \n{script}")
            # pylint: disable=exec-used
            exec(script, \
                 {'pd': pd, 'np': np, 'scipy': scipy, 'sklearn': sklearn, 'statsmodels': sm}, \
                 local_dict)
            std_out_script = stdout_capture.getvalue()
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Error running script: {str(e)}")) from e
        finally:
            sys.stdout = old_stdout  # Restore stdout

        # check if the result is a dataframe
        if save_to_disk:
            for item in save_to_disk:
                df_name = item.get("df_name")
                save_path = item.get("path")
                if df_name and save_path:
                    df_to_save = local_dict.get(df_name)
                    if isinstance(df_to_save, pd.DataFrame):
                        try:
                            df_to_save.to_csv(save_path, index=False)
                            self.notes.append(f"Successfully saved dataframe '{df_name}' to '{save_path}'")
                        except Exception as e:
                            self.notes.append(f"Error saving dataframe '{df_name}' to '{save_path}': {str(e)}")
                    else:
                        self.notes.append(f"'{df_name}' is not a DataFrame, skipping save to disk.")
                else:
                    self.notes.append(f"Invalid item in save_to_disk: {item}. Missing 'df_name' or 'path'.")

        output = std_out_script if std_out_script else "No output"
        self.notes.append(f"Result: {output}")
        return [
            TextContent(type="text", text=f"print out result: {output}")
        ]
