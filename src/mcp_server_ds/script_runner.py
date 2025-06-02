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
    persistent_dataframes: List[str] = None


### Python (Pandas, NumPy, SciPy) Script Runner
class ScriptRunner:
    def __init__(self, data_dir: Optional[str] = None, clean_hours: Optional[float] = None):
        self.notes: list[str] = []
        self.persistent_dataframes: Dict[str, pd.DataFrame] = {}
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(os.getcwd(), "data")  # Default data directory within project
        os.makedirs(self.data_dir, exist_ok=True)  # Create data directory if it doesn't exist
        self.clean_hours = clean_hours

        # Clean old files if requested, only once at initialization
        if self.clean_hours is not None:
            deleted = self.clean_old_files(hours=self.clean_hours)
            if deleted:
                self.notes.append(f"Cleaned up files older than {self.clean_hours} hours: {deleted}")

    def clean_old_files(self, hours: float = 24):
        """
        Delete files in the data directory whose creation time is older than the specified number of hours.
        :param hours: Number of hours; files older than this will be deleted.
        """
        now = time.time()
        threshold = now - hours * 3600  # 3600 seconds in an hour
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

    def safe_eval(self, script: str, persistent_dataframes: List[str] = None):
        """safely run a script, return the result if valid, otherwise return the error message
        """
        # first extract dataframes from the self.data
        local_dict = {}

        # Add persistent dataframes to the local_dict for the script to use
        local_dict.update(self.persistent_dataframes)
        self.notes.append(f"Data directory being used: {self.data_dir}")

        # Automatically load dataframes from the default data directory
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".csv"):
                    df_name = os.path.splitext(filename)[0]  # Use filename without extension as df name
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        local_dict[df_name] = pd.read_csv(file_path)
                        self.notes.append(f"Automatically loaded dataframe '{df_name}'")
                    except Exception as e:
                        self.notes.append(
                            f"Error automatically loading dataframe '{df_name}'': {str(e)}")
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
        if persistent_dataframes:
            for df_name in persistent_dataframes:
                if df_name:
                    df_to_save = local_dict.get(df_name)
                    if isinstance(df_to_save, pd.DataFrame):
                        # Save to CSV
                        csv_file_path = os.path.join(self.data_dir, f"{df_name}.csv")
                        try:
                            df_to_save.to_csv(csv_file_path, index=False)
                            self.notes.append(f"Successfully saved dataframe '{df_name}'.")
                        except Exception as e:
                            self.notes.append(f"Error saving dataframe '{df_name}': {str(e)}")
                    else:
                        self.notes.append(f"'{df_name}' is not a DataFrame, skipping save to memory.")
                else:
                    self.notes.append(f"Invalid item in persistent_dataframes: {df_name}. DataFrame name is empty.")

        output = std_out_script if std_out_script else "No output"
        self.notes.append(f"Result: {output}")
        
        # Include notes in the final output for debugging
        # full_output = "\n".join(self.notes)

        return [
            TextContent(type="text", text=f"print out result: {output}")
        ]
