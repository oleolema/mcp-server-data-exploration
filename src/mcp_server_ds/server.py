from enum import Enum
import logging
from typing import Optional, List

## import mcp server
from mcp.server.models import InitializationOptions
from mcp.types import (
    TextContent,
    Tool,
    Resource,
    INTERNAL_ERROR,
    Prompt,
    PromptArgument,
    EmbeddedResource,
    GetPromptResult,
    PromptMessage, ErrorData,
)
from mcp.server import NotificationOptions, Server
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl
import mcp.server.stdio
from pydantic import BaseModel

## import common data analysis libraries
import pandas as pd
import numpy as np
import scipy
import sklearn
import statsmodels.api as sm
from io import StringIO
import sys

logger = logging.getLogger(__name__)
logger.info("Starting mini data science exploration server")


### Prompt templates
class DataExplorationPrompts(str, Enum):
    EXPLORE_DATA = "explore-data"


class PromptArgs(str, Enum):
    CSV_PATH = "csv_path"
    TOPIC = "topic"

PROMPT_TEMPLATE = """
你是一名专业的数据科学家，负责对一个数据集进行探索性数据分析。你的目标是提供有见地的分析，提出问题，并按步骤解决问题，同时确保稳定性和结果大小的可管理性。

首先，从以下路径加载CSV文件：

<csv_path>
{csv_path}
</csv_path>

你的分析应集中在以下主题：

<analysis_topic>
{topic}
</analysis_topic>

你可以使用以下工具进行分析：
1. load_csv：用于加载CSV文件。
2. run_script：用于在MCP服务器上执行Python脚本。

请仔细按照以下步骤进行：

1. 使用load_csv工具加载CSV文件。

2. 探索数据集。提供其结构的简要总结，包括行数、列数和数据类型。包括：
   - 数据集的关键统计信息列表
   - 你在分析该数据时预见的潜在挑战

3. 你需要有一个思考过程：
   分析数据集的大小和复杂性：
   - 它有多少行和列？
   - 基于数据类型或数据量是否存在潜在的计算挑战？
   - 鉴于数据集的特点和分析主题，哪些问题是合适的？
   - 我们如何确保我们的问题不会导致过大的输出？

   基于此分析：
   - 列出与分析主题相关的10个潜在问题
   - 根据以下标准评估每个问题：
     * 直接与分析主题相关
     * 可以以合理的计算努力回答
     * 将产生可管理的结果大小
     * 提供对数据的有意义的见解
   - 选择最符合所有标准的前5个问题

4. 列出你选择的5个问题，确保它们符合上述标准。

5. 对每个问题，按照以下步骤进行：
   a. 你需要有一个思考过程：
      - 如何结构化Python脚本以有效回答这个问题？
      - 需要进行哪些数据预处理步骤？
      - 如何限制输出大小以确保稳定性？
      - 如何结合数据来说明结果？
      - 概述脚本将遵循的主要步骤

   b. 编写Python脚本回答问题。包括解释你的方法和任何限制输出大小的措施的注释。

   c. 使用run_script工具在MCP服务器上执行你的Python脚本。

   d. 详细描述run_script工具返回的结果，重点关注观察到的关键见解和模式。提供清晰简洁的文字总结，而不是使用图形表示。

6. 完成所有5个问题的分析后，提供你的发现和从数据中获得的任何总体见解的简要总结。

记住在分析中优先考虑稳定性和可管理性。如果在任何时候遇到潜在的结果集过大问题，请相应地调整你的方法。

请通过加载CSV文件并提供数据集的初步探索来开始你的分析。
"""
### Data Exploration Tools Description & Schema
class DataExplorationTools(str, Enum):
    LOAD_CSV = "load_csv"
    RUN_SCRIPT = "run_script"


LOAD_CSV_TOOL_DESCRIPTION = """
Load CSV File Tool

Purpose:
Load a local CSV file into a DataFrame.

Usage Notes:
	•	If a df_name is not provided, the tool will automatically assign names sequentially as df_1, df_2, and so on.
"""


class LoadCsv(BaseModel):
    csv_path: str
    df_name: Optional[str] = None


RUN_SCRIPT_TOOL_DESCRIPTION = """
Python Script Execution Tool

Purpose:
Execute Python scripts for specific data analytics tasks.

Allowed Actions
	1.	Print Results: Output will be displayed as the script’s stdout.
	2.	[Optional] Save DataFrames: Store DataFrames in memory for future use by specifying a save_to_memory name.

Prohibited Actions
	1.	Overwriting Original DataFrames: Do not modify existing DataFrames to preserve their integrity for future tasks.
	2.	Creating Charts: Chart generation is not permitted.
"""


class RunScript(BaseModel):
    script: str
    save_to_memory: Optional[List[str]] = None


### Python (Pandas, NumPy, SciPy) Script Runner
class ScriptRunner:
    def __init__(self):
        self.data = {}
        self.df_count = 0
        self.notes: list[str] = []

    def load_csv(self, csv_path: str, df_name: str = None):
        self.df_count += 1
        if not df_name:
            df_name = f"df_{self.df_count}"
        try:
            self.data[df_name] = pd.read_csv(csv_path)
            self.notes.append(f"Successfully loaded CSV into dataframe '{df_name}'")
            return [
                TextContent(type="text", text=f"Successfully loaded CSV into dataframe，变量名：{df_name}")
            ]
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Error loading CSV: {str(e)}")) from e

    def safe_eval(self, script: str, save_to_memory: Optional[List[str]] = None):
        """safely run a script, return the result if valid, otherwise return the error message"""
        # first extract dataframes from the self.data
        local_dict = {
            **{df_name: df for df_name, df in self.data.items()},
        }
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
            raise McpError(ErrorData(code=INTERNAL_ERROR,message= f"Error running script: {str(e)}")) from e

        # check if the result is a dataframe
        if save_to_memory:
            for df_name in save_to_memory:
                self.notes.append(f"Saving dataframe '{df_name}' to memory")
                self.data[df_name] = local_dict.get(df_name)

        output = std_out_script if std_out_script else "No output"
        self.notes.append(f"Result: {output}")
        return [
            TextContent(type="text", text=f"print out result: {output}")
        ]


### MCP Server Definition
async def main():
    script_runner = ScriptRunner()
    server = Server("local-mini-ds")

    @server.list_resources()
    async def handle_list_resources() -> list[Resource]:
        logger.debug("Handling list_resources request")
        return [
            Resource(
                uri="data-exploration://notes",
                name="Data Exploration Notes",
                description="Notes generated by the data exploration server",
                mimeType="text/plain",
            )
        ]

    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> str:
        logger.debug(f"Handling read_resource request for URI: {uri}")
        if uri == "data-exploration://notes":
            return "\n".join(script_runner.notes)
        else:
            raise ValueError(f"Unknown resource: {uri}")

    @server.list_prompts()
    async def handle_list_prompts() -> list[Prompt]:
        logger.debug("Handling list_prompts request")
        return [
            Prompt(
                name=DataExplorationPrompts.EXPLORE_DATA,
                description="A prompt to explore a csv dataset as a data scientist",
                arguments=[
                    PromptArgument(
                        name=PromptArgs.CSV_PATH,
                        description="The path to the csv file",
                        required=True,
                    ),
                    PromptArgument(
                        name=PromptArgs.TOPIC,
                        description="The topic the data exploration need to focus on",
                        required=False,
                    ),
                ],
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
        logger.debug(f"Handling get_prompt request for {name} with args {arguments}")
        if name != DataExplorationPrompts.EXPLORE_DATA:
            logger.error(f"Unknown prompt: {name}")
            raise ValueError(f"Unknown prompt: {name}")

        if not arguments or PromptArgs.CSV_PATH not in arguments:
            logger.error("Missing required argument: csv_path")
            raise ValueError("Missing required argument: csv_path")

        csv_path = arguments[PromptArgs.CSV_PATH]
        topic = arguments.get(PromptArgs.TOPIC)
        prompt = PROMPT_TEMPLATE.format(csv_path=csv_path, topic=topic)

        logger.debug(f"Generated prompt template for csv_path: {csv_path} and topic: {topic}")
        return GetPromptResult(
            description=f"Data exploration template for {topic}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt.strip()),
                )
            ],
        )

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        logger.debug("Handling list_tools request")
        return [
            Tool(
                name=DataExplorationTools.LOAD_CSV,
                description=LOAD_CSV_TOOL_DESCRIPTION,
                inputSchema=LoadCsv.model_json_schema(),
            ),
            Tool(
                name=DataExplorationTools.RUN_SCRIPT,
                description=RUN_SCRIPT_TOOL_DESCRIPTION,
                inputSchema=RunScript.model_json_schema(),
            )
        ]

    @server.call_tool()
    async def handle_call_tool(
            name: str, arguments: dict | None
    ) -> list[TextContent | EmbeddedResource]:
        logger.debug(f"Handling call_tool request for {name} with args {arguments}")
        if name == DataExplorationTools.LOAD_CSV:
            csv_path = arguments.get("csv_path")
            df_name = arguments.get("df_name")
            return script_runner.load_csv(csv_path, df_name)
        elif name == DataExplorationTools.RUN_SCRIPT:
            script = arguments.get("script")
            df_name = arguments.get("df_name")
            return script_runner.safe_eval(script, df_name)
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unknown tool: {name}"))
        return None

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.debug("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="data-exploration-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
