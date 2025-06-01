from mcp_server_ds.server import ScriptRunner


if __name__ == '__main__':
    script_runner = ScriptRunner()
    script_runner.load_csv("/Users/leqiuhong/PycharmProjects/langchain-mcp-adapters/tests/mock/sale_table.csv")
    result = script_runner.safe_eval("# 提取京东的销量数据\njd_sales = df_1[df_1['二级品类'] == '京东']['销量'].sum()\nprint(jd_sales)")  # 捕获 safe_eval 的返回值
