from mcp_server_ds.script_runner import ScriptRunner


if __name__ == '__main__':
    script_runner = ScriptRunner()
    result = script_runner.safe_eval(
        script="print(df_1.head())"
    )  # 捕获 safe_eval 的返回值
    for item in result:
        if hasattr(item, 'text'):
            print(item.text)  # 打印 safe_eval 的返回值
        else:
            print(item)