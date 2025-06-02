from mcp_server_ds.script_runner import ScriptRunner

if __name__ == '__main__':
    script_runner = ScriptRunner(data_dir='/Users/leqiuhong/IdeaProjects/bi-ai/bi-ai-insight-server/src/main/resources/temp')
    result = script_runner.safe_eval(
        script="df_sale3 = df_sale",
        persistent_dataframes=["df_sale3"]
    )  # 捕获 safe_eval 的返回值
    for item in result:
        if hasattr(item, 'text'):
            print(item.text)  # 打印 safe_eval 的返回值
        else:
            print(item)
