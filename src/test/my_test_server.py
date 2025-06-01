from mcp_server_ds.server import ScriptRunner
import os
import tempfile


def test_load_csv_and_safe_eval():

    script = f"""
print(aaa)
"""
    runner = ScriptRunner(data_dir='/Users/leqiuhong/PycharmProjects/mcp-server-data-exploration/src/test/data',
                          clean_days=100000)

    result = runner.safe_eval(script)
    print(result)




if __name__ == '__main__':
    test_load_csv_and_safe_eval()
