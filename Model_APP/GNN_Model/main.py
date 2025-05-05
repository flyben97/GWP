import subprocess

# 列出所有要运行的Python脚本路径
scripts = [
    '/home/flybenben/machine_learning_space/S04/GNN_Model/GCN.py',
    '/home/flybenben/machine_learning_space/S04/GNN_Model/MPNN.py',
    '/home/flybenben/machine_learning_space/S04/GNN_Model/GATv2.py',
    '/home/flybenben/machine_learning_space/S04/GNN_Model/AttentiveFP.py',
    '/home/flybenben/machine_learning_space/S04/GNN_Model/NF.py'
]

# 依次运行每个脚本
for script in scripts:
    try:
        # 使用subprocess.Popen来运行脚本，并实时读取输出
        process = subprocess.Popen(['python3', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 打印脚本名称
        print(f'Running {script}')

        # 实时输出stdout和stderr
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
            if error:
                print(error.strip())
        
        process.wait()
        
    except Exception as e:
        print(f'Failed to run {script}: {e}')
