import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parent
source = root.parent / 'w1-gpu-smoke-consolidated-acceptance-v3/package'
package = root / 'package'
assert not package.exists(), 'LOCAL_PACKAGE_ALREADY_EXISTS'
shutil.copytree(source, package)
for name in ('local_runtime.py', 'local_gpu_worker.py', 'local_launcher.py'):
    shutil.copy2(root / name, package / name)
base = json.loads((source / 'SERVER_ACCEPTANCE_REQUEST.json').read_text(encoding='utf-8'))
request = {**base, 'schema': 'w1-local-synthetic-acceptance-request/1.0.0',
    'acceptance_id': 'w1-gpu-smoke-local-v1', 'engineering_job_id': 'w1-gpu-smoke-local-v1-once',
    'remote_execution_root': '/home/asus/w1-gpu-smoke-local-v1/package',
    'evidence_directory': '/home/asus/w1-gpu-smoke-local-v1/gpu-evidence',
    'purpose': 'User-authorized local WSL GPU_SMOKE_TEST; synthetic fixtures only',
    'gpu': {'physical_device': 0, 'logical_device': 'cuda:0',
            'exclusive_allocation_required': False, 'desktop_gpu_shared': True,
            'peak_allocated_memory_bytes': 4 * 1024**3,
            'free_memory_minimum_bytes': int(4.5 * 1024**3)},
    'runtime': {'python_executable': '/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python',
                'python': '3.11.16', 'torch': '2.7.0+cu128', 'cuda': '12.8'},
    'registration_order': [], 'implementation_sha256': {}}
(package / 'LOCAL_ACCEPTANCE_REQUEST.json').write_text(json.dumps(request, sort_keys=True, indent=2)+'\n', encoding='utf-8')
(package / 'LOCAL_SMOKE.md').write_text('''# 本机 GPU_SMOKE_TEST

用户授权从不可达服务器迁移到本机，只运行合成工程测试，不运行研究数据、真实环境或 GPPO。旧包及授权不迁移、不消费。
GPU0 RTX3060 Laptop 6GiB，共享桌面环境；显存上限收紧至4GiB、启动需4.5GiB空闲。运行时改为本机WSL既有隔离Python3.11.16、torch2.7.0+cu128。没有声称GPU独占或完整GPU进程内存计量。
研究模型/数据/目标/种子/门槛源文件不变；2/2/8父场景、G1/G2各3seed、1epoch50候选及CPU sequence回归保持不变。wall180/CPU360全局、计算150/300、调用上限和RSS/存储上限保持不变。
入口 local_launcher.py --preflight-only 只读；--system-only 执行真实无模型系统接口测试；不带参数只执行一次GPU训练链，复用production supervise与acceptance_entry._run_bounded_tests。平台适配仅在启动、运行时和GPU计量，不替换采集、训练、checkpoint或预测阶段。
本机不受原服务器GPU1/9GiB检查约束；旧SERVER_ACCEPTANCE_REQUEST是历史参考，不是本次授权合同。本次合同仅LOCAL_ACCEPTANCE_REQUEST，保持NOT_APPROVED，运行授权在包外标准JSON与随机token绑定。
独立计量缺口：WDDM/WSL无可靠自有进程GPU内存和独占证明；PyTorch allocated/reserved峰值实测。CPU为SELF+waited CHILDREN及生产supervisor采样，非内核cgroup硬配额；总CPU上限仍监控并最终拒绝超限。最终结算尾段预留不当作实测值。
''', encoding='utf-8')
sys.path.insert(0, str(package))
from manifest_contract import write_identity_files
identity = write_identity_files(package, attempt=request['engineering_job_id'], entrypoint='local_launcher.py')
differences = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in package.rglob('*') if p.is_file() and (not (source / p.relative_to(package)).is_file()
               or p.read_bytes() != (source / p.relative_to(package)).read_bytes())}
(root / 'initial-local-identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
(root / 'local-source-differences.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
print(json.dumps({'package': str(package), 'identity': identity, 'changed_files': list(differences)}))
