import os
import sys
import subprocess

def main():
    if len(sys.argv) < 4:
        print("Usage: python dist_test.py <config> <checkpoint> <gpus> [additional args...]")
        sys.exit(1)

    config = sys.argv[1]
    checkpoint = sys.argv[2]
    gpus = sys.argv[3]
    additional_args = sys.argv[4:]

    nnodes = os.getenv('NNODES', '1')
    node_rank = os.getenv('NODE_RANK', '0')
    port = os.getenv('PORT', '29500')
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')

    pythonpath = os.path.dirname(__file__) + "/..:" + os.getenv('PYTHONPATH', '')

    command = [
        'python', '-m', 'torch.distributed.launch',
        '--nnodes', nnodes,
        '--node_rank', node_rank,
        '--master_addr', master_addr,
        '--nproc_per_node', gpus,
        '--master_port', port,
        os.path.join(os.path.dirname(__file__), 'test.py'),
        config,
        checkpoint,
        '--launcher', 'pytorch'
    ] + additional_args

    env = os.environ.copy()
    env['PYTHONPATH'] = pythonpath

    subprocess.run(command, env=env)

if __name__ == "__main__":
    main()
