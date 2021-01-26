from pathlib import Path
from joblib import Parallel, delayed
from utils import transform_pcap

PCAP_FILE_PATH = '/home/rootcs412/pcaps'
# PCAP_FILE_PATH = '/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/pcap'
PROCESSED_FILE_PATH = '/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/pcap'
NUMBER_OF_JOB = -1


# @click.command()
# @click.option('-s', '--source', help='path to the directory containing raw pcap files', required=False)
# @click.option('-t', '--target', help='path to the directory for persisting preprocessed files', required=False)
# @click.option('-n', '--njob', default=-1, help='num of executors', type=int)
def main(source=PCAP_FILE_PATH, target=PROCESSED_FILE_PATH, njob=NUMBER_OF_JOB):
    data_dir_path = Path(source)
    target_dir_path = Path(target)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=njob)(
        delayed(transform_pcap)(pcap_path, target_dir_path / (pcap_path.name + '.transformed')) for pcap_path in
        sorted(data_dir_path.iterdir()))


if __name__ == '__main__':
    main()
