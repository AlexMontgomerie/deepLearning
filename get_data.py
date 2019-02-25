from common import *

def get_data():
    hpatches_dir = './hpatches'
    splits_path = './splits.json'

    splits_json = json.load(open(splits_path, 'r'))
    split = splits_json['a']

    train_fnames = split['train']
    test_fnames = split['test']

    seqs = glob.glob(hpatches_dir+'/*')
    seqs = [os.path.abspath(p) for p in seqs]
    seqs_train = list(filter(lambda x: x.split('/')[-1] in train_fnames, seqs))
    seqs_test = list(filter(lambda x: x.split('/')[-1] in split['test'], seqs))
    return [ seqs_train, seqs_test ]
