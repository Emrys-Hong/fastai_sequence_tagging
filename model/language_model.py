from fastai.text import *
import fire
"""
fine tune the language model on coNLL 2003 dataset
get an accuracy around 35%
"""

''' build up the library'''
em_sz,nh,nl = 400,1150,3
PRE_PATH = PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'
wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)
itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
new_w = np.zeros((vs, em_sz), dtype=np.float32)
count=0
for i,w in enumerate(id2token):
    r = stoi2[w]
    if r >= 0:
        new_w[i] = enc_wgts[r]
    else:
        new_w[i] = row_m
        count += 1
print(f'there are {count} new words')

wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))wd=1e-5

'''creates the language model'''
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

class LanguageModelLoader():
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """
    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs,self.bptt,self.backwards = bs,bptt,backwards
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards: data=data[::-1]
        return T(data).long()

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)

trn_dl = LanguageModelLoader(np.concatenate(trn_sent), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_sent), bs, bptt)
tst_dl = LanguageModelLoader(np.concatenate(test_sent), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, test_dl=test_sent, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]

learner.model.load_state_dict(wgts)
lr=2e-3
lrs = lr

print('freeze to layer 1')
learner.freeze_to(-1)

learner.save('lm_last_ft')
learner.load('lm_last_ft')

print('unfreeze all the layers')
learner.unfreeze()

learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=4)
learner.save('lm1')
learner.save_encoder('lm1_enc')

