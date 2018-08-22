import torch
from pykp.dataloader import BucketIterator

word2id,id2word,vocab = torch.load('./data/AAAI/kp20k.vocab.pt')

dataloader = BucketIterator('./data/AAAI/kp20k.test.one2many.json',word2id,id2word,mode='test',batch_size=10,repeat=False,sort=False,shuffle=False)


for idx,batch in enumerate(dataloader):
	print(idx)


