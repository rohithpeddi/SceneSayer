import os
import sys
import pickle
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import pdb

from pytorch_lightning.callbacks import ModelCheckpoint

torch.cuda.empty_cache()
Tensor = TypeVar("torch.tensor")
seed_everything(42, workers=True)


class GraphDataset(Dataset):
	def __init__(
			self,
			data_root: str,
			max_num_graphs: Optional[int],
			split: str = "train",
	):
		super().__init__()
		self.data_root = data_root
		self.max_num_graphs = max_num_graphs
		if split != 'val':
			self.data_path = os.path.join(data_root, split)
		else:
			self.data_path = os.path.join(data_root, 'test')
		self.split = split
		assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
		assert self.split == "train" or self.split == "val" or self.split == "test"
		assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
		self.files = self.get_files()
	
	def __getitem__(self, index: int):
		graph_path = self.files[index]
		temp = pickle.load(open(graph_path, 'rb'))
		graph = temp['dense_graph']  # 12x35x1936
		final_graph = temp['pred_graph']
		# final_graph = (final_graph >= 0.5).astype(np.float64)
		actual_graph = temp['graph']
		# print(type(final_graph))
		#
		return torch.from_numpy(graph), torch.from_numpy(final_graph), torch.from_numpy(actual_graph)
	
	def __len__(self):
		return len(self.files)
	
	def get_files(self) -> List[str]:
		paths: List[Optional[str]] = []
		graphs = [int(graph.split('.')[0]) for graph in os.listdir(self.data_path)]
		graphs.sort()
		for graph in graphs:
			paths.append(os.path.join(self.data_path, str(graph) + '.pkl'))
		return list(filter(None, paths))


class GraphDataModule(pl.LightningDataModule):
	def __init__(
			self,
			data_root: str,
			train_batch_size: int,
			val_batch_size: int,
			num_workers: int,
			num_train_graphs: Optional[int] = None,
			num_val_graphs: Optional[int] = None,
	
	):
		super().__init__()
		self.data_root = data_root
		self.train_batch_size = train_batch_size
		self.val_batch_size = val_batch_size
		self.num_workers = num_workers
		self.num_train_graphs = num_train_graphs
		self.num_val_graphs = num_val_graphs
		
		self.train_dataset = GraphDataset(
			data_root=self.data_root,
			max_num_graphs=self.num_train_graphs,
			split="train",
		)
		self.val_dataset = GraphDataset(
			data_root=self.data_root,
			max_num_graphs=self.num_val_graphs,
			split="val",
		)
	
	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.train_batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			pin_memory=True,
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.val_batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=True,
		)


class GraphModel(nn.Module):
	def __init__(
			self,
			past_len=6,
			future_len=6,
			d_model=128,
			num_heads=8,
			num_layers=4,
			ffn_dim=512,
			num_features=67760,
			norm_first=True,
			empty_cache=False,
	):
		super().__init__()
		self.past_len = past_len
		self.future_len = future_len
		self.d_model = d_model
		self.num_features = num_features
		self.num_relations = 1936
		self.num_objects = 35
		self.in_proj = nn.Linear(num_features, d_model).double()
		self.out_proj = nn.Linear(d_model, num_features).double()
		self.y1 = nn.Linear(26, 26).double()
		self.y0 = nn.Linear(26, 26).double()
		self.a_rel_compress = nn.Linear(1936, 3).double()
		self.s_rel_compress = nn.Linear(1936, 6).double()
		self.c_rel_compress = nn.Linear(1936, 17).double()
		a_weight = torch.from_numpy(pickle.load(open('../a_weight.pkl', 'rb'))).double()
		a_bias = torch.from_numpy(pickle.load(open('../a_bias.pkl', 'rb'))).double()
		s_weight = torch.from_numpy(pickle.load(open('../s_weight.pkl', 'rb'))).double()
		s_bias = torch.from_numpy(pickle.load(open('../s_bias.pkl', 'rb'))).double()
		c_weight = torch.from_numpy(pickle.load(open('../c_weight.pkl', 'rb'))).double()
		c_bias = torch.from_numpy(pickle.load(open('../c_bias.pkl', 'rb'))).double()
		for name, params in self.a_rel_compress.named_parameters():
			if name == 'weight':
				params.data = a_weight
				# params.requires_grad = False
			if name == 'bias':
				params.data = a_bias
				# params.requires_grad = False
		for name, params in self.s_rel_compress.named_parameters():
			if name == 'weight':
				params.data = s_weight
				# params.requires_grad = False
			if name == 'bias':
				params.data = s_bias
				# params.requires_grad = False
		for name, params in self.c_rel_compress.named_parameters():
			if name == 'weight':
				params.data = c_weight
				# params.requires_grad = False
			if name == 'bias':
				params.data = c_bias
				# params.requires_grad = False
		
		self.empty_cache = empty_cache
		
		enc_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=num_heads,
			dim_feedforward=ffn_dim,
			norm_first=norm_first,
			batch_first=True,
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=num_layers).double()
	
	def forward(self, x):
		gt_final_graphs = x[1]  # predicted
		gt_actual_graphs = x[2]  # gt_graph symbolic
		x = x[0]  # dense_representation
		if torch.cuda.is_available():
			x = x.cuda()
		batch_size = x.shape[0]
		past_graphs = torch.flatten(x[:, :self.past_len], start_dim=2,
		                            end_dim=3)  # [Batch size, 6, graph height*graph width]
		gt_graphs = torch.flatten(x[:, self.past_len:], start_dim=2,
		                          end_dim=3)  # [Batch size, 6, graph height*graph width]
		# gt_final_graphs = torch.flatten(gt_final_graphs[:, self.past_len:], start_dim=2, end_dim=3)
		gt_past_graphs = torch.flatten(gt_actual_graphs[:, :self.past_len], start_dim=2, end_dim=3)  # past symbolic
		gt_actual_graphs = torch.flatten(gt_actual_graphs[:, self.past_len:], start_dim=2, end_dim=3)  # future symbolic
		inv_freq = 1. / (10000 ** (torch.arange(0.0, self.d_model, 2.0) / self.d_model))
		pos_seq = torch.arange(self.past_len - 1, -1, -1).type_as(inv_freq)
		sinusoid_inp = torch.outer(pos_seq, inv_freq)
		pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1).unsqueeze(0)
		pos_emb = nn.Parameter(pos_emb, requires_grad=False).type_as(x)
		enc_pe = pos_emb.repeat(batch_size, 1, 1)
		in_x = past_graphs
		pred_out = []
		
		for _ in range(self.future_len):
			temp = self.in_proj(in_x)
			temp = temp + enc_pe
			temp = self.transformer_encoder(temp)
			pred_graph = self.out_proj(temp[:, -1:])
			pred_out.append(pred_graph)
			in_x = torch.cat([in_x[:, 1:], pred_out[-1]], dim=1)
		future_graphs = torch.stack(pred_out, dim=1).flatten(start_dim=2, end_dim=3)
		future_graphs = future_graphs.reshape((x.shape[0], x.shape[1] // 2, x.shape[2], x.shape[3]))
		
		future_a = self.a_rel_compress(future_graphs)
		future_s = self.s_rel_compress(future_graphs)
		future_c = self.c_rel_compress(future_graphs)
		
		future_final_graphs = torch.cat((future_a, future_s, future_c), 3)
		future_final_graphs = self.y0(-future_final_graphs) + self.y1(future_final_graphs)  # linear Layer (Bx6x35x26)
		
		boundary = 0.0000000001
		future_final_graphs = (((1 - 2 * boundary) * torch.sigmoid(future_final_graphs)) + boundary)
		
		future_graphs = torch.flatten(future_graphs, start_dim=2, end_dim=3)
		future_final_graphs = torch.flatten(future_final_graphs, start_dim=2, end_dim=3)
		
		return future_graphs, gt_graphs, future_final_graphs, gt_actual_graphs
	
	def loss_function(self, input):
		future_graphs, gt_graphs, future_final_graphs, gt_actual_graphs = self.forward(input)
		temp1_loss = F.mse_loss(future_graphs, gt_graphs)
		temp2_loss = torch.nn.BCELoss()(future_final_graphs, gt_actual_graphs)
		loss1 = temp1_loss
		loss2 = temp2_loss
		loss_other = self.num_features * (temp1_loss ** 0.5)
		loss2_other = 35 * 26 * temp2_loss
		
		futures_ = (future_final_graphs >= 0.5).double()
		tp = torch.sum((gt_actual_graphs * futures_) == 1)
		fp = torch.sum((1 - gt_actual_graphs) * futures_ == 1)
		fn = torch.sum(gt_actual_graphs * (1 - futures_) == 1)
		tn = torch.sum((1 - gt_actual_graphs) * (1 - futures_) == 1)
		pdb.set_trace()
		
		# For no constraint
		aca = torch.sort(future_final_graphs, dim=2, descending=True)[0]
		
		future10 = aca[:, :, 9:10]
		future20 = aca[:, :, 19:20]
		future50 = aca[:, :, 49:50]
		
		future10_ = (future_final_graphs >= future10).double()
		future20 = (future_final_graphs >= future20).double()
		future50 = (future_final_graphs >= future50).double()
		
		tp10_ = torch.sum((gt_actual_graphs * future10) == 1)
		tp20_ = torch.sum((gt_actual_graphs * future20) == 1)
		tp50_ = torch.sum((gt_actual_graphs * future50) == 1)
		tpfn_ = torch.sum(gt_actual_graphs)
		
		# For with constraint below
		future_final_graphs = future_final_graphs.reshape(future_final_graphs.shape[0], future_final_graphs.shape[1],
		                                                  35, 26)
		
		temp1 = (future_final_graphs[:, :, :, :3] == torch.max(future_final_graphs[:, :, :, :3], dim=3, keepdim=True)[
			0])
		temp2 = (future_final_graphs[:, :, :, 3:9] == torch.max(future_final_graphs[:, :, :, 3:9], dim=3, keepdim=True)[
			0])
		temp3 = (future_final_graphs[:, :, :, 9:] == torch.max(future_final_graphs[:, :, :, 9:], dim=3, keepdim=True)[
			0])
		aca1 = temp1 * future_final_graphs[:, :, :, :3]
		aca2 = temp2 * future_final_graphs[:, :, :, 3:9]
		aca3 = temp3 * future_final_graphs[:, :, :, 9:]
		future_final_graphs = torch.flatten(future_final_graphs, start_dim=2, end_dim=3)
		temp = torch.flatten(torch.cat((aca1, aca2, aca3), 3), start_dim=2, end_dim=3)
		aca = torch.sort(temp, dim=2, descending=True)[0]
		future10 = aca[:, :, 9:10]
		future20 = aca[:, :, 19:20]
		future50 = aca[:, :, 49:50]
		future10 = (temp >= future10).double()
		future20 = (temp >= future20).double()
		future50 = (temp >= future50).double()
		tp10_with = torch.sum((gt_actual_graphs * future10) == 1)
		tp20_with = torch.sum((gt_actual_graphs * future20) == 1)
		tp50_with = torch.sum((gt_actual_graphs * future50) == 1)
		return {
			"loss": loss1 + loss2,
			"loss1": loss1,
			"loss2": loss2,
			"loss_other": loss_other,
			"loss2_other": loss2_other,
			'tp': tp,
			'fp': fp,
			'fn': fn,
			'tn': tn,
			'tp10_': tp10_,  # true positive in top 10 prediction
			'tp20_': tp20_,
			'tp50_': tp50_,
			'tpfn_': tpfn_,  # true positive with false negative
			'tp10_with': tp10_with,  # with : with constraints
			'tp20_with': tp20_with,
			'tp50_with': tp50_with
		}


lr = 0.00001
weight_decay = 0.0
warmup_steps_pct = 0.02
decay_steps_pct = 0.2
max_epochs = 200
scheduler_gamma = 0.4
num_sanity_val_steps = 1
gpus = 1


class GraphMethod(pl.LightningModule):
	def __init__(self, model: GraphModel, datamodule: pl.LightningDataModule):
		super().__init__()
		self.model = model
		self.datamodule = datamodule
		self.validation_step_outputs = []
	
	def forward(self, input: Tensor, **kwargs) -> Tensor:
		return self.model(input, **kwargs)
	
	def training_step(self, batch, batch_idx, optimizer_idx=0):
		train_loss = self.model.loss_function(batch)
		logs = {key: val.item() for key, val in train_loss.items()}
		self.log_dict(logs, sync_dist=True)
		return train_loss
	
	def validation_step(self, batch, batch_idx, optimizer_idx=0):
		val_loss = self.model.loss_function(batch)
		return val_loss
	
	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
		loss1 = torch.stack([x["loss1"] for x in outputs]).mean()
		loss2 = torch.stack([x["loss2"] for x in outputs]).mean()
		loss_other = torch.stack([x["loss_other"] for x in outputs]).mean()
		loss2_other = torch.stack([x["loss2_other"] for x in outputs]).mean()
		tp = torch.stack([x["tp"] for x in outputs]).sum()
		fp = torch.stack([x["fp"] for x in outputs]).sum()
		fn = torch.stack([x["fn"] for x in outputs]).sum()
		tn = torch.stack([x["tn"] for x in outputs]).sum()
		tp10_ = torch.stack([x["tp10_"] for x in outputs]).sum()
		tp20_ = torch.stack([x["tp20_"] for x in outputs]).sum()
		tp50_ = torch.stack([x["tp50_"] for x in outputs]).sum()
		tpfn_ = torch.stack([x["tpfn_"] for x in outputs]).sum()
		tp10_with = torch.stack([x["tp10_with"] for x in outputs]).sum()
		tp20_with = torch.stack([x["tp20_with"] for x in outputs]).sum()
		tp50_with = torch.stack([x["tp50_with"] for x in outputs]).sum()
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1 = 2 * precision * recall / (precision + recall)
		accuracy = (tp + tn) / (tn + fn + tp + fp)
		
		logs = {
			"avg_val_loss": avg_loss,
			"loss1": loss1,
			"loss2": loss2,
			"loss_other": loss_other,
			"loss2_other": loss2_other,
			'tp': tp,
			'fp': fp,
			'fn': fn,
			'tn': tn,
			'precision': precision,
			'recall': recall,
			'f1': f1,
			'accuracy': accuracy,
			'tp10_': tp10_,
			'tp20_': tp20_,
			'tp50_': tp50_,
			'tpfn_': tpfn_,
			'tp10_with': tp10_with,
			'tp20_with': tp20_with,
			'tp50_with': tp50_with,
			'recall@10': tp10_ / tpfn_,
			'recall@20': tp20_ / tpfn_,
			'recall@50': tp50_ / tpfn_,
			'recall@10_with': tp10_with / tpfn_,
			'recall@20_with': tp20_with / tpfn_,
			'recall@50_with': tp50_with / tpfn_,
		}
		self.log_dict(logs, sync_dist=True)
		self.validation_step_outputs.clear()
		print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
	
	def configure_optimizers(self):
		optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
		
		total_steps = max_epochs * len(self.datamodule.train_dataloader()) + len(self.datamodule.val_dataloader())
		ff = len(self.datamodule.train_dataloader())
		
		def warm_and_decay_lr_scheduler(step: int):
			warmup_steps = warmup_steps_pct * total_steps
			decay_steps = decay_steps_pct * total_steps
			assert step < total_steps
			if step < warmup_steps:
				factor = step / warmup_steps
			else:
				factor = 1
			
			factor *= scheduler_gamma ** (step / decay_steps)
			factor = scheduler_gamma ** (step / ff)
			factor = 1
			# print(step, ff, factor, lr*factor)
			return factor
		
		scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)
		# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
		return (
			[optimizer],
			[{"scheduler": scheduler, "interval": "step", }],
		)


torch.cuda.empty_cache()
graph_datamodule = GraphDataModule(
	data_root='../data3',
	train_batch_size=64,
	val_batch_size=64,
	num_workers=2,
)
# model = GraphModel(d_model=912, ffn_dim=3648, num_layers=7)
model = GraphModel(d_model=int(sys.argv[1]), ffn_dim=int(sys.argv[2]), num_layers=int(sys.argv[3]))
if torch.cuda.is_available():
	model = model.cuda()
method = GraphMethod(model=model, datamodule=graph_datamodule)

checkpoint_callback = ModelCheckpoint(
	save_top_k=40,
	monitor="loss",
	save_last=True,
	mode="min",
	dirpath=sys.argv[4],
	filename="ckpt-{epoch:03d}-{loss:.3f}",
)

checkpoint_callback_2 = ModelCheckpoint(
	save_top_k=40,
	monitor="epoch",
	every_n_epochs=5,
	mode="max",
	dirpath=sys.argv[4] + '_epochs',
	filename="ckpt-{epoch:03d}",
)

trainer = Trainer(
	logger=False,
	accelerator='cuda',
	# auto_select_gpus=True,
	num_sanity_val_steps=num_sanity_val_steps,
	devices=gpus,
	max_epochs=max_epochs,
	log_every_n_steps=50,
	callbacks=[checkpoint_callback, checkpoint_callback_2],
	default_root_dir=sys.argv[4],
	deterministic=True,
	gradient_clip_val=0.5,
	# detect_anomaly=True,
)
trainer.fit(method, graph_datamodule)
