# Train the model
vgg_model_path = '/scratch/shared/slow/koepke/faces/models/vggface_models/checkpoint.pth.tar' 
expr_model_path = '/scratch/shared/slow/koepke/faces/data/affectnet4_mask_dil10/checkpoint.pth.tar'

from UnwrappedFace import UnwrappedFaceWeightedAverage
from loss_functions import L1Loss

import shutil
import os
import numpy as np
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Scale, Compose
import torch.optim as optim
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='UnwrappedFace')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--sampler_lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--threads', type=int, default=1, help='Num Threads')
parser.add_argument('--batchSize', type=int, default=16, help='Batch Size')

parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--num_views', type=int, default=4, help='Num views')
parser.add_argument('--combination_function', default='WeightedAverage')
parser.add_argument('--copy_weights', type=bool, default=False)
parser.add_argument('--model_type', type=str, default='UnwrappedFaceSampler_from1view')
parser.add_argument('--inner_nc', type=int, default=512)
parser.add_argument('--old_model', type=str, default='/scratch/shared/slow/ow/faces/models/python/sampler/UnwrappedFaceSamplers_from1view/_noskip_noinnerlayers16_ImInputTruel1lambda0.001000_0.001000eyemouthx20_1_use_heatmap_expr_heatmapmodel_epoch_56.pth')
parser.add_argument('--model_epoch_path', type=str, default='/scratch/shared/slow/ow/faces/models/python/sampler/%s/', help='Batch Size')
parser.add_argument('--use_expression_content_loss', type=bool, default=False)
parser.add_argument('--use_expression_weight_loss', type=bool, default=False)
parser.add_argument('--use_voxceleb2', type=bool, default=False)
parser.add_argument('--use_mse', type=bool, default=False)
parser.add_argument('--use_content', type=bool, default=False)
parser.add_argument('--use_content_other_face', type=bool, default=False)
parser.add_argument('--use_discriminator', type=bool, default=False)
parser.add_argument('--use_uncertainty', type=bool, default=False)
parser.add_argument('--use_face_mask', type=bool, default=False)
parser.add_argument('--use_animal', type=str, default=None)
parser.add_argument('--seed', type=int, default=1)

opt = parser.parse_args()
stn_args = parser
stn_args.add_argument('--grid_size', type = int, default = 20)
stn_args.add_argument('--span_range', type = int, default = 0.99)
stn_args.add_argument('--model', type=str, default='bounded_stn')
stn_args.add_argument('--use_heatmaps', type=bool, default=True)

stn_args = stn_args.parse_args()
stn_args.span_range_height = stn_args.span_range_width = stn_args.span_range
stn_args.grid_height = stn_args.grid_width = stn_args.grid_size
stn_args.image_height = stn_args.image_width = 256


torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
from VoxCelebData_withmask import VoxCeleb

d = 1

opt.model_epoch_path = opt.model_epoch_path % opt.model_type \
						+ '_noskip_noinnerlayers%d' % opt.inner_nc \
						+ '_ImInput' + str(True) + 'l%d' % (d) \
						+ 'lambda%f_%f' % (opt.lr, opt.sampler_lr)\
						+ 'eyemouthx20_1_reweighted_scaled'

if opt.use_content:
	opt.model_epoch_path += 'use_content'
if opt.use_voxceleb2:
	opt.model_epoch_path +='voxceleb2'

if opt.use_content_other_face:
	opt.model_epoch_path += 'use_content_other_face'

if opt.use_discriminator:
	opt.model_epoch_path += 'use_discriminator0.01'

num_inputs = 3

if opt.use_uncertainty:
	num_outputs = 3
else:
	num_outputs = 2
model = UnwrappedFaceWeightedAverage(output_num_channels=num_outputs, input_num_channels=num_inputs,inner_nc=opt.inner_nc)

if opt.copy_weights:
	checkpoint_file = torch.load(opt.old_model)
	model.load_state_dict(checkpoint_file['state_dict'])
	opt.model_epoch_path = opt.model_epoch_path + 'copyWeights'
	del checkpoint_file


criterion = L1Loss()

if opt.num_views > 2:
	opt.model_epoch_path = opt.model_epoch_path + 'num_views' + str(opt.num_views) + 'combination_function' + opt.combination_function

model.stats = {'photometric_error' : np.zeros((0,1)), 'eyemoutherror' : np.zeros((0,1)), 'contenterror' : np.zeros((0,1)), 'loss' : np.zeros((0,1))}
model.val_stats = {'photometric_error' : np.zeros((0,1)), 'eyemoutherror' : np.zeros((0,1)), 'contenterror' : np.zeros((0,1)), 'loss' : np.zeros((0,1))}

model = model.cuda()

criterion = criterion.cuda()
# optimizer = optim.SGD([{'params' : model.pix2pixUnwrapped.parameters()},
# 		{'params' : model.pix2pixSampler.parameters(), 'lr' : opt.lr}], lr=opt.lr, momentum=0.9)
parameters = [{'params' : model.parameters()}]

optimizer = optim.SGD(parameters, lr=opt.lr, momentum=0.9)

def run_batch(imgs, requires_grad=False, volatile=False, other_images=None):
	for i in range(0, len(imgs)):
		imgs[i] = Variable(imgs[i], requires_grad=requires_grad, volatile=volatile).cuda()

	poses = imgs[-3]
	print('Poses', poses.size())

	if not other_images is None:
		for i in range(0, len(other_images)):
			other_images[i] = Variable(other_images[i], requires_grad=requires_grad, volatile=volatile).cuda()

		return (model(poses, *imgs[0:-3]), model(poses, *other_images[0:-3])), imgs + [poses], other_images[0:-3]

	return model(poses, *imgs[0:-3]), imgs + [poses]


folder = '/scratch/local/ssd/ow/results_face/use_image_content_1photo' + str(opt.combination_function) + '_' + str(opt.num_views) + '_exprheatmaps'

if opt.use_expression_content_loss:
	optsexpr = 'with_expr_content'
else:
	optsexpr = 'no_expr_content'

if not os.path.exists(folder):
	os.makedirs(folder)

def train(epoch, num_views):
	train_set = VoxCeleb(num_views, epoch, 1, additional_face=(opt.use_content_other_face or opt.use_discriminator))

	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        epoch_train_loss = 0
	photometricloss = 0
	contentloss = 0
	contentloss_noweight = 0
	contentloss_other = 0
	contentloss_other_noweight = 0

	if opt.use_expression_content_loss:
		exprloss = 0
		exprloss_noweight = 0
	else:
		eyemouthloss = 0

	l_discriminator = 0
	l_discriminator_identity = 0

	model.train()
	for iteration, batch in enumerate(training_data_loader, 1):
		print(len(batch), len(batch[0]))
		if opt.use_uncertainty:
			(result, confidence), inputs = run_batch(batch[0][0], True)
		elif opt.use_content_other_face or opt.use_discriminator:
			print(len(batch), len(batch[0]))
			(result, other_face), inputs, other_input = run_batch(batch[0][0], True, other_images=batch[2])
		else:
			result, inputs = run_batch(batch[0][0], True)

		if opt.use_content:
			vggmodel.eval()
			gtresult = run_gt_batch_vgg(inputs[opt.num_views-1],
	                                            layernames, vggmodel,
	                                            stored_values)
			vggmodel.train()

			resultvgg = run_batch_vgg(result, layernames, vggmodel,
	                                          stored_values)


		if opt.use_content_other_face:
			vggmodel.eval()
			gtresult_other = run_gt_batch_vgg(other_input[0],
	                                            layernamesidentity, vggmodel,
	                                            stored_identity_values)
			vggmodel.train()

			resultvgg_other = run_batch_vgg(other_face, layernamesidentity, vggmodel,
	                                            stored_identity_values)


		if opt.use_expression_content_loss:
			exprmodel.eval()
			exp_mask = Variable(batch[1], requires_grad = False).cuda()
			gtresult_expr = run_gt_batch_vgg(inputs[opt.num_views-1] * exp_mask,
                                                     layernamesc, exprmodel,
                                                     stored_values_expr)
			exprmodel.train()
			resultvgg_expr = run_batch_vgg(result * exp_mask, layernamesc,
                                                   exprmodel,
                                                   stored_values_expr)
		elif opt.use_expression_weight_loss:
			weights = Variable(batch[1].expand_as(result),
                                       requires_grad=False).cuda()
		else:
			batch[1] = batch[1].unsqueeze(1)

		if iteration % 100 == 0 or iteration == 1:
			#t_inputs = [0] * len(inputs[0:-3])
			for i in range(0, len(inputs)):
				input = inputs[i]
				if input.size(1) == 2:
					torchvision.utils.save_image(input[:,0:1,:,:].data.cpu(), '%s/img%d_%d_dim%d.png' % (folder, iteration, i, 1))
					torchvision.utils.save_image(input[:,1:2,:,:].data.cpu(), '%s/img%d_%d_dim%d.png' % (folder, iteration, i, 2))
				elif input.size(1) > 3:
					torchvision.utils.save_image(input.sum(1, keepdim=True).data.cpu(), '%s/img%d_%d_sum.png' % (folder, iteration, i, ))

				else:
					torchvision.utils.save_image(input.data.cpu(), '%s/img%d_%d.png' % (folder, iteration, i))
				#t_inputs = inputs[i].clone()

			torchvision.utils.save_image(result.data.cpu(), '%s/result_%d.png' % (folder, iteration))
			torchvision.utils.save_image(inputs[opt.num_views-1].data.cpu(), '%s/gt_%d.png' % (folder, iteration))

		lossc = criterion(result, inputs[opt.num_views-1])

		loss = lossc
		photometricloss = photometricloss + lossc.data[0]


		optimizer.zero_grad()
		epoch_train_loss += loss.data[0]
		loss.backward()
		optimizer.step()
		loss_mean = epoch_train_loss / iteration
		del batch

		if opt.use_expression_content_loss:
			print("===> Train Epoch[{}]({}/{}): Loss: {:.4f}; Expression: {:.4f}; \
				Photometric: {:.4f}; Content: {:.4f}; Content Other: {:.4f}; \n D: {:.4f}; D_ID: {:.4f}".format(epoch, iteration,
			      len(training_data_loader), loss_mean,
			      exprloss_noweight / float(iteration),
			      photometricloss / float(iteration),
			      contentloss_noweight / float(iteration),
			      contentloss_other_noweight / float(iteration)),
					l_discriminator / float(iteration),
					l_discriminator_identity / float(iteration))
		else:
			print("===> Train Epoch[{}]({}/{}): Loss: {:.4f}; EyeMouthLoss: {:.4f}; \
				Photometric: {:.4f}; Content: {:.4f}; Content Other: {:.4f}; \n D: {:.4f}; D_ID: {:.4f}".format(epoch, iteration,
			      len(training_data_loader), loss_mean,
			      eyemouthloss / float(iteration),
			      photometricloss / float(iteration),
			      contentloss_noweight / float(iteration),
			      contentloss_other_noweight / float(iteration),
			      l_discriminator / float(iteration),
			      l_discriminator_identity / float(iteration)))

		if iteration == 2000:
			break
	model.stats['photometric_error'] = np.vstack([model.stats['photometric_error'], photometricloss / iteration]);
	if opt.use_expression_content_loss:
		model.stats['expression_error'] = np.vstack([model.stats['expression_error'],
                                    exprloss_noweight / iteration]);
	else:
		model.stats['eyemoutherror'] = np.vstack([model.stats['eyemoutherror'],
		                                             eyemouthloss / iteration]);

	model.stats['contenterror'] = np.vstack([model.stats['contenterror'], contentloss_noweight / iteration]);

	if opt.use_discriminator:
		model.stats['discriminator'] = np.vstack([model.stats['discriminator'], l_discriminator / iteration])
		model.stats['discriminator_identity'] = np.vstack([model.stats['discriminator_identity'], l_discriminator_identity / iteration])

	if opt.use_content_other_face:
		model.stats['contenterror_other'] = np.vstack([model.stats['contenterror_other'], contentloss_other_noweight / iteration])
	model.stats['loss'] = np.vstack([model.stats['loss'], epoch_train_loss / iteration]);

	return l_discriminator / iteration, l_discriminator_identity / iteration

def val(epoch, num_views):
	val_set = VoxCeleb(num_views, 0, 2, additional_face=(opt.use_content_other_face or opt.use_discriminator))

	validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

	model.eval()
	epoch_val_loss = 0
	photometricloss = 0
	contentloss = 0
	contentloss_noweight = 0
	contentloss_other = 0
	contentloss_other_noweight = 0

	if opt.use_expression_content_loss:
		exprloss = 0
		exprloss_noweight = 0
	else:
		eyemouthloss = 0

	for iteration, batch in enumerate(validation_data_loader, 1):

		if opt.use_uncertainty:
			(result, confidence), inputs = run_batch(batch[0][0], False, volatile=True)
		elif opt.use_content_other_face or opt.use_discriminator:
			(result, other_face), inputs, other_input = run_batch(batch[0][0], False, other_images=batch[2], volatile=True)
		else:
			result, inputs = run_batch(batch[0][0], False, volatile=True)

		lossc = criterion(result, inputs[opt.num_views-1])

		loss = lossc
		photometricloss = photometricloss + lossc.data[0]

		epoch_val_loss += loss.data[0]
		loss_mean = epoch_val_loss / iteration

		if opt.use_expression_content_loss:
			print("===> Val Epoch[{}]({}/{}): Loss: {:.4f}; Expression: {:.4f}; Photometric: {:.4f}; Content: {:.4f}; Content Other: {:.4f};".format(epoch, iteration,
			      len(validation_data_loader), loss_mean,
			      exprloss_noweight / float(iteration),
			      photometricloss / float(iteration),
			      contentloss_noweight / float(iteration),
			      contentloss_other_noweight / float(iteration)))
		else:
			print("===> Val Epoch[{}]({}/{}): Loss: {:.4f}; EyeMouthLoss: {:.4f}; Photometric: {:.4f}; Content: {:.4f}; Content Other: {:.4f};".format(epoch, iteration,
			      len(validation_data_loader), loss_mean,
			      eyemouthloss / float(iteration),
			      photometricloss / float(iteration),
			      contentloss_noweight / float(iteration),
			      contentloss_other_noweight / float(iteration)))
		# if iteration == 2000:
		# 	break

	model.val_stats['photometric_error'] = np.vstack([model.val_stats['photometric_error'], photometricloss / iteration])
	if opt.use_expression_content_loss:
		model.val_stats['expression_error'] = np.vstack([model.val_stats['expression_error'], exprloss_noweight / iteration])
	else:
		model.val_stats['eyemoutherror'] = np.vstack([model.val_stats['eyemoutherror'], eyemouthloss / iteration])
	model.val_stats['contenterror'] = np.vstack([model.val_stats['contenterror'], contentloss_noweight / iteration])

	if opt.use_content_other_face:
		model.val_stats['contenterror_other'] = np.vstack([model.val_stats['contenterror_other'], contentloss_other_noweight / iteration])
	model.val_stats['loss'] = np.vstack([model.val_stats['loss'], epoch_val_loss / iteration])

	return epoch_val_loss / iteration

def checkpoint(model, epoch):
	dict = {'epoch' : epoch, 'state_dict' : model.state_dict(), \
			'stats' : model.val_stats, 'train_stats' : model.stats, \
			'optimizer' : optimizer.state_dict()}

	if opt.use_discriminator:
		dict['state_dict_D'] = discriminator.state_dict()
		dict['opitmizer_D'] = opitmizer_D.state_dict()
		dict['state_dict_D_identity'] = discriminator_identity.state_dict()
		dict['opitmizer_D_identity'] = opitmizer_D_identity.state_dict()

	model_out_path = "{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch)

	if not(os.path.exists(opt.model_epoch_path)):
		os.makedirs(opt.model_epoch_path)
	torch.save(dict, model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))

	# Check if new best one:
	if len(model.val_stats['photometric_error']) > 0 and (model.val_stats['photometric_error'].argmin() == ((model.val_stats['photometric_error'].size) - 1)):
		shutil.copyfile(model_out_path, "{}model_epoch_{}.pth".format(opt.model_epoch_path, 'best'))

	# remove all previous ones with a worse validation loss
	for i in range(0, epoch-1):
		#if model.val_stats['loss'][i] >=  model.val_stats['loss'][epoch-1] and \
		if os.path.exists("{}model_epoch_{}.pth".format(opt.model_epoch_path, i)):
			os.remove( "{}model_epoch_{}.pth".format(opt.model_epoch_path, i))



import torch.optim.lr_scheduler as lr_scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

if opt.use_discriminator:
	scheduler_d =lr_scheduler.ReduceLROnPlateau(opitmizer_D, 'min', patience=5)
	scheduler_d_identity =lr_scheduler.ReduceLROnPlateau(opitmizer_D_identity, 'min', patience=5)

if opt.copy_weights:
	checkpoint_file = torch.load(opt.old_model)
	model.load_state_dict(checkpoint_file['state_dict'])
start_epoch = opt.start_epoch
for epoch in range(start_epoch, 3000):
	if epoch > 0:
		checkpoint_file = torch.load("{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch-1))
		model.load_state_dict(checkpoint_file['state_dict'])
		optimizer.load_state_dict(checkpoint_file['optimizer'])

	#lossd = train(epoch, opt.num_views)
	with torch.no_grad():
		loss = val(epoch, opt.num_views)
	# checkpoint(model, epoch)
	# scheduler.step(loss)

	# if opt.use_discriminator:
	# 	scheduler_d.step(lossd[0])
	# 	scheduler_d_identity.step(lossd[1])
