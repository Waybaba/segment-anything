# Import pyrootutils and setup root
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=[".env"])

# Import Hydra
import hydra
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry



class SegmentAnythingModelWrapper:
	def __init__(self, **kwargs):
		from segment_anything import sam_model_registry
		self.model_type = kwargs["model_type"]
		self.checkpoint = kwargs["checkpoint"]
		self.device = kwargs["device"]
		self.net = sam_model_registry[kwargs["model_type"]](checkpoint=kwargs["checkpoint"])
		self.net.to(kwargs["device"])
	
	def pred_with_mask(self, image, mask):
		return self.kwargs["sam"].predict(image, mask)

# utils

def gen_rand_reg_mask_input(image):
	"""
	Generate random mask input the same size as image
	the mask should be a rectangle with random size and position
	return mask, (x, y, w, h)

	Args:
		image: torch tensor, 3xHxW
	"""
	# generate random size
	h, w = image.shape[-2:]
	h, w = h // 4, w // 4
	x = np.random.randint(0, w)
	y = np.random.randint(0, h)
	# w_ = np.random.randint(0, w-x)
	# h_ = np.random.randint(0, h-y)
	w_, h_ = 5, 5
	
	# rectangle
	mask = np.zeros((1, 1, h, w), dtype=np.float32)
	mask -= 8.0
	mask[:, :, y:y+h_, x:x+w_] = 1.

	# rectangle (torch)
	mask = torch.zeros((1, 1, h, w), dtype=torch.float32)
	mask[:, :, y:y+h_, x:x+w_] = 1.
	
	# random
	# mask = torch.randn((1, 1, h, w), dtype=torch.float32)
	
	return mask, (x, y, w_, h_)

def normalize_to_0_255(x):
	x = x - x.min()
	x = x / x.max()
	x = x * 255.
	return x

def show_anns(anns):
	if len(anns) == 0:
		return
	sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
	ax = plt.gca()
	ax.set_autoscale_on(False)

	img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
	img[:,:,3] = 0
	for ann in sorted_anns:
		m = ann['segmentation']
		color_mask = np.concatenate([np.random.random(3), [0.85]])
		img[m] = color_mask
	ax.imshow(img)

def automask_to_bwh(automask):
	"""
	Sort the by the size of the area then return the bwh mask
	input: [{
		"segmentation": np.array, # 1024,2048
		"area": int,
		...
	}]
	output: b, w, h
	"""
	w, h = automask[0]["segmentation"].shape
	automask = sorted(automask, key=lambda x: x["area"], reverse=True)
	automask_bwh = np.zeros((w, h), dtype=np.uint8)
	
	for i, mask in enumerate(automask):
		indices = np.where(mask["segmentation"] == 1)
		automask_bwh[indices] = i + 1	
	automask_bwh = automask_bwh.astype(np.uint8)
	return automask_bwh

def automask_to_bnwh(automask):
	"""
	Sort the by the size of the area then return the bwh mask
	input: [{
		"segmentation": np.array, # 1024,2048
		"area": int,
		...
	}]
	output: b, n, w, h
	"""
	w, h = automask[0]["segmentation"].shape
	automask = np.array([mask["segmentation"] for mask in automask])
	automask = automask.astype(np.uint8)
	return automask


def get_all_paths(arg_path):
	if type(arg_path) == str: arg_path = [arg_path]
	paths = []
	for parent_path in arg_path:
		parent_path = Path(parent_path)
		if parent_path.is_file():
			paths.append(parent_path)
		else:
			for path in parent_path.iterdir():
				if path.is_file():
					paths.append(path)
	return paths

### main - point_predict
def point_predict(args):
	args = hydra.utils.instantiate(args)
	args.input = root / args.input # is a folder
	args.output = root / args.output

	# load model
	model = args.model

	# load image as numpy array
	for image_path in args.input.iterdir():
		image = cv2.imread(str(image_path))
		# parse name
		img_name = image_path.stem
		image_tensor = torch.from_numpy(image).to(args.device)
		# channel first
		image_tensor = image_tensor.permute(2, 0, 1)
		# resize to 1024x1024 (float then back)
		image_tensor = image_tensor.float()
		image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False)
		image_tensor = image_tensor.squeeze(0).byte()

		# Prepare the batched_input as required by the Sam model
		point_coords = torch.tensor([[[100, 50], [200, 300]]])  # (B, N, 2), where B is batch size and N is number of points. To predict for B different classes, we need to use B
		point_labels = torch.tensor([[0, 1]])  # (B, N)
		mask_in, mask_pos = gen_rand_reg_mask_input(image_tensor)
		mask_in = mask_in.to(args.device) # Bx1xHxW
		batched_input = [
			{
				'image': image_tensor,  # The image tensor
				'original_size': (image.shape[0], image.shape[1]),
				## point
				# 'point_coords': point_coords.to(args.device),
				# 'point_labels': point_labels.to(args.device),
				## Add other required keys like point_coords, point_labels, boxes, and mask_inputs
				"mask_inputs": mask_in,
			}
		]

		multimask_output = False  # Set to True if you want multiple disambiguating masks

		# Use the forward function of the Sam model.
		results = model.net(batched_input, multimask_output)
		# results: [{
		# 	"masks": (B, N, H, W),
		# 	"iou_predictions": (B, N),
		# 	"low_res_logits": (B, N, 256, 256),
		# }]

		# save 
		for result in results:
			for mask_idx in range(len(result["masks"])):
				mask_out = result["masks"][mask_idx]
				mask_out = mask_out.permute(1, 2, 0).float().cpu().numpy()*255.  # Convert mask tensor to a numpy array
				# save img+mask_in+point (plot mask on img)
				mask_in_path = str(args.output / f"{img_name}_{mask_idx}_mask_in.png")
				mask_in = mask_in # 1x1xHxW
				mask_in = normalize_to_0_255(mask_in)
				cv2.imwrite(mask_in_path, mask_in.squeeze(0).squeeze(0).cpu().numpy()*255.)
				# save img+mask_out (plot mask on img)
				mask_out = mask_out # HxWx1
				mask_out_path = str(args.output / f"{img_name}_{mask_idx}_mask_out.png")
				cv2.imwrite(mask_out_path, result["masks"][mask_idx].squeeze(0).cpu().numpy()*255.)
				# save logits
				mask_logits = result["low_res_logits"][mask_idx]
				logits_path = str(args.output / f"{img_name}_{mask_idx}_logits.png")
				mask_logits = normalize_to_0_255(mask_logits)
				cv2.imwrite(logits_path, mask_logits.squeeze(0).cpu().numpy().astype(np.uint8))

### main - auto_mask
def auto_mask(args):
	args = hydra.utils.instantiate(args)
	# init folder
	args.output = Path(args.output)
	(args.output / "plotmask").mkdir(exist_ok=True, parents=True)
	(args.output / "automask_bnwh").mkdir(exist_ok=True, parents=True)
	(args.output / "automask_bwh").mkdir(exist_ok=True, parents=True)
	(args.output / "feats_bcwh").mkdir(exist_ok=True, parents=True)
	
	# clear output folder
	# for f in (args.output / "plotmask").iterdir():
		# f.unlink()
		# print(f"removed {f}")
	
	# load model
	sam_model = sam_model_registry[args.sam.model_type](checkpoint=args.sam.checkpoint)
	mask_generator = args.mask_generator(sam_model)
	
	# main
	image_paths = get_all_paths(args.input)
	for i, image_path in enumerate(image_paths):
		print(f"[{i+1}/{len(image_paths)}] start {image_path.stem} ... ")
		image = cv2.imread(str(image_path))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		mask_generator.predictor.set_image(image)
		feats = mask_generator.predictor.get_image_embedding()
		
		# save mask data
		# masks = mask_generator.generate(image)
		# np.save(str(args.output / "automask_bnwh" / f"{image_path.stem}.npy"), automask_to_bnwh(masks))
		# np.save(str(args.output / "automask_bwh" / f"{image_path.stem}.npy"), automask_to_bwh(masks))
		np.save(str(args.output / "feats_bcwh" / f"{image_path.stem}.npy"), feats.numpy())

		# plot and save /plotmask/... # for visualization
		# plt.figure(figsize=(20, 20))
		# plt.imshow(image)
		# show_anns(masks)
		# plt.axis('off')
		
		# plot fig
		# plt.savefig(str(args.output / "plotmask" / f"{image_path.stem}.png"))
		# print(f"saved {image_path.stem}.png")
		# plt.clf()
		# plt.close()

		# save /auto_mask/... # for further use


	print("finish")
	


@hydra.main(config_path=root/"configs", config_name="pure_model.yaml")
def my_app(args):
	if args.mode == "point_predict":
		point_predict(args)
	elif args.mode == "auto_mask":
		auto_mask(args)
	else:
		raise NotImplementedError(f"mode {args.mode} not implemented.")
	print("finished")


if __name__ == "__main__":
	my_app()
