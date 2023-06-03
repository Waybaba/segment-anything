# Import pyrootutils and setup root
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=[".env"])

# Import Hydra
import hydra
import torch
import cv2

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



@hydra.main(config_path=root/"configs", config_name="pure_model.yaml")
def my_app(args):
	args = hydra.utils.instantiate(args)
	args.input = root / args.input # is a folder
	args.output = root / args.output

	# load model
	model = args.model

	# load image as numpy array
	for image_path in args.input.iterdir():
		image = cv2.imread(str(image_path))

		image_tensor = torch.from_numpy(image).to(args.device)
		# channel first
		image_tensor = image_tensor.permute(2, 0, 1)

		# Prepare the batched_input as required by the Sam model
		batched_input = [
			{
				'image': image_tensor,  # The image tensor
				'original_size': (image.shape[0], image.shape[1]),
				# Add other required keys like point_coords, point_labels, boxes, and mask_inputs
			}
		]

		multimask_output = False  # Set to True if you want multiple disambiguating masks

		# Use the forward function of the Sam model
		results = model.net(batched_input, multimask_output)

		# Process the results as needed and save them

	print("finished")

if __name__ == "__main__":
	my_app()
