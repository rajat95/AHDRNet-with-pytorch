from torch.utils.data import Dataset,DataLoader
from utils import *
from imageio import imread

class MyDataset(Dataset):
	def __init__(self, scene_directory):
		list = os.listdir(scene_directory)
		self.image_list = []
		self.num = 0
		for scene in range(len(list)):
			expo_path=os.path.join(scene_directory, list[scene], 'exposure.txt')
			file_path=list_all_files_sorted(os.path.join(scene_directory, list[scene]), '.tif')
			label_path=os.path.join(scene_directory, list[scene])
			self.image_list += [[expo_path, file_path, label_path]]
			self.num = self.num + 1

	def __getitem__(self, idx):
		# Read Expo times in scene
		expoTimes = ReadExpoTimes(self.image_list[idx][0])
		# Read Image in scene
		imgs = ReadImages(self.image_list[idx][1])
		# Read label
		label = ReadLabel(self.image_list[idx][2])
		# inputs-process
		pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
		pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
		pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
		output0 = np.concatenate((imgs[0], pre_img0), 2)
		output1 = np.concatenate((imgs[1], pre_img1), 2)
		output2 = np.concatenate((imgs[2], pre_img2), 2)
		# label-process
		label = range_compressor(label)
		# argument
		crop_size = 128
		H, W, _ = imgs[0].shape
		x = np.random.randint(0, H - crop_size - 1)
		y = np.random.randint(0, W - crop_size - 1)

		im1 = output0[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
		im2 = output1[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
		im3 = output2[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
		im4 = label[x:x + crop_size, y:y + crop_size, :].astype(np.float32).transpose(2, 0, 1)
		im1 = torch.from_numpy(im1)
		im2 = torch.from_numpy(im2)
		im3 = torch.from_numpy(im3)
		im4 = torch.from_numpy(im4)

		sample = {'input1': im1, 'input2': im2, 'input3': im3, 'label': im4}

		return sample

	def __len__(self):
		return self.num




