import torch
from PIL import Image
from torchvision import transforms
from mnist import LeNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


def get_files(directory):
	return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
			if os.path.isfile(os.path.join(directory, f))]


def predictdir(img_dir):
	net = LeNet()
	net.load_state_dict(torch.load('./model.pkl'))
	net.eval()
	torch.no_grad()
	file = get_files(img_dir)
	for i, img_path in enumerate(file):
		img = Image.open(img_path)
		img = transform(img).unsqueeze(0)
		img_ = img.to(device)
		outputs = net(img_)
		_, predicted = torch.max(outputs, 1)
		print(img_path, classes[predicted[0]])


if __name__ == '__main__':
	dir_path = './data/MNIST/mnist_train/1'
	predictdir(dir_path)