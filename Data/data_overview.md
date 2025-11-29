- # Data
	- https://huggingface.co/datasets/mingyy/chinese_landscape_paintings - 5000 images - DROPPED DUE TO LOW PAINTING QUALITY
		- Target - generated landscape target
		- filename
		- image_caption
		- source - original painintings/sketches
		- hed -  Edge detection
	- https://huggingface.co/datasets/WUYONGF/chinese_painting - 30 high res images - WILL BE USED FOR HIDDEN TEST (FINAL TEST)
		- image - original sketches
		- text - caption
	- https://www.kaggle.com/datasets/myzhang1029/chinese-landscape-painting-dataset - 2000 images - DROPPED DUE TO LOW PAINTING QUALITY
		- image
	- https://www.kaggle.com/datasets/arnaud58/landscape-pictures - 4000 landscape images - USE AS THE DOMAIN B: LANDSCAPE PHOTO TRAINING DATA
	- https://www.kaggle.com/datasets/helloeyes/chinese-landscape-painting2photo - 2192 quality chinese landscape paintings and landscape photos for training. ==Downloaded and tracked under folder `xue2020_dataset` in the repo. USE `xue2020_dataset` AS DOMAIN A: CHINESE LANDSCAPE PAINTING FOR TRAINING==
		- Test A - paintings
> `data_processing.ipynb` NOW ONLY DOWNLOAD AND PROCESS https://huggingface.co/datasets/WUYONGF/chinese_painting AND https://www.kaggle.com/datasets/arnaud58/landscape-pictures. `xue2020_dataset` ARE ALREADY DOWNLOAED, PROCESSED AND TRACKED IN THE REPO.

- # Problem
	-
	  > “How much do explicit semantic constraints improve sketch→photo realism and structure, over a strong cycle (unpaired) GAN baseline?”  