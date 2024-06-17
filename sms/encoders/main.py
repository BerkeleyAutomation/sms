from openclip_encoder import *
from matplotlib import pyplot as plt
from PIL import Image
import time
import numpy as np

def main():
    
    # img = open('DSC_1625.jpg')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.open('dog_in_me.jpeg').convert('RGB')

    w, h = img.size
    print("Original Image Size", w, h)
    print("Pixel Count", w*h)

    # img = np.array(list(img.getdata()))
    # img = img.reshape((h,w,3))

    # img = torch.tensor(img, device=device).permute(2,0,1).unsqueeze(0).float().half()
    
    clipencoder = OpenCLIPNetworkConfig(device=device).setup()

    # import pdb; pdb.set_trace()
    img = clipencoder.preprocess(img).unsqueeze(0).half().to(device)    
    start = time.time()
    imgenc = clipencoder.model.encode_image(img).float()
    elapsed = (time.time() - start)
    print(f"Elapsed time: {elapsed}(s)")
    print(f"Frequency: {1/elapsed}(fps)")
    
    import pdb; pdb.set_trace()


    # plt.imshow(recon)
    # plt.show()


if __name__ == "__main__":
    main()
