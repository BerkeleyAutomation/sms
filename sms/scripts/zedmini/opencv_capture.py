import cv2

idx = 0
cam = cv2.VideoCapture(0)

result, image = cam.read()

if result:
    cv2.imwrite(f"img{idx}.png", image)
    print(f"Image saved to img{idx}.png")