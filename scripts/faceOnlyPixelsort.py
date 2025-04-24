import mediapipe
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from typing import Callable


img = cv2.imread("scripts/images/stockimage2.jpg")
 
fig = plt.figure(figsize = (8, 8))
plt.axis('off')
plt.imshow(img[:, :, ::-1]) # Flips from BGR to RGB format
plt.show()

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
landmarks = results.multi_face_landmarks[0]

face_oval = mp_face_mesh.FACEMESH_FACE_OVAL


df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])

routes_idx = []
 
p1 = df.iloc[0]["p1"]
p2 = df.iloc[0]["p2"]
 
for i in range(0, df.shape[0]):
     
    #print(p1, p2)
     
    obj = df[df["p1"] == p2]
    p1 = obj["p1"].values[0]
    p2 = obj["p2"].values[0]
     
    route_idx = []
    route_idx.append(p1)
    route_idx.append(p2)
    routes_idx.append(route_idx)
 
# -------------------------------
 
for route_idx in routes_idx:
    print(f"Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point")

routes = []
 
for source_idx, target_idx in routes_idx:
     
    source = landmarks.landmark[source_idx]
    target = landmarks.landmark[target_idx]
         
    relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
    relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
 
    #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
     
    routes.append(relative_source)
    routes.append(relative_target)

#--------------------------------
# print("Not crashed yet")

 
mask = np.zeros((img.shape[0], img.shape[1]))
mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
mask = mask.astype(bool)
  
out = np.zeros_like(img)
out[mask] = img[mask]
# print("Not crashed yet2")

fig = plt.figure(figsize = (15, 15))
plt.axis('off')
plt.imshow(out[:, :, ::-1])
cv2.imwrite((f'scripts/images/stockoutface2.png'), out)

# This sort algorthim is made by GregT
def sort_pixels(image: Image, value: Callable, condition: Callable, rotation: int = 0) -> Image:
    pixels = np.rot90(np.array(image), rotation)
    values = value(pixels)
    edges = np.apply_along_axis(lambda row: np.convolve(row, [-1, 1], 'same'), 0, condition(values))
    intervals = [np.flatnonzero(row) for row in edges]

    for row, key in enumerate(values):
        order = np.split(key, intervals[row])
        for index, interval in enumerate(order[1:]):
            order[index + 1] = np.argsort(interval) + intervals[row][index]
        order[0] = range(order[0].size)
        order = np.concatenate(order)

        for channel in range(3):
            pixels[row, :, channel] = pixels[row, order.astype('uint32'), channel]

    return Image.fromarray(np.rot90(pixels, -rotation))

image_me_sorted = sort_pixels(Image.open('scripts/images/stockoutface2.png'),
            lambda pixels: np.average(pixels, axis=2) / 255,
            lambda lum: (lum > 2 / 6) & (lum < 4 / 6), 1) #.save('scripts/images/outFaceFinal.png')

if image_me_sorted is not None:
    image_me_sorted.save('scripts/images/stockout2.png')
    pixelsorted = np.array(image_me_sorted) 
else:
    print("Error: `sort_pixels` returned None")
    exit()
# print("Not crashed yet3")

print("Running dumb merge")
def dumbmerge(baseimage, faceimage, pixelsorted):
    height, width, _ = baseimage.shape
    for y in range(height):
        print( round(y  / height, 2)*100, "% done")
        for x in range(width):
            # Access pixel at (x, y)
            if not np.all(faceimage[y, x] == [0, 0, 0]):
                # Check if the pixel sorted image is NOT black
                if not np.all(pixelsorted[y, x] == [0, 0, 0]):
                    baseimage[y, x] = pixelsorted[y, x] 
    return baseimage

# cv2.imwrite('scripts/images/ps.png',image_me_sorted)
pixelsorted = np.array(image_me_sorted)  # Convert PIL image to NumPy array
pixelsorted = cv2.imread("scripts/images/stockout2.png")


finalimage = dumbmerge(img, out, pixelsorted)
cv2.imwrite('scripts/images/stockFinal2.png',finalimage)



# def sort_pixels2(image):
#     height, width, _ = image.shape
#     sorted_image = np.copy(image)

#     for y in range(height):
#         row = image[y]
#         mask = np.any(row != [0, 0, 0], axis=1)
        
#         non_black_pixels = row[mask]

#         if len(non_black_pixels) > 0:  # Ensure there are pixels to sort
#             sorted_pixels = sorted(non_black_pixels, key=lambda pixel: np.sum(pixel))
#             sorted_image[y][mask] = sorted_pixels

#     return sorted_image

# Apply Pixel Sorting to Face-Only Image
# faceOnlyTest = sort_pixels2(out)

# cv2.imwrite('scripts/images/outFaceFinal.png',faceOnlyTest)