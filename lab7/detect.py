import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img1 = cv2.imread('test1.jpg')
img2 = cv2.imread('test2.jpg')
img3 = cv2.imread('DSCF0112.JPG')


#Resize the image
# resized_img = cv2.resize(img, (300, 300))
ratio = float(500) / img1.shape[1]
ratio1 = float(500) / img3.shape[1]

resized_img1 = cv2.resize(img1, (0, 0), fx=ratio, fy=ratio)
resized_img2 = cv2.resize(img2, (0, 0), fx=ratio, fy=ratio)
resized_img3 = cv2.resize(img3, (0, 0), fx=ratio1, fy=ratio1)


# Convert into grayscale
gray1 = cv2.cvtColor(resized_img1, cv2.COLOR_BGRA2GRAY)
gray2 = cv2.cvtColor(resized_img2, cv2.COLOR_BGRA2GRAY)
gray3 = cv2.cvtColor(resized_img3, cv2.COLOR_BGRA2GRAY)

# Detect faces
faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4)
faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)
faces3 = face_cascade.detectMultiScale(gray3, 1.1, 4)


# Draw rectangle around the faces
for (x, y, w, h) in faces1:
    cv2.rectangle(resized_img1, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces2:
    cv2.rectangle(resized_img2, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces3:
    cv2.rectangle(resized_img3, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
print("there are %d faces in the test image 1" %(faces1.shape[0]))
print("there are %d faces in the test image 2" %(faces2.shape[0]))
print("there are %d faces in the test image 3" %(faces3.shape[0]))

cv2.imshow('resized_img1', resized_img1)
cv2.imshow('resized_img2', resized_img2)
cv2.imshow('resized_img3', resized_img3)

cv2.waitKey()