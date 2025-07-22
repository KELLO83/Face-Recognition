from deepface import DeepFace
import cv2
import numpy as np
import sklearn

image = cv2.imread("1.jpg", cv2.IMREAD_COLOR)
image2 = cv2.imread("2.jpg", cv2.IMREAD_COLOR)


"""
ArcFace, Facenet512, GhostFaceNet, SFace 
"""
embedding = DeepFace.represent(image, model_name="ArcFace", enforce_detection=False , normalization="ArcFace",detector_backend='mtcnn'
                               ,align=True)
embedding2 = DeepFace.represent(image2, model_name="ArcFace", enforce_detection=False , normalization="ArcFace",\
                                detector_backend='mtcnn',
                                align=True)

for key , value in embedding[0].items():
    if key == "embedding":
        value = np.array(value)
        print(value.shape)
    else:
        print(f"{key}: {value}")



similarity = sklearn.metrics.pairwise.cosine_similarity(np.array([embedding[0]['embedding']]).reshape(-1,1)\
                                                        , np.array([embedding2[0]['embedding']])[0][0].reshape(-1,1))[0][0]
print("Similarity:", similarity)
