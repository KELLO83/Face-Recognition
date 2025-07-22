
from insightface.app import FaceAnalysis
import cv2
import sklearn
import sklearn.metrics
app = FaceAnalysis(name="antelopev2",
     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
     root=".",)

print("Preparing face analysis app...")


app.prepare(ctx_id=0, det_size=(640, 640))


image = cv2.imread('1.jpg')
compare = cv2.imread('2.jpg')

faces = app.get(image)

c = app.get(compare)

value = faces[0]
value2 = c[0]
embedding_vector = value['embedding']
compare_vector = value2['embedding']

embedding_vector = embedding_vector.reshape(1, -1)
compare_vector = compare_vector.reshape(1, -1)
for k , v in value.items():
    print("key : ",k)

print(embedding_vector.shape)
print('='*10)



print("유사도 : ",sklearn.metrics.pairwise.cosine_similarity(embedding_vector, compare_vector)[0][0])