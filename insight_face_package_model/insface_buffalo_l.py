
from insightface.app import FaceAnalysis
import cv2
import sklearn
import sklearn.metrics

"""

HEAD : ArcFace 

antelopev2	ResNet-100	Glint360K	407MB	최고 성능
buffalo_l	ResNet-50	WebFace600K	326MB	높은 성능
buffalo_m	ResNet-50	WebFace600K	313MB	중간 성능
buffalo_s	MobileFaceNet	WebFace600K	159MB	빠른 속도
buffalo_sc	MobileFaceNet	WebFace600K	16MB	초경량

"""

app = FaceAnalysis(name="buffalo_l",
     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
     root=".",)

app = FaceAnalysis(name="antelopev2", 
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                   root=".")


image = cv2.imread('man/1.jpg')
compare = cv2.imread('man/2.jpg')

if image is None or compare is None:
    raise ValueError("이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")

app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(image)

c = app.get(compare)

value = faces[0]
value2 = c[0]
embedding_vector = value['embedding']
compare_vector = value2['embedding']

embedding_vector = embedding_vector.reshape(1, -1)
compare_vector = compare_vector.reshape(1, -1)

print(embedding_vector.shape)
print('='*10)



print("유사도 : ",sklearn.metrics.pairwise.cosine_similarity(embedding_vector, compare_vector)[0][0])