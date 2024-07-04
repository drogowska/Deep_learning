import cv2

from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # 2 classes: Male and Female
        )


    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc(x)
        return x



def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(100, 100), max_size=(200, 200))

    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']
        boxes.append((x, y, w, h))
    return boxes


def draw(frame, boxes):
    for x, y, w, h in boxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)



if __name__ == '__main__':
    # file = lbp_frontal_face_cascade_filename()
    file =  "E:\\Folder\\studia\\Informatyka Stosowana II st\\semersr 2\\UG\\zd1\\face.xml" #"./face.xml"
    detector = Cascade(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenderCNN().to(device)
    model.load_state_dict(torch.load(r'E:\Folder\studia\Informatyka Stosowana II st\semersr 2\UG\DeepLearning\gender_classification_model.pth', map_location=torch.device('cpu')))
    model.eval()

    glas = models.resnet18(pretrained=True)
    num_ftrs = glas.fc.in_features
    glas.fc = nn.Linear(num_ftrs, 1)  # 2 atrybuty do przewidywania
    glas.load_state_dict(torch.load(r'E:\Folder\studia\Informatyka Stosowana II st\semersr 2\UG\DeepLearning\model_attributes_balanced.pth', map_location=torch.device('cpu')))
    glas.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # glas.to(device)
    transform_gals = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(0)
    skip = 5
    i = 0
    boxes = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 50)  # Position of the text
    font_scale = 0.5
    color = (0, 0, 0)  # Text color in BGR format (white in this case)
    thickness = 1
    while (True):
        ret, frame = cap.read()
        if i % skip == 0:
            boxes = detect(frame, detector)
        for x, y, w, h in boxes: 
            roi = frame[y-h:y+2*h, x-w:x+2*w]
            pil_image = Image.fromarray(roi)
            frame_t = transform(pil_image)
            frame_g = transform_gals(pil_image)
            gender = model(frame_t)
            # print(frame_g)
            glassses = glas(frame_g.unsqueeze(0))
            glassses = torch.sigmoid(glassses)
            # print(boxes)
            draw(frame, boxes)
            if gender >0.5: 
                text='Gender: Female'
            else: 
                text = 'Gender: male'
            cv2.putText(frame, text, (x,y-10), font, font_scale, color, thickness)
            if glassses >0.5: 
                text='Glasses: Yes'
            else: 
                text = 'Glasses: No'
            cv2.putText(frame, text, (x, y-25), font, font_scale, color, thickness)
        cv2.imshow('Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    # cap.release()
    # cv2.destroyAllWindows()
