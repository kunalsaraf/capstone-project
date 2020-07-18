import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
global model
MODEL_PATH = './capstone_inner/model/model.h5'
model = load_model(MODEL_PATH)
    
def get_label(file_path):
    processed_image = process_image(file_path)
    classes = model.predict(processed_image)
#    print(classes)
    predicted_labels = classes.tolist()[0]
    denomination=[200, 2000, 500]
    percentage_labels=[0.0 ,0.0 ,0.0]
    sum_total=sum(predicted_labels)
    for i in range(0,len(predicted_labels)):
        percentage_labels[i]=round((predicted_labels[i]/sum_total)*100,2)
    result_dict = dict(zip(denomination,percentage_labels))
    print(result_dict)
    denomination.clear()
    percentage_labels.clear()
    for i in sorted(result_dict):
        denomination.append(str(i))
        percentage_labels.append(result_dict[i])
        print((i, result_dict[i]), end =" ") 
    plt.bar(denomination, percentage_labels, color = ['red', 'green', 'yellow'])
    plt.title('Probability of each denomination')
    plt.xlabel('Denominations')
    plt.ylabel('Percentage (%)')
    plt.grid(b=True)
    plt.savefig("./capstone_inner/static/result/currency.jpg")
    plt.close()
    # plt.show()

def process_image(file_path):
    # img = load_img(file_path, target_size=(256, 256, 3))
    img = load_img(file_path, target_size=(128, 96, 3))
    img = img_to_array(img)
    # img = img.reshape(1,256,256,3).astype('float')
    img = img.reshape(1,128,96,3).astype('float')
    img /= 255
    return img

if __name__=="__main__":
    get_label('./model/demo3.jpg')
