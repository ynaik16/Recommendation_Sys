
"""


"""
import sys
import csv
import json
import time
import math
import random
import pyspark
import numpy as np
import xgboost as xgb
from pyspark import SparkContext


sc = SparkContext.getOrCreate()

sc.setLogLevel("ERROR")

## Defining the path to the input folder.
folder = sys.argv[1]

## Defining the path to the testing data. 
input_test_file = sys.argv[2]

## Output file path.
output_file = sys.argv[3]

## Reading the training data.
input_train_file = folder + "/yelp_train.csv"

## Loading the raw data.
#input_test_file = "/content/drive/MyDrive/Colab Notebooks/DS 553/project/Copy of yelp_val.csv"
#input_train_file = "/content/drive/MyDrive/Colab Notebooks/DS 553/project/Copy of yelp_train.csv"

train_file = sc.textFile(input_train_file)
test_file = sc.textFile(input_test_file)

## Filter out lines.
first_line_train = train_file.first()
first_line_val = test_file.first()

s = time.time()

## Loading the CSV data.
train_rdd = train_file.filter(lambda x: x != first_line_train).map(lambda f: f.split(","))
test_rdd = test_file.filter(lambda x: x != first_line_val).map(lambda f: f.split(","))

dict1 = train_rdd.map(lambda f : (f[0], f[1])).groupByKey().mapValues(set).collectAsMap()    # userBusinessMap

dict2 = train_rdd.map(lambda f : (f[1], f[0])).groupByKey().mapValues(set).collectAsMap()                   # businessUserMap

stars_dict = train_rdd.map(lambda f : ((f[0], f[1]), float(f[2]))).collectAsMap()            # userBusinessRateDict

dict3 = train_rdd.map(lambda f : (f[1], (float(f[2])))).combineByKey(lambda val: (val,1), lambda x,val : (x[0] + val, x[1] + 1), lambda x,y: (x[0] + y[0], x[1] + y[1] )).mapValues(lambda x: x[0]/x[1]).collectAsMap()  #business Avergae



def get_list(X):
    uid, bid = X[0], X[1]

    return uid, bid

def get_corated(uid, userBusiness):
    corated = userBusiness[uid]

    return corated

def compute_pearsons(uid, bid, corated_items, userBusiness, businessUser, ubRating):

    similarity_index = []

    for item in corated_items:

        x = ubRating[(uid, item)]

        corated_user = businessUser[bid].intersection(businessUser[item])

        if (len(corated_user) == 0 or len(corated_user) == 1):
            variance = abs(dict3[bid] - dict3[item])
			
            if (0 <= variance <= 1):
                similarity_index.append([1.0, 1.0*x, 1.0])
                continue

            elif (1 < variance <= 2):
                similarity_index.append([0.5, 0.5*x, 0.5])
                continue				

            else:
                similarity_index.append([0.0, 0.0, 0.0])	
                continue


        temp1 = []
        temp2 = []


        for users in corated_user:


            temp1.append(ubRating[(users, bid)])
            temp2.append(ubRating[(users, item)])

        temp1array = np.asarray(temp1, dtype = np.float32)
        temp2array = np.asarray(temp2, dtype = np.float32)

        b1 = temp1array - dict3[bid]
        b2 = temp2array - dict3[item]


        top = np.sum(np.multiply(b1, b2))
        bottom = np.sqrt(np.sum(b1 ** 2)) * np.sqrt(np.sum(b2 ** 2))

        if (top == 0 or top == 0):
            similarity_index.append([0.0, 0.0, 0.0])	
            continue

        frac = top/bottom

        if (frac < 0):
            continue			

        similarity_index.append([frac, frac * x, abs(frac)])

  
    return similarity_index

def check_prediction(pearson, N):

    ordered = sorted(pearson, key = lambda x : -x[0])
    ordered = ordered[ : N]
    matrix = np.array(ordered)
    total = matrix.sum(axis = 0)
 
    return total

def get_pred(chunk, userBusiness, businessUser, ubRating, N):

    uid, bid = get_list(chunk)


    if (bid not in businessUser.keys()):
        return 3.0


    if (uid not in userBusiness.keys()):
        return 3.0

    corated_items = get_corated(uid, userBusiness)

    pearson = compute_pearsons(uid, bid, corated_items, userBusiness, businessUser, ubRating)

    if (len(pearson) == 0):
        return 3.0

    prediction = check_prediction(pearson, N)


    if (prediction[1] == 0.0 or prediction[2] == 0.0):
        return 3.0			

    out = prediction[1] / prediction[2]


    return out









N = 10

test_pred = test_rdd.map(lambda chunk : get_pred(chunk, dict1, dict2, stars_dict, N)).collect()

test_pred = np.asarray(test_pred, dtype=np.float32)


#user_path = "/content/drive/MyDrive/Colab Notebooks/DS 553/project/Copy of user.json"

user_path = folder + '/user.json'
user_file = sc.textFile(user_path)

## Loading the data in json format.
user_rdd = user_file.map(lambda f: json.loads(f)).map(lambda f : ((f['user_id'], (f['review_count'], f['average_stars'])))).collectAsMap()

# ## Reading the business information file.
business_path = folder + '/business.json'
#business_path = "/content/drive/MyDrive/Colab Notebooks/DS 553/project/Copy of business.json"
business_file = sc.textFile(business_path)


business_rdd = business_file.map(lambda f: json.loads(f)).map(lambda f : ((f['business_id'], (f['review_count'], f['stars'])))).collectAsMap()

def get_info(feature, id):
    info = feature[id]

    return info


def extract_details(chunk, user_rdd, business_rdd, testInput = False):

	
    if (testInput):
        uid, bid, stars = chunk[0], chunk[1], -1.0
    else:
        uid, bid, stars = chunk[0], chunk[1], chunk[2]

	## Case of cold starts.
    if (uid not in user_rdd.keys() or bid not in business_rdd.keys()):
        return [uid, bid, None, None, None, None, None]

    user_review, user_star = get_info(user_rdd, uid)

    business_review, business_star = get_info(business_rdd, bid)

    return [uid, bid, float(user_review), float(user_star), float(business_review), float(business_star), float(stars)]

def extract_features(yelp_matrix):

    x = yelp_matrix[:, 2 : -1]
    y = yelp_matrix[:, -1]


    x = np.array(x, dtype = 'float')
    y = np.array(y, dtype = 'float')

    return x, y

def convert_to_matrix(x):
    ax = np.array(x)
    return ax

yelp_train = train_rdd.map(lambda chunk : extract_details(chunk, user_rdd, business_rdd)).collect()

yelp_matrix = convert_to_matrix(yelp_train)

train_feature_x, train_feature_y = extract_features(yelp_matrix)


model = xgb.XGBRegressor(objective = 'reg:linear')
model.fit(train_feature_x,train_feature_y)

yelp_test = test_rdd.map(lambda chunk : extract_details(chunk, user_rdd, business_rdd, True)).collect()

test_matrix = convert_to_matrix(yelp_test)

test_feature_x, test_feature_y = extract_features(test_matrix)

predictions = model.predict(test_feature_x)

Result = 0.99 * predictions + 0.01 * test_pred

## Concatenate the names and predictions.
total_result = np.c_[test_matrix[:, : 2], Result]

output = 'user_id, business_id, prediction\n'

for item in total_result:
    output+= str(item[0])+','+ str(item[1])+','+ str(item[2])+'\n'

with open(output_file, 'w') as fo:
    fo.write(output)
    
e = time.time() - s

print("Duration: {} sec".format(e))















################################################################################################################
"""
target_vals = test_rdd.map(lambda x: (float(x[2]))).collect()

count_1 =0
count_2 =0
count_3 =0
count_4=0
count_5=0

for i in range(len(target_vals)):

    variance = abs(predictions[i] - target_vals[i])
			
    if (0 >= variance < 1):
        count_1 +=1

    elif (1>= variance < 2):
        count_2 +=1				

    elif (2 >= variance < 3):
        count_3 +=1
    
    elif (3 >= variance < 4):
        count_4 +=1	
    
    elif (4 >= variance):
        count_5 +=1	

  
print(">=0 and <1: ", count_1)
print(">=1 and <2: ", count_2)
print(">=2 and <3: ", count_3)
print(">=3 and <4: ", count_4)
print(">=4: ", count_5)

MSE = np.square(np.subtract(Result,test_pred)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)


Duration = 183.29938650131226 sec

"""