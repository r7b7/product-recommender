# product-recommender-poc
This app implements content-based filtering on synthetic data and collaborative filtering model is trained on movie rating data provided by symphony.
- API is implemented using FAST API. This app exposes one GET endpoint.
- User-id passed in GET enpoint is not used anywhere in the code, currently its given for decoration purposes.
- Purpose of this app is to understand how basic recommendation systems work wuthout using neural networks.

## Install following libraries to run this app
1. <python-version> -m pip install fastapi
2. <python-version> -m pip install uvicorn
3. <python-version> -m pip install surprise
4. <python-version> -m pip install scikit-learn
5. <python-version> -m pip install pandas

## Run the app
run the following command in terminal 
```<python-version> -m uvicorn main:recommendApp --reload```

Test the app using this endpoint - http://localhost:8000/recommendation/1234
